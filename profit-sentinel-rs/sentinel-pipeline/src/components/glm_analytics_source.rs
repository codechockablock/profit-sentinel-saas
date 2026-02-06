use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use sentinel_vsa::bundling::{bundle_inventory_batch, InventoryRow};
use sentinel_vsa::codebook::Codebook;
use sentinel_vsa::primitives::VsaPrimitives;
use sentinel_vsa::similarity;

use crate::inventory_loader::InventoryRecord;
use crate::issue_classifier::classify_and_aggregate;
use crate::source::Source;
use crate::types::{AgentQuery, IssueCandidate, UserRole};

/// Default hypervector dimensions for VSA engine.
const VSA_DIM: usize = 1024;
/// Default seed for reproducible VSA results.
const VSA_SEED: u64 = 42;
/// Similarity threshold for boosting confidence of related issues.
const CLUSTER_THRESHOLD: f64 = 0.7;
/// Maximum SKUs to sample per group for pairwise similarity.
/// 50 SKUs = 1,225 pairs (vs 26M for 7K SKUs). Stride-based sampling
/// preserves statistical representation while keeping cost O(1).
const MAX_SIMILARITY_SAMPLE: usize = 50;

/// Source that produces `IssueCandidate` items from real inventory data
/// via VSA (Vector Symbolic Architecture) analysis.
///
/// The source:
/// 1. Takes inventory records grouped by store
/// 2. Runs VSA bundling to encode each row as a hypervector
/// 3. Classifies issues by signal thresholds (matching the VSA primitives)
/// 4. Computes dollar impact from actual inventory data
/// 5. Uses VSA similarity search to cluster related anomalies
pub struct GlmAnalyticsSource {
    /// Inventory records to analyze.
    records: Vec<InventoryRecord>,
    /// Shared VSA primitives (11 anomaly signal vectors).
    primitives: Arc<VsaPrimitives>,
    /// Shared codebook for SKU → phasor vector mapping.
    codebook: Arc<Codebook>,
}

impl GlmAnalyticsSource {
    /// Create a new source with inventory data.
    pub fn new(records: Vec<InventoryRecord>) -> Self {
        Self {
            records,
            primitives: Arc::new(VsaPrimitives::new(VSA_DIM, VSA_SEED)),
            codebook: Arc::new(Codebook::new(VSA_DIM, VSA_SEED)),
        }
    }

    /// Create a new source with custom VSA dimensions and seed.
    pub fn with_config(records: Vec<InventoryRecord>, dimensions: usize, seed: u64) -> Self {
        Self {
            records,
            primitives: Arc::new(VsaPrimitives::new(dimensions, seed)),
            codebook: Arc::new(Codebook::new(dimensions, seed)),
        }
    }

    /// Run VSA analysis and issue classification for a set of store IDs.
    fn analyze(&self, store_ids: &[String], timestamp: &str) -> Vec<IssueCandidate> {
        let mut all_candidates = Vec::new();

        for store_id in store_ids {
            // Filter records for this store
            let store_records: Vec<&InventoryRecord> = self
                .records
                .iter()
                .filter(|r| r.store_id == *store_id)
                .collect();

            if store_records.is_empty() {
                continue;
            }

            // Convert to InventoryRow for VSA
            let rows: Vec<InventoryRow> = store_records
                .iter()
                .map(|r| r.to_inventory_row())
                .collect();

            // Run VSA bundling — this validates the data through the full
            // VSA pipeline and produces hypervectors for similarity analysis.
            let bundles = bundle_inventory_batch(&rows, &self.primitives, &self.codebook);

            // Classify issues from the raw inventory data
            let mut candidates = classify_and_aggregate(&rows, store_id, timestamp);

            // Enrich candidates with VSA similarity insights.
            // If multiple SKUs in the same issue group have similar VSA
            // signatures, it increases confidence (systemic pattern).
            //
            // Optimization: HashMap for O(1) SKU→index lookup, and stride-based
            // sampling caps pairwise comparisons at MAX_SIMILARITY_SAMPLE² pairs.
            let sku_index: HashMap<&str, usize> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| (r.sku.as_str(), i))
                .collect();

            for candidate in &mut candidates {
                if candidate.sku_ids.len() > 1 {
                    let group_indices: Vec<usize> = candidate
                        .sku_ids
                        .iter()
                        .filter_map(|sku| sku_index.get(sku.as_str()).copied())
                        .collect();

                    if group_indices.len() >= 2 {
                        // Sample large groups to avoid O(N²) blowup.
                        // Stride-based sampling maintains statistical representation.
                        let sampled: Vec<usize> = if group_indices.len() > MAX_SIMILARITY_SAMPLE {
                            let stride = group_indices.len() / MAX_SIMILARITY_SAMPLE;
                            group_indices
                                .iter()
                                .step_by(stride.max(1))
                                .take(MAX_SIMILARITY_SAMPLE)
                                .copied()
                                .collect()
                        } else {
                            group_indices
                        };

                        let mut total_sim = 0.0;
                        let mut pair_count = 0;

                        for (i, &idx_a) in sampled.iter().enumerate() {
                            for &idx_b in sampled.iter().skip(i + 1) {
                                let sim =
                                    similarity::cosine_similarity(&bundles[idx_a], &bundles[idx_b]);
                                total_sim += sim.abs();
                                pair_count += 1;
                            }
                        }

                        if pair_count > 0 {
                            let avg_sim = total_sim / pair_count as f64;
                            if avg_sim > CLUSTER_THRESHOLD {
                                candidate.confidence =
                                    (candidate.confidence * 1.1).min(1.0);
                            }
                        }
                    }
                }
            }

            all_candidates.extend(candidates);
        }

        all_candidates
    }
}

#[async_trait]
impl Source<AgentQuery, IssueCandidate> for GlmAnalyticsSource {
    fn enable(&self, query: &AgentQuery) -> bool {
        !query.store_ids.is_empty() && !self.records.is_empty()
    }

    async fn get_candidates(&self, query: &AgentQuery) -> Result<Vec<IssueCandidate>, String> {
        let timestamp = if query.time_range.end.is_empty() {
            "2025-01-15T00:00:00Z".to_string()
        } else {
            query.time_range.end.clone()
        };

        let mut candidates = self.analyze(&query.store_ids, &timestamp);

        // If the user is a store manager, only return issues for their store.
        if let UserRole::StoreManager { ref store_id } = query.user_role {
            candidates.retain(|c| c.store_id == *store_id);
        }

        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{IssueType, TimeRange, UserRole};

    fn sample_records() -> Vec<InventoryRecord> {
        vec![
            InventoryRecord {
                store_id: "store-7".into(),
                sku: "ELC-4401".into(),
                qty_on_hand: -47.0,
                unit_cost: 23.50,
                margin_pct: 0.35,
                sales_last_30d: 10.0,
                days_since_receipt: 30.0,
                retail_price: 31.73,
                is_damaged: false,
                on_order_qty: 0.0,
                is_seasonal: false,
            },
            InventoryRecord {
                store_id: "store-7".into(),
                sku: "SEA-1201".into(),
                qty_on_hand: 100.0,
                unit_cost: 50.0,
                margin_pct: 0.35,
                sales_last_30d: 0.0,
                days_since_receipt: 180.0,
                retail_price: 67.50,
                is_damaged: false,
                on_order_qty: 0.0,
                is_seasonal: false,
            },
            InventoryRecord {
                store_id: "store-12".into(),
                sku: "PNT-1001".into(),
                qty_on_hand: 50.0,
                unit_cost: 100.0,
                margin_pct: 0.05,
                sales_last_30d: 10.0,
                days_since_receipt: 30.0,
                retail_price: 105.0,
                is_damaged: false,
                on_order_qty: 0.0,
                is_seasonal: false,
            },
            InventoryRecord {
                store_id: "store-12".into(),
                sku: "NRM-0001".into(),
                qty_on_hand: 50.0,
                unit_cost: 100.0,
                margin_pct: 0.35,
                sales_last_30d: 20.0,
                days_since_receipt: 30.0,
                retail_price: 135.0,
                is_damaged: false,
                on_order_qty: 0.0,
                is_seasonal: false,
            },
        ]
    }

    fn make_query(stores: Vec<&str>) -> AgentQuery {
        AgentQuery {
            request_id: "test-001".into(),
            user_id: "exec".into(),
            user_role: UserRole::Executive,
            store_ids: stores.into_iter().map(String::from).collect(),
            time_range: TimeRange {
                start: "2025-01-01T00:00:00Z".into(),
                end: "2025-01-15T00:00:00Z".into(),
            },
            priority_filters: None,
        }
    }

    #[tokio::test]
    async fn source_produces_real_issues() {
        let source = GlmAnalyticsSource::new(sample_records());
        let query = make_query(vec!["store-7", "store-12"]);
        let candidates = source.get_candidates(&query).await.unwrap();

        // store-7: NegativeInventory (ELC-4401) + DeadStock (SEA-1201)
        // store-12: MarginErosion (PNT-1001)
        assert!(
            candidates.len() >= 3,
            "Should have at least 3 issues, got {}",
            candidates.len()
        );

        // Verify the negative inventory dollar impact is correct
        let neg_inv = candidates
            .iter()
            .find(|c| c.issue_type == IssueType::NegativeInventory)
            .expect("Should have NegativeInventory");
        // 47 × $23.50 = $1,104.50
        assert!(
            (neg_inv.dollar_impact - 1104.50).abs() < 0.01,
            "Dollar impact should be $1,104.50, got {}",
            neg_inv.dollar_impact
        );
        assert!(neg_inv.sku_ids.contains(&"ELC-4401".to_string()));
    }

    #[tokio::test]
    async fn source_filters_by_store_manager() {
        let source = GlmAnalyticsSource::new(sample_records());
        let query = AgentQuery {
            request_id: "test-002".into(),
            user_id: "mgr".into(),
            user_role: UserRole::StoreManager {
                store_id: "store-7".into(),
            },
            store_ids: vec!["store-7".into(), "store-12".into()],
            time_range: TimeRange {
                start: "2025-01-01T00:00:00Z".into(),
                end: "2025-01-15T00:00:00Z".into(),
            },
            priority_filters: None,
        };
        let candidates = source.get_candidates(&query).await.unwrap();
        assert!(candidates.iter().all(|c| c.store_id == "store-7"));
    }

    #[tokio::test]
    async fn source_disabled_for_empty_data() {
        let source = GlmAnalyticsSource::new(vec![]);
        let query = make_query(vec!["store-7"]);
        assert!(!source.enable(&query));
    }

    #[tokio::test]
    async fn normal_rows_produce_no_candidates() {
        let records = vec![InventoryRecord {
            store_id: "store-1".into(),
            sku: "NRM-001".into(),
            qty_on_hand: 50.0,
            unit_cost: 100.0,
            margin_pct: 0.35,
            sales_last_30d: 20.0,
            days_since_receipt: 30.0,
            retail_price: 135.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        }];
        let source = GlmAnalyticsSource::new(records);
        let query = make_query(vec!["store-1"]);
        let candidates = source.get_candidates(&query).await.unwrap();
        assert!(candidates.is_empty(), "Normal rows should produce no issues");
    }
}
