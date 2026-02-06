//! Issue classification from inventory data.
//!
//! Classifies each inventory row into zero or more `IssueCandidate` structs
//! by analyzing the same signal thresholds used by the VSA bundling engine.
//! This is the bridge between raw inventory data and the pipeline's typed
//! issue model.
//!
//! Dollar impact is computed from the underlying data:
//! - Negative inventory: |qty| x unit_cost
//! - Dead stock: qty x unit_cost (capital tied up in unsellable goods)
//! - Margin erosion: qty x (unit_cost x (expected_margin - actual_margin))
//! - Overstock/Seasonal: qty x unit_cost x carrying_cost_factor
//! - Damaged: qty x unit_cost
//! - Receiving gap: negative retail -> qty x unit_cost (pricing error exposure)
//! - Shrinkage pattern: high-value items with low margin and near-zero sales
//! - Zero cost anomaly: items with $0 cost but positive retail (missing cost data)
//! - Price discrepancy: cost exceeds retail price (selling below cost)
//! - Overstock: inventory exceeds 12 months' supply based on sales velocity

use sentinel_vsa::bundling::InventoryRow;
use sentinel_vsa::evidence::EvidenceScorer;

use crate::types::{IssueCandidate, IssueType, TrendDirection};

// ---------------------------------------------------------------------------
// Detection thresholds (ported from original Python config.py)
// ---------------------------------------------------------------------------

/// Margin threshold below which items are flagged for margin erosion.
const MARGIN_EROSION_THRESHOLD: f64 = 0.20;
/// DIB benchmark margin used for dollar impact gap calculation.
const DIB_BENCHMARK_MARGIN: f64 = 0.35;
/// Dead stock: minimum days since last receipt to qualify.
const DEAD_STOCK_DAYS: f64 = 90.0;
/// Purchasing leakage: minimum unit cost to qualify.
const HIGH_COST_THRESHOLD: f64 = 500.0;
/// Patronage miss: minimum qty to qualify as seasonal overstock.
const PATRONAGE_QTY_THRESHOLD: f64 = 200.0;
/// Monthly carrying cost factor (2% per month).
const CARRYING_COST_MONTHLY: f64 = 0.02;
/// Shrinkage pattern: minimum inventory value (qty x cost) to flag.
const SHRINKAGE_MIN_VALUE: f64 = 1000.0;
/// Shrinkage pattern: maximum margin to be suspicious.
const SHRINKAGE_MAX_MARGIN: f64 = 0.10;
/// Shrinkage pattern: maximum sales velocity to be suspicious.
const SHRINKAGE_MAX_SALES: f64 = 10.0;
/// Shrinkage impact cap per SKU (prevents outlier inflation).
const SHRINKAGE_MAX_IMPACT: f64 = 2000.0;
/// Overstock: months-of-supply threshold (> 12 months = overstocked).
const OVERSTOCK_MONTHS_SUPPLY: f64 = 12.0;
/// Overstock: minimum qty for low-sales overstock detection.
const OVERSTOCK_MIN_QTY: f64 = 500.0;
/// Overstock: optimal stock level assumed for carrying cost calculation.
const OVERSTOCK_OPTIMAL_QTY: f64 = 30.0;
/// Overstock: annual carrying cost rate (20% annual = 1.67% monthly).
const OVERSTOCK_CARRYING_RATE: f64 = 0.0167;
/// Overstock impact cap per SKU.
const OVERSTOCK_MAX_IMPACT: f64 = 2000.0;
/// Zero cost anomaly impact cap per SKU.
const ZERO_COST_MAX_IMPACT: f64 = 2000.0;
/// Price discrepancy impact cap per SKU.
const PRICE_DISCREPANCY_MAX_IMPACT: f64 = 2000.0;

/// Classification result for a single inventory row.
#[derive(Clone, Debug)]
pub struct ClassifiedIssue {
    pub issue_type: IssueType,
    pub sku: String,
    pub store_id: String,
    pub dollar_impact: f64,
    pub confidence: f64,
    pub trend_direction: TrendDirection,
    /// Which signals fired for this row (for diagnostics).
    pub active_signals: Vec<&'static str>,
}

/// Classify a single inventory row into zero or more issues.
///
/// Each row can produce multiple issues (e.g., a row with negative inventory
/// AND low margin produces both a NegativeInventory and MarginErosion issue).
/// The dollar impact is computed from the actual data, not from the VSA vector.
pub fn classify_row(row: &InventoryRow, store_id: &str) -> Vec<ClassifiedIssue> {
    let mut issues = Vec::new();
    let signals = detect_signals(row);

    // --- Negative Inventory ---
    // A row with qty_on_hand < 0 means we've oversold or have a system error.
    // Dollar impact = |qty| x unit_cost (the value of the missing inventory).
    if row.qty_on_hand < 0.0 {
        let dollar_impact = row.qty_on_hand.abs() * row.unit_cost;
        issues.push(ClassifiedIssue {
            issue_type: IssueType::NegativeInventory,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: compute_confidence(&signals, &["negative_qty"]),
            trend_direction: infer_trend(row),
            active_signals: signals.clone(),
        });
    }

    // --- Dead Stock ---
    // Zero sales in last 30 days + old receipt (>90 days) = capital tied up.
    // Dollar impact = qty x unit_cost (total value of dead inventory).
    if row.sales_last_30d == 0.0
        && row.days_since_receipt > DEAD_STOCK_DAYS
        && row.qty_on_hand > 0.0
    {
        let dollar_impact = row.qty_on_hand * row.unit_cost;
        issues.push(ClassifiedIssue {
            issue_type: IssueType::DeadStock,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: compute_confidence(&signals, &["zero_sales", "old_receipt"]),
            trend_direction: TrendDirection::Worsening, // dead stock always worsening
            active_signals: signals.clone(),
        });
    }

    // --- Margin Erosion ---
    // Low margin (<20%) on in-stock items.
    // Dollar impact = qty x unit_cost x (expected_margin - actual_margin).
    // This represents the profit gap versus what the item should be earning.
    if row.margin_pct < MARGIN_EROSION_THRESHOLD && row.qty_on_hand > 0.0 {
        let margin_gap = DIB_BENCHMARK_MARGIN - row.margin_pct.max(0.0);
        let dollar_impact = row.qty_on_hand * row.unit_cost * margin_gap;
        issues.push(ClassifiedIssue {
            issue_type: IssueType::MarginErosion,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: compute_confidence(&signals, &["low_margin"]),
            trend_direction: if row.margin_pct < 0.10 {
                TrendDirection::Worsening
            } else {
                TrendDirection::Stable
            },
            active_signals: signals.clone(),
        });
    }

    // --- Receiving Gap ---
    // Negative retail price signals a data/pricing error.
    // Dollar impact = qty x unit_cost (exposure from mispriced goods).
    if row.retail_price < 0.0 && row.qty_on_hand > 0.0 {
        let dollar_impact = row.qty_on_hand * row.unit_cost;
        issues.push(ClassifiedIssue {
            issue_type: IssueType::ReceivingGap,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: 0.95, // pricing errors are high-confidence
            trend_direction: TrendDirection::Stable,
            active_signals: signals.clone(),
        });
    }

    // --- Vendor Short Ship ---
    // Damaged goods with items on order suggests vendor fulfillment issues.
    // Dollar impact = qty_damaged x unit_cost + on_order x unit_cost x risk_factor.
    if row.is_damaged && row.on_order_qty > 0.0 {
        let dollar_impact = row.qty_on_hand * row.unit_cost
            + row.on_order_qty * row.unit_cost * 0.25; // 25% risk of repeat
        issues.push(ClassifiedIssue {
            issue_type: IssueType::VendorShortShip,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: compute_confidence(&signals, &["damaged", "on_order"]),
            trend_direction: TrendDirection::Worsening,
            active_signals: signals.clone(),
        });
    } else if row.is_damaged {
        // Damaged goods without reorder — just the damaged value.
        let dollar_impact = row.qty_on_hand * row.unit_cost;
        if dollar_impact > 0.0 {
            issues.push(ClassifiedIssue {
                issue_type: IssueType::VendorShortShip,
                sku: row.sku.clone(),
                store_id: store_id.to_string(),
                dollar_impact,
                confidence: compute_confidence(&signals, &["damaged"]),
                trend_direction: TrendDirection::Stable,
                active_signals: signals.clone(),
            });
        }
    }

    // --- Purchasing Leakage ---
    // High cost items (>$500) with low margin. The purchasing department is
    // paying too much or not negotiating volume discounts.
    if row.unit_cost > HIGH_COST_THRESHOLD
        && row.margin_pct < MARGIN_EROSION_THRESHOLD
        && row.qty_on_hand > 0.0
    {
        let overpay_est = (row.unit_cost - HIGH_COST_THRESHOLD) * 0.15; // estimated 15% overpay
        let dollar_impact = row.qty_on_hand * overpay_est;
        issues.push(ClassifiedIssue {
            issue_type: IssueType::PurchasingLeakage,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: compute_confidence(&signals, &["high_cost", "low_margin"]),
            trend_direction: TrendDirection::Stable,
            active_signals: signals.clone(),
        });
    }

    // --- Patronage Miss (Overstock + Seasonal) ---
    // High qty seasonal items risk patronage dividend loss if not sold through.
    if row.qty_on_hand > PATRONAGE_QTY_THRESHOLD && row.is_seasonal {
        let months_held = row.days_since_receipt / 30.0;
        let dollar_impact =
            row.qty_on_hand * row.unit_cost * CARRYING_COST_MONTHLY * months_held;
        issues.push(ClassifiedIssue {
            issue_type: IssueType::PatronageMiss,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: compute_confidence(&signals, &["high_qty", "seasonal"]),
            trend_direction: if months_held > 3.0 {
                TrendDirection::Worsening
            } else {
                TrendDirection::Stable
            },
            active_signals: signals.clone(),
        });
    }

    // --- Shrinkage Pattern ---
    // Multi-signal shrinkage detection: high-value items with near-zero margin
    // and minimal sales suggest inventory leakage (theft, damage, spoilage).
    // Original Python: sub_total > $1000 AND margin < 10% AND sold < 10
    if row.qty_on_hand > 0.0 && row.unit_cost > 0.0 {
        let inventory_value = row.qty_on_hand * row.unit_cost;
        if inventory_value > SHRINKAGE_MIN_VALUE
            && row.margin_pct < SHRINKAGE_MAX_MARGIN
            && row.sales_last_30d < SHRINKAGE_MAX_SALES
        {
            // Dollar impact capped at SHRINKAGE_MAX_IMPACT per SKU to prevent outliers
            let dollar_impact = (inventory_value * 0.10).min(SHRINKAGE_MAX_IMPACT);
            issues.push(ClassifiedIssue {
                issue_type: IssueType::ShrinkagePattern,
                sku: row.sku.clone(),
                store_id: store_id.to_string(),
                dollar_impact,
                confidence: compute_confidence(
                    &signals,
                    &["low_margin", "zero_sales"],
                ),
                trend_direction: if row.sales_last_30d == 0.0 {
                    TrendDirection::Worsening
                } else {
                    TrendDirection::Stable
                },
                active_signals: signals.clone(),
            });
        }
    }

    // --- Zero Cost Anomaly ---
    // Items with $0 unit cost but positive retail price indicate missing cost data.
    // This corrupts margin calculations and hides profitability problems.
    // Original Python: cost == 0 AND revenue > 0
    if row.unit_cost == 0.0 && row.retail_price > 0.0 && row.qty_on_hand > 0.0 {
        // Impact: 10% of retail value as proxy for untracked cost exposure
        let dollar_impact =
            (row.qty_on_hand * row.retail_price * 0.10).min(ZERO_COST_MAX_IMPACT);
        // Higher confidence if actively selling (more urgent to fix)
        let confidence = if row.sales_last_30d > 0.0 { 0.85 } else { 0.70 };
        issues.push(ClassifiedIssue {
            issue_type: IssueType::ZeroCostAnomaly,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence,
            trend_direction: if row.sales_last_30d > 0.0 {
                TrendDirection::Worsening // actively selling with wrong data
            } else {
                TrendDirection::Stable
            },
            active_signals: signals.clone(),
        });
    }

    // --- Price Discrepancy ---
    // Cost exceeds retail price, meaning we're selling below cost.
    // Original Python: cost > revenue (selling at a loss)
    // Excludes negative retail (already caught by ReceivingGap) and zero-cost items.
    if row.unit_cost > 0.0
        && row.retail_price > 0.0
        && row.unit_cost > row.retail_price
        && row.qty_on_hand > 0.0
    {
        let loss_per_unit = row.unit_cost - row.retail_price;
        // Impact = loss per unit x qty, capped
        let dollar_impact =
            (loss_per_unit * row.qty_on_hand).min(PRICE_DISCREPANCY_MAX_IMPACT);
        issues.push(ClassifiedIssue {
            issue_type: IssueType::PriceDiscrepancy,
            sku: row.sku.clone(),
            store_id: store_id.to_string(),
            dollar_impact,
            confidence: 0.85, // pricing errors are high-confidence
            trend_direction: if row.sales_last_30d > 0.0 {
                TrendDirection::Worsening // actively selling at a loss
            } else {
                TrendDirection::Stable
            },
            active_signals: signals.clone(),
        });
    }

    // --- Overstock ---
    // Inventory exceeds 12 months' supply based on sales velocity, OR
    // high quantity (>500) with near-zero sales (<10/month).
    // Excludes seasonal items (already handled by PatronageMiss) and damaged.
    if row.qty_on_hand > 0.0
        && row.unit_cost > 0.0
        && !row.is_seasonal
        && !row.is_damaged
    {
        let is_overstock = if row.sales_last_30d > 0.0 {
            // Months-of-supply check: qty / monthly_sales > 12 months
            row.qty_on_hand / row.sales_last_30d > OVERSTOCK_MONTHS_SUPPLY
        } else {
            // No sales: high qty with no movement
            row.qty_on_hand > OVERSTOCK_MIN_QTY
        };

        if is_overstock {
            // Carrying cost on excess inventory above optimal stock level
            let excess = (row.qty_on_hand - OVERSTOCK_OPTIMAL_QTY).max(0.0);
            let dollar_impact =
                (excess * row.unit_cost * OVERSTOCK_CARRYING_RATE).min(OVERSTOCK_MAX_IMPACT);
            if dollar_impact > 0.0 {
                issues.push(ClassifiedIssue {
                    issue_type: IssueType::Overstock,
                    sku: row.sku.clone(),
                    store_id: store_id.to_string(),
                    dollar_impact,
                    confidence: compute_confidence(
                        &signals,
                        &["high_qty", "zero_sales"],
                    ),
                    trend_direction: if row.sales_last_30d == 0.0
                        && row.days_since_receipt > DEAD_STOCK_DAYS
                    {
                        TrendDirection::Worsening
                    } else {
                        TrendDirection::Stable
                    },
                    active_signals: signals.clone(),
                });
            }
        }
    }

    issues
}

/// VSA vector dimensionality for evidence scoring.
/// 1024 dimensions provide a good balance of accuracy and performance.
const EVIDENCE_DIMENSIONS: usize = 1024;

/// Classify a batch of inventory rows into issues, grouping by store and type.
///
/// Rows that trigger the same issue type at the same store are grouped into
/// a single `IssueCandidate` with aggregated dollar impact and combined SKU list.
/// Each candidate receives a root cause attribution via evidence scoring.
pub fn classify_and_aggregate(
    rows: &[InventoryRow],
    store_id: &str,
    timestamp: &str,
) -> Vec<IssueCandidate> {
    // Classify every row
    let mut all_issues: Vec<ClassifiedIssue> = rows
        .iter()
        .flat_map(|row| classify_row(row, store_id))
        .collect();

    // Group by (store_id, issue_type) and aggregate
    all_issues.sort_by(|a, b| {
        format!("{:?}{}", a.issue_type, a.store_id)
            .cmp(&format!("{:?}{}", b.issue_type, b.store_id))
    });

    // Create evidence scorer once for all candidates
    let scorer = EvidenceScorer::new(EVIDENCE_DIMENSIONS);

    let mut candidates: Vec<IssueCandidate> = Vec::new();
    let mut idx = 0;

    while idx < all_issues.len() {
        let issue = &all_issues[idx];
        let key_type = issue.issue_type.clone();
        let key_store = issue.store_id.clone();

        // Collect all issues with the same type and store
        let mut group_skus: Vec<String> = Vec::new();
        let mut total_dollar_impact = 0.0;
        let mut total_confidence = 0.0;
        let mut worst_trend = TrendDirection::Improving;
        let mut count = 0;
        let mut group_signals: Vec<Vec<&str>> = Vec::new();

        while idx < all_issues.len()
            && all_issues[idx].issue_type == key_type
            && all_issues[idx].store_id == key_store
        {
            let issue = &all_issues[idx];
            group_skus.push(issue.sku.clone());
            total_dollar_impact += issue.dollar_impact;
            total_confidence += issue.confidence;
            count += 1;

            // Capture signals for evidence scoring
            group_signals.push(issue.active_signals.clone());

            // Take the worst trend in the group
            worst_trend = worse_trend(&worst_trend, &issue.trend_direction);

            idx += 1;
        }

        let avg_confidence = total_confidence / count as f64;

        // Collect unique active signals across all SKUs in this group
        // (must happen before group_signals is potentially moved)
        let mut unique_signals: Vec<String> = group_signals
            .iter()
            .flat_map(|s| s.iter().map(|&sig| sig.to_string()))
            .collect();
        unique_signals.sort();
        unique_signals.dedup();

        // Score root cause from aggregated evidence signals.
        // For large groups (>50 SKUs), sample to keep scoring fast.
        let scoring_signals: Vec<Vec<&str>> = if group_signals.len() > 50 {
            // Stride-based sampling to maintain statistical representation
            let stride = group_signals.len() / 50;
            group_signals
                .iter()
                .step_by(stride.max(1))
                .take(50)
                .cloned()
                .collect()
        } else {
            group_signals
        };

        let scoring_result = scorer.score(&scoring_signals);

        // Generate a deterministic ID
        let id = format!(
            "{}-{:?}-{:03}",
            key_store,
            key_type,
            candidates.len() + 1
        );

        candidates.push(IssueCandidate {
            id,
            issue_type: key_type,
            store_id: key_store,
            sku_ids: group_skus,
            dollar_impact: total_dollar_impact,
            confidence: avg_confidence.min(1.0),
            trend_direction: worst_trend,
            detection_timestamp: timestamp.to_string(),
            priority_score: None,
            urgency_score: None,
            executive_relevance: None,
            root_cause: scoring_result.top_cause,
            root_cause_confidence: Some(scoring_result.confidence),
            cause_scores: scoring_result.scores.clone(),
            root_cause_ambiguity: Some(scoring_result.ambiguity),
            active_signals: unique_signals,
        });
    }

    candidates
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Detect which signals are active for a row (mirrors bundling.rs thresholds).
fn detect_signals(row: &InventoryRow) -> Vec<&'static str> {
    let mut signals = Vec::new();

    if row.qty_on_hand < 0.0 {
        signals.push("negative_qty");
    }
    if row.unit_cost > HIGH_COST_THRESHOLD {
        signals.push("high_cost");
    }
    if row.margin_pct < MARGIN_EROSION_THRESHOLD {
        signals.push("low_margin");
    }
    if row.sales_last_30d == 0.0 {
        signals.push("zero_sales");
    }
    if row.qty_on_hand > PATRONAGE_QTY_THRESHOLD {
        signals.push("high_qty");
    }
    if row.days_since_receipt <= 7.0 {
        signals.push("recent_receipt");
    }
    if row.days_since_receipt > DEAD_STOCK_DAYS {
        signals.push("old_receipt");
    }
    if row.retail_price < 0.0 {
        signals.push("negative_retail");
    }
    if row.is_damaged {
        signals.push("damaged");
    }
    if row.on_order_qty > 0.0 {
        signals.push("on_order");
    }
    if row.is_seasonal {
        signals.push("seasonal");
    }
    // New signals for Phase 9 detections
    if row.unit_cost == 0.0 && row.retail_price > 0.0 {
        signals.push("zero_cost");
    }
    if row.unit_cost > 0.0 && row.retail_price > 0.0 && row.unit_cost > row.retail_price {
        signals.push("cost_exceeds_retail");
    }

    signals
}

/// Compute confidence based on how many of the expected signals fired.
fn compute_confidence(active: &[&str], expected: &[&str]) -> f64 {
    if expected.is_empty() {
        return 0.5;
    }
    let matched = expected.iter().filter(|e| active.contains(e)).count();
    let base = matched as f64 / expected.len() as f64;
    // Scale confidence: more active signals = more context = higher confidence
    let signal_bonus = (active.len() as f64 * 0.02).min(0.15);
    (base * 0.85 + signal_bonus).min(1.0)
}

/// Infer trend direction from the row's characteristics.
fn infer_trend(row: &InventoryRow) -> TrendDirection {
    if row.qty_on_hand < -50.0 || (row.sales_last_30d == 0.0 && row.days_since_receipt > 90.0) {
        TrendDirection::Worsening
    } else if row.days_since_receipt <= 7.0 {
        TrendDirection::Improving
    } else {
        TrendDirection::Stable
    }
}

/// Return the worse of two trends.
fn worse_trend(a: &TrendDirection, b: &TrendDirection) -> TrendDirection {
    match (a, b) {
        (TrendDirection::Worsening, _) | (_, TrendDirection::Worsening) => {
            TrendDirection::Worsening
        }
        (TrendDirection::Stable, _) | (_, TrendDirection::Stable) => TrendDirection::Stable,
        _ => TrendDirection::Improving,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(
        sku: &str,
        qty: f64,
        cost: f64,
        margin: f64,
        sales: f64,
        days: f64,
    ) -> InventoryRow {
        InventoryRow {
            sku: sku.to_string(),
            qty_on_hand: qty,
            unit_cost: cost,
            margin_pct: margin,
            sales_last_30d: sales,
            days_since_receipt: days,
            retail_price: cost * (1.0 + margin),
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        }
    }

    #[test]
    fn negative_inventory_calculates_correct_dollar_impact() {
        let row = make_row("SKU-001", -47.0, 23.50, 0.35, 10.0, 30.0);
        let issues = classify_row(&row, "store-7");
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].issue_type, IssueType::NegativeInventory);
        // -47 x $23.50 = $1,104.50
        assert!((issues[0].dollar_impact - 1104.50).abs() < 0.01);
    }

    #[test]
    fn dead_stock_identified_correctly() {
        let row = make_row("SKU-002", 100.0, 50.0, 0.35, 0.0, 180.0);
        let issues = classify_row(&row, "store-7");
        let dead_stock: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::DeadStock)
            .collect();
        assert_eq!(dead_stock.len(), 1);
        // 100 x $50 = $5,000
        assert!((dead_stock[0].dollar_impact - 5000.0).abs() < 0.01);
        assert_eq!(dead_stock[0].trend_direction, TrendDirection::Worsening);
    }

    #[test]
    fn margin_erosion_calculates_profit_gap() {
        let row = make_row("SKU-003", 50.0, 100.0, 0.05, 10.0, 30.0);
        let issues = classify_row(&row, "store-7");
        let margin: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::MarginErosion)
            .collect();
        assert_eq!(margin.len(), 1);
        // 50 x $100 x (0.35 - 0.05) = $1,500
        assert!((margin[0].dollar_impact - 1500.0).abs() < 0.01);
    }

    #[test]
    fn normal_row_produces_no_issues() {
        let row = make_row("SKU-004", 50.0, 100.0, 0.35, 20.0, 30.0);
        let issues = classify_row(&row, "store-7");
        assert!(issues.is_empty(), "Normal row should produce no issues");
    }

    #[test]
    fn multi_signal_row_produces_multiple_issues() {
        // This row has negative qty AND low margin
        let row = make_row("SKU-005", -30.0, 800.0, 0.05, 10.0, 30.0);
        let issues = classify_row(&row, "store-7");
        let types: Vec<&IssueType> = issues.iter().map(|i| &i.issue_type).collect();
        assert!(types.contains(&&IssueType::NegativeInventory));
        // Negative qty row won't have MarginErosion since qty < 0
        // but will have NegativeInventory: 30 x $800 = $24,000
        let neg = issues
            .iter()
            .find(|i| i.issue_type == IssueType::NegativeInventory)
            .unwrap();
        assert!((neg.dollar_impact - 24000.0).abs() < 0.01);
    }

    #[test]
    fn classify_and_aggregate_groups_by_type() {
        let rows = vec![
            make_row("SKU-A", -10.0, 50.0, 0.35, 10.0, 30.0),
            make_row("SKU-B", -20.0, 30.0, 0.35, 10.0, 30.0),
            make_row("SKU-C", 100.0, 50.0, 0.35, 0.0, 180.0), // dead stock
        ];
        let candidates = classify_and_aggregate(&rows, "store-7", "2025-01-15T00:00:00Z");

        // Should have NegativeInventory (2 SKUs) and DeadStock (1 SKU)
        let neg = candidates
            .iter()
            .find(|c| c.issue_type == IssueType::NegativeInventory)
            .unwrap();
        assert_eq!(neg.sku_ids.len(), 2);
        // $500 + $600 = $1,100
        assert!((neg.dollar_impact - 1100.0).abs() < 0.01);

        let dead = candidates
            .iter()
            .find(|c| c.issue_type == IssueType::DeadStock)
            .unwrap();
        assert_eq!(dead.sku_ids.len(), 1);
        assert!((dead.dollar_impact - 5000.0).abs() < 0.01);
    }

    #[test]
    fn damaged_with_on_order_is_vendor_short_ship() {
        let mut row = make_row("SKU-006", 10.0, 200.0, 0.30, 5.0, 15.0);
        row.is_damaged = true;
        row.on_order_qty = 25.0;
        let issues = classify_row(&row, "store-7");
        let vendor: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::VendorShortShip)
            .collect();
        assert_eq!(vendor.len(), 1);
        // 10 x $200 + 25 x $200 x 0.25 = $2,000 + $1,250 = $3,250
        assert!((vendor[0].dollar_impact - 3250.0).abs() < 0.01);
    }

    #[test]
    fn seasonal_overstock_is_patronage_miss() {
        let mut row = make_row("SKU-007", 300.0, 25.0, 0.35, 5.0, 90.0);
        row.is_seasonal = true;
        let issues = classify_row(&row, "store-7");
        let patronage: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::PatronageMiss)
            .collect();
        assert_eq!(patronage.len(), 1);
        // 300 x $25 x 0.02 x (90/30) = $450
        assert!((patronage[0].dollar_impact - 450.0).abs() < 0.01);
    }

    // --- Phase 9: New detection tests ---

    #[test]
    fn shrinkage_pattern_high_value_low_margin_low_sales() {
        // qty=200, cost=$10, margin=5%, sales=2/month
        // inventory_value = 200 * 10 = $2000 > $1000
        // margin 0.05 < 0.10, sales 2.0 < 10.0 => shrinkage
        let row = make_row("SKU-SHRINK", 200.0, 10.0, 0.05, 2.0, 60.0);
        let issues = classify_row(&row, "store-7");
        let shrinkage: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::ShrinkagePattern)
            .collect();
        assert_eq!(shrinkage.len(), 1);
        // impact = min(2000 * 0.10, 2000) = $200
        assert!((shrinkage[0].dollar_impact - 200.0).abs() < 0.01);
    }

    #[test]
    fn shrinkage_pattern_not_triggered_below_value_threshold() {
        // qty=5, cost=$10 => value=$50, below $1000 threshold
        let row = make_row("SKU-LOW", 5.0, 10.0, 0.05, 2.0, 60.0);
        let issues = classify_row(&row, "store-7");
        let shrinkage: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::ShrinkagePattern)
            .collect();
        assert_eq!(shrinkage.len(), 0);
    }

    #[test]
    fn shrinkage_capped_at_max_impact() {
        // qty=50000, cost=$100 => value=$5,000,000
        // impact = min(5000000 * 0.10, 2000) = $2000 (capped)
        let row = make_row("SKU-BIG-SHRINK", 50000.0, 100.0, 0.05, 5.0, 60.0);
        let issues = classify_row(&row, "store-7");
        let shrinkage: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::ShrinkagePattern)
            .collect();
        assert_eq!(shrinkage.len(), 1);
        assert!((shrinkage[0].dollar_impact - 2000.0).abs() < 0.01);
    }

    #[test]
    fn zero_cost_anomaly_detected() {
        // cost=$0, retail=$50, qty=20, sales=5
        let row = InventoryRow {
            sku: "SKU-ZEROCOST".into(),
            qty_on_hand: 20.0,
            unit_cost: 0.0,
            margin_pct: 1.0, // 100% margin with zero cost
            sales_last_30d: 5.0,
            days_since_receipt: 30.0,
            retail_price: 50.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        };
        let issues = classify_row(&row, "store-7");
        let zero_cost: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::ZeroCostAnomaly)
            .collect();
        assert_eq!(zero_cost.len(), 1);
        // impact = min(20 * 50 * 0.10, 2000) = min(100, 2000) = $100
        assert!((zero_cost[0].dollar_impact - 100.0).abs() < 0.01);
        assert_eq!(zero_cost[0].confidence, 0.85); // actively selling
        assert_eq!(zero_cost[0].trend_direction, TrendDirection::Worsening);
    }

    #[test]
    fn zero_cost_anomaly_lower_confidence_without_sales() {
        let row = InventoryRow {
            sku: "SKU-ZEROCOST2".into(),
            qty_on_hand: 10.0,
            unit_cost: 0.0,
            margin_pct: 1.0,
            sales_last_30d: 0.0,
            days_since_receipt: 30.0,
            retail_price: 25.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        };
        let issues = classify_row(&row, "store-7");
        let zero_cost: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::ZeroCostAnomaly)
            .collect();
        assert_eq!(zero_cost.len(), 1);
        assert_eq!(zero_cost[0].confidence, 0.70);
        assert_eq!(zero_cost[0].trend_direction, TrendDirection::Stable);
    }

    #[test]
    fn price_discrepancy_cost_exceeds_retail() {
        // cost=$80, retail=$60, qty=10 => selling below cost
        let row = InventoryRow {
            sku: "SKU-PRICEDISC".into(),
            qty_on_hand: 10.0,
            unit_cost: 80.0,
            margin_pct: -0.33, // negative margin
            sales_last_30d: 3.0,
            days_since_receipt: 30.0,
            retail_price: 60.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        };
        let issues = classify_row(&row, "store-7");
        let price_disc: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::PriceDiscrepancy)
            .collect();
        assert_eq!(price_disc.len(), 1);
        // loss_per_unit = 80 - 60 = $20, impact = min(20 * 10, 2000) = $200
        assert!((price_disc[0].dollar_impact - 200.0).abs() < 0.01);
        assert_eq!(price_disc[0].trend_direction, TrendDirection::Worsening); // actively selling
    }

    #[test]
    fn price_discrepancy_not_triggered_when_retail_above_cost() {
        let row = make_row("SKU-NORMAL", 50.0, 100.0, 0.35, 20.0, 30.0);
        let issues = classify_row(&row, "store-7");
        let price_disc: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::PriceDiscrepancy)
            .collect();
        assert_eq!(price_disc.len(), 0);
    }

    #[test]
    fn overstock_by_months_supply() {
        // qty=500, sales=3/month => 166 months supply >> 12
        // Not seasonal, not damaged
        let row = make_row("SKU-OVER1", 500.0, 20.0, 0.35, 3.0, 60.0);
        let issues = classify_row(&row, "store-7");
        let overstock: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::Overstock)
            .collect();
        assert_eq!(overstock.len(), 1);
        // excess = 500 - 30 = 470, impact = min(470 * 20 * 0.0167, 2000) = min(157.0, 2000) = $156.98
        let expected = (470.0 * 20.0 * OVERSTOCK_CARRYING_RATE).min(OVERSTOCK_MAX_IMPACT);
        assert!((overstock[0].dollar_impact - expected).abs() < 0.01);
    }

    #[test]
    fn overstock_high_qty_zero_sales() {
        // qty=600 > 500, sales=0 => overstock with no movement
        let row = make_row("SKU-OVER2", 600.0, 15.0, 0.35, 0.0, 180.0);
        let issues = classify_row(&row, "store-7");
        let overstock: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::Overstock)
            .collect();
        assert_eq!(overstock.len(), 1);
        assert_eq!(overstock[0].trend_direction, TrendDirection::Worsening);
    }

    #[test]
    fn overstock_not_triggered_for_seasonal_items() {
        // High qty but seasonal => should be PatronageMiss, not Overstock
        let mut row = make_row("SKU-SEASONAL", 600.0, 15.0, 0.35, 0.0, 180.0);
        row.is_seasonal = true;
        let issues = classify_row(&row, "store-7");
        let overstock: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::Overstock)
            .collect();
        assert_eq!(overstock.len(), 0);
        // But should have PatronageMiss
        let patronage: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::PatronageMiss)
            .collect();
        assert_eq!(patronage.len(), 1);
    }

    #[test]
    fn overstock_not_triggered_for_normal_stock_level() {
        // qty=50, sales=20/month => 2.5 months supply, well under 12
        let row = make_row("SKU-NORMAL-STOCK", 50.0, 100.0, 0.35, 20.0, 30.0);
        let issues = classify_row(&row, "store-7");
        let overstock: Vec<_> = issues
            .iter()
            .filter(|i| i.issue_type == IssueType::Overstock)
            .collect();
        assert_eq!(overstock.len(), 0);
    }
}
