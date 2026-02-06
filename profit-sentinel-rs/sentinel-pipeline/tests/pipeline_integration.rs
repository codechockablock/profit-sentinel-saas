use sentinel_pipeline::candidate_pipeline::CandidatePipeline;
use sentinel_pipeline::components::dollar_impact_scorer::DollarImpactScorer;
use sentinel_pipeline::components::glm_analytics_source::GlmAnalyticsSource;
use sentinel_pipeline::components::low_impact_filter::LowImpactFilter;
use sentinel_pipeline::components::store_diversity_scorer::StoreDiversityScorer;
use sentinel_pipeline::components::top_k_selector::TopKSelector;
use sentinel_pipeline::filter::FilterResult;
use sentinel_pipeline::inventory_loader::InventoryRecord;
use sentinel_pipeline::pipelines::executive_digest::ExecutiveDigestPipeline;
use sentinel_pipeline::scorer::Scorer;
use sentinel_pipeline::selector::Selector;
use sentinel_pipeline::source::Source;
use sentinel_pipeline::types::*;

// ---------------------------------------------------------------------------
// Test data fixtures
// ---------------------------------------------------------------------------

/// Creates a realistic inventory dataset across multiple stores.
fn sample_records() -> Vec<InventoryRecord> {
    vec![
        // store-7: negative inventory — ELC-4401 at -47 qty, $23.50 cost
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
        // store-7: dead stock — SEA-1201, 100 qty, no sales, 180 days old
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
        // store-7: another dead stock — SEA-1202
        InventoryRecord {
            store_id: "store-7".into(),
            sku: "SEA-1202".into(),
            qty_on_hand: 75.0,
            unit_cost: 42.0,
            margin_pct: 0.35,
            sales_last_30d: 0.0,
            days_since_receipt: 200.0,
            retail_price: 56.70,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        },
        // store-12: margin erosion — PNT-1001, 5% margin
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
        // store-12: margin erosion — PNT-1002, 8% margin
        InventoryRecord {
            store_id: "store-12".into(),
            sku: "PNT-1002".into(),
            qty_on_hand: 30.0,
            unit_cost: 150.0,
            margin_pct: 0.08,
            sales_last_30d: 8.0,
            days_since_receipt: 25.0,
            retail_price: 162.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        },
        // store-12: normal row (no anomalies)
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
        // store-3: damaged goods with reorder — vendor short ship
        InventoryRecord {
            store_id: "store-3".into(),
            sku: "HRD-9901".into(),
            qty_on_hand: 10.0,
            unit_cost: 200.0,
            margin_pct: 0.30,
            sales_last_30d: 5.0,
            days_since_receipt: 15.0,
            retail_price: 260.0,
            is_damaged: true,
            on_order_qty: 25.0,
            is_seasonal: false,
        },
        // store-3: seasonal overstock — patronage miss
        InventoryRecord {
            store_id: "store-3".into(),
            sku: "GRD-5501".into(),
            qty_on_hand: 300.0,
            unit_cost: 25.0,
            margin_pct: 0.35,
            sales_last_30d: 5.0,
            days_since_receipt: 90.0,
            retail_price: 33.75,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: true,
        },
    ]
}

fn make_executive_query(store_ids: Vec<&str>) -> AgentQuery {
    AgentQuery {
        request_id: "test-001".into(),
        user_id: "exec_test".into(),
        user_role: UserRole::Executive,
        store_ids: store_ids.into_iter().map(String::from).collect(),
        time_range: TimeRange {
            start: "2025-01-01T00:00:00Z".into(),
            end: "2025-01-15T00:00:00Z".into(),
        },
        priority_filters: None,
    }
}

fn make_store_manager_query(store_id: &str) -> AgentQuery {
    AgentQuery {
        request_id: "test-002".into(),
        user_id: "mgr_test".into(),
        user_role: UserRole::StoreManager {
            store_id: store_id.into(),
        },
        store_ids: vec![store_id.into()],
        time_range: TimeRange {
            start: "2025-01-01T00:00:00Z".into(),
            end: "2025-01-15T00:00:00Z".into(),
        },
        priority_filters: None,
    }
}

// ---------------------------------------------------------------------------
// Source tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn glm_source_produces_real_issues() {
    let source = GlmAnalyticsSource::new(sample_records());
    let query = make_executive_query(vec!["store-7", "store-12"]);
    let candidates = source.get_candidates(&query).await.unwrap();

    // store-7 should have NegativeInventory + DeadStock
    // store-12 should have MarginErosion
    assert!(
        candidates.len() >= 3,
        "Should have at least 3 issues across 2 stores, got {}",
        candidates.len()
    );

    // Verify dollar impact on the negative inventory
    let neg = candidates
        .iter()
        .find(|c| c.issue_type == IssueType::NegativeInventory)
        .expect("Should detect NegativeInventory");
    // 47 × $23.50 = $1,104.50
    assert!(
        (neg.dollar_impact - 1104.50).abs() < 0.01,
        "NegativeInventory dollar impact should be $1,104.50, got ${:.2}",
        neg.dollar_impact
    );
}

#[tokio::test]
async fn glm_source_filters_by_store_manager_role() {
    let source = GlmAnalyticsSource::new(sample_records());
    let query = make_store_manager_query("store-7");
    let candidates = source.get_candidates(&query).await.unwrap();
    assert!(candidates.iter().all(|c| c.store_id == "store-7"));
    assert!(!candidates.is_empty());
}

#[tokio::test]
async fn glm_source_disabled_for_empty_data() {
    let source = GlmAnalyticsSource::new(vec![]);
    let query = AgentQuery {
        request_id: "test-empty".into(),
        user_id: "exec_test".into(),
        user_role: UserRole::Executive,
        store_ids: vec!["store-7".into()],
        time_range: TimeRange {
            start: "2025-01-01T00:00:00Z".into(),
            end: "2025-01-02T00:00:00Z".into(),
        },
        priority_filters: None,
    };
    assert!(!source.enable(&query));
}

// ---------------------------------------------------------------------------
// Filter tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn low_impact_filter_removes_cheap_issues() {
    let filter = LowImpactFilter::new(1000.0);
    let candidates = vec![
        IssueCandidate {
            id: "big".into(),
            dollar_impact: 2400.0,
            ..IssueCandidate::default()
        },
        IssueCandidate {
            id: "small".into(),
            dollar_impact: 50.0,
            ..IssueCandidate::default()
        },
    ];
    let query = make_executive_query(vec!["store-7"]);
    let FilterResult { kept, removed } =
        sentinel_pipeline::filter::Filter::filter(&filter, &query, candidates)
            .await
            .unwrap();
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0].id, "big");
    assert_eq!(removed.len(), 1);
    assert_eq!(removed[0].id, "small");
}

// ---------------------------------------------------------------------------
// Scorer tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dollar_impact_scorer_assigns_scores() {
    let scorer = DollarImpactScorer;
    let query = make_executive_query(vec!["store-7"]);
    let candidates = vec![
        IssueCandidate {
            dollar_impact: 2400.0,
            confidence: 0.9,
            trend_direction: TrendDirection::Worsening,
            ..IssueCandidate::default()
        },
        IssueCandidate {
            dollar_impact: 100.0,
            confidence: 0.5,
            trend_direction: TrendDirection::Improving,
            ..IssueCandidate::default()
        },
    ];
    let scored = scorer.score(&query, &candidates).await.unwrap();
    // Higher impact + worsening trend should produce higher score.
    assert!(scored[0].priority_score.unwrap() > scored[1].priority_score.unwrap());
}

#[tokio::test]
async fn store_diversity_scorer_attenuates_repeated_stores() {
    let scorer = StoreDiversityScorer::default();
    let query = make_executive_query(vec!["store-7"]);
    let candidates = vec![
        IssueCandidate {
            store_id: "store-7".into(),
            priority_score: Some(10.0),
            ..IssueCandidate::default()
        },
        IssueCandidate {
            store_id: "store-7".into(),
            priority_score: Some(9.0),
            ..IssueCandidate::default()
        },
        IssueCandidate {
            store_id: "store-12".into(),
            priority_score: Some(8.0),
            ..IssueCandidate::default()
        },
    ];
    let scored = scorer.score(&query, &candidates).await.unwrap();
    let s7_first = scored[0].priority_score.unwrap();
    let s7_second = scored[1].priority_score.unwrap();
    let s12 = scored[2].priority_score.unwrap();

    assert!(
        s7_first > s7_second,
        "second store-7 should be attenuated: {} vs {}",
        s7_first,
        s7_second
    );
    assert!(
        s12 > s7_second,
        "diverse store should beat attenuated repeat: {} vs {}",
        s12,
        s7_second
    );
}

// ---------------------------------------------------------------------------
// Selector tests
// ---------------------------------------------------------------------------

#[test]
fn top_k_selector_picks_highest_scores() {
    let selector = TopKSelector { k: 2 };
    let query = make_executive_query(vec!["store-7"]);
    let candidates = vec![
        IssueCandidate {
            id: "low".into(),
            priority_score: Some(1.0),
            ..IssueCandidate::default()
        },
        IssueCandidate {
            id: "high".into(),
            priority_score: Some(10.0),
            ..IssueCandidate::default()
        },
        IssueCandidate {
            id: "mid".into(),
            priority_score: Some(5.0),
            ..IssueCandidate::default()
        },
    ];
    let selected = selector.select(&query, candidates);
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].id, "high");
    assert_eq!(selected[1].id, "mid");
}

// ---------------------------------------------------------------------------
// Full pipeline integration tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn executive_digest_pipeline_end_to_end() {
    let pipeline = ExecutiveDigestPipeline::with_inventory(sample_records());
    let query = make_executive_query(vec!["store-7", "store-12", "store-3"]);

    let result = pipeline.execute(query).await;

    // Should have retrieved real issues from 3 stores
    assert!(
        result.retrieved_candidates.len() >= 3,
        "Should retrieve at least 3 issues, got {}",
        result.retrieved_candidates.len()
    );

    // Top 5 selected (default result_size)
    assert!(result.selected_candidates.len() <= 5);
    assert!(!result.selected_candidates.is_empty());

    // All selected candidates should have a priority score
    for c in &result.selected_candidates {
        assert!(
            c.priority_score.is_some(),
            "candidate {} should have a priority score",
            c.id
        );
    }

    // Candidates should be sorted by priority score descending
    let scores: Vec<f64> = result
        .selected_candidates
        .iter()
        .map(|c| c.priority_score.unwrap())
        .collect();
    for w in scores.windows(2) {
        assert!(
            w[0] >= w[1],
            "candidates should be sorted descending: {} < {}",
            w[0],
            w[1]
        );
    }

    // Hydrator should have populated urgency scores
    for c in &result.selected_candidates {
        assert!(
            c.urgency_score.is_some(),
            "candidate {} should have urgency_score from hydrator",
            c.id
        );
    }

    // Dollar impacts should be real computed values, not dummy data
    for c in &result.selected_candidates {
        assert!(
            c.dollar_impact > 0.0,
            "candidate {} should have positive dollar impact",
            c.id
        );
    }

    // Verify that the highest-impact issues float to the top
    // (after scoring and diversity adjustment)
    let top = &result.selected_candidates[0];
    assert!(
        top.dollar_impact > 100.0,
        "top issue should have significant dollar impact, got ${:.2}",
        top.dollar_impact
    );
}

#[tokio::test]
async fn pipeline_result_size_is_respected() {
    let pipeline = ExecutiveDigestPipeline::with_inventory_and_size(sample_records(), 2);
    let query = make_executive_query(vec!["store-7", "store-12"]);
    let result = pipeline.execute(query).await;
    assert!(result.selected_candidates.len() <= 2);
}

#[tokio::test]
async fn pipeline_with_single_store() {
    let pipeline = ExecutiveDigestPipeline::with_inventory(sample_records());
    let query = make_executive_query(vec!["store-7"]);
    let result = pipeline.execute(query).await;

    assert!(!result.retrieved_candidates.is_empty());
    assert!(result.selected_candidates.len() <= 5);

    // All candidates from the same store
    for c in &result.selected_candidates {
        assert_eq!(c.store_id, "store-7");
    }
}

#[tokio::test]
async fn pipeline_produces_correct_dollar_impacts() {
    let pipeline = ExecutiveDigestPipeline::with_inventory_and_size(sample_records(), 20);
    let query = make_executive_query(vec!["store-7", "store-12", "store-3"]);
    let result = pipeline.execute(query).await;

    // Find the NegativeInventory issue for store-7
    let neg = result
        .selected_candidates
        .iter()
        .find(|c| c.issue_type == IssueType::NegativeInventory && c.store_id == "store-7");

    if let Some(neg) = neg {
        // 47 × $23.50 = $1,104.50
        assert!(
            (neg.dollar_impact - 1104.50).abs() < 0.01,
            "NegativeInventory should be $1,104.50, got ${:.2}",
            neg.dollar_impact
        );
    }

    // Find the DeadStock issue for store-7
    let dead = result
        .selected_candidates
        .iter()
        .find(|c| c.issue_type == IssueType::DeadStock && c.store_id == "store-7");

    if let Some(dead) = dead {
        // SEA-1201: 100 × $50 = $5,000 + SEA-1202: 75 × $42 = $3,150 = $8,150 total
        assert!(
            (dead.dollar_impact - 8150.0).abs() < 0.01,
            "DeadStock should be $8,150.00, got ${:.2}",
            dead.dollar_impact
        );
    }

    // Find the VendorShortShip for store-3
    let vendor = result
        .selected_candidates
        .iter()
        .find(|c| c.issue_type == IssueType::VendorShortShip && c.store_id == "store-3");

    if let Some(vendor) = vendor {
        // 10 × $200 + 25 × $200 × 0.25 = $2,000 + $1,250 = $3,250
        assert!(
            (vendor.dollar_impact - 3250.0).abs() < 0.01,
            "VendorShortShip should be $3,250.00, got ${:.2}",
            vendor.dollar_impact
        );
    }
}

// ---------------------------------------------------------------------------
// Types tests
// ---------------------------------------------------------------------------

#[test]
fn issue_candidate_default_has_no_scores() {
    let c = IssueCandidate::default();
    assert!(c.priority_score.is_none());
    assert!(c.urgency_score.is_none());
    assert!(c.executive_relevance.is_none());
}

#[test]
fn agent_query_has_request_id() {
    use sentinel_pipeline::candidate_pipeline::HasRequestId;
    let q = make_executive_query(vec!["store-7"]);
    assert_eq!(q.request_id(), "test-001");
}
