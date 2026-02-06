//! Correctness tests for sentinel-vsa.
//!
//! Validates that:
//! 1. Anomalous rows produce non-zero bundle vectors
//! 2. Normal rows produce zero bundle vectors
//! 3. Rows with the same anomaly pattern cluster together under similarity search
//! 4. Rows with different anomaly patterns are dissimilar
//! 5. Determinism: same inputs always produce the same outputs

use sentinel_vsa::bundling::{bundle_inventory_batch, InventoryRow};
use sentinel_vsa::codebook::Codebook;
use sentinel_vsa::primitives::VsaPrimitives;
use sentinel_vsa::similarity::{cosine_similarity, find_similar};

const DIM: usize = 1024;
const SEED: u64 = 42;

fn prims() -> VsaPrimitives {
    VsaPrimitives::new(DIM, SEED)
}

fn cb() -> Codebook {
    Codebook::new(DIM, SEED)
}

fn norm(bundle: &ndarray::Array1<num_complex::Complex64>) -> f64 {
    bundle.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// Helper row generators
// ---------------------------------------------------------------------------

fn normal_row(sku: &str) -> InventoryRow {
    InventoryRow {
        sku: sku.to_string(),
        qty_on_hand: 50.0,
        unit_cost: 100.0,
        margin_pct: 0.35,
        sales_last_30d: 20.0,
        days_since_receipt: 30.0,
        retail_price: 150.0,
        is_damaged: false,
        on_order_qty: 0.0,
        is_seasonal: false,
    }
}

fn negative_inventory_row(sku: &str) -> InventoryRow {
    InventoryRow {
        sku: sku.to_string(),
        qty_on_hand: -25.0,
        unit_cost: 100.0,
        margin_pct: 0.35,
        sales_last_30d: 20.0,
        days_since_receipt: 30.0,
        retail_price: 150.0,
        is_damaged: false,
        on_order_qty: 0.0,
        is_seasonal: false,
    }
}

fn dead_stock_row(sku: &str) -> InventoryRow {
    InventoryRow {
        sku: sku.to_string(),
        qty_on_hand: 100.0,
        unit_cost: 50.0,
        margin_pct: 0.35,
        sales_last_30d: 0.0,
        days_since_receipt: 180.0,
        retail_price: 75.0,
        is_damaged: false,
        on_order_qty: 0.0,
        is_seasonal: false,
    }
}

fn margin_erosion_row(sku: &str) -> InventoryRow {
    InventoryRow {
        sku: sku.to_string(),
        qty_on_hand: 50.0,
        unit_cost: 800.0,
        margin_pct: 0.05,
        sales_last_30d: 10.0,
        days_since_receipt: 30.0,
        retail_price: 840.0,
        is_damaged: false,
        on_order_qty: 0.0,
        is_seasonal: false,
    }
}

fn damaged_row(sku: &str) -> InventoryRow {
    InventoryRow {
        sku: sku.to_string(),
        qty_on_hand: 10.0,
        unit_cost: 200.0,
        margin_pct: 0.30,
        sales_last_30d: 5.0,
        days_since_receipt: 15.0,
        retail_price: 260.0,
        is_damaged: true,
        on_order_qty: 25.0,
        is_seasonal: false,
    }
}

// ---------------------------------------------------------------------------
// Anomaly vector non-zero tests
// ---------------------------------------------------------------------------

#[test]
fn anomalous_rows_produce_nonzero_bundles() {
    let p = prims();
    let c = cb();

    let anomalous_rows = vec![
        negative_inventory_row("SKU-NEG-001"),
        dead_stock_row("SKU-DEAD-001"),
        margin_erosion_row("SKU-MARG-001"),
        damaged_row("SKU-DMG-001"),
    ];

    let bundles = bundle_inventory_batch(&anomalous_rows, &p, &c);

    for (i, bundle) in bundles.iter().enumerate() {
        let n = norm(bundle);
        assert!(
            n > 1e-10,
            "Anomalous row {} (sku={}) should produce non-zero bundle, got norm={}",
            i,
            anomalous_rows[i].sku,
            n
        );
    }
}

#[test]
fn normal_rows_produce_zero_bundles() {
    let p = prims();
    let c = cb();

    let normal_rows = vec![
        normal_row("SKU-NORM-001"),
        normal_row("SKU-NORM-002"),
        normal_row("SKU-NORM-003"),
    ];

    let bundles = bundle_inventory_batch(&normal_rows, &p, &c);

    for (i, bundle) in bundles.iter().enumerate() {
        let n = norm(bundle);
        assert!(
            n < 1e-10,
            "Normal row {} (sku={}) should produce zero bundle, got norm={}",
            i,
            normal_rows[i].sku,
            n
        );
    }
}

// ---------------------------------------------------------------------------
// Similarity clustering tests
//
// VSA with binding: each bundle = sum_i(primitive_i * sku_vec * strength_i).
// Because different SKUs have near-orthogonal codebook vectors, two bundles
// for different SKUs will be near-orthogonal *regardless* of anomaly pattern.
// This is by design — it lets you identify which specific SKU has which issue.
//
// Similarity is high between bundles of the SAME SKU with the SAME anomaly
// pattern (e.g., same SKU observed at two different times). We test that here.
// ---------------------------------------------------------------------------

#[test]
fn same_sku_same_anomaly_is_highly_similar() {
    let p = prims();
    let c = cb();

    // Same SKU, same anomaly pattern → identical bundle
    let row_a = negative_inventory_row("SKU-001");
    let row_b = negative_inventory_row("SKU-001");

    let bundles = bundle_inventory_batch(&[row_a, row_b], &p, &c);

    let sim = cosine_similarity(&bundles[0], &bundles[1]);
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "Same SKU + same anomaly should have similarity ~1.0, got {}",
        sim
    );
}

#[test]
fn same_sku_different_severity_still_similar() {
    let p = prims();
    let c = cb();

    // Same SKU, same anomaly type but different severity
    let mut mild = negative_inventory_row("SKU-001");
    mild.qty_on_hand = -2.0; // mild negative

    let mut severe = negative_inventory_row("SKU-001");
    severe.qty_on_hand = -80.0; // severe negative

    let bundles = bundle_inventory_batch(&[mild, severe], &p, &c);

    let sim = cosine_similarity(&bundles[0], &bundles[1]);
    // Both activate the same primitive bound to the same SKU, just different
    // strengths. Since they also share zero_sales+old_receipt signals, the
    // bundles should be quite similar.
    assert!(
        sim > 0.5,
        "Same SKU with different severity should still be similar, got {}",
        sim
    );
}

#[test]
fn same_sku_different_anomaly_is_less_similar() {
    let p = prims();
    let c = cb();

    // Same SKU, but completely different anomaly patterns
    let neg = negative_inventory_row("SKU-001");
    let damaged = damaged_row("SKU-001");

    let bundles = bundle_inventory_batch(&[neg, damaged], &p, &c);

    let sim = cosine_similarity(&bundles[0], &bundles[1]);
    // Different primitives activated, but bound to the same SKU vector.
    // The binding with the same SKU creates some correlation, but the
    // different primitive vectors should reduce it.
    assert!(
        sim < 0.9,
        "Same SKU different anomaly should be less similar than same anomaly, got {}",
        sim
    );
}

#[test]
fn different_skus_are_near_orthogonal() {
    let p = prims();
    let c = cb();

    // Different SKUs, same anomaly → near-orthogonal because of binding
    let row_a = negative_inventory_row("SKU-AAA");
    let row_b = negative_inventory_row("SKU-ZZZ");

    let bundles = bundle_inventory_batch(&[row_a, row_b], &p, &c);

    let sim = cosine_similarity(&bundles[0], &bundles[1]);
    assert!(
        sim.abs() < 0.15,
        "Different SKUs should be near-orthogonal even with same anomaly, got {}",
        sim
    );
}

#[test]
fn find_similar_ranks_self_highest() {
    let p = prims();
    let c = cb();

    let rows = vec![
        negative_inventory_row("SKU-001"), // 0 - query
        normal_row("SKU-002"),             // 1 - normal (zero bundle)
        dead_stock_row("SKU-003"),         // 2 - different anomaly, different SKU
        margin_erosion_row("SKU-004"),     // 3 - different anomaly, different SKU
    ];

    let bundles = bundle_inventory_batch(&rows, &p, &c);

    // Query with index 0
    let results = find_similar(&bundles[0], &bundles, -1.0, 10);

    // Self should be the top match with similarity 1.0
    assert_eq!(results[0].0, 0, "Self should be the top match");
    assert!(
        (results[0].1 - 1.0).abs() < 1e-10,
        "Self-similarity should be 1.0, got {}",
        results[0].1
    );

    // Normal row (zero bundle) should have similarity 0.0
    let normal_result = results.iter().find(|(idx, _)| *idx == 1);
    if let Some((_, score)) = normal_result {
        assert!(
            score.abs() < 1e-10,
            "Normal (zero) row should have ~0 similarity, got {}",
            score
        );
    }
}

#[test]
fn find_similar_among_same_sku_snapshots() {
    let p = prims();
    let c = cb();

    // Simulate the same SKU observed at different times with varying conditions.
    // This is the real use case: tracking how a SKU's anomaly profile evolves.
    let rows = vec![
        // SKU-001 with negative qty (snapshot 1)
        InventoryRow {
            sku: "SKU-001".into(),
            qty_on_hand: -10.0,
            unit_cost: 100.0,
            margin_pct: 0.35,
            sales_last_30d: 5.0,
            days_since_receipt: 30.0,
            retail_price: 150.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        },
        // SKU-001 with negative qty (snapshot 2, slightly different)
        InventoryRow {
            sku: "SKU-001".into(),
            qty_on_hand: -15.0,
            unit_cost: 100.0,
            margin_pct: 0.35,
            sales_last_30d: 3.0,
            days_since_receipt: 35.0,
            retail_price: 150.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        },
        // SKU-002 with negative qty (different SKU)
        InventoryRow {
            sku: "SKU-002".into(),
            qty_on_hand: -10.0,
            unit_cost: 100.0,
            margin_pct: 0.35,
            sales_last_30d: 5.0,
            days_since_receipt: 30.0,
            retail_price: 150.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        },
        // SKU-003 with dead stock (different anomaly)
        dead_stock_row("SKU-003"),
    ];

    let bundles = bundle_inventory_batch(&rows, &p, &c);

    // Query: SKU-001 snapshot 1
    let results = find_similar(&bundles[0], &bundles, 0.0, 10);
    let result_indices: Vec<usize> = results.iter().map(|(idx, _)| *idx).collect();

    // The two SKU-001 snapshots should be the top 2 results (both have
    // similarity ~1.0 since they differ only in signal strength, which
    // produces collinear bundle vectors). Order between them may vary.
    assert!(
        result_indices[..2].contains(&0) && result_indices[..2].contains(&1),
        "Top 2 results should be the two SKU-001 snapshots (indices 0 and 1). Got {:?}",
        results
    );

    // Both top results should have similarity ~1.0
    assert!(
        results[0].1 > 0.99,
        "Top result should have similarity ~1.0, got {}",
        results[0].1
    );
    assert!(
        results[1].1 > 0.99,
        "Second result should have similarity ~1.0, got {}",
        results[1].1
    );

    // SKU-002 and SKU-003 (different SKUs) should be much lower
    let other_results: Vec<&(usize, f64)> = results
        .iter()
        .filter(|(idx, _)| *idx >= 2)
        .collect();
    for (idx, score) in &other_results {
        assert!(
            score.abs() < 0.15,
            "Different SKU (index {}) should be near-orthogonal, got {}",
            idx,
            score
        );
    }
}

// ---------------------------------------------------------------------------
// Determinism tests
// ---------------------------------------------------------------------------

#[test]
fn bundling_is_deterministic() {
    let p = prims();
    let c = cb();

    let rows = vec![
        negative_inventory_row("SKU-001"),
        dead_stock_row("SKU-002"),
        margin_erosion_row("SKU-003"),
    ];

    let bundles_a = bundle_inventory_batch(&rows, &p, &c);
    let bundles_b = bundle_inventory_batch(&rows, &p, &c);

    for (a, b) in bundles_a.iter().zip(bundles_b.iter()) {
        assert_eq!(a, b, "Bundling should be deterministic");
    }
}

#[test]
fn similarity_is_deterministic() {
    let p = prims();
    let c = cb();

    let rows = vec![
        negative_inventory_row("SKU-001"),
        negative_inventory_row("SKU-002"),
        dead_stock_row("SKU-003"),
    ];

    let bundles = bundle_inventory_batch(&rows, &p, &c);

    let results_a = find_similar(&bundles[0], &bundles, 0.0, 10);
    let results_b = find_similar(&bundles[0], &bundles, 0.0, 10);

    assert_eq!(results_a.len(), results_b.len());
    for ((idx_a, score_a), (idx_b, score_b)) in results_a.iter().zip(results_b.iter()) {
        assert_eq!(idx_a, idx_b);
        assert!(
            (score_a - score_b).abs() < 1e-15,
            "Similarity scores should be identical: {} vs {}",
            score_a,
            score_b
        );
    }
}

// ---------------------------------------------------------------------------
// Scale test (smaller than benchmark, but validates correctness at volume)
// ---------------------------------------------------------------------------

#[test]
fn bundle_10k_rows_all_nonzero_anomalous() {
    let p = prims();
    let c = cb();

    let rows: Vec<InventoryRow> = (0..10_000)
        .map(|i| negative_inventory_row(&format!("SKU-{:06}", i)))
        .collect();

    let bundles = bundle_inventory_batch(&rows, &p, &c);
    assert_eq!(bundles.len(), 10_000);

    for (i, b) in bundles.iter().enumerate() {
        assert!(
            norm(b) > 1e-10,
            "All 10K anomalous bundles should be non-zero, failed at {}",
            i
        );
    }
}
