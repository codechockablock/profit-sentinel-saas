//! Performance benchmark for sentinel-vsa bundling.
//!
//! Generates 156K synthetic inventory rows and measures the time to bundle them
//! into hypervectors. Target: <2 seconds (Python baseline: 19 seconds).
//!
//! Run with:
//!   cargo run --example benchmark --release -p sentinel-vsa

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sentinel_vsa::bundling::{bundle_inventory_batch, InventoryRow};
use sentinel_vsa::codebook::Codebook;
use sentinel_vsa::primitives::VsaPrimitives;
use sentinel_vsa::similarity;
use std::time::Instant;

const NUM_ROWS: usize = 156_000;
const DIMENSIONS: usize = 1024;
const SEED: u64 = 42;

fn main() {
    println!("=== Sentinel-VSA Performance Benchmark ===");
    println!();

    // -----------------------------------------------------------------------
    // 1. Generate synthetic data
    // -----------------------------------------------------------------------
    println!("Generating {} synthetic inventory rows...", NUM_ROWS);
    let gen_start = Instant::now();
    let rows = generate_synthetic_rows(NUM_ROWS, SEED);
    let gen_elapsed = gen_start.elapsed();
    println!(
        "  Data generation: {:.3}s ({:.0} rows/s)",
        gen_elapsed.as_secs_f64(),
        NUM_ROWS as f64 / gen_elapsed.as_secs_f64()
    );

    // Count anomalous vs normal rows
    let anomalous = rows
        .iter()
        .filter(|r| has_any_signal(r))
        .count();
    println!(
        "  Anomalous rows: {} ({:.1}%)",
        anomalous,
        anomalous as f64 / NUM_ROWS as f64 * 100.0
    );
    println!();

    // -----------------------------------------------------------------------
    // 2. Initialize primitives and codebook
    // -----------------------------------------------------------------------
    println!(
        "Initializing VsaPrimitives and Codebook (dimensions={})...",
        DIMENSIONS
    );
    let init_start = Instant::now();
    let primitives = VsaPrimitives::new(DIMENSIONS, SEED);
    let codebook = Codebook::new(DIMENSIONS, SEED);
    let init_elapsed = init_start.elapsed();
    println!("  Initialization: {:.3}s", init_elapsed.as_secs_f64());
    println!();

    // -----------------------------------------------------------------------
    // 3. Bundle all rows (THE HOT PATH)
    // -----------------------------------------------------------------------
    println!("Bundling {} rows x {} dimensions...", NUM_ROWS, DIMENSIONS);

    // Warm-up run
    let warmup_start = Instant::now();
    let _ = bundle_inventory_batch(&rows[..1000], &primitives, &codebook);
    let warmup_elapsed = warmup_start.elapsed();
    println!("  Warm-up (1K rows): {:.3}s", warmup_elapsed.as_secs_f64());

    // Timed run
    let bundle_start = Instant::now();
    let bundles = bundle_inventory_batch(&rows, &primitives, &codebook);
    let bundle_elapsed = bundle_start.elapsed();

    let rows_per_sec = NUM_ROWS as f64 / bundle_elapsed.as_secs_f64();
    let ops_per_sec = (NUM_ROWS * 11) as f64 / bundle_elapsed.as_secs_f64();

    println!();
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  BUNDLING RESULTS                                       │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!(
        "  │  Rows:            {:>10}                            │",
        NUM_ROWS
    );
    println!(
        "  │  Dimensions:      {:>10}                            │",
        DIMENSIONS
    );
    println!(
        "  │  Time:            {:>10.3}s                           │",
        bundle_elapsed.as_secs_f64()
    );
    println!(
        "  │  Throughput:      {:>10.0} rows/s                    │",
        rows_per_sec
    );
    println!(
        "  │  Primitive ops/s: {:>10.0}                            │",
        ops_per_sec
    );

    let target_met = bundle_elapsed.as_secs_f64() < 2.0;
    if target_met {
        println!(
            "  │  Target (<2s):    ✓ PASSED ({:.1}x faster than Python) │",
            19.0 / bundle_elapsed.as_secs_f64()
        );
    } else {
        println!(
            "  │  Target (<2s):    ✗ MISSED ({:.1}x vs Python 19s)      │",
            19.0 / bundle_elapsed.as_secs_f64()
        );
    }
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // -----------------------------------------------------------------------
    // 4. Correctness validation
    // -----------------------------------------------------------------------
    println!("Running correctness validation...");

    // 4a. Non-zero check for anomalous rows
    let mut nonzero_count = 0;
    let mut zero_anomaly_count = 0;
    for (i, row) in rows.iter().enumerate() {
        let norm: f64 = bundles[i].iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if has_any_signal(row) {
            if norm > 1e-10 {
                nonzero_count += 1;
            } else {
                zero_anomaly_count += 1;
            }
        }
    }
    println!(
        "  Anomalous rows with non-zero bundles: {}/{} (missing: {})",
        nonzero_count, anomalous, zero_anomaly_count
    );
    assert_eq!(
        zero_anomaly_count, 0,
        "All anomalous rows should have non-zero bundles"
    );

    // 4b. Zero check for normal rows
    let normal_count = NUM_ROWS - anomalous;
    let mut zero_normal_count = 0;
    for (i, row) in rows.iter().enumerate() {
        let norm: f64 = bundles[i].iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        if !has_any_signal(row) && norm < 1e-10 {
            zero_normal_count += 1;
        }
    }
    println!(
        "  Normal rows with zero bundles: {}/{}",
        zero_normal_count, normal_count
    );
    assert_eq!(
        zero_normal_count, normal_count,
        "All normal rows should have zero bundles"
    );

    // -----------------------------------------------------------------------
    // 5. Similarity search benchmark
    // -----------------------------------------------------------------------
    println!();
    println!("Running similarity search (first anomalous row as query)...");

    // Find first anomalous row
    let query_idx = rows
        .iter()
        .position(|r| has_any_signal(r))
        .expect("should have at least one anomalous row");

    let sim_start = Instant::now();
    let similar = similarity::find_similar(&bundles[query_idx], &bundles, 0.3, 20);
    let sim_elapsed = sim_start.elapsed();

    println!(
        "  Query: row {} (sku={}, qty={}, cost={:.0})",
        query_idx, rows[query_idx].sku, rows[query_idx].qty_on_hand, rows[query_idx].unit_cost
    );
    println!(
        "  Search time: {:.3}s across {} vectors",
        sim_elapsed.as_secs_f64(),
        bundles.len()
    );
    println!("  Results above threshold 0.3:");
    for (rank, (idx, score)) in similar.iter().enumerate().take(10) {
        let row = &rows[*idx];
        println!(
            "    #{}: idx={} score={:.4} sku={} qty={} cost={:.0} margin={:.0}%",
            rank + 1,
            idx,
            score,
            row.sku,
            row.qty_on_hand,
            row.unit_cost,
            row.margin_pct * 100.0,
        );
    }

    // Verify that similar rows share anomaly patterns with the query
    if !similar.is_empty() {
        let query_signals = active_signals(&rows[query_idx]);
        let mut pattern_match_count = 0;
        for (idx, _score) in &similar {
            if *idx == query_idx {
                continue;
            }
            let match_signals = active_signals(&rows[*idx]);
            // Check if at least one signal overlaps
            if query_signals.iter().any(|s| match_signals.contains(s)) {
                pattern_match_count += 1;
            }
        }
        let non_self_results = similar.len().saturating_sub(1);
        if non_self_results > 0 {
            println!(
                "  Pattern overlap: {}/{} results share at least one anomaly signal with query",
                pattern_match_count, non_self_results
            );
        }
    }

    println!();
    println!("=== Benchmark complete ===");
}

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------

fn generate_synthetic_rows(count: usize, seed: u64) -> Vec<InventoryRow> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|i| {
            let sku = format!("SKU-{:06}", i);

            // ~30% of rows have some kind of anomaly
            let is_anomalous = rng.gen_bool(0.30);

            if is_anomalous {
                // Generate a row with one or more anomaly signals
                generate_anomalous_row(&mut rng, sku)
            } else {
                // Generate a normal row (no signals should fire)
                generate_normal_row(&mut rng, sku)
            }
        })
        .collect()
}

fn generate_normal_row(rng: &mut StdRng, sku: String) -> InventoryRow {
    InventoryRow {
        sku,
        qty_on_hand: rng.gen_range(10.0..200.0),   // positive, under 200
        unit_cost: rng.gen_range(5.0..500.0),        // under $500
        margin_pct: rng.gen_range(0.20..0.60),       // above 20%
        sales_last_30d: rng.gen_range(1.0..100.0),   // non-zero
        days_since_receipt: rng.gen_range(8.0..90.0), // between 8 and 90
        retail_price: rng.gen_range(10.0..1000.0),    // positive
        is_damaged: false,
        on_order_qty: 0.0,
        is_seasonal: false,
    }
}

fn generate_anomalous_row(rng: &mut StdRng, sku: String) -> InventoryRow {
    let mut row = generate_normal_row(rng, sku);

    // Randomly activate anomaly signals
    let anomaly_type = rng.gen_range(0u8..7);
    match anomaly_type {
        0 => {
            // Negative inventory
            row.qty_on_hand = rng.gen_range(-100.0..-1.0);
        }
        1 => {
            // High cost + low margin (potential margin erosion)
            row.unit_cost = rng.gen_range(500.0..2000.0);
            row.margin_pct = rng.gen_range(0.01..0.19);
        }
        2 => {
            // Dead stock: zero sales, old receipt
            row.sales_last_30d = 0.0;
            row.days_since_receipt = rng.gen_range(91.0..365.0);
        }
        3 => {
            // Overstock: high quantity, seasonal
            row.qty_on_hand = rng.gen_range(201.0..1000.0);
            row.is_seasonal = true;
        }
        4 => {
            // Damaged goods
            row.is_damaged = true;
            row.on_order_qty = rng.gen_range(1.0..50.0);
        }
        5 => {
            // Receiving gap: negative retail, recent receipt
            row.retail_price = -1.0;
            row.days_since_receipt = rng.gen_range(0.0..7.0);
        }
        _ => {
            // Multiple signals: negative qty + low margin + zero sales
            row.qty_on_hand = rng.gen_range(-50.0..-1.0);
            row.margin_pct = rng.gen_range(0.01..0.10);
            row.sales_last_30d = 0.0;
        }
    }

    row
}

// ---------------------------------------------------------------------------
// Signal detection helpers (mirrors bundling logic)
// ---------------------------------------------------------------------------

fn has_any_signal(row: &InventoryRow) -> bool {
    row.qty_on_hand < 0.0
        || row.unit_cost > 500.0
        || row.margin_pct < 0.20
        || row.sales_last_30d == 0.0
        || row.qty_on_hand > 200.0
        || row.days_since_receipt <= 7.0
        || row.days_since_receipt > 90.0
        || row.retail_price < 0.0
        || row.is_damaged
        || row.on_order_qty > 0.0
        || row.is_seasonal
}

fn active_signals(row: &InventoryRow) -> Vec<&'static str> {
    let mut signals = Vec::new();
    if row.qty_on_hand < 0.0 {
        signals.push("negative_qty");
    }
    if row.unit_cost > 500.0 {
        signals.push("high_cost");
    }
    if row.margin_pct < 0.20 {
        signals.push("low_margin");
    }
    if row.sales_last_30d == 0.0 {
        signals.push("zero_sales");
    }
    if row.qty_on_hand > 200.0 {
        signals.push("high_qty");
    }
    if row.days_since_receipt <= 7.0 {
        signals.push("recent_receipt");
    }
    if row.days_since_receipt > 90.0 {
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
    signals
}
