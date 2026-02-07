use ndarray::Array1;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::codebook::Codebook;
use crate::primitives::VsaPrimitives;

/// A single row of inventory data to be encoded as a hypervector bundle.
#[derive(Clone, Debug)]
pub struct InventoryRow {
    pub sku: String,
    pub qty_on_hand: f64,
    pub unit_cost: f64,
    pub margin_pct: f64,
    pub sales_last_30d: f64,
    pub days_since_receipt: f64,
    pub retail_price: f64,
    pub is_damaged: bool,
    pub on_order_qty: f64,
    pub is_seasonal: bool,
}

/// Bundle a batch of inventory rows into hypervectors using Rayon for parallelism.
///
/// This is the hot path — for 156K rows × 11 primitives we want sub-2s execution.
///
/// Optimization strategy:
/// 1. Pre-warm the codebook and snapshot it into a lock-free HashMap
/// 2. The parallel phase accesses the snapshot with zero synchronization overhead
/// 3. Inlined accumulation loop avoids temporary ndarray allocations
/// 4. Raw slice access enables compiler auto-vectorization (SIMD)
pub fn bundle_inventory_batch(
    rows: &[InventoryRow],
    primitives: &VsaPrimitives,
    codebook: &Codebook,
) -> Vec<Array1<Complex64>> {
    // Phase 1: Parallel warmup — generate codebook vectors for all unique SKUs.
    // Uses Rayon to parallelize vector generation across CPU cores.
    codebook.warmup_parallel(rows.iter().map(|r| r.sku.as_str()));
    let snapshot = codebook.snapshot();

    // Phase 2: Parallel bundling — completely lock-free.
    rows.par_iter()
        .map(|row| bundle_single_row(row, primitives, &snapshot))
        .collect()
}

/// A lock-free codebook snapshot for use in the parallel hot path.
type CodebookSnapshot = HashMap<String, Arc<Array1<Complex64>>>;

/// Encode a single inventory row as a superposition of weighted primitive vectors
/// bound with the SKU's codebook vector.
///
/// Each activated primitive is weighted by a signal strength in [0, 1], then
/// multiplied element-wise (binding) with the SKU vector and accumulated
/// (bundling) into the result.
fn bundle_single_row(
    row: &InventoryRow,
    primitives: &VsaPrimitives,
    snapshot: &CodebookSnapshot,
) -> Array1<Complex64> {
    let sku_arc = snapshot.get(&row.sku).expect("codebook was not warmed up");
    let sku_vec: &[Complex64] = sku_arc.as_slice().expect("SKU vector must be contiguous");
    let dim = primitives.dimensions();
    let mut bundle = Array1::<Complex64>::zeros(dim);

    // Negative quantity signal
    if row.qty_on_hand < 0.0 {
        let strength = (row.qty_on_hand.abs() / 100.0).min(1.0);
        accumulate(&mut bundle, &primitives.negative_qty, sku_vec, strength);
    }

    // High cost signal (above $500)
    if row.unit_cost > 500.0 {
        let strength = ((row.unit_cost - 500.0) / 500.0).min(1.0);
        accumulate(&mut bundle, &primitives.high_cost, sku_vec, strength);
    }

    // Low margin signal (below 20%)
    if row.margin_pct < 0.20 {
        let strength = 1.0 - (row.margin_pct / 0.20).max(0.0);
        accumulate(&mut bundle, &primitives.low_margin, sku_vec, strength);
    }

    // Zero sales signal (dead stock indicator)
    if row.sales_last_30d == 0.0 {
        accumulate(&mut bundle, &primitives.zero_sales, sku_vec, 1.0);
    }

    // High quantity signal (above 200)
    if row.qty_on_hand > 200.0 {
        let strength = ((row.qty_on_hand - 200.0) / 200.0).min(1.0);
        accumulate(&mut bundle, &primitives.high_qty, sku_vec, strength);
    }

    // Recent receipt signal (within 7 days)
    if row.days_since_receipt <= 7.0 {
        let strength = 1.0 - (row.days_since_receipt / 7.0);
        accumulate(&mut bundle, &primitives.recent_receipt, sku_vec, strength);
    }

    // Old receipt signal (over 90 days)
    if row.days_since_receipt > 90.0 {
        let strength = ((row.days_since_receipt - 90.0) / 90.0).min(1.0);
        accumulate(&mut bundle, &primitives.old_receipt, sku_vec, strength);
    }

    // Negative retail price signal
    if row.retail_price < 0.0 {
        accumulate(&mut bundle, &primitives.negative_retail, sku_vec, 1.0);
    }

    // Damaged signal
    if row.is_damaged {
        accumulate(&mut bundle, &primitives.damaged, sku_vec, 1.0);
    }

    // On order signal
    if row.on_order_qty > 0.0 {
        let strength = (row.on_order_qty / 100.0).min(1.0);
        accumulate(&mut bundle, &primitives.on_order, sku_vec, strength);
    }

    // Seasonal signal
    if row.is_seasonal {
        accumulate(&mut bundle, &primitives.seasonal, sku_vec, 1.0);
    }

    bundle
}

/// Accumulate: bundle += primitive * sku_vec * strength
///
/// Operates on raw slices to avoid ndarray bounds-check overhead and enable
/// compiler auto-vectorization (SIMD). Each Complex64 is (re, im) = two f64s,
/// and the complex multiply + accumulate compiles to tight FMA loops on ARM NEON.
#[inline(always)]
fn accumulate(
    bundle: &mut Array1<Complex64>,
    primitive: &Array1<Complex64>,
    sku_vec: &[Complex64],
    strength: f64,
) {
    let scale = Complex64::new(strength, 0.0);
    let b = bundle.as_slice_mut().expect("bundle must be contiguous for SIMD accumulation");
    let p = primitive.as_slice().expect("primitive must be contiguous");
    let len = b.len();
    for i in 0..len {
        b[i] += p[i] * sku_vec[i] * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::Codebook;

    fn sample_row(sku: &str) -> InventoryRow {
        InventoryRow {
            sku: sku.to_string(),
            qty_on_hand: -5.0,
            unit_cost: 750.0,
            margin_pct: 0.10,
            sales_last_30d: 0.0,
            days_since_receipt: 120.0,
            retail_price: 49.99,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        }
    }

    #[test]
    fn bundle_produces_correct_dimensions() {
        let prims = VsaPrimitives::new(512, 42);
        let cb = Codebook::new(512, 42);
        let row = sample_row("SKU-001");
        let results = bundle_inventory_batch(&[row], &prims, &cb);
        assert_eq!(results[0].len(), 512);
    }

    #[test]
    fn bundle_batch_parallelism_works() {
        let prims = VsaPrimitives::new(256, 42);
        let cb = Codebook::new(256, 42);
        let rows: Vec<InventoryRow> = (0..100)
            .map(|i| sample_row(&format!("SKU-{:04}", i)))
            .collect();
        let results = bundle_inventory_batch(&rows, &prims, &cb);
        assert_eq!(results.len(), 100);
        for r in &results {
            assert_eq!(r.len(), 256);
        }
    }

    #[test]
    fn bundle_nonzero_for_active_signals() {
        let prims = VsaPrimitives::new(256, 42);
        let cb = Codebook::new(256, 42);
        let row = sample_row("SKU-001");
        let results = bundle_inventory_batch(&[row], &prims, &cb);
        let result = &results[0];
        // Row has negative qty, high cost, low margin, zero sales, old receipt
        // so the bundle should be non-zero.
        let norm: f64 = result.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(norm > 0.0, "bundle should be non-zero for active signals");
    }

    #[test]
    fn bundle_zero_for_no_signals() {
        let prims = VsaPrimitives::new(256, 42);
        let cb = Codebook::new(256, 42);
        let row = InventoryRow {
            sku: "SKU-NORMAL".to_string(),
            qty_on_hand: 50.0,
            unit_cost: 100.0,
            margin_pct: 0.35,
            sales_last_30d: 10.0,
            days_since_receipt: 30.0,
            retail_price: 150.0,
            is_damaged: false,
            on_order_qty: 0.0,
            is_seasonal: false,
        };
        let results = bundle_inventory_batch(&[row], &prims, &cb);
        let result = &results[0];
        let norm: f64 = result.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            norm < 1e-10,
            "bundle should be zero when no signals are active"
        );
    }
}
