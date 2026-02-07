//! Centralized detection thresholds for profit leak classification.
//!
//! These values are calibrated for hardware retail (Do It Best / True Value class).
//! Changing a threshold here affects BOTH VSA signal bundling (in `bundling.rs`)
//! and issue classification (in `sentinel-pipeline/issue_classifier.rs`).

/// Dollar amount above which a cost is considered "high" for alerting purposes.
pub const HIGH_COST_THRESHOLD: f64 = 500.0;

/// Margin threshold below which items are flagged for margin erosion.
/// Items below this margin are encoded with the `low_margin` primitive.
pub const MARGIN_EROSION_THRESHOLD: f64 = 0.20;

/// Quantity threshold for patronage/bulk purchase detection.
/// Items above this quantity trigger the `high_qty` signal.
pub const PATRONAGE_QTY_THRESHOLD: f64 = 200.0;

/// Days with zero movement before an item triggers the `old_receipt` signal
/// and qualifies for dead stock classification.
pub const DEAD_STOCK_DAYS: f64 = 90.0;

/// Days threshold for the `recent_receipt` signal (within this many days).
pub const RECENT_RECEIPT_DAYS: f64 = 7.0;

/// Do It Best benchmark margin for hardware retail.
pub const DIB_BENCHMARK_MARGIN: f64 = 0.35;

/// Monthly carrying cost factor (2% per month).
pub const CARRYING_COST_MONTHLY: f64 = 0.02;

/// Dimensionality for all VSA hypervectors. Must be consistent across
/// bundling, evidence scoring, and analytics source components.
pub const VSA_DIMENSIONS: usize = 1024;
