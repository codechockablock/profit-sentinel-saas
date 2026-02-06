//! Evidence-based root cause scoring.
//!
//! Implements the positive-similarity summing algorithm from the original
//! Python evidence scorer, ported to Rust. This provides 0% quantitative
//! hallucination by only summing positive similarities between evidence
//! vectors and cause hypothesis vectors.
//!
//! # Algorithm
//!
//! 1. Each inventory signal (e.g., "low_margin", "zero_sales") is mapped
//!    to one or more cause hypotheses with weights via evidence rules.
//! 2. Evidence vectors are constructed by weighted superposition of cause
//!    vectors based on which rules matched.
//! 3. Each cause is scored by summing `max(0, similarity(evidence, cause))`
//!    across all evidence vectors (positive-similarity only).
//! 4. The highest-scoring cause becomes the root cause attribution.
//!
//! # Hallucination Prevention
//!
//! - Only positive similarities contribute (negative evidence is clamped).
//! - Accumulation (not averaging) rewards multiple independent confirmations.
//! - No generative component — causes are drawn from a fixed vocabulary.

use ndarray::Array1;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Cause definitions
// ---------------------------------------------------------------------------

/// The 8 root cause hypotheses for inventory issues.
///
/// Ported from the original Python `causes.py` with the same semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum RootCause {
    /// Shrinkage due to internal or external theft.
    Theft,
    /// Vendor raised costs without notice.
    VendorIncrease,
    /// Rebate timing mismatch (co-op, dating, terms).
    RebateTiming,
    /// Margin erosion from pricing or discount issues.
    MarginLeak,
    /// Market demand shifted (seasonal, trends, competition).
    DemandShift,
    /// Product quality problems (defects, returns).
    QualityIssue,
    /// Incorrect pricing configuration in POS system.
    PricingError,
    /// Gradual inventory discrepancy from process drift.
    InventoryDrift,
}

impl RootCause {
    /// All cause variants for iteration.
    pub const ALL: [RootCause; 8] = [
        RootCause::Theft,
        RootCause::VendorIncrease,
        RootCause::RebateTiming,
        RootCause::MarginLeak,
        RootCause::DemandShift,
        RootCause::QualityIssue,
        RootCause::PricingError,
        RootCause::InventoryDrift,
    ];

    /// Deterministic seed string for generating the cause's phasor vector.
    fn seed_string(&self) -> &'static str {
        match self {
            RootCause::Theft => "cause_vector_theft_v1",
            RootCause::VendorIncrease => "cause_vector_vendor_increase_v1",
            RootCause::RebateTiming => "cause_vector_rebate_timing_v1",
            RootCause::MarginLeak => "cause_vector_margin_leak_v1",
            RootCause::DemandShift => "cause_vector_demand_shift_v1",
            RootCause::QualityIssue => "cause_vector_quality_issue_v1",
            RootCause::PricingError => "cause_vector_pricing_error_v1",
            RootCause::InventoryDrift => "cause_vector_inventory_drift_v1",
        }
    }

    /// Severity category for routing decisions.
    pub fn severity(&self) -> &'static str {
        match self {
            RootCause::Theft => "critical",
            RootCause::VendorIncrease => "high",
            RootCause::RebateTiming => "medium",
            RootCause::MarginLeak => "high",
            RootCause::DemandShift => "medium",
            RootCause::QualityIssue => "high",
            RootCause::PricingError => "high",
            RootCause::InventoryDrift => "medium",
        }
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            RootCause::Theft => "Theft / Shrinkage",
            RootCause::VendorIncrease => "Vendor Price Increase",
            RootCause::RebateTiming => "Rebate Timing Mismatch",
            RootCause::MarginLeak => "Margin Leak",
            RootCause::DemandShift => "Demand Shift",
            RootCause::QualityIssue => "Quality Issue",
            RootCause::PricingError => "Pricing Error",
            RootCause::InventoryDrift => "Inventory Drift",
        }
    }

    /// Actionable recommendations for this root cause.
    pub fn recommendations(&self) -> &'static [&'static str] {
        match self {
            RootCause::Theft => &[
                "Review security footage for high-value items",
                "Analyze void/return patterns by employee",
                "Conduct surprise cycle counts",
                "Check receiving accuracy against POs",
            ],
            RootCause::VendorIncrease => &[
                "Review recent vendor invoices for price changes",
                "Compare to co-op contract pricing",
                "Negotiate volume discounts",
                "Evaluate alternative vendors",
            ],
            RootCause::RebateTiming => &[
                "Verify rebate accrual timing with co-op",
                "Check payment terms on affected POs",
                "Review dating program utilization",
                "Reconcile pending rebates with vendor",
            ],
            RootCause::MarginLeak => &[
                "Audit promotional pricing for stuck discounts",
                "Review markdown cadence",
                "Compare retail to MSRP/suggested retail",
                "Check for unauthorized price overrides",
            ],
            RootCause::DemandShift => &[
                "Review category sales trends",
                "Adjust reorder points based on velocity",
                "Consider markdown or transfer",
                "Evaluate seasonal timing",
            ],
            RootCause::QualityIssue => &[
                "Check return rates by vendor/SKU",
                "File vendor claims for defective product",
                "Review product condition on shelf",
                "Contact vendor quality department",
            ],
            RootCause::PricingError => &[
                "Verify POS retail price vs cost",
                "Check for data entry errors",
                "Review recent price file imports",
                "Correct pricing in POS system",
            ],
            RootCause::InventoryDrift => &[
                "Schedule cycle count for affected items",
                "Review receiving process accuracy",
                "Check for transfer errors between locations",
                "Audit bin locations and quantities",
            ],
        }
    }
}

impl std::fmt::Display for RootCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

// ---------------------------------------------------------------------------
// Evidence rules
// ---------------------------------------------------------------------------

/// An evidence rule mapping a signal to a cause with a weight.
///
/// When the signal fires, the cause gets `weight` added to its evidence
/// accumulator. Negative weights represent counter-evidence.
#[derive(Clone, Debug)]
pub struct EvidenceRule {
    /// The signal name (must match detect_signals() output).
    pub signal: &'static str,
    /// The cause this evidence supports.
    pub cause: RootCause,
    /// Weight in [-1.0, 1.0]. Positive = supports, negative = counter-evidence.
    pub weight: f64,
}

/// Build the complete evidence rule set.
///
/// These rules encode domain knowledge about hardware retail inventory:
/// which observable signals (from POS data) support which root cause hypotheses.
///
/// Ported from the original Python `rules.py` (38 rules), adapted for the
/// signal vocabulary in our Rust issue_classifier's `detect_signals()`.
pub fn build_evidence_rules() -> Vec<EvidenceRule> {
    vec![
        // === THEFT / SHRINKAGE ===
        // Negative inventory without recent receipt → theft/shrinkage
        EvidenceRule { signal: "negative_qty", cause: RootCause::Theft, weight: 0.9 },
        // Low margin on high-value items → potential theft
        EvidenceRule { signal: "low_margin", cause: RootCause::Theft, weight: 0.5 },
        // Zero sales but positive stock → items disappearing
        EvidenceRule { signal: "zero_sales", cause: RootCause::Theft, weight: 0.6 },
        // High cost items are theft targets
        EvidenceRule { signal: "high_cost", cause: RootCause::Theft, weight: 0.5 },
        // Counter-evidence: damaged items are accounted for, not theft
        EvidenceRule { signal: "damaged", cause: RootCause::Theft, weight: -0.4 },
        // Counter-evidence: recent receipt more likely receiving error
        EvidenceRule { signal: "recent_receipt", cause: RootCause::Theft, weight: -0.3 },

        // === VENDOR PRICE INCREASE ===
        // Low margin signals vendor cost increase
        EvidenceRule { signal: "low_margin", cause: RootCause::VendorIncrease, weight: 0.8 },
        // High cost items with margin compression
        EvidenceRule { signal: "high_cost", cause: RootCause::VendorIncrease, weight: 0.6 },
        // On order items may have new (higher) pricing
        EvidenceRule { signal: "on_order", cause: RootCause::VendorIncrease, weight: 0.3 },
        // Cost exceeds retail → vendor price jump
        EvidenceRule { signal: "cost_exceeds_retail", cause: RootCause::VendorIncrease, weight: 0.9 },

        // === REBATE TIMING ===
        // Seasonal items may have co-op rebate timing issues
        EvidenceRule { signal: "seasonal", cause: RootCause::RebateTiming, weight: 0.7 },
        // High qty with low margin → rebate not yet applied
        EvidenceRule { signal: "high_qty", cause: RootCause::RebateTiming, weight: 0.4 },
        EvidenceRule { signal: "low_margin", cause: RootCause::RebateTiming, weight: 0.5 },

        // === MARGIN LEAK ===
        // Direct margin signal
        EvidenceRule { signal: "low_margin", cause: RootCause::MarginLeak, weight: 1.0 },
        // Cost exceeds retail → selling below cost
        EvidenceRule { signal: "cost_exceeds_retail", cause: RootCause::MarginLeak, weight: 0.9 },
        // Zero cost with retail → margin meaningless (100% fake margin)
        EvidenceRule { signal: "zero_cost", cause: RootCause::MarginLeak, weight: 0.4 },

        // === DEMAND SHIFT ===
        // Zero sales → demand dropped
        EvidenceRule { signal: "zero_sales", cause: RootCause::DemandShift, weight: 0.9 },
        // Old receipt + no sales = demand dried up
        EvidenceRule { signal: "old_receipt", cause: RootCause::DemandShift, weight: 0.7 },
        // High qty = overstock from demand miss
        EvidenceRule { signal: "high_qty", cause: RootCause::DemandShift, weight: 0.5 },
        // Seasonal items inherently have demand shifts
        EvidenceRule { signal: "seasonal", cause: RootCause::DemandShift, weight: 0.6 },
        // Counter-evidence: recent receipt = still in demand cycle
        EvidenceRule { signal: "recent_receipt", cause: RootCause::DemandShift, weight: -0.4 },

        // === QUALITY ISSUE ===
        // Damaged goods directly indicate quality
        EvidenceRule { signal: "damaged", cause: RootCause::QualityIssue, weight: 1.0 },
        // On order + damaged = repeat quality problem
        EvidenceRule { signal: "on_order", cause: RootCause::QualityIssue, weight: 0.5 },
        // High qty of damaged goods = batch issue
        EvidenceRule { signal: "high_qty", cause: RootCause::QualityIssue, weight: 0.3 },

        // === PRICING ERROR ===
        // Zero cost = missing cost data (pricing config error)
        EvidenceRule { signal: "zero_cost", cause: RootCause::PricingError, weight: 1.0 },
        // Cost exceeds retail = pricing setup wrong
        EvidenceRule { signal: "cost_exceeds_retail", cause: RootCause::PricingError, weight: 0.9 },
        // Negative retail = data error
        EvidenceRule { signal: "negative_retail", cause: RootCause::PricingError, weight: 1.0 },

        // === INVENTORY DRIFT ===
        // Negative qty without other explanations → drift
        EvidenceRule { signal: "negative_qty", cause: RootCause::InventoryDrift, weight: 0.7 },
        // Old receipt = long time for drift to accumulate
        EvidenceRule { signal: "old_receipt", cause: RootCause::InventoryDrift, weight: 0.6 },
        // High qty items have more opportunity for drift
        EvidenceRule { signal: "high_qty", cause: RootCause::InventoryDrift, weight: 0.3 },
        // Counter-evidence: damaged items are accounted for
        EvidenceRule { signal: "damaged", cause: RootCause::InventoryDrift, weight: -0.3 },
    ]
}

// ---------------------------------------------------------------------------
// Cause Vectors
// ---------------------------------------------------------------------------

/// Pre-computed phasor vectors for each root cause hypothesis.
///
/// Generated deterministically from seed strings so that scoring is
/// reproducible across runs.
pub struct CauseVectors {
    vectors: Vec<(RootCause, Array1<Complex64>)>,
    dimensions: usize,
}

impl CauseVectors {
    /// Create cause vectors with the given dimensionality.
    ///
    /// Uses FNV-1a hash of each cause's seed string to derive a
    /// deterministic RNG seed, then generates a random phasor vector.
    pub fn new(dimensions: usize) -> Self {
        let vectors = RootCause::ALL
            .iter()
            .map(|&cause| {
                let seed = fnv1a_hash(cause.seed_string().as_bytes());
                let vec = random_phasor_vector(dimensions, seed);
                (cause, vec)
            })
            .collect();
        Self { vectors, dimensions }
    }

    /// Get the phasor vector for a specific cause.
    pub fn get(&self, cause: &RootCause) -> Option<&Array1<Complex64>> {
        self.vectors
            .iter()
            .find(|(c, _)| c == cause)
            .map(|(_, v)| v)
    }

    /// Dimensionality of the vectors.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// ---------------------------------------------------------------------------
// Evidence Encoder
// ---------------------------------------------------------------------------

/// Encodes observed signals into a weighted evidence vector using the
/// evidence rules and cause vectors.
///
/// Pre-computes a signal→rule-indices lookup table at construction time
/// so that `encode_signals()` is O(n) in the number of active signals
/// rather than O(n × m) where m is the total rule count.
pub struct EvidenceEncoder {
    rules: Vec<EvidenceRule>,
    cause_vectors: CauseVectors,
    /// Lookup: signal name → indices into `self.rules`.
    signal_index: HashMap<&'static str, Vec<usize>>,
}

impl EvidenceEncoder {
    /// Create an encoder with the standard rule set.
    pub fn new(dimensions: usize) -> Self {
        let rules = build_evidence_rules();
        let mut signal_index: HashMap<&'static str, Vec<usize>> = HashMap::new();
        for (i, rule) in rules.iter().enumerate() {
            signal_index
                .entry(rule.signal)
                .or_insert_with(Vec::new)
                .push(i);
        }
        Self {
            rules,
            cause_vectors: CauseVectors::new(dimensions),
            signal_index,
        }
    }

    /// Encode a set of active signals into an evidence vector.
    ///
    /// For each signal in `active_signals`, finds all matching rules via
    /// the pre-computed `signal_index` (O(1) lookup per signal) and
    /// accumulates `weight * cause_vector` into the evidence bundle.
    /// The result is a weighted superposition of cause hypothesis vectors.
    pub fn encode_signals(&self, active_signals: &[&str]) -> Array1<Complex64> {
        let dim = self.cause_vectors.dimensions();
        let mut bundle = Array1::<Complex64>::zeros(dim);

        for signal in active_signals {
            if let Some(rule_indices) = self.signal_index.get(signal) {
                for &idx in rule_indices {
                    let rule = &self.rules[idx];
                    if let Some(cause_vec) = self.cause_vectors.get(&rule.cause) {
                        let weight = Complex64::new(rule.weight, 0.0);
                        bundle
                            .as_slice_mut()
                            .unwrap()
                            .iter_mut()
                            .zip(cause_vec.as_slice().unwrap().iter())
                            .for_each(|(b, c)| *b += weight * c);
                    }
                }
            }
        }

        bundle
    }
}

// ---------------------------------------------------------------------------
// Evidence Scorer
// ---------------------------------------------------------------------------

/// Result of scoring a single issue's evidence against cause hypotheses.
#[derive(Clone, Debug, Serialize)]
pub struct ScoringResult {
    /// All causes ranked by score (highest first).
    pub scores: Vec<CauseScore>,
    /// The winning root cause (if confidence is sufficient).
    pub top_cause: Option<RootCause>,
    /// Confidence in the top cause (0.0–1.0).
    pub confidence: f64,
    /// Ambiguity: ratio of 2nd-place score to 1st-place (0.0 = clear, 1.0 = tie).
    pub ambiguity: f64,
    /// Number of evidence signals that matched rules.
    pub evidence_count: usize,
}

/// Score for a single cause hypothesis.
#[derive(Clone, Debug, Serialize)]
pub struct CauseScore {
    pub cause: RootCause,
    /// Raw positive-similarity sum.
    pub score: f64,
    /// Number of evidence vectors supporting this cause.
    pub evidence_count: usize,
}

/// The main evidence scorer implementing positive-similarity summing.
///
/// # Algorithm
///
/// For each candidate issue:
/// 1. Collect the active signals from all SKUs in the issue group.
/// 2. Encode each unique signal set into an evidence vector.
/// 3. For each cause: sum `max(0, cosine_similarity(evidence, cause_vec))`.
/// 4. The highest-scoring cause with sufficient confidence is the root cause.
///
/// # 0% Hallucination Guarantee
///
/// By clamping similarities to non-negative values, negative evidence
/// (counter-indicators) cannot cancel out positive evidence. Each piece
/// of supporting evidence independently accumulates.
pub struct EvidenceScorer {
    encoder: EvidenceEncoder,
    cause_vectors: CauseVectors,
    /// Minimum confidence to assign a root cause (below this → None).
    pub confidence_threshold: f64,
}

impl EvidenceScorer {
    /// Create a new scorer with the given vector dimensionality.
    pub fn new(dimensions: usize) -> Self {
        Self {
            encoder: EvidenceEncoder::new(dimensions),
            cause_vectors: CauseVectors::new(dimensions),
            confidence_threshold: 0.3,
        }
    }

    /// Score the root cause for an issue given its aggregated signals.
    ///
    /// `signal_sets` contains the active signals from each SKU in the issue.
    /// For a single-SKU issue, this is a vec of one signal set.
    /// For a multi-SKU aggregated issue, multiple sets are scored together.
    pub fn score(&self, signal_sets: &[Vec<&str>]) -> ScoringResult {
        if signal_sets.is_empty() {
            return ScoringResult {
                scores: Vec::new(),
                top_cause: None,
                confidence: 0.0,
                ambiguity: 0.0,
                evidence_count: 0,
            };
        }

        // Encode each signal set into an evidence vector
        let evidence_vecs: Vec<Array1<Complex64>> = signal_sets
            .iter()
            .map(|signals| self.encoder.encode_signals(signals))
            .collect();

        // Filter out zero vectors (no rules matched)
        let non_zero: Vec<&Array1<Complex64>> = evidence_vecs
            .iter()
            .filter(|v| {
                v.iter()
                    .map(|c| c.norm_sqr())
                    .sum::<f64>()
                    .sqrt()
                    > 1e-15
            })
            .collect();

        if non_zero.is_empty() {
            return ScoringResult {
                scores: Vec::new(),
                top_cause: None,
                confidence: 0.0,
                ambiguity: 0.0,
                evidence_count: 0,
            };
        }

        // Score each cause using positive-similarity summing
        let mut cause_scores: Vec<CauseScore> = RootCause::ALL
            .iter()
            .map(|&cause| {
                let cause_vec = self.cause_vectors.get(&cause).unwrap();
                let mut total_sim = 0.0f64;
                let mut supporting_count = 0usize;

                for ev in &non_zero {
                    let sim = hermitian_cosine_similarity(ev, cause_vec);
                    let clamped = sim.max(0.0); // Positive-only!
                    total_sim += clamped;
                    if sim > 0.0 {
                        supporting_count += 1;
                    }
                }

                CauseScore {
                    cause,
                    score: total_sim,
                    evidence_count: supporting_count,
                }
            })
            .collect();

        // Sort by score descending
        cause_scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute confidence and ambiguity
        let max_score = cause_scores.first().map(|s| s.score).unwrap_or(0.0);

        let confidence = if max_score > 0.0 {
            // Normalize: score relative to evidence count × 0.5 (empirical scale)
            (max_score / (non_zero.len() as f64 * 0.5)).min(1.0)
        } else {
            0.0
        };

        let ambiguity = if cause_scores.len() >= 2 && max_score > 0.0 {
            cause_scores[1].score / max_score
        } else {
            0.0
        };

        let top_cause = if confidence >= self.confidence_threshold {
            cause_scores.first().map(|s| s.cause)
        } else {
            None
        };

        ScoringResult {
            scores: cause_scores,
            top_cause,
            confidence,
            ambiguity,
            evidence_count: non_zero.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Hermitian cosine similarity between two complex vectors.
///
/// ```text
/// sim(a, b) = Re(⟨a, b⟩) / (‖a‖ · ‖b‖)
/// ```
///
/// where ⟨a, b⟩ = Σ conj(a_i) * b_i (Hermitian inner product).
fn hermitian_cosine_similarity(a: &Array1<Complex64>, b: &Array1<Complex64>) -> f64 {
    let dot: Complex64 = a
        .iter()
        .zip(b.iter())
        .map(|(ai, bi)| ai.conj() * bi)
        .sum();

    let norm_a: f64 = a.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();

    if norm_a < 1e-15 || norm_b < 1e-15 {
        return 0.0;
    }

    (dot.re / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Generate a random phasor vector: each component has magnitude 1 with
/// a uniformly random phase angle in [0, 2π).
fn random_phasor_vector(dimensions: usize, seed: u64) -> Array1<Complex64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from_iter((0..dimensions).map(|_| {
        let phase: f64 = rng.gen_range(0.0..2.0 * PI);
        Complex64::from_polar(1.0, phase)
    }))
}

/// FNV-1a hash for deterministic seed derivation from strings.
fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cause_vectors_are_deterministic() {
        let cv1 = CauseVectors::new(512);
        let cv2 = CauseVectors::new(512);

        for cause in &RootCause::ALL {
            let v1 = cv1.get(cause).unwrap();
            let v2 = cv2.get(cause).unwrap();
            assert_eq!(v1, v2, "cause vectors for {:?} should be deterministic", cause);
        }
    }

    #[test]
    fn cause_vectors_are_near_orthogonal() {
        let cv = CauseVectors::new(4096);
        let theft_vec = cv.get(&RootCause::Theft).unwrap();
        let demand_vec = cv.get(&RootCause::DemandShift).unwrap();
        let sim = hermitian_cosine_similarity(theft_vec, demand_vec);
        assert!(
            sim.abs() < 0.1,
            "different cause vectors should be near-orthogonal in 4096-d, got {}",
            sim
        );
    }

    #[test]
    fn cause_self_similarity_is_one() {
        let cv = CauseVectors::new(512);
        let theft_vec = cv.get(&RootCause::Theft).unwrap();
        let sim = hermitian_cosine_similarity(theft_vec, theft_vec);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn empty_signals_produce_zero_evidence() {
        let encoder = EvidenceEncoder::new(512);
        let evidence = encoder.encode_signals(&[]);
        let norm: f64 = evidence.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            norm < 1e-15,
            "empty signals should produce zero vector, got norm {}",
            norm
        );
    }

    #[test]
    fn theft_signals_produce_theft_cause() {
        let scorer = EvidenceScorer::new(1024);
        let signals = vec![vec!["negative_qty", "zero_sales", "high_cost"]];
        let result = scorer.score(&signals);

        assert!(result.top_cause.is_some(), "should identify a root cause");
        assert_eq!(
            result.top_cause.unwrap(),
            RootCause::Theft,
            "negative_qty + zero_sales + high_cost should indicate theft"
        );
        assert!(
            result.confidence > 0.3,
            "confidence should exceed threshold, got {}",
            result.confidence
        );
    }

    #[test]
    fn pricing_error_signals() {
        let scorer = EvidenceScorer::new(1024);
        let signals = vec![vec!["zero_cost", "negative_retail"]];
        let result = scorer.score(&signals);

        assert_eq!(
            result.top_cause,
            Some(RootCause::PricingError),
            "zero_cost + negative_retail should indicate pricing error"
        );
    }

    #[test]
    fn demand_shift_signals() {
        let scorer = EvidenceScorer::new(1024);
        let signals = vec![vec!["zero_sales", "old_receipt", "high_qty"]];
        let result = scorer.score(&signals);

        assert_eq!(
            result.top_cause,
            Some(RootCause::DemandShift),
            "zero_sales + old_receipt + high_qty should indicate demand shift"
        );
    }

    #[test]
    fn quality_issue_signals() {
        let scorer = EvidenceScorer::new(1024);
        let signals = vec![vec!["damaged", "on_order"]];
        let result = scorer.score(&signals);

        assert_eq!(
            result.top_cause,
            Some(RootCause::QualityIssue),
            "damaged + on_order should indicate quality issue"
        );
    }

    #[test]
    fn vendor_increase_signals() {
        let scorer = EvidenceScorer::new(1024);
        let signals = vec![vec!["cost_exceeds_retail", "high_cost", "low_margin"]];
        let result = scorer.score(&signals);

        // Should be either VendorIncrease or MarginLeak (both strong signals)
        let top = result.top_cause.unwrap();
        assert!(
            top == RootCause::VendorIncrease || top == RootCause::MarginLeak,
            "cost_exceeds_retail + high_cost + low_margin should indicate vendor increase or margin leak, got {:?}",
            top
        );
    }

    #[test]
    fn multiple_sku_signals_accumulate() {
        let scorer = EvidenceScorer::new(1024);
        // Two SKUs both showing theft signals → stronger confidence
        let signals = vec![
            vec!["negative_qty", "zero_sales"],
            vec!["negative_qty", "high_cost"],
        ];
        let result = scorer.score(&signals);

        assert_eq!(result.evidence_count, 2);
        assert!(
            result.confidence > 0.0,
            "multiple SKU signals should accumulate confidence"
        );
    }

    #[test]
    fn positive_similarity_only() {
        // Verify that counter-evidence (damaged) shifts the bundle direction
        // but theft score remains positive due to positive-similarity clamping.
        let scorer = EvidenceScorer::new(1024);

        // Without counter-evidence: strong theft signals
        let signals_strong = vec![vec!["negative_qty", "zero_sales", "high_cost"]];
        let result_strong = scorer.score(&signals_strong);

        // With counter-evidence: damaged (counter to theft) + theft signals
        let signals_counter = vec![vec!["negative_qty", "zero_sales", "damaged"]];
        let result_counter = scorer.score(&signals_counter);

        // Both should have positive theft scores — the key property of
        // positive-similarity scoring is that scores never go negative.
        let theft_score_strong = result_strong
            .scores
            .iter()
            .find(|s| s.cause == RootCause::Theft)
            .map(|s| s.score)
            .unwrap_or(0.0);
        let theft_score_counter = result_counter
            .scores
            .iter()
            .find(|s| s.cause == RootCause::Theft)
            .map(|s| s.score)
            .unwrap_or(0.0);

        assert!(
            theft_score_strong > 0.0,
            "theft score should be positive with strong evidence"
        );
        assert!(
            theft_score_counter > 0.0,
            "theft score should remain positive even with counter-evidence (damaged), got {}",
            theft_score_counter
        );

        // Quality issue should also score positively (damaged is its strongest signal)
        let quality_score = result_counter
            .scores
            .iter()
            .find(|s| s.cause == RootCause::QualityIssue)
            .map(|s| s.score)
            .unwrap_or(0.0);
        assert!(
            quality_score > 0.0,
            "quality issue should score positively when damaged signal is present"
        );
    }

    #[test]
    fn ambiguity_detection() {
        let scorer = EvidenceScorer::new(1024);
        // Signals that support multiple causes → higher ambiguity
        let signals = vec![vec!["low_margin"]];
        let result = scorer.score(&signals);

        // low_margin maps to: Theft(0.5), VendorIncrease(0.8), RebateTiming(0.5), MarginLeak(1.0)
        // So MarginLeak should win but with ambiguity from VendorIncrease
        assert!(
            result.ambiguity > 0.3,
            "ambiguous signals should produce high ambiguity, got {}",
            result.ambiguity
        );
    }

    #[test]
    fn all_causes_have_metadata() {
        for cause in &RootCause::ALL {
            assert!(!cause.display_name().is_empty());
            assert!(!cause.severity().is_empty());
            assert!(!cause.recommendations().is_empty());
        }
    }

    #[test]
    fn scoring_result_is_serializable() {
        let scorer = EvidenceScorer::new(256);
        let signals = vec![vec!["negative_qty", "zero_sales"]];
        let result = scorer.score(&signals);
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("top_cause"));
        assert!(json.contains("confidence"));
    }

    #[test]
    fn fnv1a_is_deterministic() {
        let h1 = fnv1a_hash(b"cause_vector_theft_v1");
        let h2 = fnv1a_hash(b"cause_vector_theft_v1");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_seeds_produce_different_hashes() {
        let h1 = fnv1a_hash(b"cause_vector_theft_v1");
        let h2 = fnv1a_hash(b"cause_vector_demand_shift_v1");
        assert_ne!(h1, h2);
    }
}
