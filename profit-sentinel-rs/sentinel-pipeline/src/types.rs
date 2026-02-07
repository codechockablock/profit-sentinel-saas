use std::fmt;

use serde::Serialize;
use sentinel_vsa::evidence::{CauseScore, RootCause};

use crate::candidate_pipeline::HasRequestId;

// ---------------------------------------------------------------------------
// Query types
// ---------------------------------------------------------------------------

/// Time range for analysis queries.
#[derive(Clone, Debug, Serialize)]
pub struct TimeRange {
    /// ISO-8601 start timestamp.
    pub start: String,
    /// ISO-8601 end timestamp.
    pub end: String,
}

/// Priority filters the agent can apply to narrow results.
#[derive(Clone, Debug, Default)]
pub struct PriorityFilters {
    pub min_dollar_impact: Option<f64>,
    pub issue_types: Option<Vec<IssueType>>,
    pub store_ids: Option<Vec<String>>,
}

/// The role of the user making the query.
#[derive(Clone, Debug)]
pub enum UserRole {
    Executive,
    StoreManager { store_id: String },
}

/// Query from an executive or store manager agent.
#[derive(Clone, Debug)]
pub struct AgentQuery {
    pub request_id: String,
    pub user_id: String,
    pub user_role: UserRole,
    pub store_ids: Vec<String>,
    pub time_range: TimeRange,
    pub priority_filters: Option<PriorityFilters>,
}

impl HasRequestId for AgentQuery {
    fn request_id(&self) -> &str {
        &self.request_id
    }
}

// ---------------------------------------------------------------------------
// Candidate types
// ---------------------------------------------------------------------------

/// The type of issue detected.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum IssueType {
    ReceivingGap,
    DeadStock,
    MarginErosion,
    NegativeInventory,
    VendorShortShip,
    PurchasingLeakage,
    PatronageMiss,
    ShrinkagePattern,
    ZeroCostAnomaly,
    PriceDiscrepancy,
    Overstock,
}

impl fmt::Display for IssueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IssueType::ReceivingGap => write!(f, "Receiving Gap"),
            IssueType::DeadStock => write!(f, "Dead Stock"),
            IssueType::MarginErosion => write!(f, "Margin Erosion"),
            IssueType::NegativeInventory => write!(f, "Negative Inventory"),
            IssueType::VendorShortShip => write!(f, "Vendor Short Ship"),
            IssueType::PurchasingLeakage => write!(f, "Purchasing Leakage"),
            IssueType::PatronageMiss => write!(f, "Patronage Miss"),
            IssueType::ShrinkagePattern => write!(f, "Shrinkage Pattern"),
            IssueType::ZeroCostAnomaly => write!(f, "Zero Cost Anomaly"),
            IssueType::PriceDiscrepancy => write!(f, "Price Discrepancy"),
            IssueType::Overstock => write!(f, "Overstock"),
        }
    }
}

/// Which direction the trend is heading.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum TrendDirection {
    Worsening,
    Stable,
    Improving,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendDirection::Worsening => write!(f, "\u{2191} Worsening"),
            TrendDirection::Stable => write!(f, "\u{2192} Stable"),
            TrendDirection::Improving => write!(f, "\u{2193} Improving"),
        }
    }
}

/// A candidate issue/opportunity to surface to the agent.
#[derive(Clone, Debug, Serialize)]
pub struct IssueCandidate {
    pub id: String,
    pub issue_type: IssueType,
    pub store_id: String,
    pub sku_ids: Vec<String>,
    pub dollar_impact: f64,
    pub confidence: f64,
    pub trend_direction: TrendDirection,
    pub detection_timestamp: String,

    // Scoring fields (populated by scorers)
    pub priority_score: Option<f64>,
    pub urgency_score: Option<f64>,
    pub executive_relevance: Option<f64>,

    // Evidence-based root cause attribution (populated by classify_and_aggregate)
    /// The most likely root cause for this issue, determined by positive-similarity
    /// scoring against evidence vectors from the active signals.
    pub root_cause: Option<RootCause>,
    /// Confidence in the root cause attribution (0.0–1.0).
    pub root_cause_confidence: Option<f64>,

    // Phase 13: Detailed evidence for symbolic bridge
    /// All 8 cause scores, ranked by score (highest first).
    pub cause_scores: Vec<CauseScore>,
    /// Ambiguity ratio: 2nd-place score / 1st-place score (0.0=clear, 1.0=tie).
    pub root_cause_ambiguity: Option<f64>,
    /// Unique active signals aggregated across all SKUs in this issue.
    pub active_signals: Vec<String>,
}

impl Default for IssueCandidate {
    fn default() -> Self {
        Self {
            id: String::new(),
            issue_type: IssueType::ReceivingGap,
            store_id: String::new(),
            sku_ids: Vec::new(),
            dollar_impact: 0.0,
            confidence: 0.0,
            trend_direction: TrendDirection::Stable,
            detection_timestamp: String::new(),
            priority_score: None,
            urgency_score: None,
            executive_relevance: None,
            root_cause: None,
            root_cause_confidence: None,
            cause_scores: Vec::new(),
            root_cause_ambiguity: None,
            active_signals: Vec::new(),
        }
    }
}
