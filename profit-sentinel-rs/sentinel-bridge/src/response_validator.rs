//! Response Validator — The output constraint layer.
//!
//! The bridge constrains what the LLM sends TO the world model.
//! This module constrains what the LLM sends TO the user.
//!
//! The LLM never talks to the customer directly. It produces a
//! structured `CustomerResponse` that gets validated before rendering.
//! If validation fails, the response is rejected and the LLM must
//! try again or fall back to dashboard-only mode.
//!
//! This closes the gap: inputs are constrained by the operation enum,
//! outputs are constrained by the response validator. The LLM is
//! bound on both sides.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A structured response the LLM must produce instead of free text.
/// Every field is validated before the customer sees anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerResponse {
    /// The type of response determines what validation rules apply.
    pub response_type: ResponseType,

    /// The actual message to show the customer.
    /// Validated for banned terms, required elements, etc.
    pub message: String,

    /// Findings included in this response (if any).
    /// Each finding is individually validated.
    pub findings: Vec<Finding>,

    /// Transfer recommendations (if any).
    /// Each must include financial comparison or it's rejected.
    pub transfers: Vec<TransferSummary>,

    /// What operations were actually executed to produce this response.
    /// If empty when findings are present, the response is REJECTED.
    /// This prevents the "lazy LLM" problem — it must show its work.
    pub operations_executed: Vec<String>,

    /// Confidence level for the overall response.
    pub confidence: ConfidenceLevel,

    /// Was the battery healthy when this response was generated?
    pub battery_healthy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseType {
    /// Dashboard summary — always allowed.
    Dashboard,
    /// Contains specific findings — requires healthy battery.
    AnalysisWithFindings,
    /// Contains transfer recommendations — requires findings + financials.
    TransferRecommendation,
    /// Answering a specific question — requires operations executed.
    QueryResponse,
    /// System is recalibrating — no findings allowed.
    Recalibrating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    /// What type of anomaly.
    pub finding_type: FindingType,
    /// The SKU or entity this is about.
    pub entity_id: String,
    /// Human-readable description.
    pub description: String,
    /// Dollar impact — REQUIRED. Findings without dollar amounts are rejected.
    pub dollar_impact: Option<f64>,
    /// Confidence level for this specific finding.
    pub confidence: ConfidenceLevel,
    /// Recommended action.
    pub recommended_action: String,
    /// What operation produced this finding (must match operations_executed).
    pub source_operation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FindingType {
    DeadStock,
    Shrinkage,
    MarginErosion,
    PhantomInventory,
    VendorAnomaly,
    SeasonalMisalignment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferSummary {
    pub source_store: String,
    pub dest_store: String,
    pub item_description: String,
    pub units: u32,
    /// REQUIRED — transfers without clearance comparison are rejected.
    pub clearance_recovery: f64,
    /// REQUIRED.
    pub transfer_recovery: f64,
    /// REQUIRED — must equal transfer_recovery - clearance_recovery.
    pub net_benefit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfidenceLevel {
    High,
    Medium,
    Low,
}

/// Validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub violations: Vec<Violation>,
}

#[derive(Debug, Clone)]
pub struct Violation {
    pub rule: &'static str,
    pub detail: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Severity {
    /// Response must be rejected and regenerated.
    Reject,
    /// Response can proceed but violation is logged.
    Warn,
}

/// The banned terms list — words that should never appear in customer-facing output.
/// This is not a prompt instruction. This is a string match that rejects the response.
const BANNED_TERMS: &[&str] = &[
    "resonator",
    "phasor",
    "binding",
    "unbinding",
    "bundling",
    "primitive",
    "eigenvector",
    "hyperdimensional",
    "codebook",
    "cosine similarity",
    "VSA",
    "vector symbolic",
    "transition primitive",
    "battery health",
    "algebraic integrity",
    "convergence rate",
    "state entropy",
    "filler vector",
    "role-filler",
    "superposition",
    "cleanup memory",
    "resonator convergence",
    "contrastive learning",
    "proprioceptive",
    "sovereign collapse",
];

/// Validate a customer response before it reaches the user.
pub fn validate_response(
    response: &CustomerResponse,
    battery_healthy: bool,
) -> ValidationResult {
    let mut violations = Vec::new();

    // ================================================================
    // RULE 1: No banned terminology in customer-facing text
    // ================================================================
    let message_lower = response.message.to_lowercase();
    for term in BANNED_TERMS {
        if message_lower.contains(&term.to_lowercase()) {
            violations.push(Violation {
                rule: "NO_TECHNICAL_JARGON",
                detail: format!(
                    "Response contains banned term '{}'. \
                     Rewrite without technical jargon.",
                    term
                ),
                severity: Severity::Reject,
            });
        }
    }

    // Check findings descriptions too
    for finding in &response.findings {
        let desc_lower = finding.description.to_lowercase();
        for term in BANNED_TERMS {
            if desc_lower.contains(&term.to_lowercase()) {
                violations.push(Violation {
                    rule: "NO_TECHNICAL_JARGON_IN_FINDING",
                    detail: format!(
                        "Finding for {} contains banned term '{}'.",
                        finding.entity_id, term
                    ),
                    severity: Severity::Reject,
                });
            }
        }
    }

    // ================================================================
    // RULE 2: No findings when battery is unhealthy
    // ================================================================
    if !battery_healthy && !response.findings.is_empty() {
        violations.push(Violation {
            rule: "NO_FINDINGS_WHEN_UNHEALTHY",
            detail: "Cannot surface findings when battery health is not \
                     confirmed. Use ResponseType::Recalibrating instead."
                .into(),
            severity: Severity::Reject,
        });
    }

    // ================================================================
    // RULE 3: Every finding must have a dollar amount
    // ================================================================
    for finding in &response.findings {
        if finding.dollar_impact.is_none() {
            violations.push(Violation {
                rule: "FINDING_REQUIRES_DOLLAR_IMPACT",
                detail: format!(
                    "Finding for {} has no dollar impact. \
                     Every finding must connect to dollars.",
                    finding.entity_id
                ),
                severity: Severity::Reject,
            });
        }
    }

    // ================================================================
    // RULE 4: Every finding must have a recommended action
    // ================================================================
    for finding in &response.findings {
        if finding.recommended_action.trim().is_empty() {
            violations.push(Violation {
                rule: "FINDING_REQUIRES_ACTION",
                detail: format!(
                    "Finding for {} has no recommended action.",
                    finding.entity_id
                ),
                severity: Severity::Reject,
            });
        }
    }

    // ================================================================
    // RULE 5: Findings require operations to have been executed
    //         (prevents lazy LLM that generates findings from nothing)
    // ================================================================
    if !response.findings.is_empty() && response.operations_executed.is_empty()
    {
        violations.push(Violation {
            rule: "FINDINGS_REQUIRE_OPERATIONS",
            detail: "Findings present but no operations were executed. \
                     The LLM must actually query the world model, not \
                     generate findings from imagination."
                .into(),
            severity: Severity::Reject,
        });
    }

    // Check each finding's source_operation appears in operations_executed
    if !response.operations_executed.is_empty() {
        let executed: HashSet<&str> = response
            .operations_executed
            .iter()
            .map(|s| s.as_str())
            .collect();
        for finding in &response.findings {
            if !executed.contains(finding.source_operation.as_str()) {
                violations.push(Violation {
                    rule: "FINDING_SOURCE_NOT_EXECUTED",
                    detail: format!(
                        "Finding for {} claims source operation '{}' \
                         but that operation wasn't executed.",
                        finding.entity_id, finding.source_operation
                    ),
                    severity: Severity::Reject,
                });
            }
        }
    }

    // ================================================================
    // RULE 6: Transfer recommendations must include financial comparison
    // ================================================================
    for transfer in &response.transfers {
        if transfer.clearance_recovery <= 0.0 {
            violations.push(Violation {
                rule: "TRANSFER_REQUIRES_CLEARANCE_COMPARISON",
                detail: format!(
                    "Transfer to {} missing clearance recovery amount.",
                    transfer.dest_store
                ),
                severity: Severity::Reject,
            });
        }

        if transfer.transfer_recovery <= 0.0 {
            violations.push(Violation {
                rule: "TRANSFER_REQUIRES_TRANSFER_RECOVERY",
                detail: format!(
                    "Transfer to {} missing transfer recovery amount.",
                    transfer.dest_store
                ),
                severity: Severity::Reject,
            });
        }

        // Verify net_benefit math is correct
        let expected_benefit =
            transfer.transfer_recovery - transfer.clearance_recovery;
        if (transfer.net_benefit - expected_benefit).abs() > 0.01 {
            violations.push(Violation {
                rule: "TRANSFER_NET_BENEFIT_MATH",
                detail: format!(
                    "Transfer to {}: net_benefit ({:.2}) doesn't match \
                     transfer_recovery ({:.2}) - clearance_recovery ({:.2}) \
                     = {:.2}",
                    transfer.dest_store,
                    transfer.net_benefit,
                    transfer.transfer_recovery,
                    transfer.clearance_recovery,
                    expected_benefit
                ),
                severity: Severity::Reject,
            });
        }
    }

    // ================================================================
    // RULE 7: Low confidence findings must use hedging language
    // ================================================================
    for finding in &response.findings {
        if finding.confidence == ConfidenceLevel::Low {
            let desc_lower = finding.description.to_lowercase();
            let has_hedge = desc_lower.contains("may")
                || desc_lower.contains("might")
                || desc_lower.contains("suggest")
                || desc_lower.contains("possible")
                || desc_lower.contains("recommend checking")
                || desc_lower.contains("appears");

            if !has_hedge {
                violations.push(Violation {
                    rule: "LOW_CONFIDENCE_REQUIRES_HEDGE",
                    detail: format!(
                        "Finding for {} has Low confidence but uses \
                         definitive language. Must include hedging \
                         (may, might, suggests, possible, etc.).",
                        finding.entity_id
                    ),
                    severity: Severity::Reject,
                });
            }
        }
    }

    // ================================================================
    // RULE 8: Response type must match content
    // ================================================================
    match &response.response_type {
        ResponseType::Recalibrating => {
            if !response.findings.is_empty()
                || !response.transfers.is_empty()
            {
                violations.push(Violation {
                    rule: "RECALIBRATING_NO_FINDINGS",
                    detail: "Recalibrating response must not contain \
                             findings or transfers."
                        .into(),
                    severity: Severity::Reject,
                });
            }
        }
        ResponseType::Dashboard => {
            if !response.findings.is_empty() {
                violations.push(Violation {
                    rule: "DASHBOARD_NO_FINDINGS",
                    detail: "Dashboard response should not contain \
                             findings. Use AnalysisWithFindings type."
                        .into(),
                    severity: Severity::Warn,
                });
            }
        }
        ResponseType::TransferRecommendation => {
            if response.transfers.is_empty() {
                violations.push(Violation {
                    rule: "TRANSFER_TYPE_REQUIRES_TRANSFERS",
                    detail: "TransferRecommendation response has no \
                             transfers."
                        .into(),
                    severity: Severity::Reject,
                });
            }
        }
        _ => {}
    }

    let valid = !violations
        .iter()
        .any(|v| v.severity == Severity::Reject);

    ValidationResult { valid, violations }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_finding() -> Finding {
        Finding {
            finding_type: FindingType::DeadStock,
            entity_id: "SKU-7742".into(),
            description: "Item hasn't sold in 95 days. 47 units \
                          tying up $587 in capital."
                .into(),
            dollar_impact: Some(587.0),
            confidence: ConfidenceLevel::High,
            recommended_action: "Consider transferring to Store 2 \
                                 where similar items are selling."
                .into(),
            source_operation: "Explain".into(),
        }
    }

    fn valid_response() -> CustomerResponse {
        CustomerResponse {
            response_type: ResponseType::AnalysisWithFindings,
            message: "I found 1 item that needs attention.".into(),
            findings: vec![healthy_finding()],
            transfers: vec![],
            operations_executed: vec![
                "Introspect".into(),
                "Snapshot".into(),
                "Explain".into(),
            ],
            confidence: ConfidenceLevel::High,
            battery_healthy: true,
        }
    }

    #[test]
    fn test_valid_response_passes() {
        let result = validate_response(&valid_response(), true);
        assert!(result.valid, "Violations: {:?}", result.violations);
    }

    #[test]
    fn test_banned_term_rejected() {
        let mut resp = valid_response();
        resp.message =
            "The resonator detected an anomaly in your inventory.".into();
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "NO_TECHNICAL_JARGON"));
    }

    #[test]
    fn test_banned_term_in_finding_rejected() {
        let mut resp = valid_response();
        resp.findings[0].description =
            "The phasor binding shows drift in this SKU.".into();
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "NO_TECHNICAL_JARGON_IN_FINDING"));
    }

    #[test]
    fn test_findings_blocked_when_unhealthy() {
        let resp = valid_response();
        let result = validate_response(&resp, false);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "NO_FINDINGS_WHEN_UNHEALTHY"));
    }

    #[test]
    fn test_finding_without_dollars_rejected() {
        let mut resp = valid_response();
        resp.findings[0].dollar_impact = None;
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "FINDING_REQUIRES_DOLLAR_IMPACT"));
    }

    #[test]
    fn test_finding_without_action_rejected() {
        let mut resp = valid_response();
        resp.findings[0].recommended_action = "".into();
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "FINDING_REQUIRES_ACTION"));
    }

    #[test]
    fn test_findings_without_operations_rejected() {
        let mut resp = valid_response();
        resp.operations_executed = vec![];
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "FINDINGS_REQUIRE_OPERATIONS"));
    }

    #[test]
    fn test_finding_source_must_match_executed() {
        let mut resp = valid_response();
        resp.findings[0].source_operation = "MagicOperation".into();
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "FINDING_SOURCE_NOT_EXECUTED"));
    }

    #[test]
    fn test_transfer_requires_financials() {
        let mut resp = valid_response();
        resp.response_type = ResponseType::TransferRecommendation;
        resp.transfers = vec![TransferSummary {
            source_store: "Store 1".into(),
            dest_store: "Store 2".into(),
            item_description: "Cabinet Pulls".into(),
            units: 47,
            clearance_recovery: 0.0, // Missing!
            transfer_recovery: 1174.53,
            net_benefit: 1174.53,
        }];
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "TRANSFER_REQUIRES_CLEARANCE_COMPARISON"));
    }

    #[test]
    fn test_transfer_math_verified() {
        let mut resp = valid_response();
        resp.response_type = ResponseType::TransferRecommendation;
        resp.transfers = vec![TransferSummary {
            source_store: "Store 1".into(),
            dest_store: "Store 2".into(),
            item_description: "Cabinet Pulls".into(),
            units: 47,
            clearance_recovery: 587.26,
            transfer_recovery: 1174.53,
            net_benefit: 999.99, // Wrong math!
        }];
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "TRANSFER_NET_BENEFIT_MATH"));
    }

    #[test]
    fn test_low_confidence_requires_hedging() {
        let mut resp = valid_response();
        resp.findings[0].confidence = ConfidenceLevel::Low;
        resp.findings[0].description =
            "This SKU is definitely dead stock.".into();
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "LOW_CONFIDENCE_REQUIRES_HEDGE"));
    }

    #[test]
    fn test_low_confidence_with_hedge_passes() {
        let mut resp = valid_response();
        resp.findings[0].confidence = ConfidenceLevel::Low;
        resp.findings[0].description =
            "This SKU may be experiencing reduced demand.".into();
        let result = validate_response(&resp, true);
        assert!(result.valid, "Violations: {:?}", result.violations);
    }

    #[test]
    fn test_recalibrating_no_findings() {
        let mut resp = valid_response();
        resp.response_type = ResponseType::Recalibrating;
        // Findings still present — violation
        let result = validate_response(&resp, true);
        assert!(!result.valid);
        assert!(result
            .violations
            .iter()
            .any(|v| v.rule == "RECALIBRATING_NO_FINDINGS"));
    }

    #[test]
    fn test_dashboard_only_passes() {
        let resp = CustomerResponse {
            response_type: ResponseType::Dashboard,
            message: "All stores looking healthy today.".into(),
            findings: vec![],
            transfers: vec![],
            operations_executed: vec!["Snapshot".into()],
            confidence: ConfidenceLevel::High,
            battery_healthy: true,
        };
        let result = validate_response(&resp, true);
        assert!(result.valid);
    }

    #[test]
    fn test_valid_transfer_passes() {
        let resp = CustomerResponse {
            response_type: ResponseType::TransferRecommendation,
            message: "Found a transfer opportunity.".into(),
            findings: vec![healthy_finding()],
            transfers: vec![TransferSummary {
                source_store: "Store 1".into(),
                dest_store: "Store 2".into(),
                item_description: "Cabinet Pulls".into(),
                units: 47,
                clearance_recovery: 587.26,
                transfer_recovery: 1174.53,
                net_benefit: 587.27,
            }],
            operations_executed: vec![
                "Introspect".into(),
                "Explain".into(),
                "FindTransfers".into(),
            ],
            confidence: ConfidenceLevel::High,
            battery_healthy: true,
        };
        let result = validate_response(&resp, true);
        assert!(result.valid, "Violations: {:?}", result.violations);
    }
}
