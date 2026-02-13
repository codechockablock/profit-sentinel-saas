//! Bridge Protocol — Request parsing, execution, and response formatting.
//!
//! This is where the constraint enforcement happens:
//! 1. LLM text -> parse into VSAOperation (reject if invalid)
//! 2. Validate parameters (reject if out of bounds)
//! 3. Check proprioceptive state (refuse if model is unhealthy)
//! 4. Execute operation
//! 5. Format response for LLM consumption
//! 6. Log the operation for audit trail

use crate::error::{BridgeError, BridgeResult};
use crate::ops::VSAOperation;
use crate::state::*;
use serde::{Deserialize, Serialize};

/// A request from the LLM to the VSA world model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeRequest {
    /// The operation to perform.
    pub operation: VSAOperation,

    /// Request ID for tracking.
    pub request_id: String,

    /// Optional context: why is the LLM making this request?
    pub context: Option<String>,
}

/// A response from the VSA world model to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeResponse {
    /// The operation result.
    pub result: OperationResult,

    /// Request ID (echoed back).
    pub request_id: String,

    /// Was this operation read-only?
    pub read_only: bool,

    /// Current model status after operation.
    pub model_status: ModelStatus,

    /// Proprioceptive context to include in LLM's next prompt.
    /// This is the Session 18c feedback mechanism.
    pub feedback: Option<String>,
}

/// Audit log entry for state mutations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: u64,
    pub request_id: String,
    pub operation: String,
    pub was_read_only: bool,
    pub success: bool,
    pub error: Option<String>,
    pub model_status_before: ModelStatus,
    pub model_status_after: ModelStatus,
}

/// The Bridge — orchestrates communication between LLM and VSA.
pub struct Bridge {
    /// Audit log of all operations.
    pub audit_log: Vec<AuditEntry>,

    /// Maximum prediction horizon (prevent runaway dreaming).
    pub max_predict_steps: u32,

    /// Maximum counterfactual horizon.
    pub max_counterfactual_horizon: u32,

    /// Should mutations be refused when model status is unhealthy?
    pub sovereign_collapse_enabled: bool,

    /// Current model status (updated by the VSA world model).
    pub current_status: ModelStatus,

    /// Step counter for audit.
    step: u64,
}

impl Bridge {
    pub fn new() -> Self {
        Bridge {
            audit_log: Vec::new(),
            max_predict_steps: 50,
            max_counterfactual_horizon: 20,
            sovereign_collapse_enabled: true,
            current_status: ModelStatus::Warming,
            step: 0,
        }
    }

    /// Parse raw LLM output into a validated BridgeRequest.
    ///
    /// This is the first line of defense: if the LLM's output
    /// doesn't parse into a valid operation, it's rejected here.
    pub fn parse_request(&self, raw_json: &str) -> BridgeResult<BridgeRequest> {
        let request: BridgeRequest =
            serde_json::from_str(raw_json).map_err(|e| {
                BridgeError::UnknownOperation(format!(
                    "Failed to parse request: {}",
                    e
                ))
            })?;

        // Validate parameters
        self.validate_operation(&request.operation)?;

        Ok(request)
    }

    /// Validate operation parameters.
    fn validate_operation(&self, op: &VSAOperation) -> BridgeResult<()> {
        match op {
            VSAOperation::Predict { steps } => {
                if *steps > self.max_predict_steps {
                    return Err(BridgeError::InvalidParameter {
                        op: "Predict".into(),
                        reason: format!(
                            "steps={} exceeds max={}",
                            steps, self.max_predict_steps
                        ),
                    });
                }
                if *steps == 0 {
                    return Err(BridgeError::InvalidParameter {
                        op: "Predict".into(),
                        reason: "steps must be > 0".into(),
                    });
                }
            }
            VSAOperation::Counterfactual { horizon, .. } => {
                if *horizon > self.max_counterfactual_horizon {
                    return Err(BridgeError::InvalidParameter {
                        op: "Counterfactual".into(),
                        reason: format!(
                            "horizon={} exceeds max={}",
                            horizon, self.max_counterfactual_horizon
                        ),
                    });
                }
            }
            VSAOperation::Trajectory { window, .. } => {
                if *window == 0 || *window > 1000 {
                    return Err(BridgeError::InvalidParameter {
                        op: "Trajectory".into(),
                        reason: format!(
                            "window={} out of range [1, 1000]",
                            window
                        ),
                    });
                }
            }
            _ => {} // Other operations have no parameter constraints
        }
        Ok(())
    }

    /// Check sovereign collapse: should this operation be refused?
    ///
    /// Mutations are refused when the model is in an unhealthy state.
    /// Read-only operations are always allowed (you can always observe).
    /// This prevents the model from making changes it can't verify.
    fn check_sovereign_collapse(
        &self,
        op: &VSAOperation,
    ) -> BridgeResult<()> {
        if !self.sovereign_collapse_enabled {
            return Ok(());
        }

        // Read-only operations are always allowed
        if op.is_read_only() {
            return Ok(());
        }

        // Mutations during unhealthy states are refused
        match &self.current_status {
            ModelStatus::SilentFailure => {
                Err(BridgeError::Refused {
                    reason: "Model in SILENT_FAILURE state \u{2014} mutations \
                             refused until battery health is verified. \
                             Run RunBattery first."
                        .into(),
                    confidence: 0.0,
                })
            }
            ModelStatus::AlgebraicAnomaly => {
                Err(BridgeError::Refused {
                    reason: "Algebraic anomaly detected \u{2014} operations may \
                             produce incorrect results. Run RunBattery \
                             to diagnose."
                        .into(),
                    confidence: 0.0,
                })
            }
            // Allow mutations in other states, but with warnings
            _ => Ok(()),
        }
    }

    /// Process a validated request.
    ///
    /// This is where the operation would be dispatched to the actual
    /// VSA world model. In this bridge layer, we handle:
    /// - Sovereign collapse checks
    /// - Audit logging
    /// - Proprioceptive feedback generation
    ///
    /// The actual VSA execution happens in the Python world model
    /// (or future Rust world model), called via the dispatch method.
    pub fn process(
        &mut self,
        request: &BridgeRequest,
    ) -> BridgeResult<BridgeResponse> {
        let status_before = self.current_status.clone();

        // Sovereign collapse check
        self.check_sovereign_collapse(&request.operation)?;

        // Log the attempt
        self.step += 1;

        // At this point, the operation would be dispatched to the
        // VSA world model. The bridge doesn't execute VSA operations
        // itself — it validates, logs, and wraps the results.
        //
        // The actual dispatch would look like:
        //   let result = world_model.execute(&request.operation)?;
        //
        // For now, we return a placeholder that shows the protocol:
        let result = self.dispatch_placeholder(&request.operation);

        // Generate proprioceptive feedback for the LLM
        let feedback = self.generate_feedback(&request.operation);

        // Audit log
        self.audit_log.push(AuditEntry {
            timestamp: self.step,
            request_id: request.request_id.clone(),
            operation: request.operation.describe(),
            was_read_only: request.operation.is_read_only(),
            success: result.is_ok(),
            error: result.as_ref().err().map(|e| e.to_string()),
            model_status_before: status_before,
            model_status_after: self.current_status.clone(),
        });

        Ok(BridgeResponse {
            result: result?,
            request_id: request.request_id.clone(),
            read_only: request.operation.is_read_only(),
            model_status: self.current_status.clone(),
            feedback,
        })
    }

    /// Generate proprioceptive feedback text for the LLM.
    ///
    /// This is the Session 18c mechanism: structured text that
    /// changes how the LLM reasons about its next action.
    fn generate_feedback(&self, _op: &VSAOperation) -> Option<String> {
        match &self.current_status {
            ModelStatus::Healthy => None, // No feedback needed
            ModelStatus::Warming => Some(
                "NOTE: Model is still warming up. Predictions may be \
                 unreliable. Consider running more observations before \
                 making decisions."
                    .into(),
            ),
            ModelStatus::Degrading => Some(
                "WARNING: Prediction error is increasing. The model's \
                 understanding of the environment may be becoming stale. \
                 Consider running LearnStructure to update transition \
                 primitives."
                    .into(),
            ),
            ModelStatus::SilentFailure => Some(
                "CRITICAL: Silent failure detected \u{2014} the model appears \
                 confident but predictions are inaccurate. State \
                 mutations are BLOCKED. Run RunBattery to diagnose \
                 before proceeding."
                    .into(),
            ),
            ModelStatus::ModeLocked => Some(
                "WARNING: Model is using only one transition primitive. \
                 This suggests the dynamics model has collapsed to a \
                 single attractor. Consider running LearnStructure."
                    .into(),
            ),
            ModelStatus::AlgebraicAnomaly => Some(
                "CRITICAL: Behavioral battery detected algebraic \
                 integrity issues. Operations may produce incorrect \
                 results. State mutations are BLOCKED."
                    .into(),
            ),
        }
    }

    /// **Experimental** — placeholder dispatch for the VSA bridge protocol.
    ///
    /// This method demonstrates the dispatch interface that will route
    /// operations to either the Python VSA world model (via FFI) or a
    /// future Rust-native implementation. Currently returns structured
    /// placeholders showing the expected response format.
    ///
    /// # Stability
    ///
    /// This interface is experimental and may change as the Rust world
    /// model matures. The operation enum variants and result types are
    /// considered stable; the dispatch routing is not.
    fn dispatch_placeholder(
        &self,
        op: &VSAOperation,
    ) -> BridgeResult<OperationResult> {
        // This is where the bridge connects to the actual VSA world model.
        // In production, this would call into the Python model via FFI
        // or into a Rust-native VSA implementation.
        //
        // For now, return structured placeholders that demonstrate
        // the response format.
        match op {
            VSAOperation::Introspect => Ok(OperationResult::IntrospectionResult {
                proprioception: ProprioceptiveState {
                    status: self.current_status.clone(),
                    mean_error: 0.0,
                    error_trend: 0.0,
                    convergence_rate: 0.0,
                    active_primitives: 0,
                    alerts: vec![],
                },
                battery: BatteryHealth {
                    binding: 1.0,
                    chain: 1.0,
                    bundling: 0.5,
                    algebraic: 1.0,
                    transition: 0.0,
                    convergence: 0.0,
                    anomalies: vec![],
                },
            }),
            _ => Err(BridgeError::StateError(
                "VSA world model not connected \u{2014} bridge is in \
                 protocol-only mode"
                    .into(),
            )),
        }
    }

    /// Update model status (called by the VSA world model after each step).
    pub fn update_status(&mut self, status: ModelStatus) {
        self.current_status = status;
    }
}

impl Default for Bridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(op: VSAOperation) -> BridgeRequest {
        BridgeRequest {
            operation: op,
            request_id: "test-001".into(),
            context: None,
        }
    }

    #[test]
    fn test_parse_valid_request() {
        let bridge = Bridge::new();
        let json = r#"{
            "operation": {"op": "Snapshot"},
            "request_id": "req-001",
            "context": "initial state check"
        }"#;
        let req = bridge.parse_request(json);
        assert!(req.is_ok());
    }

    #[test]
    fn test_reject_invalid_json() {
        let bridge = Bridge::new();
        let result = bridge.parse_request("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_excessive_predict_steps() {
        let bridge = Bridge::new();
        let result =
            bridge.validate_operation(&VSAOperation::Predict { steps: 999 });
        assert!(result.is_err());
    }

    #[test]
    fn test_sovereign_collapse_blocks_mutations() {
        let mut bridge = Bridge::new();
        bridge.current_status = ModelStatus::SilentFailure;

        // Read-only should still work
        let read_op = VSAOperation::Snapshot;
        assert!(bridge.check_sovereign_collapse(&read_op).is_ok());

        // Mutations should be blocked
        let mutate_op = VSAOperation::Reset { entity_id: None };
        let result = bridge.check_sovereign_collapse(&mutate_op);
        assert!(result.is_err());
        if let Err(BridgeError::Refused { reason, .. }) = result {
            assert!(reason.contains("SILENT_FAILURE"));
        }
    }

    #[test]
    fn test_sovereign_collapse_allows_healthy_mutations() {
        let mut bridge = Bridge::new();
        bridge.current_status = ModelStatus::Healthy;

        let mutate_op = VSAOperation::LearnStructure;
        assert!(bridge.check_sovereign_collapse(&mutate_op).is_ok());
    }

    #[test]
    fn test_feedback_generated_for_unhealthy_states() {
        let mut bridge = Bridge::new();

        bridge.current_status = ModelStatus::Healthy;
        assert!(bridge
            .generate_feedback(&VSAOperation::Snapshot)
            .is_none());

        bridge.current_status = ModelStatus::SilentFailure;
        let fb = bridge.generate_feedback(&VSAOperation::Snapshot);
        assert!(fb.is_some());
        assert!(fb.unwrap().contains("CRITICAL"));
    }

    #[test]
    fn test_audit_log_populated() {
        let mut bridge = Bridge::new();
        bridge.current_status = ModelStatus::Healthy;

        let req = make_request(VSAOperation::Introspect);
        let _ = bridge.process(&req);

        assert_eq!(bridge.audit_log.len(), 1);
        assert!(bridge.audit_log[0].was_read_only);
        assert!(bridge.audit_log[0].success);
    }

    #[test]
    fn test_full_round_trip() {
        let mut bridge = Bridge::new();
        bridge.current_status = ModelStatus::Healthy;

        // Parse
        let json = r#"{
            "operation": {"op": "Introspect"},
            "request_id": "rt-001",
            "context": null
        }"#;
        let req = bridge.parse_request(json).unwrap();

        // Process
        let resp = bridge.process(&req).unwrap();

        // Verify
        assert_eq!(resp.request_id, "rt-001");
        assert!(resp.read_only);
        assert!(resp.feedback.is_none()); // Healthy = no feedback
    }
}
