//! VSA Operations — The complete vocabulary of valid actions.
//!
//! The LLM's output gets parsed into exactly one of these variants.
//! If it doesn't parse, the operation is REJECTED. No partial execution,
//! no silent failures, no malformed state mutations.
//!
//! This enum is exhaustive. The compiler guarantees every variant
//! has a handler in the protocol module. Adding a new operation
//! requires handling it everywhere — you can't forget.

use serde::{Deserialize, Serialize};

/// Every valid operation the LLM can request of the VSA world model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", content = "params")]
pub enum VSAOperation {
    // ========================================
    // STATE QUERIES (read-only, always safe)
    // ========================================

    /// Query the current content of a role slot.
    /// Returns: the filler vector's closest codebook match + similarity.
    Query { role: String },

    /// Query all slots, returning the full state snapshot.
    /// Returns: per-slot content, entropy, attention levels.
    Snapshot,

    /// Get the prediction error for a specific entity.
    /// Returns: mean error, error trend, rank among all entities.
    EntityError { entity_id: String },

    /// Get the current attention map.
    /// Returns: per-slot attention levels, high-attention slots.
    AttentionMap,

    /// Get proprioceptive state summary.
    /// Returns: battery health, alerts, error trajectory.
    Introspect,

    // ========================================
    // PREDICTION (forward model, no mutation)
    // ========================================

    /// Predict next state N steps forward without new observations.
    /// This is "dreaming" — running the dynamics model forward.
    /// Returns: predicted state, confidence at each step, divergence.
    Predict { steps: u32 },

    /// Compare two entities: how different are their current states
    /// and recent trajectories?
    /// Returns: per-slot similarity, error difference, trajectory divergence.
    Compare {
        entity_a: String,
        entity_b: String,
    },

    /// Counterfactual: what would the state look like if we changed
    /// one slot's value? Runs the transition model forward with the
    /// hypothetical change.
    /// Returns: predicted state delta, affected slots, confidence.
    Counterfactual {
        entity_id: String,
        role: String,
        new_value: String,
        horizon: u32,
    },

    // ========================================
    // EXPLANATION (interpretation, no mutation)
    // ========================================

    /// Explain why an entity has high prediction error.
    /// Returns: per-slot error breakdown, which transition primitive
    /// fired, how it differs from expected, similar historical patterns.
    Explain { entity_id: String },

    /// What transition primitive is dominant right now?
    /// Returns: primitive name, usage count, what it represents.
    CurrentDynamics,

    /// Show the trajectory of a specific metric over the last N steps.
    /// Returns: time series of the requested metric.
    Trajectory {
        metric: MetricType,
        window: u32,
    },

    // ========================================
    // STATE MUTATION (controlled, audited)
    // ========================================

    /// Feed a new observation into the world model.
    /// This is the primary input — equivalent to receiving new data.
    /// Returns: prediction error, attention update, transition used.
    Observe { observation: Observation },

    /// Force-set a slot value. Used for corrections or hypotheticals.
    /// AUDITED: creates an entry in the modification log.
    /// Returns: previous value, new state entropy.
    SetFiller {
        role: String,
        value: String,
        reason: String,
    },

    /// Reset the model's state for a specific entity.
    /// AUDITED: clears accumulated predictions and errors.
    /// Returns: confirmation + what was cleared.
    Reset { entity_id: Option<String> },

    // ========================================
    // LEARNING (model modification, audited)
    // ========================================

    /// Trigger structure learning: merge/split/add transition primitives
    /// based on accumulated experience.
    /// AUDITED: records what changed and why.
    /// Returns: primitives added/removed/modified, before/after battery.
    LearnStructure,

    /// Run the full behavioral battery and report health.
    /// Returns: all 22 measurements, health scores, anomalies.
    RunBattery,
}

/// Types of metrics that can be tracked over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    PredictionError,
    StateEntropy,
    ConvergenceRate,
    AttentionSpread,
    PrimitiveEntropy,
    BatteryHealth,
}

/// A structured observation that can be fed into the world model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Entity this observation is about (e.g., SKU ID).
    pub entity_id: String,
    /// Role-value pairs (e.g., {"stock_level": "low", "margin": "negative"}).
    pub slots: Vec<SlotValue>,
}

/// A single slot-value pair in an observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotValue {
    pub role: String,
    pub value: String,
}

impl VSAOperation {
    /// Is this operation read-only? Read-only operations never mutate state.
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            VSAOperation::Query { .. }
                | VSAOperation::Snapshot
                | VSAOperation::EntityError { .. }
                | VSAOperation::AttentionMap
                | VSAOperation::Introspect
                | VSAOperation::Predict { .. }
                | VSAOperation::Compare { .. }
                | VSAOperation::Counterfactual { .. }
                | VSAOperation::Explain { .. }
                | VSAOperation::CurrentDynamics
                | VSAOperation::Trajectory { .. }
                | VSAOperation::RunBattery
        )
    }

    /// Does this operation modify model state?
    pub fn is_mutation(&self) -> bool {
        matches!(
            self,
            VSAOperation::Observe { .. }
                | VSAOperation::SetFiller { .. }
                | VSAOperation::Reset { .. }
                | VSAOperation::LearnStructure
        )
    }

    /// Human-readable description of what this operation does.
    pub fn describe(&self) -> String {
        match self {
            VSAOperation::Query { role } => format!("Query slot '{role}'"),
            VSAOperation::Snapshot => "Full state snapshot".into(),
            VSAOperation::EntityError { entity_id } => {
                format!("Get prediction error for '{entity_id}'")
            }
            VSAOperation::AttentionMap => "Get current attention map".into(),
            VSAOperation::Introspect => "Proprioceptive self-check".into(),
            VSAOperation::Predict { steps } => {
                format!("Predict {steps} steps forward")
            }
            VSAOperation::Compare { entity_a, entity_b } => {
                format!("Compare '{entity_a}' vs '{entity_b}'")
            }
            VSAOperation::Counterfactual {
                entity_id,
                role,
                new_value,
                horizon,
            } => format!(
                "What if {entity_id}.{role} = {new_value}? ({horizon} steps)"
            ),
            VSAOperation::Explain { entity_id } => {
                format!("Explain high error for '{entity_id}'")
            }
            VSAOperation::CurrentDynamics => {
                "Current transition dynamics".into()
            }
            VSAOperation::Trajectory { metric, window } => {
                format!("Trajectory of {metric:?} over {window} steps")
            }
            VSAOperation::Observe { observation } => {
                format!("Observe {} with {} slots",
                    observation.entity_id, observation.slots.len())
            }
            VSAOperation::SetFiller { role, value, reason } => {
                format!("Set {role}={value} ({reason})")
            }
            VSAOperation::Reset { entity_id } => match entity_id {
                Some(id) => format!("Reset state for '{id}'"),
                None => "Reset all state".into(),
            },
            VSAOperation::LearnStructure => {
                "Trigger structure learning".into()
            }
            VSAOperation::RunBattery => "Run behavioral battery".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_query() {
        let json = r#"{"op": "Query", "params": {"role": "stock_level"}}"#;
        let op: VSAOperation = serde_json::from_str(json).unwrap();
        assert!(op.is_read_only());
        assert!(!op.is_mutation());
    }

    #[test]
    fn test_parse_observe() {
        let json = r#"{
            "op": "Observe",
            "params": {
                "observation": {
                    "entity_id": "SKU_0001",
                    "slots": [
                        {"role": "stock_level", "value": "low"},
                        {"role": "margin", "value": "negative"}
                    ]
                }
            }
        }"#;
        let op: VSAOperation = serde_json::from_str(json).unwrap();
        assert!(op.is_mutation());
    }

    #[test]
    fn test_parse_counterfactual() {
        let json = r#"{
            "op": "Counterfactual",
            "params": {
                "entity_id": "SKU_0005",
                "role": "stock_level",
                "new_value": "high",
                "horizon": 5
            }
        }"#;
        let op: VSAOperation = serde_json::from_str(json).unwrap();
        assert!(op.is_read_only());
    }

    #[test]
    fn test_invalid_op_rejected() {
        let json = r#"{"op": "DeleteEverything", "params": {}}"#;
        let result: Result<VSAOperation, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_ops_described() {
        // Ensure every operation has a non-empty description
        let ops = vec![
            VSAOperation::Query { role: "test".into() },
            VSAOperation::Snapshot,
            VSAOperation::EntityError { entity_id: "x".into() },
            VSAOperation::AttentionMap,
            VSAOperation::Introspect,
            VSAOperation::Predict { steps: 5 },
            VSAOperation::Compare {
                entity_a: "a".into(),
                entity_b: "b".into(),
            },
            VSAOperation::Counterfactual {
                entity_id: "x".into(),
                role: "r".into(),
                new_value: "v".into(),
                horizon: 3,
            },
            VSAOperation::Explain { entity_id: "x".into() },
            VSAOperation::CurrentDynamics,
            VSAOperation::Trajectory {
                metric: MetricType::PredictionError,
                window: 20,
            },
            VSAOperation::Observe {
                observation: Observation {
                    entity_id: "x".into(),
                    slots: vec![],
                },
            },
            VSAOperation::SetFiller {
                role: "r".into(),
                value: "v".into(),
                reason: "test".into(),
            },
            VSAOperation::Reset { entity_id: None },
            VSAOperation::LearnStructure,
            VSAOperation::RunBattery,
        ];

        for op in &ops {
            let desc = op.describe();
            assert!(!desc.is_empty(), "Empty description for {:?}", op);
        }
    }
}
