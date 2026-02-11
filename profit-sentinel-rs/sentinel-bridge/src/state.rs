//! World model state serialization.
//!
//! Converts VSA state into structured, LLM-readable summaries.
//! The LLM never sees raw vectors — it sees interpretable state
//! descriptions with confidence levels and attention markers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete world state as the LLM sees it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    /// Current step number.
    pub step: u64,

    /// Per-slot state.
    pub slots: HashMap<String, SlotState>,

    /// Overall state entropy (how spread is information across slots).
    pub entropy: f64,

    /// Overall surprise level (mean prediction error).
    pub surprise: f64,

    /// Current attention map.
    pub attention: HashMap<String, f64>,

    /// High-attention slots (above threshold).
    pub high_attention: Vec<String>,

    /// Current battery health.
    pub battery: BatteryHealth,

    /// Proprioceptive state.
    pub proprioception: ProprioceptiveState,

    /// Active transition primitives and usage.
    pub dynamics: DynamicsState,
}

/// State of a single role slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotState {
    /// Role name.
    pub role: String,

    /// Best codebook match for current filler.
    pub value: String,

    /// Similarity to best match (confidence).
    pub confidence: f64,

    /// Prediction error for this slot at last observation.
    pub last_error: f64,

    /// Attention level for this slot.
    pub attention: f64,

    /// Is this slot empty (filler approximately equals identity)?
    pub is_empty: bool,
}

/// Battery health summary — derived from the 22-measurement battery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatteryHealth {
    /// Binding operations working correctly? (0.0-1.0)
    pub binding: f64,

    /// Long chains maintaining integrity? (0.0-1.0)
    pub chain: f64,

    /// Bundling maintaining capacity? (0.0-1.0)
    pub bundling: f64,

    /// Algebraic laws holding? (0.0-1.0)
    pub algebraic: f64,

    /// Transition model functioning? (0.0-1.0)
    pub transition: f64,

    /// Resonator converging? (0.0-1.0)
    pub convergence: f64,

    /// Any anomalies detected by the battery.
    pub anomalies: Vec<String>,
}

/// Proprioceptive state — the model's awareness of its own operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProprioceptiveState {
    /// Current status.
    pub status: ModelStatus,

    /// Mean prediction error over recent window.
    pub mean_error: f64,

    /// Error trend (positive = getting worse).
    pub error_trend: f64,

    /// Convergence rate of the resonator.
    pub convergence_rate: f64,

    /// Number of unique primitives used recently.
    pub active_primitives: u32,

    /// Active alerts.
    pub alerts: Vec<Alert>,
}

/// Model operational status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model is operating normally.
    Healthy,
    /// Model is in warmup phase (learning transitions).
    Warming,
    /// Model is degrading (rising error).
    Degrading,
    /// Model has detected silent failure.
    SilentFailure,
    /// Model is stuck in one mode.
    ModeLocked,
    /// Battery detected algebraic issues.
    AlgebraicAnomaly,
}

/// A proprioceptive alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert type.
    pub alert_type: AlertType,
    /// Step when detected.
    pub step: u64,
    /// Human-readable message.
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    Degradation,
    EntropyCollapse,
    ConvergenceFailure,
    ModeLock,
    SilentFailure,
    RecoveryDegradation,
    BundleSaturation,
    AlgebraicIntegrity,
}

/// Current dynamics state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicsState {
    /// All transition primitives.
    pub primitives: Vec<PrimitiveInfo>,

    /// Most recently used primitive.
    pub last_primitive: Option<String>,

    /// Usage uniformity (0.0 = all one primitive, 1.0 = perfectly uniform).
    pub uniformity: f64,
}

/// Info about a transition primitive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveInfo {
    pub name: String,
    pub usage_count: u64,
    pub usage_fraction: f64,
}

/// Response to a specific operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OperationResult {
    /// Result of a Query operation.
    QueryResult {
        role: String,
        value: String,
        confidence: f64,
        attention: f64,
    },

    /// Full state snapshot.
    SnapshotResult { state: WorldState },

    /// Entity error report.
    EntityErrorResult {
        entity_id: String,
        mean_error: f64,
        error_trend: f64,
        rank: u32,
        total_entities: u32,
    },

    /// Attention map.
    AttentionMapResult {
        attention: HashMap<String, f64>,
        high_attention: Vec<String>,
    },

    /// Proprioceptive report.
    IntrospectionResult {
        proprioception: ProprioceptiveState,
        battery: BatteryHealth,
    },

    /// Prediction result.
    PredictionResult {
        steps: Vec<PredictionStep>,
        final_confidence: f64,
        diverged_at: Option<u32>,
    },

    /// Comparison result.
    ComparisonResult {
        entity_a: String,
        entity_b: String,
        per_slot_similarity: HashMap<String, f64>,
        overall_similarity: f64,
        error_difference: f64,
    },

    /// Counterfactual result.
    CounterfactualResult {
        original_state: HashMap<String, String>,
        predicted_state: HashMap<String, String>,
        affected_slots: Vec<String>,
        confidence: f64,
    },

    /// Explanation of high error.
    ExplanationResult {
        entity_id: String,
        per_slot_errors: HashMap<String, f64>,
        dominant_primitive: String,
        similar_historical: Vec<String>,
        suggested_cause: String,
    },

    /// Current dynamics report.
    DynamicsResult { dynamics: DynamicsState },

    /// Trajectory data.
    TrajectoryResult {
        metric: String,
        values: Vec<f64>,
        trend: f64,
    },

    /// Observation result.
    ObserveResult {
        prediction_error: f64,
        per_slot_errors: HashMap<String, f64>,
        transition_used: String,
        attention_update: HashMap<String, f64>,
    },

    /// Set filler result.
    SetFillerResult {
        role: String,
        previous_value: String,
        new_value: String,
        new_entropy: f64,
    },

    /// Reset result.
    ResetResult {
        entities_cleared: Vec<String>,
        steps_cleared: u64,
    },

    /// Structure learning result.
    LearnStructureResult {
        primitives_added: Vec<String>,
        primitives_removed: Vec<String>,
        primitives_modified: Vec<String>,
        before_health: BatteryHealth,
        after_health: BatteryHealth,
    },

    /// Battery report.
    BatteryResult {
        measurements: HashMap<String, f64>,
        health: BatteryHealth,
    },

    /// Operation was refused.
    RefusedResult { reason: String, confidence: f64 },
}

/// A single step in a multi-step prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionStep {
    pub step: u32,
    pub predicted_state: HashMap<String, String>,
    pub confidence: f64,
    pub primitive_used: String,
}

impl WorldState {
    /// Generate an LLM-readable text summary of the current state.
    ///
    /// This is what goes into the LLM's context window.
    /// Concise, structured, no raw vectors.
    pub fn to_context_prompt(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("[WORLD MODEL STATE \u{2014} Step {}]", self.step));
        lines.push(format!(
            "Entropy: {:.3} | Surprise: {:.3}",
            self.entropy, self.surprise
        ));

        // High attention slots first
        if !self.high_attention.is_empty() {
            lines.push(format!(
                "\u{26a0} HIGH ATTENTION: {}",
                self.high_attention.join(", ")
            ));
        }

        // Slot summary
        lines.push(String::new());
        lines.push("Slots:".into());
        for (name, slot) in &self.slots {
            let attn_marker = if slot.attention > 0.3 { "!" } else { " " };
            lines.push(format!(
                "  {}{}: {} (conf={:.2}, err={:.3}, attn={:.3})",
                attn_marker, name, slot.value, slot.confidence,
                slot.last_error, slot.attention
            ));
        }

        // Battery health (only report if issues)
        let battery = &self.battery;
        if !battery.anomalies.is_empty() {
            lines.push(String::new());
            lines.push("\u{26a0} BATTERY ANOMALIES:".into());
            for a in &battery.anomalies {
                lines.push(format!("  - {}", a));
            }
        }

        // Proprioceptive alerts
        if !self.proprioception.alerts.is_empty() {
            lines.push(String::new());
            lines.push("\u{26a0} PROPRIOCEPTION ALERTS:".into());
            for alert in &self.proprioception.alerts {
                lines.push(format!(
                    "  [{:?}] {}",
                    alert.alert_type, alert.message
                ));
            }
        }

        // Status
        lines.push(String::new());
        lines.push(format!("Status: {:?}", self.proprioception.status));
        lines.push(format!(
            "Dynamics: {} primitives, uniformity={:.2}",
            self.dynamics.primitives.len(),
            self.dynamics.uniformity
        ));

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_state_serialization() {
        let state = WorldState {
            step: 42,
            slots: HashMap::new(),
            entropy: 1.386,
            surprise: 0.607,
            attention: HashMap::new(),
            high_attention: vec!["stock_level".into()],
            battery: BatteryHealth {
                binding: 1.0,
                chain: 1.0,
                bundling: 0.518,
                algebraic: 1.0,
                transition: 0.842,
                convergence: 0.06,
                anomalies: vec![],
            },
            proprioception: ProprioceptiveState {
                status: ModelStatus::Healthy,
                mean_error: 0.607,
                error_trend: 0.003,
                convergence_rate: 0.94,
                active_primitives: 12,
                alerts: vec![],
            },
            dynamics: DynamicsState {
                primitives: vec![],
                last_primitive: Some("T_07_n57".into()),
                uniformity: 0.833,
            },
        };

        let json = serde_json::to_string_pretty(&state).unwrap();
        let parsed: WorldState = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.step, 42);
        assert_eq!(parsed.battery.binding, 1.0);
    }

    #[test]
    fn test_context_prompt_generation() {
        let state = WorldState {
            step: 100,
            slots: {
                let mut m = HashMap::new();
                m.insert(
                    "stock_level".into(),
                    SlotState {
                        role: "stock_level".into(),
                        value: "low".into(),
                        confidence: 0.95,
                        last_error: 0.45,
                        attention: 0.6,
                        is_empty: false,
                    },
                );
                m
            },
            entropy: 1.2,
            surprise: 0.65,
            attention: HashMap::new(),
            high_attention: vec!["stock_level".into()],
            battery: BatteryHealth {
                binding: 1.0,
                chain: 1.0,
                bundling: 0.5,
                algebraic: 1.0,
                transition: 0.8,
                convergence: 0.9,
                anomalies: vec![],
            },
            proprioception: ProprioceptiveState {
                status: ModelStatus::Healthy,
                mean_error: 0.65,
                error_trend: 0.001,
                convergence_rate: 0.9,
                active_primitives: 10,
                alerts: vec![],
            },
            dynamics: DynamicsState {
                primitives: vec![],
                last_primitive: None,
                uniformity: 0.8,
            },
        };

        let prompt = state.to_context_prompt();
        assert!(prompt.contains("Step 100"));
        assert!(prompt.contains("HIGH ATTENTION"));
        assert!(prompt.contains("stock_level"));
    }
}
