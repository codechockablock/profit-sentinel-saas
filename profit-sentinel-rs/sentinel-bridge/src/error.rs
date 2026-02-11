//! Bridge error types.
//!
//! Every failure mode has a named variant. No stringly-typed errors.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("Unknown operation: {0}")]
    UnknownOperation(String),

    #[error("Invalid parameter for {op}: {reason}")]
    InvalidParameter { op: String, reason: String },

    #[error("Role not found: {0}")]
    UnknownRole(String),

    #[error("Entity not found: {0}")]
    UnknownEntity(String),

    #[error("Operation refused: {reason} (confidence: {confidence:.3})")]
    Refused { reason: String, confidence: f64 },

    #[error("VSA state error: {0}")]
    StateError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
}

/// Result type alias for bridge operations.
pub type BridgeResult<T> = Result<T, BridgeError>;
