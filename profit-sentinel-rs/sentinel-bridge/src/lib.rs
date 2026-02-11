//! VSA Bridge — The constraint layer between VSA world model and LLM
//!
//! The LLM outputs text. This bridge parses that text into exactly one
//! of N valid operations, executes it against the VSA state, and returns
//! a structured result the LLM can read.
//!
//! The type system IS the safety layer:
//! - Every operation is an enum variant with validated parameters
//! - Every response is a structured type, not free-form text
//! - Invalid operations are rejected at parse time, not at runtime
//! - The compiler guarantees every operation has a handler
//!
//! This is "sovereign collapse" as a compiler guarantee.

pub mod ops;
pub mod state;
pub mod protocol;
pub mod error;
pub mod response_validator;

pub use ops::VSAOperation;
pub use state::{WorldState, SlotState, BatteryHealth, ProprioceptiveState};
pub use protocol::{BridgeRequest, BridgeResponse, Bridge};
pub use error::BridgeError;
pub use response_validator::{CustomerResponse, validate_response, ValidationResult};
