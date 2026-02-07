pub mod bundling;
pub mod codebook;
pub mod evidence;
pub mod math;
pub mod primitives;
pub mod similarity;
pub mod thresholds;

#[cfg(feature = "python")]
pub mod py_bindings;

pub use bundling::{bundle_inventory_batch, InventoryRow};
pub use codebook::Codebook;
pub use evidence::{CauseScore, EvidenceScorer, RootCause, ScoringResult};
pub use primitives::VsaPrimitives;
pub use similarity::{cosine_similarity, find_similar};
