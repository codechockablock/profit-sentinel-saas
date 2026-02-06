use async_trait::async_trait;
use std::sync::Arc;

use crate::side_effect::{SideEffect, SideEffectInput};
use crate::types::{AgentQuery, IssueCandidate};

/// Caches the digest result so repeated queries return instantly.
///
/// In production this would write to Redis or a local cache.
/// For now it just logs the event.
pub struct DigestCacheSideEffect;

#[async_trait]
impl SideEffect<AgentQuery, IssueCandidate> for DigestCacheSideEffect {
    async fn run(
        &self,
        input: Arc<SideEffectInput<AgentQuery, IssueCandidate>>,
    ) -> Result<(), String> {
        log::info!(
            "request_id={} cached digest with {} candidates",
            input.query.request_id,
            input.selected_candidates.len()
        );
        Ok(())
    }
}
