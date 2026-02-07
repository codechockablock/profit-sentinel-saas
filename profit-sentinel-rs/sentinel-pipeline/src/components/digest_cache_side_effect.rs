use async_trait::async_trait;
use std::sync::Arc;

use crate::side_effect::{SideEffect, SideEffectInput};
use crate::types::{AgentQuery, IssueCandidate};

/// STUB: Caches the digest result so repeated queries return instantly.
///
/// In production this would write to Redis or a local cache.
/// Currently only logs the event — no actual caching occurs.
pub struct DigestCacheSideEffect;

#[async_trait]
impl SideEffect<AgentQuery, IssueCandidate> for DigestCacheSideEffect {
    async fn run(
        &self,
        input: Arc<SideEffectInput<AgentQuery, IssueCandidate>>,
    ) -> Result<(), String> {
        log::warn!(
            "digest_cache_side_effect is a stub — cache write skipped for request_id={}",
            input.query.request_id,
        );
        log::info!(
            "request_id={} would cache digest with {} candidates",
            input.query.request_id,
            input.selected_candidates.len()
        );
        Ok(())
    }
}
