use async_trait::async_trait;
use std::sync::Arc;

use crate::util;

/// Input provided to side effects after the pipeline completes selection.
#[derive(Clone)]
pub struct SideEffectInput<Q, C> {
    pub query: Arc<Q>,
    pub selected_candidates: Vec<C>,
}

/// A side effect is an action that runs after selection and does not
/// affect the pipeline result. Examples: caching, sending notifications.
#[async_trait]
pub trait SideEffect<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// Decide if this side effect should run.
    fn enable(&self, _query: Arc<Q>) -> bool {
        true
    }

    /// Execute the side effect.
    async fn run(&self, input: Arc<SideEffectInput<Q, C>>) -> Result<(), String>;

    /// Returns a stable name for logging/metrics.
    fn name(&self) -> &str {
        util::short_type_name(std::any::type_name::<Self>())
    }
}
