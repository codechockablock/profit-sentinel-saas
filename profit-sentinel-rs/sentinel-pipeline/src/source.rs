use async_trait::async_trait;

use crate::util;

#[async_trait]
pub trait Source<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// Decide if this source should run for the given query.
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// Fetch candidates for the given query.
    async fn get_candidates(&self, query: &Q) -> Result<Vec<C>, String>;

    /// Returns a stable name for logging/metrics.
    fn name(&self) -> &str {
        util::short_type_name(std::any::type_name::<Self>())
    }
}
