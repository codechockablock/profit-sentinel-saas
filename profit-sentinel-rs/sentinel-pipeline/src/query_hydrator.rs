use async_trait::async_trait;

use crate::util;

/// Query hydrators run in parallel before candidate fetching and
/// enrich the query object with additional context.
#[async_trait]
pub trait QueryHydrator<Q>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
{
    /// Decide if this query hydrator should run for the given query.
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// Hydrate the query by performing async operations.
    /// Returns a new query with this hydrator's fields populated.
    async fn hydrate(&self, query: &Q) -> Result<Q, String>;

    /// Update the query with the hydrated fields.
    /// Only the fields this hydrator is responsible for should be copied.
    fn update(&self, query: &mut Q, hydrated: Q);

    /// Returns a stable name for logging/metrics.
    fn name(&self) -> &str {
        util::short_type_name(std::any::type_name::<Self>())
    }
}
