use async_trait::async_trait;

use crate::query_hydrator::QueryHydrator;
use crate::types::AgentQuery;

/// Hydrates the query with a default time range if none is provided.
///
/// Demonstrates the QueryHydrator trait. In production this would
/// resolve relative time ranges ("last 24 hours") to absolute timestamps.
pub struct TimeRangeQueryHydrator;

#[async_trait]
impl QueryHydrator<AgentQuery> for TimeRangeQueryHydrator {
    async fn hydrate(&self, query: &AgentQuery) -> Result<AgentQuery, String> {
        // If the time range is empty, fill in a default.
        if query.time_range.start.is_empty() || query.time_range.end.is_empty() {
            Ok(AgentQuery {
                time_range: crate::types::TimeRange {
                    start: "2025-01-01T00:00:00Z".to_string(),
                    end: "2025-01-02T00:00:00Z".to_string(),
                },
                ..query.clone()
            })
        } else {
            Ok(query.clone())
        }
    }

    fn update(&self, query: &mut AgentQuery, hydrated: AgentQuery) {
        query.time_range = hydrated.time_range;
    }
}
