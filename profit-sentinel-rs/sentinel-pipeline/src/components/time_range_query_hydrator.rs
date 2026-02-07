use async_trait::async_trait;
use chrono::Utc;

use crate::query_hydrator::QueryHydrator;
use crate::types::AgentQuery;

/// Default lookback window when no time range is provided.
const DEFAULT_LOOKBACK_HOURS: i64 = 24;

/// Hydrates the query with a default time range if none is provided.
///
/// When the query has an empty time range, fills in a window of
/// `lookback_hours` ending at the current UTC time.
pub struct TimeRangeQueryHydrator {
    lookback_hours: i64,
}

impl TimeRangeQueryHydrator {
    pub fn new() -> Self {
        Self {
            lookback_hours: DEFAULT_LOOKBACK_HOURS,
        }
    }

    pub fn with_lookback_hours(lookback_hours: i64) -> Self {
        Self { lookback_hours }
    }
}

impl Default for TimeRangeQueryHydrator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl QueryHydrator<AgentQuery> for TimeRangeQueryHydrator {
    async fn hydrate(&self, query: &AgentQuery) -> Result<AgentQuery, String> {
        if query.time_range.start.is_empty() || query.time_range.end.is_empty() {
            let end = Utc::now();
            let start = end - chrono::Duration::hours(self.lookback_hours);
            Ok(AgentQuery {
                time_range: crate::types::TimeRange {
                    start: start.to_rfc3339(),
                    end: end.to_rfc3339(),
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
