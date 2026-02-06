use async_trait::async_trait;

use crate::filter::{Filter, FilterResult};
use crate::types::{AgentQuery, IssueCandidate};

/// Filters out issues below a minimum dollar impact threshold.
///
/// Modeled after the X Algorithm's AgeFilter: a simple predicate that
/// partitions candidates into kept/removed sets.
pub struct LowImpactFilter {
    pub min_dollar_impact: f64,
}

impl LowImpactFilter {
    pub fn new(min_dollar_impact: f64) -> Self {
        Self { min_dollar_impact }
    }
}

impl Default for LowImpactFilter {
    fn default() -> Self {
        Self {
            min_dollar_impact: 100.0,
        }
    }
}

#[async_trait]
impl Filter<AgentQuery, IssueCandidate> for LowImpactFilter {
    async fn filter(
        &self,
        _query: &AgentQuery,
        candidates: Vec<IssueCandidate>,
    ) -> Result<FilterResult<IssueCandidate>, String> {
        let (kept, removed): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|c| c.dollar_impact >= self.min_dollar_impact);

        Ok(FilterResult { kept, removed })
    }
}
