use async_trait::async_trait;

use crate::hydrator::Hydrator;
use crate::types::{AgentQuery, IssueCandidate};

/// Hydrates candidates with additional store context.
///
/// In production this would fetch store metadata (name, region, manager)
/// and attach it to each candidate. For now it sets the urgency_score
/// field as a demonstration that hydrators can enrich candidates.
pub struct StoreContextHydrator;

#[async_trait]
impl Hydrator<AgentQuery, IssueCandidate> for StoreContextHydrator {
    async fn hydrate(
        &self,
        _query: &AgentQuery,
        candidates: &[IssueCandidate],
    ) -> Result<Vec<IssueCandidate>, String> {
        let hydrated = candidates
            .iter()
            .map(|c| {
                // Simulate urgency based on trend direction.
                let urgency = match c.trend_direction {
                    crate::types::TrendDirection::Worsening => Some(0.9),
                    crate::types::TrendDirection::Stable => Some(0.5),
                    crate::types::TrendDirection::Improving => Some(0.2),
                };
                IssueCandidate {
                    urgency_score: urgency,
                    ..IssueCandidate::default()
                }
            })
            .collect();
        Ok(hydrated)
    }

    fn update(&self, candidate: &mut IssueCandidate, hydrated: IssueCandidate) {
        candidate.urgency_score = hydrated.urgency_score;
    }
}
