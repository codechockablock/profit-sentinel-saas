use async_trait::async_trait;

use crate::hydrator::Hydrator;
use crate::types::{AgentQuery, IssueCandidate};

/// STUB: Hydrates candidates with additional store context.
///
/// In production this would fetch store metadata (name, region, manager)
/// from the database and attach it to each candidate. Currently sets
/// urgency_score from trend direction as a placeholder.
pub struct StoreContextHydrator;

#[async_trait]
impl Hydrator<AgentQuery, IssueCandidate> for StoreContextHydrator {
    async fn hydrate(
        &self,
        _query: &AgentQuery,
        candidates: &[IssueCandidate],
    ) -> Result<Vec<IssueCandidate>, String> {
        log::warn!(
            "store_context_hydrator is a stub — returning simulated urgency scores for {} candidates",
            candidates.len()
        );
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
