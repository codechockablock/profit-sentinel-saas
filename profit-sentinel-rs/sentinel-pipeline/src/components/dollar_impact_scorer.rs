use async_trait::async_trait;

use crate::scorer::Scorer;
use crate::types::{AgentQuery, IssueCandidate, TrendDirection};

/// Scores candidates by dollar impact on a log scale, with a trend multiplier.
///
/// Modeled after the X Algorithm's WeightedScorer: each signal dimension
/// gets a weight, and the final score is a weighted combination.
pub struct DollarImpactScorer;

#[async_trait]
impl Scorer<AgentQuery, IssueCandidate> for DollarImpactScorer {
    async fn score(
        &self,
        _query: &AgentQuery,
        candidates: &[IssueCandidate],
    ) -> Result<Vec<IssueCandidate>, String> {
        let scored = candidates
            .iter()
            .map(|c| {
                let base_score = (c.dollar_impact + 1.0).ln(); // log scale, +1 to handle $0
                let trend_multiplier = match c.trend_direction {
                    TrendDirection::Worsening => 1.5,
                    TrendDirection::Stable => 1.0,
                    TrendDirection::Improving => 0.7,
                };
                let confidence_weight = c.confidence;

                IssueCandidate {
                    priority_score: Some(base_score * trend_multiplier * confidence_weight),
                    ..IssueCandidate::default()
                }
            })
            .collect();

        Ok(scored)
    }

    fn update(&self, candidate: &mut IssueCandidate, scored: IssueCandidate) {
        candidate.priority_score = scored.priority_score;
    }
}
