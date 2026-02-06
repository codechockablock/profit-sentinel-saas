use async_trait::async_trait;
use std::cmp::Ordering;
use std::collections::HashMap;

use crate::scorer::Scorer;
use crate::types::{AgentQuery, IssueCandidate};

/// Attenuates scores for repeated stores, ensuring the executive digest
/// covers multiple locations instead of all issues from one store.
///
/// Modeled directly on the X Algorithm's AuthorDiversityScorer: candidates
/// are sorted by current score, then each subsequent appearance of the same
/// store_id is attenuated by `decay_factor^position`.
pub struct StoreDiversityScorer {
    pub decay_factor: f64,
    pub floor: f64,
}

impl Default for StoreDiversityScorer {
    fn default() -> Self {
        Self {
            decay_factor: 0.7,
            floor: 0.1,
        }
    }
}

impl StoreDiversityScorer {
    fn multiplier(&self, position: usize) -> f64 {
        (1.0 - self.floor) * self.decay_factor.powf(position as f64) + self.floor
    }
}

#[async_trait]
impl Scorer<AgentQuery, IssueCandidate> for StoreDiversityScorer {
    async fn score(
        &self,
        _query: &AgentQuery,
        candidates: &[IssueCandidate],
    ) -> Result<Vec<IssueCandidate>, String> {
        let mut store_counts: HashMap<String, usize> = HashMap::new();
        let mut scored = vec![IssueCandidate::default(); candidates.len()];

        // Sort by current priority score descending.
        let mut ordered: Vec<(usize, &IssueCandidate)> =
            candidates.iter().enumerate().collect();
        ordered.sort_by(|(_, a), (_, b)| {
            let a_score = a.priority_score.unwrap_or(f64::NEG_INFINITY);
            let b_score = b.priority_score.unwrap_or(f64::NEG_INFINITY);
            b_score.partial_cmp(&a_score).unwrap_or(Ordering::Equal)
        });

        for (original_idx, candidate) in ordered {
            let entry = store_counts
                .entry(candidate.store_id.clone())
                .or_insert(0);
            let position = *entry;
            *entry += 1;

            let multiplier = self.multiplier(position);
            let adjusted = candidate.priority_score.map(|s| s * multiplier);

            scored[original_idx] = IssueCandidate {
                priority_score: adjusted,
                ..IssueCandidate::default()
            };
        }

        Ok(scored)
    }

    fn update(&self, candidate: &mut IssueCandidate, scored: IssueCandidate) {
        candidate.priority_score = scored.priority_score;
    }
}
