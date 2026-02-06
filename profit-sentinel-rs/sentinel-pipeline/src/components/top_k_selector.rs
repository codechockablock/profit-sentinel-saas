use crate::selector::Selector;
use crate::types::{AgentQuery, IssueCandidate};

/// Selects the top K candidates by priority score.
///
/// Direct port of the X Algorithm's TopKScoreSelector.
pub struct TopKSelector {
    pub k: usize,
}

impl Default for TopKSelector {
    fn default() -> Self {
        Self { k: 5 }
    }
}

impl Selector<AgentQuery, IssueCandidate> for TopKSelector {
    fn score(&self, candidate: &IssueCandidate) -> f64 {
        candidate.priority_score.unwrap_or(f64::NEG_INFINITY)
    }

    fn size(&self) -> Option<usize> {
        Some(self.k)
    }
}
