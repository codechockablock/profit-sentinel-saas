use crate::util;

/// Selectors sort and truncate the candidate list after scoring.
pub trait Selector<Q, C>: Send + Sync
where
    Q: Clone + Send + Sync + 'static,
    C: Clone + Send + Sync + 'static,
{
    /// Default selection: sort and truncate based on provided configs.
    fn select(&self, _query: &Q, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = self.sort(candidates);
        if let Some(limit) = self.size() {
            sorted.truncate(limit);
        }
        sorted
    }

    /// Decide if this selector should run for the given query.
    fn enable(&self, _query: &Q) -> bool {
        true
    }

    /// Extract the score from a candidate to use for sorting.
    fn score(&self, candidate: &C) -> f64;

    /// Sort candidates by their scores in descending order.
    ///
    /// NaN scores are pushed to the end of the list so they never appear
    /// as top candidates. This guards against division-by-zero or missing
    /// data producing garbage results at the top of the output.
    fn sort(&self, candidates: Vec<C>) -> Vec<C> {
        let mut sorted = candidates;
        sorted.sort_by(|a, b| {
            let sa = self.score(a);
            let sb = self.score(b);
            // Explicit total ordering: NaN goes to end (Greater)
            match (sa.is_nan(), sb.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        sorted
    }

    /// Optionally provide a maximum number of candidates to select.
    /// Defaults to no truncation if not overridden.
    fn size(&self) -> Option<usize> {
        None
    }

    /// Returns a stable name for logging/metrics.
    fn name(&self) -> &str {
        util::short_type_name(std::any::type_name::<Self>())
    }
}
