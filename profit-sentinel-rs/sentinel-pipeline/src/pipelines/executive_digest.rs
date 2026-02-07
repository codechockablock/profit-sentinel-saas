use async_trait::async_trait;
use std::sync::Arc;

use crate::candidate_pipeline::CandidatePipeline;
use crate::components::digest_cache_side_effect::DigestCacheSideEffect;
use crate::components::dollar_impact_scorer::DollarImpactScorer;
use crate::components::glm_analytics_source::GlmAnalyticsSource;
use crate::components::low_impact_filter::LowImpactFilter;
use crate::components::store_context_hydrator::StoreContextHydrator;
use crate::components::store_diversity_scorer::StoreDiversityScorer;
use crate::components::time_range_query_hydrator::TimeRangeQueryHydrator;
use crate::components::top_k_selector::TopKSelector;
use crate::filter::Filter;
use crate::hydrator::Hydrator;
use crate::inventory_loader::InventoryRecord;
use crate::query_hydrator::QueryHydrator;
use crate::scorer::Scorer;
use crate::selector::Selector;
use crate::side_effect::SideEffect;
use crate::source::Source;
use crate::types::{AgentQuery, IssueCandidate};

/// The executive morning digest pipeline.
///
/// Wired up exactly like the X Algorithm's PhoenixCandidatePipeline:
/// each stage has concrete component implementations plugged in.
///
/// Pipeline flow:
/// 1. TimeRangeQueryHydrator fills in default time range
/// 2. GlmAnalyticsSource runs VSA analysis on real inventory data
/// 3. StoreContextHydrator enriches candidates with store metadata
/// 4. LowImpactFilter removes issues below $100
/// 5. DollarImpactScorer assigns priority scores
/// 6. StoreDiversityScorer attenuates repeated stores
/// 7. TopKSelector picks the top N
/// 8. DigestCacheSideEffect caches the result
pub struct ExecutiveDigestPipeline {
    query_hydrators: Vec<Box<dyn QueryHydrator<AgentQuery>>>,
    sources: Vec<Box<dyn Source<AgentQuery, IssueCandidate>>>,
    hydrators: Vec<Box<dyn Hydrator<AgentQuery, IssueCandidate>>>,
    filters: Vec<Box<dyn Filter<AgentQuery, IssueCandidate>>>,
    scorers: Vec<Box<dyn Scorer<AgentQuery, IssueCandidate>>>,
    selector: TopKSelector,
    post_selection_hydrators: Vec<Box<dyn Hydrator<AgentQuery, IssueCandidate>>>,
    post_selection_filters: Vec<Box<dyn Filter<AgentQuery, IssueCandidate>>>,
    side_effects: Arc<Vec<Box<dyn SideEffect<AgentQuery, IssueCandidate>>>>,
    result_size: usize,
}

impl ExecutiveDigestPipeline {
    /// Create a pipeline with real inventory data.
    ///
    /// This is the primary constructor for production use.
    pub fn with_inventory(records: Vec<InventoryRecord>) -> Self {
        Self::with_inventory_and_size(records, 5)
    }

    /// Create a pipeline with inventory data and custom result size.
    pub fn with_inventory_and_size(records: Vec<InventoryRecord>, result_size: usize) -> Self {
        let query_hydrators: Vec<Box<dyn QueryHydrator<AgentQuery>>> =
            vec![Box::new(TimeRangeQueryHydrator::new())];

        let sources: Vec<Box<dyn Source<AgentQuery, IssueCandidate>>> =
            vec![Box::new(GlmAnalyticsSource::new(records))];

        let hydrators: Vec<Box<dyn Hydrator<AgentQuery, IssueCandidate>>> =
            vec![Box::new(StoreContextHydrator)];

        let filters: Vec<Box<dyn Filter<AgentQuery, IssueCandidate>>> =
            vec![Box::new(LowImpactFilter::default())];

        let scorers: Vec<Box<dyn Scorer<AgentQuery, IssueCandidate>>> = vec![
            Box::new(DollarImpactScorer),
            Box::new(StoreDiversityScorer::default()),
        ];

        let selector = TopKSelector { k: result_size };

        let side_effects: Arc<Vec<Box<dyn SideEffect<AgentQuery, IssueCandidate>>>> =
            Arc::new(vec![Box::new(DigestCacheSideEffect)]);

        Self {
            query_hydrators,
            sources,
            hydrators,
            filters,
            scorers,
            selector,
            post_selection_hydrators: Vec::new(),
            post_selection_filters: Vec::new(),
            side_effects,
            result_size,
        }
    }
}

#[async_trait]
impl CandidatePipeline<AgentQuery, IssueCandidate> for ExecutiveDigestPipeline {
    fn query_hydrators(&self) -> &[Box<dyn QueryHydrator<AgentQuery>>] {
        &self.query_hydrators
    }

    fn sources(&self) -> &[Box<dyn Source<AgentQuery, IssueCandidate>>] {
        &self.sources
    }

    fn hydrators(&self) -> &[Box<dyn Hydrator<AgentQuery, IssueCandidate>>] {
        &self.hydrators
    }

    fn filters(&self) -> &[Box<dyn Filter<AgentQuery, IssueCandidate>>] {
        &self.filters
    }

    fn scorers(&self) -> &[Box<dyn Scorer<AgentQuery, IssueCandidate>>] {
        &self.scorers
    }

    fn selector(&self) -> &dyn Selector<AgentQuery, IssueCandidate> {
        &self.selector
    }

    fn post_selection_hydrators(&self) -> &[Box<dyn Hydrator<AgentQuery, IssueCandidate>>] {
        &self.post_selection_hydrators
    }

    fn post_selection_filters(&self) -> &[Box<dyn Filter<AgentQuery, IssueCandidate>>] {
        &self.post_selection_filters
    }

    fn side_effects(&self) -> Arc<Vec<Box<dyn SideEffect<AgentQuery, IssueCandidate>>>> {
        Arc::clone(&self.side_effects)
    }

    fn result_size(&self) -> usize {
        self.result_size
    }
}
