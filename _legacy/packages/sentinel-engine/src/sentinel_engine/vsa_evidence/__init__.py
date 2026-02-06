"""
VSA Evidence Module - Grounded evidence retrieval for profit leak diagnosis.

Based on 16 validated hypotheses from VSA research:
- 0% quantitative hallucination (vs 39.6% ungrounded)
- 100% multi-hop reasoning accuracy
- +586% improvement over random baseline
- 5,059x hot path speedup (0.003ms vs 500ms cold path)
- 95% reduction in retrieval operations vs chunk RAG

Key insight: Encode facts by what causes they support, not generic semantic content.

Usage:
    from sentinel_engine.context import create_analysis_context
    from sentinel_engine.vsa_evidence import (
        create_cause_scorer,
        create_evidence_encoder,
        ScoringResult,
    )

    ctx = create_analysis_context()
    scorer = create_cause_scorer(ctx)

    # Score POS rows for cause identification
    result = scorer.score_rows(rows, context={"avg_margin": 0.3})

    if result.needs_cold_path:
        # Route to LLM for deeper analysis
        pass
    else:
        # Hot path resolution
        print(f"Top cause: {result.top_cause}")
        print(f"Confidence: {result.confidence}")

Reference: RESEARCH_SUMMARY.md
"""

from .causes import (
    CAUSE_KEYS,
    CAUSE_METADATA,
    CauseVectors,
    create_cause_vectors,
    get_cause_metadata,
)
from .encoder import (
    EvidenceEncoder,
    HierarchicalEvidenceEncoder,
    create_evidence_encoder,
    create_hierarchical_encoder,
)
from .knowledge import (
    CAUSE_CATEGORIES,
    RETAIL_KNOWLEDGE,
    KnowledgeGraph,
    create_knowledge_graph,
    get_retail_knowledge,
)
from .rules import (
    RETAIL_EVIDENCE_RULES,
    EvidenceRule,
    RuleEngine,
    create_rule_engine,
    extract_evidence_facts,
)
from .scorer import (
    BatchScorer,
    CauseScore,
    CauseScorer,
    ScoringResult,
    create_batch_scorer,
    create_cause_scorer,
)

__all__ = [
    # Causes
    "CAUSE_KEYS",
    "CAUSE_METADATA",
    "CauseVectors",
    "create_cause_vectors",
    "get_cause_metadata",
    # Rules
    "RETAIL_EVIDENCE_RULES",
    "EvidenceRule",
    "RuleEngine",
    "create_rule_engine",
    "extract_evidence_facts",
    # Encoder
    "EvidenceEncoder",
    "HierarchicalEvidenceEncoder",
    "create_evidence_encoder",
    "create_hierarchical_encoder",
    # Scorer
    "CauseScore",
    "CauseScorer",
    "ScoringResult",
    "BatchScorer",
    "create_cause_scorer",
    "create_batch_scorer",
    # Knowledge (v3.7)
    "RETAIL_KNOWLEDGE",
    "CAUSE_CATEGORIES",
    "KnowledgeGraph",
    "create_knowledge_graph",
    "get_retail_knowledge",
]

__version__ = "1.0.0"
