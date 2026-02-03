"""
sentinel_engine - Dorian Knowledge Engine for Retail Analytics

v5.0.0 - Complete Dorian Integration
======================================
This module provides production-grade VSA inference with FAISS indexing:
- Dorian Knowledge Engine (10K dimensions, 10M+ facts)
- Conversational diagnostic engine for shrinkage analysis
- PDF report generation
- Knowledge loaders (Wikidata, arXiv, ConceptNet)

Example (v5.0 - Dorian):
    from sentinel_engine import DorianCore, _DORIAN_AVAILABLE

    if _DORIAN_AVAILABLE:
        dorian = DorianCore(dimensions=10000)
        dorian.add_fact("product", "has_category", "electronics")
        results = dorian.query("product")

Example (v5.0 - Diagnostic):
    from sentinel_engine import ConversationalDiagnostic, _DIAGNOSTIC_AVAILABLE

    if _DIAGNOSTIC_AVAILABLE:
        diagnostic = ConversationalDiagnostic()
        session = diagnostic.start_session(csv_data, "My Store")
        question = diagnostic.get_current_question(session)
        diagnostic.submit_answer(session, "yes")
        report = diagnostic.generate_report(session)

Legacy API (v2.1-v4.x - Deprecated)
===================================
The old context-based API is still available for backward compatibility
but is deprecated. Migrate to Dorian for new development.

    from sentinel_engine import _CORE_AVAILABLE
    if _CORE_AVAILABLE:
        from sentinel_engine import bundle_pos_facts, query_bundle

For internal/advanced use, import from submodules directly:
    from sentinel_engine.vsa_evidence import CauseScorer
    from sentinel_engine.streaming import process_large_file
"""

__version__ = "5.0.0"  # Dorian integration

# Default constants (used by legacy API and Dorian)
# v3.7: Changed from 10000 to 4096 for Dorian (faster, still effective)
DEFAULT_DIMENSIONS = 4096
DEFAULT_MAX_CODEBOOK_SIZE = 100000
HIERARCHICAL_CODEBOOK_THRESHOLD = 10000

# =============================================================================
# Availability Flags (centralized in _availability.py)
# =============================================================================
from ._availability import (
    _BRIDGE_AVAILABLE,
    _CONTEXT_AVAILABLE,
    _CONTRADICTION_DETECTOR_AVAILABLE,
    _CORE_AVAILABLE,
    _DIAGNOSTIC_AVAILABLE,
    _DORIAN_AVAILABLE,
    _FLAGGING_AVAILABLE,
    _PIPELINE_AVAILABLE,
    _STREAMING_AVAILABLE,
    _VSA_EVIDENCE_AVAILABLE,
)

# =============================================================================
# Primary API (v5.0.0) - Dorian & Diagnostic
# =============================================================================
if _DORIAN_AVAILABLE:
    from .dorian import (
        DorianCore,
        FactStore,
        InferenceEngine,
        KnowledgePipeline,
        VSAEngine,
    )
else:
    DorianCore = None
    VSAEngine = None
    FactStore = None
    InferenceEngine = None
    KnowledgePipeline = None

if _DIAGNOSTIC_AVAILABLE:
    from .diagnostic import (
        ConversationalDiagnostic,
        DetectedPattern,
        DiagnosticSession,
        ProfitSentinelReport,
        generate_report_from_session,
    )
else:
    ConversationalDiagnostic = None
    DiagnosticSession = None
    DetectedPattern = None
    ProfitSentinelReport = None
    generate_report_from_session = None

# =============================================================================
# Legacy Context API (v2.1 - Deprecated, use Dorian)
# =============================================================================
if _CONTEXT_AVAILABLE:
    from .legacy.context import AnalysisContext, create_analysis_context
else:
    AnalysisContext = None
    create_analysis_context = None

# =============================================================================
# Core Functions (proprietary - may not be available)
# =============================================================================
if _CORE_AVAILABLE:
    from .core import (
        CATEGORY_ALIASES,
        COST_ALIASES,
        LAST_SALE_ALIASES,
        LEAK_METADATA,
        MARGIN_ALIASES,
        PRIMITIVES,
        QTY_DIFF_ALIASES,
        QUANTITY_ALIASES,
        REVENUE_ALIASES,
        SKU_ALIASES,
        SOLD_ALIASES,
        THRESHOLDS,
        VENDOR_ALIASES,
        add_to_codebook,
        bundle_pos_facts,
        codebook_dict,
        convergence_lock_resonator_gpu,
        get_all_primitives,
        get_primitive_metadata,
        normalize_torch,
        query_bundle,
        reset_codebook,
        seed_hash,
    )
else:
    CATEGORY_ALIASES = None
    COST_ALIASES = None
    LAST_SALE_ALIASES = None
    LEAK_METADATA = None
    MARGIN_ALIASES = None
    PRIMITIVES = None
    QTY_DIFF_ALIASES = None
    QUANTITY_ALIASES = None
    REVENUE_ALIASES = None
    SKU_ALIASES = None
    SOLD_ALIASES = None
    THRESHOLDS = None
    VENDOR_ALIASES = None
    add_to_codebook = None
    bundle_pos_facts = None
    codebook_dict = None
    convergence_lock_resonator_gpu = None
    get_all_primitives = None
    get_primitive_metadata = None
    normalize_torch = None
    query_bundle = None
    reset_codebook = None
    seed_hash = None

# =============================================================================
# Pipeline Components
# =============================================================================
if _PIPELINE_AVAILABLE:
    from .batch import BatchProcessor, StreamProcessor
    from .codebook import CodebookManager, PersistentCodebook
    from .pipeline import PipelineResult, PipelineStage, TieredPipeline
else:
    TieredPipeline = None
    PipelineStage = None
    PipelineResult = None
    PersistentCodebook = None
    CodebookManager = None
    BatchProcessor = None
    StreamProcessor = None

# =============================================================================
# Bridge
# =============================================================================
if _BRIDGE_AVAILABLE:
    from .bridge import VSASymbolicBridge
else:
    VSASymbolicBridge = None

# =============================================================================
# Contradiction Detector (v2.1.0)
# =============================================================================
if _CONTRADICTION_DETECTOR_AVAILABLE:
    from .contradiction_detector import (
        CONTRADICTORY_PAIRS,
        Contradiction,
        detect_contradictions,
        generate_contradiction_report,
        resolve_contradictions,
    )
else:
    CONTRADICTORY_PAIRS = None
    Contradiction = None
    detect_contradictions = None
    generate_contradiction_report = None
    resolve_contradictions = None

# =============================================================================
# Streaming (v3.0.0)
# =============================================================================
if _STREAMING_AVAILABLE:
    from .streaming import (
        StreamingResult,
        StreamingStats,
        bundle_pos_facts_streaming,
        compute_streaming_stats,
        process_dataframe,
        process_large_file,
        read_file_chunked,
    )
else:
    StreamingResult = None
    StreamingStats = None
    bundle_pos_facts_streaming = None
    compute_streaming_stats = None
    process_dataframe = None
    process_large_file = None
    read_file_chunked = None

# =============================================================================
# VSA Evidence Grounding (v4.0.0)
# =============================================================================
if _VSA_EVIDENCE_AVAILABLE:
    from .routing import (
        AnalysisResult,
        ColdPathRequest,
        HotPathResult,
        RoutingDecision,
        SmartRouter,
        create_smart_router,
    )
    from .vsa_evidence import (
        CAUSE_KEYS,
        CAUSE_METADATA,
        RETAIL_EVIDENCE_RULES,
        BatchScorer,
        CauseScore,
        CauseScorer,
        CauseVectors,
        EvidenceEncoder,
        EvidenceRule,
        HierarchicalEvidenceEncoder,
        RuleEngine,
        ScoringResult,
        create_batch_scorer,
        create_cause_scorer,
        create_cause_vectors,
        create_evidence_encoder,
        create_hierarchical_encoder,
        create_rule_engine,
        extract_evidence_facts,
        get_cause_metadata,
    )
else:
    AnalysisResult = None
    ColdPathRequest = None
    HotPathResult = None
    RoutingDecision = None
    SmartRouter = None
    create_smart_router = None
    CAUSE_KEYS = None
    CAUSE_METADATA = None
    RETAIL_EVIDENCE_RULES = None
    BatchScorer = None
    CauseScore = None
    CauseScorer = None
    CauseVectors = None
    EvidenceEncoder = None
    EvidenceRule = None
    HierarchicalEvidenceEncoder = None
    RuleEngine = None
    ScoringResult = None
    create_batch_scorer = None
    create_cause_scorer = None
    create_cause_vectors = None
    create_evidence_encoder = None
    create_hierarchical_encoder = None
    create_rule_engine = None
    extract_evidence_facts = None
    get_cause_metadata = None

# =============================================================================
# Semantic Flagging (v4.1.0)
# =============================================================================
if _FLAGGING_AVAILABLE:
    from .flagging import (
        FlagCategory,
        FlaggedQuery,
        FlagSeverity,
        ProfitSentinelFlagIntegration,
        SemanticFlag,
        SemanticFlagDetector,
    )
else:
    FlagCategory = None
    FlaggedQuery = None
    FlagSeverity = None
    ProfitSentinelFlagIntegration = None
    SemanticFlag = None
    SemanticFlagDetector = None


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Constants
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MAX_CODEBOOK_SIZE",
    "HIERARCHICAL_CODEBOOK_THRESHOLD",
    # Availability flags
    "_CONTEXT_AVAILABLE",
    "_CORE_AVAILABLE",
    "_PIPELINE_AVAILABLE",
    "_BRIDGE_AVAILABLE",
    "_CONTRADICTION_DETECTOR_AVAILABLE",
    "_STREAMING_AVAILABLE",
    "_VSA_EVIDENCE_AVAILABLE",
    "_FLAGGING_AVAILABLE",
    "_DORIAN_AVAILABLE",
    "_DIAGNOSTIC_AVAILABLE",
    # Dorian (v5.0.0) - PRIMARY API
    "DorianCore",
    "VSAEngine",
    "FactStore",
    "InferenceEngine",
    "KnowledgePipeline",
    # Diagnostic (v5.0.0) - PRIMARY API
    "ConversationalDiagnostic",
    "DiagnosticSession",
    "DetectedPattern",
    "ProfitSentinelReport",
    "generate_report_from_session",
    # Legacy Context API (v2.1 - Deprecated)
    "AnalysisContext",
    "create_analysis_context",
    # Core functions
    "bundle_pos_facts",
    "query_bundle",
    "reset_codebook",
    "get_primitive_metadata",
    "get_all_primitives",
    "seed_hash",
    "normalize_torch",
    "add_to_codebook",
    "convergence_lock_resonator_gpu",
    # Data structures
    "PRIMITIVES",
    "codebook_dict",
    "LEAK_METADATA",
    "THRESHOLDS",
    # Column aliases
    "QUANTITY_ALIASES",
    "COST_ALIASES",
    "REVENUE_ALIASES",
    "SOLD_ALIASES",
    "QTY_DIFF_ALIASES",
    "MARGIN_ALIASES",
    "LAST_SALE_ALIASES",
    "SKU_ALIASES",
    "VENDOR_ALIASES",
    "CATEGORY_ALIASES",
    # Pipeline
    "TieredPipeline",
    "PipelineStage",
    "PipelineResult",
    "PersistentCodebook",
    "CodebookManager",
    "BatchProcessor",
    "StreamProcessor",
    # Bridge
    "VSASymbolicBridge",
    # Contradiction Detector (v2.1.0)
    "detect_contradictions",
    "resolve_contradictions",
    "generate_contradiction_report",
    "Contradiction",
    "CONTRADICTORY_PAIRS",
    "_CONTRADICTION_DETECTOR_AVAILABLE",
    # Streaming (v3.0.0)
    "process_large_file",
    "process_dataframe",
    "StreamingResult",
    "StreamingStats",
    "read_file_chunked",
    "compute_streaming_stats",
    "bundle_pos_facts_streaming",
    # VSA Evidence Grounding (v4.0.0)
    "CAUSE_KEYS",
    "CAUSE_METADATA",
    "RETAIL_EVIDENCE_RULES",
    "CauseVectors",
    "create_cause_vectors",
    "get_cause_metadata",
    "EvidenceRule",
    "RuleEngine",
    "create_rule_engine",
    "extract_evidence_facts",
    "EvidenceEncoder",
    "HierarchicalEvidenceEncoder",
    "create_evidence_encoder",
    "create_hierarchical_encoder",
    "CauseScore",
    "CauseScorer",
    "ScoringResult",
    "BatchScorer",
    "create_cause_scorer",
    "create_batch_scorer",
    # Routing (v4.0.0)
    "SmartRouter",
    "RoutingDecision",
    "HotPathResult",
    "ColdPathRequest",
    "AnalysisResult",
    "create_smart_router",
    # Flagging (v4.1.0)
    "FlagCategory",
    "FlaggedQuery",
    "FlagSeverity",
    "ProfitSentinelFlagIntegration",
    "SemanticFlag",
    "SemanticFlagDetector",
]
