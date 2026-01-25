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
"""

__version__ = "5.0.0"  # Dorian integration

# Default constants (used by legacy API and Dorian)
DEFAULT_DIMENSIONS = 10000
DEFAULT_MAX_CODEBOOK_SIZE = 100000
HIERARCHICAL_CODEBOOK_THRESHOLD = 10000

# Legacy context-based API (v2.1 - Deprecated, use Dorian instead)
try:
    from .legacy.context import (
        AnalysisContext,
        create_analysis_context,
    )

    _CONTEXT_AVAILABLE = True
except ImportError:
    import logging

    logging.getLogger(__name__).debug(
        "Legacy context module not available - using Dorian"
    )
    _CONTEXT_AVAILABLE = False
    AnalysisContext = None
    create_analysis_context = None

# Core analysis functions (proprietary - may not be available in all environments)
try:
    from .core import (
        CATEGORY_ALIASES,
        COST_ALIASES,
        LAST_SALE_ALIASES,
        LEAK_METADATA,
        MARGIN_ALIASES,
        # Data structures
        PRIMITIVES,
        QTY_DIFF_ALIASES,
        # Column aliases (for reference)
        QUANTITY_ALIASES,
        REVENUE_ALIASES,
        SKU_ALIASES,
        SOLD_ALIASES,
        THRESHOLDS,
        VENDOR_ALIASES,
        add_to_codebook,
        # Main functions
        bundle_pos_facts,
        codebook_dict,  # Deprecated - use ctx.codebook instead
        convergence_lock_resonator_gpu,
        get_all_primitives,
        get_primitive_metadata,
        normalize_torch,
        query_bundle,
        reset_codebook,  # Now a no-op for backward compatibility
        # VSA utilities (legacy - prefer context methods)
        seed_hash,
    )

    _CORE_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"Core module not available: {e}")
    _CORE_AVAILABLE = False
    # Set placeholders for optional core imports
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

# Pipeline components (if they exist and are complete)
try:
    from .batch import BatchProcessor, StreamProcessor
    from .codebook import CodebookManager, PersistentCodebook
    from .pipeline import PipelineResult, PipelineStage, TieredPipeline

    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False
    TieredPipeline = None
    PipelineStage = None
    PipelineResult = None
    PersistentCodebook = None
    CodebookManager = None
    BatchProcessor = None
    StreamProcessor = None

# Bridge (if available)
try:
    from .bridge import VSASymbolicBridge

    _BRIDGE_AVAILABLE = True
except ImportError:
    _BRIDGE_AVAILABLE = False
    VSASymbolicBridge = None

# Contradiction Detector (v2.1.0)
try:
    from .contradiction_detector import (
        CONTRADICTORY_PAIRS,
        Contradiction,
        detect_contradictions,
        generate_contradiction_report,
        resolve_contradictions,
    )

    _CONTRADICTION_DETECTOR_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"Contradiction detector not available: {e}")
    _CONTRADICTION_DETECTOR_AVAILABLE = False
    CONTRADICTORY_PAIRS = None
    Contradiction = None
    detect_contradictions = None
    generate_contradiction_report = None
    resolve_contradictions = None

# Streaming module for large files (v3.0.0)
try:
    from .streaming import (
        StreamingResult,
        StreamingStats,
        bundle_pos_facts_streaming,
        compute_streaming_stats,
        process_dataframe,
        process_large_file,
        read_file_chunked,
    )

    _STREAMING_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"Streaming module not available: {e}")
    _STREAMING_AVAILABLE = False
    StreamingResult = None
    StreamingStats = None
    bundle_pos_facts_streaming = None
    compute_streaming_stats = None
    process_dataframe = None
    process_large_file = None
    read_file_chunked = None

# VSA Evidence Grounding (v4.0.0) - 0% hallucination, 100% multi-hop accuracy
try:
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

    _VSA_EVIDENCE_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"VSA Evidence module not available: {e}")
    _VSA_EVIDENCE_AVAILABLE = False
    # Set placeholders for optional imports
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

# Semantic Flagging (v4.1.0) - Employer review system
try:
    from .flagging import (
        FlagCategory,
        FlaggedQuery,
        FlagSeverity,
        ProfitSentinelFlagIntegration,
        SemanticFlag,
        SemanticFlagDetector,
    )

    _FLAGGING_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"Flagging module not available: {e}")
    _FLAGGING_AVAILABLE = False
    FlagCategory = None
    FlaggedQuery = None
    FlagSeverity = None
    ProfitSentinelFlagIntegration = None
    SemanticFlag = None
    SemanticFlagDetector = None

# Dorian Knowledge Engine (v5.0.0) - Production VSA with FAISS indexing
try:
    from .dorian import (
        DorianCore,
        FactStore,
        InferenceEngine,
        KnowledgePipeline,
        VSAEngine,
    )

    _DORIAN_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"Dorian module not available: {e}")
    _DORIAN_AVAILABLE = False
    DorianCore = None
    VSAEngine = None
    FactStore = None
    InferenceEngine = None
    KnowledgePipeline = None

# Diagnostic Engine (v5.0.0) - Conversational shrinkage diagnostic
try:
    from .diagnostic import (
        ConversationalDiagnostic,
        DetectedPattern,
        DiagnosticSession,
        ProfitSentinelReport,
        generate_report_from_session,
    )

    _DIAGNOSTIC_AVAILABLE = True
except ImportError as e:
    import logging

    logging.getLogger(__name__).debug(f"Diagnostic module not available: {e}")
    _DIAGNOSTIC_AVAILABLE = False
    ConversationalDiagnostic = None
    DiagnosticSession = None
    DetectedPattern = None
    ProfitSentinelReport = None
    generate_report_from_session = None

__all__ = [
    # Version
    "__version__",
    # Dorian (v5.0.0) - PRIMARY API
    "_DORIAN_AVAILABLE",
    "DorianCore",
    "VSAEngine",
    "FactStore",
    "InferenceEngine",
    "KnowledgePipeline",
    # Diagnostic (v5.0.0) - PRIMARY API
    "_DIAGNOSTIC_AVAILABLE",
    "ConversationalDiagnostic",
    "DiagnosticSession",
    "DetectedPattern",
    "ProfitSentinelReport",
    "generate_report_from_session",
    # Constants
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MAX_CODEBOOK_SIZE",
    "HIERARCHICAL_CODEBOOK_THRESHOLD",
    # Legacy Context API (v2.1 - Deprecated)
    "_CONTEXT_AVAILABLE",
    "AnalysisContext",
    "create_analysis_context",
    # Core availability flag (legacy)
    "_CORE_AVAILABLE",
    # Flagging (v4.1.0)
    "_FLAGGING_AVAILABLE",
    "FlagCategory",
    "FlaggedQuery",
    "FlagSeverity",
    "ProfitSentinelFlagIntegration",
    "SemanticFlag",
    "SemanticFlagDetector",
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
    # Pipeline (if available)
    "TieredPipeline",
    "PipelineStage",
    "PipelineResult",
    "PersistentCodebook",
    "CodebookManager",
    "BatchProcessor",
    "StreamProcessor",
    # Bridge (if available)
    "VSASymbolicBridge",
    # Contradiction Detector (v2.1.0)
    "detect_contradictions",
    "resolve_contradictions",
    "generate_contradiction_report",
    "Contradiction",
    "CONTRADICTORY_PAIRS",
    # Streaming (v3.0.0)
    "_STREAMING_AVAILABLE",
    "process_large_file",
    "process_dataframe",
    "StreamingResult",
    "StreamingStats",
    "read_file_chunked",
    "compute_streaming_stats",
    "bundle_pos_facts_streaming",
    # Contradiction Detector (v2.1.0)
    "_CONTRADICTION_DETECTOR_AVAILABLE",
    # VSA Evidence Grounding (v4.0.0)
    "_VSA_EVIDENCE_AVAILABLE",
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
]
