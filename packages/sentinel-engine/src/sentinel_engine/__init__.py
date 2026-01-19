"""
sentinel_engine - Aggressive VSA-based Profit Leak Detection Engine

This module provides the scalable infrastructure for VSA inference:
- Core analysis functions (bundle_pos_facts, query_bundle)
- 8 detection primitives for comprehensive leak detection
- Universal POS column support (Paladin, Square, Lightspeed, etc.)
- $ impact estimation and recommendations

v2.1 Features (Request Isolation):
- AnalysisContext for per-request state isolation
- No global mutable state - thread-safe for concurrent requests
- Backward compatibility with deprecation warnings

v2.0 Features:
- 8 leak detection primitives (was 4)
- Aggressive thresholds tuned for real retail data
- Universal column synonym handling
- Metadata with recommendations per leak type

Example (v2.1 - Recommended):
    from sentinel_engine import bundle_pos_facts, query_bundle, LEAK_METADATA
    from sentinel_engine.context import create_analysis_context

    # Create isolated context for this request
    ctx = create_analysis_context()

    # Bundle POS data facts
    bundle = bundle_pos_facts(ctx, rows)

    # Query for anomalies
    items, scores = query_bundle(ctx, bundle, "low_stock")

    # Get recommendations
    metadata = LEAK_METADATA["low_stock"]

Example (Legacy - Deprecated):
    # Still works but uses global state (not thread-safe)
    bundle = bundle_pos_facts(rows)  # Deprecated signature
"""

__version__ = "3.0.0"

# Context-based API (v2.1 - Recommended)
from .context import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MAX_CODEBOOK_SIZE,
    AnalysisContext,
    analysis_context,
    create_analysis_context,
)

# Core analysis functions
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
from .contradiction_detector import (
    CONTRADICTORY_PAIRS,
    Contradiction,
    detect_contradictions,
    generate_contradiction_report,
    resolve_contradictions,
)

# Streaming module for large files (v3.0.0)
from .streaming import (
    StreamingResult,
    StreamingStats,
    bundle_pos_facts_streaming,
    compute_streaming_stats,
    process_dataframe,
    process_large_file,
    read_file_chunked,
)

__all__ = [
    # Version
    "__version__",
    # Context API (v2.1 - Recommended)
    "AnalysisContext",
    "create_analysis_context",
    "analysis_context",
    "DEFAULT_DIMENSIONS",
    "DEFAULT_MAX_CODEBOOK_SIZE",
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
    "process_large_file",
    "process_dataframe",
    "StreamingResult",
    "StreamingStats",
    "read_file_chunked",
    "compute_streaming_stats",
    "bundle_pos_facts_streaming",
]
