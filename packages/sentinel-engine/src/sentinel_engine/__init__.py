"""
sentinel_engine - Aggressive VSA-based Profit Leak Detection Engine

This module provides the scalable infrastructure for VSA inference:
- Core analysis functions (bundle_pos_facts, query_bundle)
- 8 detection primitives for comprehensive leak detection
- Universal POS column support (Paladin, Square, Lightspeed, etc.)
- $ impact estimation and recommendations

v2.0 Features:
- 8 leak detection primitives (was 4)
- Aggressive thresholds tuned for real retail data
- Universal column synonym handling
- Metadata with recommendations per leak type

Example:
    from sentinel_engine import bundle_pos_facts, query_bundle, LEAK_METADATA

    # Bundle POS data facts
    bundle = bundle_pos_facts(rows)

    # Query for anomalies
    items, scores = query_bundle(bundle, "low_stock")

    # Get recommendations
    metadata = LEAK_METADATA["low_stock"]
"""

__version__ = "2.0.0"

# Core analysis functions
from .core import (
    # Main functions
    bundle_pos_facts,
    query_bundle,
    reset_codebook,
    get_primitive_metadata,
    get_all_primitives,
    # VSA utilities
    seed_hash,
    normalize_torch,
    add_to_codebook,
    convergence_lock_resonator_gpu,
    # Data structures
    PRIMITIVES,
    codebook_dict,
    LEAK_METADATA,
    THRESHOLDS,
    # Column aliases (for reference)
    QUANTITY_ALIASES,
    COST_ALIASES,
    REVENUE_ALIASES,
    SOLD_ALIASES,
    QTY_DIFF_ALIASES,
    MARGIN_ALIASES,
    LAST_SALE_ALIASES,
    SKU_ALIASES,
    VENDOR_ALIASES,
    CATEGORY_ALIASES,
)

# Pipeline components (if they exist and are complete)
try:
    from .pipeline import TieredPipeline, PipelineStage, PipelineResult
    from .codebook import PersistentCodebook, CodebookManager
    from .batch import BatchProcessor, StreamProcessor
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

__all__ = [
    # Version
    "__version__",
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
]
