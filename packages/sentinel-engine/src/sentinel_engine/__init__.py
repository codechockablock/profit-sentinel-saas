"""
sentinel_engine - Scalable VSA Inference Engine

This module provides the scalable infrastructure for VSA inference:
- Core analysis functions (bundle_pos_facts, query_bundle)
- Tiered pipeline (statistical pre-filter -> VSA deep dive)
- FAISS integration for approximate nearest neighbor
- Persistent codebook with session management
- Batch processing utilities

Designed to handle 150k-1.5M+ entities efficiently.

Example:
    from sentinel_engine import bundle_pos_facts, query_bundle

    # Bundle POS data facts
    bundle = bundle_pos_facts(rows)

    # Query for anomalies
    items, scores = query_bundle(bundle, "low_stock")
"""

__version__ = "1.0.0"

# Core analysis functions
from .core import (
    bundle_pos_facts,
    query_bundle,
    seed_hash,
    normalize_torch,
    add_to_codebook,
    convergence_lock_resonator_gpu,
    PRIMITIVES,
    codebook_dict,
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
    "seed_hash",
    "normalize_torch",
    "add_to_codebook",
    "convergence_lock_resonator_gpu",
    "PRIMITIVES",
    "codebook_dict",
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
