"""
Availability flags for optional sentinel_engine components.

This module centralizes all availability checks for optional dependencies.
Import these flags to check if a component is available before using it.

Example:
    from sentinel_engine._availability import _DORIAN_AVAILABLE

    if _DORIAN_AVAILABLE:
        from sentinel_engine import DorianCore
        dorian = DorianCore()
"""

import logging

logger = logging.getLogger(__name__)

# Context-based API (v2.1+)
_CONTEXT_AVAILABLE = False
try:
    from .context import AnalysisContext, create_analysis_context  # noqa: F401

    _CONTEXT_AVAILABLE = True
except ImportError:
    logger.debug("Context module not available - using Dorian")

# Core analysis functions
_CORE_AVAILABLE = False
try:
    from .core import bundle_pos_facts, query_bundle  # noqa: F401

    _CORE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Core module not available: {e}")

# Pipeline components
_PIPELINE_AVAILABLE = False
try:
    from .batch import BatchProcessor  # noqa: F401
    from .pipeline import TieredPipeline  # noqa: F401

    _PIPELINE_AVAILABLE = True
except ImportError:
    pass

# Bridge
_BRIDGE_AVAILABLE = False
try:
    from .bridge import VSASymbolicBridge  # noqa: F401

    _BRIDGE_AVAILABLE = True
except ImportError:
    pass

# Contradiction Detector (v2.1.0)
_CONTRADICTION_DETECTOR_AVAILABLE = False
try:
    from .contradiction_detector import detect_contradictions  # noqa: F401

    _CONTRADICTION_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Contradiction detector not available: {e}")

# Streaming module (v3.0.0)
_STREAMING_AVAILABLE = False
try:
    from .streaming import process_large_file  # noqa: F401

    _STREAMING_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Streaming module not available: {e}")

# VSA Evidence Grounding (v4.0.0)
_VSA_EVIDENCE_AVAILABLE = False
try:
    from .routing import SmartRouter  # noqa: F401
    from .vsa_evidence import CauseScorer  # noqa: F401

    _VSA_EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"VSA Evidence module not available: {e}")

# Semantic Flagging (v4.1.0)
_FLAGGING_AVAILABLE = False
try:
    from .flagging import SemanticFlagDetector  # noqa: F401

    _FLAGGING_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Flagging module not available: {e}")

# Dorian Knowledge Engine (v5.0.0)
_DORIAN_AVAILABLE = False
try:
    from .dorian import DorianCore  # noqa: F401

    _DORIAN_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Dorian module not available: {e}")

# Diagnostic Engine (v5.0.0)
_DIAGNOSTIC_AVAILABLE = False
try:
    from .diagnostic import ConversationalDiagnostic  # noqa: F401

    _DIAGNOSTIC_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Diagnostic module not available: {e}")


__all__ = [
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
]
