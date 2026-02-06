"""
Dorian - Vector Symbolic Architecture Knowledge Engine

A geometric knowledge system that encodes meaning as high-dimensional vectors.
Supports 10M+ facts with sub-millisecond query times.

Core Components:
- VSAEngine: Vector operations (bind, bundle, similarity)
- FactStore: Triple storage with FAISS indexing
- InferenceEngine: Multi-hop reasoning

Example:
    from dorian.core import DorianCore

    dorian = DorianCore(dimensions=10000)
    dorian.add_fact("lumber", "has_tracking_behavior", "receiving_gap")
    results = dorian.query(subject="lumber")
"""

from .core import DorianCore, FactStore, InferenceEngine, VSAEngine
from .persistence import (
    DorianPersistence,
    PersistedFact,
    anonymize_sku_to_category,
    detect_industry,
    get_persistence,
)
from .pipeline import KnowledgePipeline

__all__ = [
    "DorianCore",
    "VSAEngine",
    "FactStore",
    "InferenceEngine",
    "KnowledgePipeline",
    # Persistence
    "DorianPersistence",
    "PersistedFact",
    "get_persistence",
    "anonymize_sku_to_category",
    "detect_industry",
]

__version__ = "1.0.0"
