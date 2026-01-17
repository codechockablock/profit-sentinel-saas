"""
Analysis Context - Request-scoped state container for VSA analysis.

This module provides thread-safe, request-isolated state management for
the Sentinel Engine. Each API request creates its own AnalysisContext,
ensuring no data leakage between concurrent analyses.

CRITICAL SAFETY PROPERTY:
    All mutable state (codebook, leak counts, etc.) lives in the context.
    No module-level mutable state is used during analysis.
    Parallel requests cannot contaminate each other.

Usage:
    from sentinel_engine.context import create_analysis_context

    # In your API handler:
    ctx = create_analysis_context()
    bundle = bundle_pos_facts(ctx, rows)
    items, scores = query_bundle(ctx, bundle, "low_stock")
    # ctx is garbage collected after request completes
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import torch

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS (Immutable - safe to share)
# =============================================================================

DEFAULT_DIMENSIONS = 8192   # Production v3.0: reduced from 16384 for 2x speed, <1% accuracy loss
DEFAULT_MAX_CODEBOOK_SIZE = 50000
DEFAULT_DTYPE = torch.complex64

# Resonator parameters (calibrated v3.0 - optimized for 8192-D)
RESONATOR_ALPHA = 0.85      # Blend factor (old vs new)
RESONATOR_POWER = 0.64      # Resonance power
RESONATOR_ITERS = 100       # Production v3.0: reduced from 300 (early-stopping handles convergence)
RESONATOR_MULTI_STEPS = 3   # Multi-step cycles
RESONATOR_TOP_K = 32        # Production v3.0: reduced from 64 for efficiency
RESONATOR_CONVERGENCE_THRESHOLD = 0.0001  # Production v3.0: tighter threshold with early-stopping


# =============================================================================
# ANALYSIS CONTEXT
# =============================================================================

@dataclass
class AnalysisContext:
    """
    Encapsulates all mutable state for a single analysis run.

    Thread-safe by isolation: each request gets its own instance.
    No shared mutable state between contexts.

    Attributes:
        dimensions: Vector dimensionality (default 16384)
        max_codebook_size: Maximum entities in codebook before FIFO eviction
        device: Torch device (CPU or CUDA)
        dtype: Torch dtype for complex vectors
        codebook: OrderedDict mapping entity names to hypervectors
        leak_counts: Count of detections per primitive type
        rows_processed: Number of rows processed in this analysis
        dataset_stats: Computed statistics from the dataset
    """

    # Configuration (immutable after creation)
    dimensions: int = DEFAULT_DIMENSIONS
    max_codebook_size: int = DEFAULT_MAX_CODEBOOK_SIZE
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    dtype: torch.dtype = DEFAULT_DTYPE

    # Resonator parameters (can be tuned per-analysis if needed)
    alpha: float = RESONATOR_ALPHA
    power: float = RESONATOR_POWER
    iters: int = RESONATOR_ITERS
    multi_steps: int = RESONATOR_MULTI_STEPS
    top_k: int = RESONATOR_TOP_K
    convergence_threshold: float = RESONATOR_CONVERGENCE_THRESHOLD

    # Codebook filtering options (calibrated v2.1.0)
    sku_only_codebook: bool = False  # Set True to filter codebook to SKUs only

    # Mutable state (isolated per request)
    codebook: OrderedDict = field(default_factory=OrderedDict)
    leak_counts: Dict[str, int] = field(default_factory=dict)
    rows_processed: int = 0

    # Dataset statistics (computed during bundling)
    dataset_stats: Dict[str, float] = field(default_factory=dict)

    # Cached primitive vectors (lazily initialized per context)
    _primitives: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)

    # Random generator for deterministic vector generation
    _generator: Optional[torch.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize leak counts and generator after dataclass creation."""
        self.leak_counts = {
            "low_stock": 0,
            "high_margin_leak": 0,
            "dead_item": 0,
            "negative_inventory": 0,
            "overstock": 0,
            "price_discrepancy": 0,
            "shrinkage_pattern": 0,
            "margin_erosion": 0,
            "high_velocity": 0,
            "seasonal": 0,
        }
        self.dataset_stats = {
            "avg_quantity": 0.0,
            "avg_margin": 0.0,
            "avg_sold": 0.0,
        }
        # Initialize generator for this context
        self._generator = torch.Generator(device=self.device)

    # =========================================================================
    # CODEBOOK MANAGEMENT
    # =========================================================================

    def add_to_codebook(self, entity: str, is_sku: bool = False) -> Optional[torch.Tensor]:
        """
        Add entity to this context's codebook with FIFO eviction.

        Args:
            entity: Entity name (SKU, description, vendor, etc.)
            is_sku: Whether this entity is a SKU (used for SKU-only filtering)

        Returns:
            The hypervector for this entity, or None if entity is invalid
        """
        # Normalize entity
        entity = entity.strip().lower()

        # Skip invalid entities
        if not entity or entity in ('unknown', 'unknown_sku', 'unknown_desc',
                                     'unknown_vendor', 'unknown_category', ''):
            return None

        # SKU-only filtering: skip non-SKU entities if enabled
        if self.sku_only_codebook and not is_sku:
            # Still return a vector (ephemeral) but don't store in codebook
            return self.seed_hash(entity)

        # Add if not present
        if entity not in self.codebook:
            self.codebook[entity] = self.seed_hash(entity)

            # FIFO eviction if over capacity
            if len(self.codebook) > self.max_codebook_size:
                self.codebook.popitem(last=False)

        return self.codebook[entity]

    def get_from_codebook(self, entity: str) -> Optional[torch.Tensor]:
        """
        Get hypervector for entity from codebook.

        Args:
            entity: Entity name

        Returns:
            Hypervector if found, None otherwise
        """
        entity = entity.strip().lower()
        return self.codebook.get(entity)

    def get_or_create(self, entity: str) -> torch.Tensor:
        """
        Get hypervector for entity, creating if not in codebook.

        Unlike add_to_codebook, this always returns a vector (creates
        ephemeral one if entity is invalid for codebook).

        Args:
            entity: Entity name

        Returns:
            Hypervector for entity
        """
        vec = self.get_from_codebook(entity)
        if vec is not None:
            return vec

        # Try to add
        vec = self.add_to_codebook(entity)
        if vec is not None:
            return vec

        # Entity is invalid for codebook, create ephemeral vector
        return self.seed_hash(entity)

    def get_codebook_tensor(self) -> Optional[torch.Tensor]:
        """
        Get codebook as stacked tensor for resonator operations.

        Returns:
            Tensor of shape (n_entities, dimensions) or None if empty
        """
        if not self.codebook:
            return None
        return torch.stack(list(self.codebook.values()))

    def get_codebook_keys(self) -> List[str]:
        """Get list of entity names in codebook order."""
        return list(self.codebook.keys())

    # =========================================================================
    # VECTOR OPERATIONS
    # =========================================================================

    def seed_hash(self, string: str) -> torch.Tensor:
        """
        Generate deterministic hypervector from string.

        Uses SHA-256 hash to seed random phase generation, ensuring
        the same string always produces the same vector.

        Args:
            string: Seed string

        Returns:
            Normalized complex phasor hypervector
        """
        hash_obj = hashlib.sha256(string.encode())
        seed = int.from_bytes(hash_obj.digest(), 'big') % (2**32)

        # Use context's generator for isolation
        self._generator.manual_seed(seed)

        phases = torch.rand(
            self.dimensions,
            device=self.device,
            generator=self._generator,
            dtype=torch.float32
        ) * 2 * torch.pi

        v = torch.exp(1j * phases).to(self.dtype)
        return self.normalize(v)

    def normalize(self, v: torch.Tensor) -> torch.Tensor:
        """
        Normalize vector to unit length.

        Args:
            v: Input tensor

        Returns:
            Normalized tensor
        """
        norm = torch.norm(v, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        return v / norm

    def zeros(self) -> torch.Tensor:
        """Create zero vector on context's device."""
        return torch.zeros(self.dimensions, device=self.device, dtype=self.dtype)

    # =========================================================================
    # PRIMITIVE VECTORS
    # =========================================================================

    def get_primitives(self) -> Dict[str, torch.Tensor]:
        """
        Get primitive vectors for this context.

        Primitives are lazily initialized on first access and cached
        for the lifetime of this context.

        Returns:
            Dict mapping primitive names to hypervectors
        """
        if self._primitives is None:
            self._primitives = {
                # Core detection primitives
                "low_stock": self.seed_hash("primitive_low_stock_v2"),
                "high_margin_leak": self.seed_hash("primitive_high_margin_leak_v2"),
                "dead_item": self.seed_hash("primitive_dead_item_v2"),
                "negative_inventory": self.seed_hash("primitive_negative_inventory_v2"),
                "overstock": self.seed_hash("primitive_overstock_v2"),
                "price_discrepancy": self.seed_hash("primitive_price_discrepancy_v2"),
                "shrinkage_pattern": self.seed_hash("primitive_shrinkage_pattern_v2"),
                "margin_erosion": self.seed_hash("primitive_margin_erosion_v2"),
                # Utility primitives
                "high_velocity": self.seed_hash("primitive_high_velocity_v2"),
                "seasonal": self.seed_hash("primitive_seasonal_v2"),
            }
        return self._primitives

    def get_primitive(self, name: str) -> Optional[torch.Tensor]:
        """
        Get a specific primitive vector by name.

        Args:
            name: Primitive name (e.g., "low_stock")

        Returns:
            Primitive hypervector or None if not found
        """
        return self.get_primitives().get(name)

    # =========================================================================
    # STATISTICS AND CLEANUP
    # =========================================================================

    def update_stats(self, avg_quantity: float, avg_margin: float, avg_sold: float):
        """Update dataset statistics for relative threshold calculations."""
        self.dataset_stats["avg_quantity"] = avg_quantity
        self.dataset_stats["avg_margin"] = avg_margin
        self.dataset_stats["avg_sold"] = avg_sold

    def increment_leak_count(self, primitive: str):
        """Increment detection count for a primitive."""
        if primitive in self.leak_counts:
            self.leak_counts[primitive] += 1

    def reset(self):
        """
        Clear all mutable state.

        Useful for testing or reusing context (not recommended in production).
        """
        self.codebook.clear()
        self.rows_processed = 0
        for key in self.leak_counts:
            self.leak_counts[key] = 0
        self.dataset_stats = {
            "avg_quantity": 0.0,
            "avg_margin": 0.0,
            "avg_sold": 0.0,
        }
        self._primitives = None
        logger.debug("Analysis context reset")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of context state for logging/debugging.

        Returns:
            Dict with codebook size, leak counts, etc.
        """
        return {
            "codebook_size": len(self.codebook),
            "rows_processed": self.rows_processed,
            "leak_counts": self.leak_counts.copy(),
            "dataset_stats": self.dataset_stats.copy(),
            "device": str(self.device),
            "dimensions": self.dimensions,
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_analysis_context(
    dimensions: int = DEFAULT_DIMENSIONS,
    max_codebook_size: int = DEFAULT_MAX_CODEBOOK_SIZE,
    use_gpu: bool = True,
    device: Optional[str] = None,
    sku_only_codebook: bool = False,
    convergence_threshold: float = RESONATOR_CONVERGENCE_THRESHOLD,
    iters: int = RESONATOR_ITERS,
) -> AnalysisContext:
    """
    Factory function to create a new analysis context.

    Call this once per API request. Pass the context to all analysis
    functions. Let it be garbage collected after the response.

    Args:
        dimensions: Vector dimensionality (default 16384)
        max_codebook_size: Maximum codebook entries (default 50000)
        use_gpu: Whether to use GPU if available (default True)
        device: Explicit device string (overrides use_gpu if provided)
        sku_only_codebook: Filter codebook to SKUs only (calibrated v2.1.0)
        convergence_threshold: Resonator convergence threshold (calibrated: 0.005)
        iters: Resonator iterations (calibrated: 300)

    Returns:
        Fresh AnalysisContext instance

    Example:
        ctx = create_analysis_context(sku_only_codebook=True)
        try:
            bundle = bundle_pos_facts(ctx, rows)
            items, scores = query_bundle(ctx, bundle, "low_stock")
            return {"items": items, "scores": scores}
        finally:
            ctx.reset()  # Optional - GC handles it
    """
    # Determine device
    if device is not None:
        torch_device = torch.device(device)
    elif use_gpu and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")

    ctx = AnalysisContext(
        dimensions=dimensions,
        max_codebook_size=max_codebook_size,
        device=torch_device,
        sku_only_codebook=sku_only_codebook,
        convergence_threshold=convergence_threshold,
        iters=iters,
    )

    logger.debug(f"Created analysis context: device={torch_device}, dims={dimensions}, sku_only={sku_only_codebook}")

    return ctx


# =============================================================================
# CONTEXT MANAGER (Alternative usage pattern)
# =============================================================================

class analysis_context:
    """
    Context manager for analysis operations.

    Automatically creates and cleans up context.

    Example:
        with analysis_context() as ctx:
            bundle = bundle_pos_facts(ctx, rows)
            items, scores = query_bundle(ctx, bundle, "low_stock")
    """

    def __init__(
        self,
        dimensions: int = DEFAULT_DIMENSIONS,
        max_codebook_size: int = DEFAULT_MAX_CODEBOOK_SIZE,
        use_gpu: bool = True,
    ):
        self.dimensions = dimensions
        self.max_codebook_size = max_codebook_size
        self.use_gpu = use_gpu
        self.ctx: Optional[AnalysisContext] = None

    def __enter__(self) -> AnalysisContext:
        self.ctx = create_analysis_context(
            dimensions=self.dimensions,
            max_codebook_size=self.max_codebook_size,
            use_gpu=self.use_gpu,
        )
        return self.ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ctx is not None:
            self.ctx.reset()
            self.ctx = None
        return False  # Don't suppress exceptions
