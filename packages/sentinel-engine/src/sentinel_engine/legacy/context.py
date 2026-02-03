"""
AnalysisContext - Request-scoped state for VSA-based profit leak detection.

This module provides the context object that carries all state for a single
analysis request, including:
- Codebook of entity hypervectors
- Primitive hypervectors for leak types
- Statistics for adaptive thresholds
- Resonator hyperparameters

CRITICAL: Each request gets its own context. No global mutable state.
Thread-safe for concurrent request handling.

GLM (Geometric Language Model) Extensions:
- Geometric validation gates with confidence thresholds
- Similarity computation for complex phasor vectors
- Permutation for asymmetric causal binding
- Grounded query interface for LLM validation

Sparse VSA Extensions:
- SparseVector: Sparse representation with (indices, phases) pairs
- SparseVSA: Operations optimized for sparse vectors
- Batch operations for efficient validation

Dirac VSA Extensions (True Dirac-Inspired Semantic Algebra):
- DiracVector: 4-component vector (spatial, temporal, phase, entropy)
- DiracVSA: Full algebraic operations with temporal asymmetry
- Causal binding: cause→effect asymmetry via permutation
- Negation: Phase rotation by π preserves spatial similarity
- Entropy tracking: Monotonically increasing for irreversible ops
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# FAISS for fast similarity search
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION THRESHOLDS (Geometric Language Model Constraints)
# =============================================================================

VALIDATION_THRESHOLDS = {
    "noise_floor": 0.01,  # ~1/√d for d=16384
    "rejection_threshold": 0.10,  # Below this = reject claim
    "low_confidence": 0.25,  # Flag for review
    "moderate_confidence": 0.40,  # Acceptable for suggestions
    "high_confidence": 0.60,  # Strong claims allowed
    "very_high_confidence": 0.80,  # Definitive statements
}

# Default VSA dimensions (2^14 = 16384)
DEFAULT_DIMENSIONS = 16384

# Default codebook size limit before hierarchical encoding kicks in
DEFAULT_MAX_CODEBOOK_SIZE = 100_000

# Threshold at which hierarchical codebook structure is used
HIERARCHICAL_CODEBOOK_THRESHOLD = 50_000


# =============================================================================
# SPARSE VSA - SPARSE VECTOR REPRESENTATION (Optimization)
# =============================================================================


@dataclass
class SparseVector:
    """
    Sparse representation of a hypervector using (indices, phases) pairs.

    NOTE: This is a sparse OPTIMIZATION for standard VSA vectors.
    For true Dirac-inspired semantic extension with temporal asymmetry,
    negation via phase rotation, and entropy tracking, use DiracVector.

    Instead of storing all D dimensions, we store only K non-zero indices
    and their associated phases. This enables O(K) operations instead of O(D)
    for many common operations, where K << D.

    The dense representation can be reconstructed as:
        v[indices[i]] = exp(i * phases[i]) for each i
        v[j] = 0 for j not in indices

    Attributes:
        indices: Tensor of shape (K,) containing non-zero indices
        phases: Tensor of shape (K,) containing phases in [0, 2π)
        dimensions: Total dimensionality D of the full vector
        density: Sparsity level K/D (fraction of non-zero elements)
    """

    indices: torch.Tensor  # Shape: (K,) - indices of non-zero elements
    phases: torch.Tensor  # Shape: (K,) - phases at those indices
    dimensions: int  # Total dimensionality D
    density: float = field(default=0.1)  # K/D ratio

    def __post_init__(self):
        """Validate tensor shapes and types."""
        if self.indices.shape != self.phases.shape:
            raise ValueError(
                f"indices shape {self.indices.shape} != phases shape {self.phases.shape}"
            )
        if len(self.indices.shape) != 1:
            raise ValueError(f"Expected 1D tensors, got shape {self.indices.shape}")

    @property
    def sparsity(self) -> int:
        """Return K, the number of non-zero elements."""
        return len(self.indices)

    def to_dense(
        self, device: torch.device | None = None, dtype: torch.dtype = torch.complex64
    ) -> torch.Tensor:
        """
        Convert sparse vector to dense complex vector.

        Args:
            device: Target device (default: same as indices)
            dtype: Complex dtype (default: complex64)

        Returns:
            Dense tensor of shape (dimensions,)
        """
        if device is None:
            device = self.indices.device

        dense = torch.zeros(self.dimensions, dtype=dtype, device=device)
        # e^(i*phase) = cos(phase) + i*sin(phase)
        values = torch.complex(torch.cos(self.phases), torch.sin(self.phases)).to(
            dtype=dtype, device=device
        )
        dense[self.indices.long()] = values
        return dense

    @classmethod
    def from_dense(
        cls,
        dense: torch.Tensor,
        top_k: int | None = None,
        threshold: float = 0.1,
    ) -> SparseVector:
        """
        Create DiracVector from dense complex vector.

        Selects either top_k largest magnitude elements or elements
        above threshold magnitude.

        Args:
            dense: Dense complex tensor of shape (D,)
            top_k: If specified, keep top K elements by magnitude
            threshold: If top_k not specified, keep elements with |v| > threshold

        Returns:
            SparseVector with sparse representation
        """
        magnitudes = torch.abs(dense)
        dimensions = len(dense)

        if top_k is not None:
            k = min(top_k, dimensions)
            _, indices = torch.topk(magnitudes, k)
        else:
            indices = torch.where(magnitudes > threshold)[0]

        # Extract phases: angle of complex number
        phases = torch.angle(dense[indices])
        density = len(indices) / dimensions

        return cls(
            indices=indices,
            phases=phases,
            dimensions=dimensions,
            density=density,
        )

    def similarity(self, other: SparseVector) -> float:
        """
        Compute sparse similarity with another SparseVector.

        Only overlapping indices contribute to similarity.
        This is O(K) instead of O(D).

        Args:
            other: Another SparseVector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        if self.dimensions != other.dimensions:
            raise ValueError(
                f"Dimension mismatch: {self.dimensions} vs {other.dimensions}"
            )

        # Find overlapping indices
        # Convert to sets for intersection
        self_idx_set = set(self.indices.tolist())
        other_idx_set = set(other.indices.tolist())
        common = self_idx_set & other_idx_set

        if not common:
            return 0.0

        # Build index maps for fast lookup
        self_idx_to_pos = {idx: pos for pos, idx in enumerate(self.indices.tolist())}
        other_idx_to_pos = {idx: pos for pos, idx in enumerate(other.indices.tolist())}

        # Compute dot product over common indices
        dot_real = 0.0
        for idx in common:
            phase_self = self.phases[self_idx_to_pos[idx]].item()
            phase_other = other.phases[other_idx_to_pos[idx]].item()
            # Re(e^(i*a) * conj(e^(i*b))) = Re(e^(i*(a-b))) = cos(a-b)
            dot_real += math.cos(phase_self - phase_other)

        # Normalize by geometric mean of sparsities
        norm = math.sqrt(self.sparsity * other.sparsity)
        return dot_real / norm if norm > 0 else 0.0


class SparseVSA:
    """
    Vector Symbolic Architecture operations optimized for SparseVectors.

    NOTE: This is a sparse OPTIMIZATION for standard VSA operations.
    For true Dirac-inspired semantic extension with temporal asymmetry,
    negation via phase rotation, and entropy tracking, use DiracVSA.

    Provides bind, unbind, and bundle operations that work directly
    on sparse representations without converting to dense.

    This enables O(K) complexity for most operations where K << D.
    """

    def __init__(self, dimensions: int = 16384, default_sparsity: int = 1638):
        """
        Initialize SparseVSA.

        Args:
            dimensions: Total dimensionality D
            default_sparsity: Default K for new vectors (typically D/10)
        """
        self.dimensions = dimensions
        self.default_sparsity = default_sparsity

    def random_vector(
        self,
        sparsity: int | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> SparseVector:
        """
        Generate a random sparse vector.

        Args:
            sparsity: Number of non-zero elements (default: default_sparsity)
            device: Target device

        Returns:
            Random SparseVector
        """
        k = sparsity or self.default_sparsity
        k = min(k, self.dimensions)

        # Random unique indices
        indices = torch.randperm(self.dimensions, device=device)[:k]
        # Random phases in [0, 2π)
        phases = torch.rand(k, device=device) * 2 * math.pi

        return SparseVector(
            indices=indices,
            phases=phases,
            dimensions=self.dimensions,
            density=k / self.dimensions,
        )

    def seed_hash(
        self, string: str, device: torch.device = torch.device("cpu")
    ) -> SparseVector:
        """
        Generate deterministic SparseVector from string.

        Uses SHA256 to seed the random selection of indices and phases.

        Args:
            string: Input string
            device: Target device

        Returns:
            Deterministic SparseVector
        """
        # Create deterministic seed from string
        hash_bytes = hashlib.sha256(string.encode()).digest()
        seed = int.from_bytes(hash_bytes[:8], byteorder="big")

        # Use generator for reproducibility
        gen = torch.Generator(device="cpu").manual_seed(seed)

        k = self.default_sparsity
        indices = torch.randperm(self.dimensions, generator=gen)[:k].to(device)
        phases = torch.rand(k, generator=gen).to(device) * 2 * math.pi

        return SparseVector(
            indices=indices,
            phases=phases,
            dimensions=self.dimensions,
            density=k / self.dimensions,
        )

    def bind(self, a: SparseVector, b: SparseVector) -> SparseVector:
        """
        Bind two SparseVectors.

        For sparse vectors, binding is performed only at overlapping indices.
        The result has sparsity ≤ min(K_a, K_b).

        Binding formula: (a ⊗ b)[i] = a[i] * b[i] = e^(i*(phase_a + phase_b))

        Args:
            a: First SparseVector
            b: Second SparseVector

        Returns:
            Bound SparseVector
        """
        if a.dimensions != b.dimensions:
            raise ValueError(f"Dimension mismatch: {a.dimensions} vs {b.dimensions}")

        # Find common indices
        a_idx_set = set(a.indices.tolist())
        b_idx_set = set(b.indices.tolist())
        common = sorted(a_idx_set & b_idx_set)

        if not common:
            # Return empty sparse vector
            return SparseVector(
                indices=torch.tensor([], dtype=torch.long, device=a.indices.device),
                phases=torch.tensor([], dtype=torch.float32, device=a.phases.device),
                dimensions=a.dimensions,
                density=0.0,
            )

        # Build index maps
        a_idx_to_pos = {idx: pos for pos, idx in enumerate(a.indices.tolist())}
        b_idx_to_pos = {idx: pos for pos, idx in enumerate(b.indices.tolist())}

        # Compute bound phases at common indices
        result_indices = torch.tensor(common, dtype=torch.long, device=a.indices.device)
        result_phases = torch.zeros(
            len(common), dtype=torch.float32, device=a.phases.device
        )

        for i, idx in enumerate(common):
            phase_a = a.phases[a_idx_to_pos[idx]]
            phase_b = b.phases[b_idx_to_pos[idx]]
            # Binding adds phases (mod 2π)
            result_phases[i] = (phase_a + phase_b) % (2 * math.pi)

        return SparseVector(
            indices=result_indices,
            phases=result_phases,
            dimensions=a.dimensions,
            density=len(common) / a.dimensions,
        )

    def unbind(self, bound: SparseVector, key: SparseVector) -> SparseVector:
        """
        Unbind a key from a bound vector.

        Unbinding is the inverse of binding: a ⊗ b ⊗ b* ≈ a
        For phases: unbind subtracts the key phase.

        Args:
            bound: The bound vector
            key: The key to unbind

        Returns:
            Unbound SparseVector (approximate original)
        """
        if bound.dimensions != key.dimensions:
            raise ValueError(
                f"Dimension mismatch: {bound.dimensions} vs {key.dimensions}"
            )

        # Find common indices
        bound_idx_set = set(bound.indices.tolist())
        key_idx_set = set(key.indices.tolist())
        common = sorted(bound_idx_set & key_idx_set)

        if not common:
            return SparseVector(
                indices=torch.tensor([], dtype=torch.long, device=bound.indices.device),
                phases=torch.tensor(
                    [], dtype=torch.float32, device=bound.phases.device
                ),
                dimensions=bound.dimensions,
                density=0.0,
            )

        bound_idx_to_pos = {idx: pos for pos, idx in enumerate(bound.indices.tolist())}
        key_idx_to_pos = {idx: pos for pos, idx in enumerate(key.indices.tolist())}

        result_indices = torch.tensor(
            common, dtype=torch.long, device=bound.indices.device
        )
        result_phases = torch.zeros(
            len(common), dtype=torch.float32, device=bound.phases.device
        )

        for i, idx in enumerate(common):
            phase_bound = bound.phases[bound_idx_to_pos[idx]]
            phase_key = key.phases[key_idx_to_pos[idx]]
            # Unbinding subtracts phases (mod 2π)
            result_phases[i] = (phase_bound - phase_key) % (2 * math.pi)

        return SparseVector(
            indices=result_indices,
            phases=result_phases,
            dimensions=bound.dimensions,
            density=len(common) / bound.dimensions,
        )

    def bundle(
        self, vectors: Sequence[SparseVector], weights: Sequence[float] | None = None
    ) -> SparseVector:
        """
        Bundle multiple SparseVectors into a superposition.

        For sparse vectors, bundling combines phases at each index
        using weighted circular mean.

        Args:
            vectors: Sequence of SparseVectors to bundle
            weights: Optional weights for each vector (default: uniform)

        Returns:
            Bundled SparseVector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        dimensions = vectors[0].dimensions
        if not all(v.dimensions == dimensions for v in vectors):
            raise ValueError("All vectors must have same dimensions")

        if weights is None:
            weights = [1.0] * len(vectors)

        # Collect all indices across all vectors
        all_indices: dict[
            int, list[tuple[float, float]]
        ] = {}  # idx -> [(phase, weight), ...]

        for vec, weight in zip(vectors, weights):
            for i, idx in enumerate(vec.indices.tolist()):
                if idx not in all_indices:
                    all_indices[idx] = []
                all_indices[idx].append((vec.phases[i].item(), weight))

        # Compute weighted circular mean for each index
        result_indices = []
        result_phases = []

        for idx, phase_weights in all_indices.items():
            # Circular mean: average of unit vectors
            sum_cos = sum(w * math.cos(p) for p, w in phase_weights)
            sum_sin = sum(w * math.sin(p) for p, w in phase_weights)
            total_weight = sum(w for _, w in phase_weights)

            if total_weight > 0:
                avg_cos = sum_cos / total_weight
                avg_sin = sum_sin / total_weight
                # Convert back to phase
                phase = math.atan2(avg_sin, avg_cos)
                if phase < 0:
                    phase += 2 * math.pi

                result_indices.append(idx)
                result_phases.append(phase)

        device = vectors[0].indices.device
        return SparseVector(
            indices=torch.tensor(result_indices, dtype=torch.long, device=device),
            phases=torch.tensor(result_phases, dtype=torch.float32, device=device),
            dimensions=dimensions,
            density=len(result_indices) / dimensions,
        )


# =============================================================================
# DIRAC VSA - TRUE DIRAC-INSPIRED SEMANTIC EXTENSION
# =============================================================================


@dataclass
class DiracVector:
    """
    True Dirac-inspired 4-component semantic vector.

    Extends beyond flat VSA geometry to capture:
    - Negation (phase rotation by π)
    - Temporal direction (cause→effect asymmetry via permutation)
    - Information dynamics (entropy tracking)
    - Oscillatory patterns (phase component)

    The four components form a complete semantic representation:
    - spatial: Content embedding in d-dimensional complex space
    - temporal: Directional component with non-commutative binding
    - phase: Single complex number for oscillation/negation tracking
    - entropy: Non-negative scalar tracking information loss

    Key Properties:
    - NOT(x) has phase rotated by π, preserves spatial similarity
    - bind(A,B) ≠ bind(B,A) when temporal asymmetry is enabled
    - Entropy monotonically increases for irreversible operations
    - Unbinding cause recovers effect cleanly; unbinding effect is noisy

    Attributes:
        spatial: Complex tensor of shape (d,) - content representation
        temporal: Complex tensor of shape (d,) - directional component
        phase: Single complex number - oscillation/negation state
        entropy: Non-negative float - information loss tracking
    """

    spatial: torch.Tensor  # Shape: (d,), dtype: complex64
    temporal: torch.Tensor  # Shape: (d,), dtype: complex64
    phase: complex  # Single complex number (oscillation tracking)
    entropy: float  # Non-negative scalar (information loss)

    def __post_init__(self):
        """Validate tensor shapes and types."""
        if self.spatial.shape != self.temporal.shape:
            raise ValueError(
                f"spatial shape {self.spatial.shape} != temporal shape {self.temporal.shape}"
            )
        if len(self.spatial.shape) != 1:
            raise ValueError(f"Expected 1D tensors, got shape {self.spatial.shape}")
        if not torch.is_complex(self.spatial):
            raise ValueError("spatial must be complex tensor")
        if not torch.is_complex(self.temporal):
            raise ValueError("temporal must be complex tensor")
        if self.entropy < 0:
            raise ValueError(f"entropy must be non-negative, got {self.entropy}")

    @property
    def dimensions(self) -> int:
        """Return dimensionality d."""
        return self.spatial.shape[0]

    @property
    def device(self) -> torch.device:
        """Return device of tensors."""
        return self.spatial.device

    def to_dense(self) -> torch.Tensor:
        """
        Combine spatial and temporal components into unified representation.

        Returns:
            Complex tensor of shape (d,) combining both components
        """
        # Weighted combination with phase rotation applied
        phase_tensor = torch.tensor(
            self.phase, dtype=self.spatial.dtype, device=self.device
        )
        return self.spatial * phase_tensor + self.temporal

    def negate(self) -> DiracVector:
        """
        Return negation via π phase rotation.

        NOT(x) has phase rotated by π, preserving spatial similarity
        while indicating logical negation.

        Returns:
            Negated DiracVector with phase rotated by π
        """
        import cmath

        new_phase = self.phase * cmath.exp(1j * math.pi)  # Rotate by π
        return DiracVector(
            spatial=self.spatial.clone(),
            temporal=self.temporal.clone(),
            phase=new_phase,
            entropy=self.entropy,  # Negation is reversible, no entropy increase
        )

    def similarity(self, other: DiracVector) -> float:
        """
        Compute similarity between DiracVectors.

        Combines spatial and temporal components to capture full semantic
        similarity including causal asymmetry.

        Spatial component: Content similarity
        Temporal component: Directional/causal similarity

        The combined similarity is a weighted average favoring spatial
        (content) but including temporal (direction) for causal relationships.

        Args:
            other: Another DiracVector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        if self.dimensions != other.dimensions:
            raise ValueError(
                f"Dimension mismatch: {self.dimensions} vs {other.dimensions}"
            )

        # Normalize spatial components
        a_spatial_norm = self.spatial / (torch.norm(self.spatial) + 1e-8)
        b_spatial_norm = other.spatial / (torch.norm(other.spatial) + 1e-8)

        # Normalize temporal components
        a_temporal_norm = self.temporal / (torch.norm(self.temporal) + 1e-8)
        b_temporal_norm = other.temporal / (torch.norm(other.temporal) + 1e-8)

        # Complex dot products
        spatial_dot = torch.sum(a_spatial_norm * b_spatial_norm.conj())
        temporal_dot = torch.sum(a_temporal_norm * b_temporal_norm.conj())

        # Account for phase difference
        phase_diff = self.phase * (1 / other.phase) if abs(other.phase) > 1e-8 else 1.0
        phase_factor = abs(phase_diff)

        # Weighted combination: 70% spatial, 30% temporal
        # This captures content similarity primarily while still reflecting
        # temporal/causal asymmetry
        spatial_sim = torch.real(spatial_dot).item() * phase_factor
        temporal_sim = torch.real(temporal_dot).item() * phase_factor

        combined_sim = 0.7 * spatial_sim + 0.3 * temporal_sim
        return combined_sim

    def clone(self) -> DiracVector:
        """Create a deep copy of this DiracVector."""
        return DiracVector(
            spatial=self.spatial.clone(),
            temporal=self.temporal.clone(),
            phase=self.phase,
            entropy=self.entropy,
        )


class DiracVSA:
    """
    Full Dirac-inspired Vector Symbolic Architecture.

    Implements algebraic operations that capture:
    - Temporal asymmetry: bind(A,B) ≠ bind(B,A) via permutation
    - Entropy tracking: Monotonically increasing for irreversible ops
    - Negation: Phase rotation by π
    - Causal binding: Cause→effect with clean recovery

    Key Operations:
    - bind(): Asymmetric binding with entropy increase
    - unbind_cause(): Clean recovery of effect from (cause→effect)
    - unbind_effect(): Noisy recovery of cause (intentionally degraded)
    - bundle(): Superposition with entropy accumulation
    - negate(): Phase rotation by π

    Validation Targets (21σ separation):
    - temporal_asymmetry_score > 0.1 (noise floor ~0.01)
    - entropy_cause_effect > 0.0 for irreversible bindings
    - unbind_cause similarity > 0.9
    - unbind_effect similarity < 0.5 (intentionally noisy)
    """

    # Permutation shift for temporal asymmetry
    TEMPORAL_SHIFT: int = 1

    # Entropy increment for irreversible operations
    ENTROPY_INCREMENT: float = 0.05

    def __init__(
        self,
        dimensions: int = 16384,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
    ):
        """
        Initialize DiracVSA.

        Args:
            dimensions: Vector dimensionality d
            device: Target device (default: CPU)
            dtype: Complex dtype (default: complex64)
        """
        self.dimensions = dimensions
        self.device = device or torch.device("cpu")
        self.dtype = dtype

    def random_vector(
        self,
        entropy: float = 0.0,
    ) -> DiracVector:
        """
        Generate a random DiracVector.

        Args:
            entropy: Initial entropy value (default: 0.0)

        Returns:
            Random DiracVector with unit-magnitude phasor components
        """
        # Random phases for spatial component
        spatial_phases = torch.rand(self.dimensions, device=self.device) * 2 * math.pi
        spatial = torch.complex(
            torch.cos(spatial_phases), torch.sin(spatial_phases)
        ).to(self.dtype)

        # Random phases for temporal component
        temporal_phases = torch.rand(self.dimensions, device=self.device) * 2 * math.pi
        temporal = torch.complex(
            torch.cos(temporal_phases), torch.sin(temporal_phases)
        ).to(self.dtype)

        # Initial phase = 1+0j (no rotation)
        phase = complex(1.0, 0.0)

        return DiracVector(
            spatial=spatial,
            temporal=temporal,
            phase=phase,
            entropy=entropy,
        )

    def seed_hash(self, string: str) -> DiracVector:
        """
        Generate deterministic DiracVector from string via SHA256.

        Args:
            string: Input string to hash

        Returns:
            Deterministic DiracVector
        """
        import hashlib

        # Create deterministic seed from string
        hash_bytes = hashlib.sha256(string.encode()).digest()
        seed = int.from_bytes(hash_bytes[:8], byteorder="big")

        # Use generator for reproducibility
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # Generate spatial phases
        spatial_phases = torch.rand(self.dimensions, generator=gen) * 2 * math.pi
        spatial = torch.complex(
            torch.cos(spatial_phases), torch.sin(spatial_phases)
        ).to(dtype=self.dtype, device=self.device)

        # Generate temporal phases (different seed)
        gen2 = torch.Generator(device="cpu").manual_seed(seed ^ 0xDEADBEEF)
        temporal_phases = torch.rand(self.dimensions, generator=gen2) * 2 * math.pi
        temporal = torch.complex(
            torch.cos(temporal_phases), torch.sin(temporal_phases)
        ).to(dtype=self.dtype, device=self.device)

        return DiracVector(
            spatial=spatial,
            temporal=temporal,
            phase=complex(1.0, 0.0),
            entropy=0.0,
        )

    def normalize(self, v: torch.Tensor) -> torch.Tensor:
        """L2 normalize a complex tensor."""
        norm = torch.norm(v) + 1e-8
        return v / norm

    def permute(self, v: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """Circular permutation for temporal asymmetry."""
        return torch.roll(v, shifts=shifts, dims=-1)

    def inverse_permute(self, v: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """Inverse of permute operation."""
        return torch.roll(v, shifts=-shifts, dims=-1)

    def bind(
        self,
        cause: DiracVector,
        effect: DiracVector,
        symmetric: bool = False,
    ) -> DiracVector:
        """
        Asymmetric causal binding: cause → effect.

        Uses permutation on cause's temporal component to create
        asymmetry: bind(A,B) ≠ bind(B,A).

        The binding increases entropy (irreversible information loss).

        Args:
            cause: The cause DiracVector
            effect: The effect DiracVector
            symmetric: If True, skip permutation (standard binding)

        Returns:
            Bound DiracVector with increased entropy
        """
        if cause.dimensions != effect.dimensions:
            raise ValueError(
                f"Dimension mismatch: {cause.dimensions} vs {effect.dimensions}"
            )

        # Bind spatial components (element-wise multiplication)
        bound_spatial = self.normalize(cause.spatial * effect.spatial)

        # Bind temporal components with asymmetric permutation
        if symmetric:
            bound_temporal = self.normalize(cause.temporal * effect.temporal)
        else:
            # Permute cause's temporal before binding → asymmetry
            permuted_cause_temporal = self.permute(
                cause.temporal, shifts=self.TEMPORAL_SHIFT
            )
            bound_temporal = self.normalize(permuted_cause_temporal * effect.temporal)

        # Combine phases
        bound_phase = cause.phase * effect.phase

        # Entropy increases for irreversible binding
        bound_entropy = cause.entropy + effect.entropy + self.ENTROPY_INCREMENT

        return DiracVector(
            spatial=bound_spatial,
            temporal=bound_temporal,
            phase=bound_phase,
            entropy=bound_entropy,
        )

    def unbind_cause(
        self,
        bound: DiracVector,
        cause: DiracVector,
    ) -> DiracVector:
        """
        Unbind cause to recover effect (clean recovery).

        Given bound = bind(cause, effect), recover effect.
        This should yield high similarity to original effect.

        Args:
            bound: The bound vector (cause → effect)
            cause: The cause to unbind

        Returns:
            Recovered effect DiracVector
        """
        # Unbind spatial: bound.spatial * conj(cause.spatial)
        recovered_spatial = self.normalize(bound.spatial * cause.spatial.conj())

        # Unbind temporal with inverse permutation to undo asymmetry
        permuted_cause_temporal = self.permute(
            cause.temporal, shifts=self.TEMPORAL_SHIFT
        )
        recovered_temporal = self.normalize(
            bound.temporal * permuted_cause_temporal.conj()
        )

        # Unbind phases
        recovered_phase = (
            bound.phase / cause.phase if abs(cause.phase) > 1e-8 else bound.phase
        )

        # Entropy doesn't decrease (second law)
        recovered_entropy = bound.entropy

        return DiracVector(
            spatial=recovered_spatial,
            temporal=recovered_temporal,
            phase=recovered_phase,
            entropy=recovered_entropy,
        )

    def unbind_effect(
        self,
        bound: DiracVector,
        effect: DiracVector,
    ) -> DiracVector:
        """
        Unbind effect to recover cause (noisy recovery).

        Given bound = bind(cause, effect), attempt to recover cause.
        This is INTENTIONALLY noisy due to causal asymmetry.

        The temporal permutation applied during binding cannot be
        cleanly undone when unbinding from the effect side.

        Args:
            bound: The bound vector (cause → effect)
            effect: The effect to unbind

        Returns:
            Noisy approximation of cause DiracVector
        """
        # Unbind spatial (this part works relatively well)
        recovered_spatial = self.normalize(bound.spatial * effect.spatial.conj())

        # Unbind temporal - but we can't undo the permutation correctly
        # because we don't know which temporal was permuted
        # This introduces intentional noise
        recovered_temporal = self.normalize(bound.temporal * effect.temporal.conj())
        # No inverse permutation - this creates the asymmetry degradation

        # Unbind phases
        recovered_phase = (
            bound.phase / effect.phase if abs(effect.phase) > 1e-8 else bound.phase
        )

        # Entropy further increases due to noisy recovery
        recovered_entropy = bound.entropy + self.ENTROPY_INCREMENT

        return DiracVector(
            spatial=recovered_spatial,
            temporal=recovered_temporal,
            phase=recovered_phase,
            entropy=recovered_entropy,
        )

    def bundle(
        self,
        vectors: Sequence[DiracVector],
        weights: Sequence[float] | None = None,
    ) -> DiracVector:
        """
        Bundle multiple DiracVectors into superposition.

        Creates a vector similar to all inputs. Entropy accumulates
        from all bundled vectors.

        Args:
            vectors: Sequence of DiracVectors to bundle
            weights: Optional weights (default: uniform)

        Returns:
            Bundled DiracVector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty sequence")

        dimensions = vectors[0].dimensions
        if not all(v.dimensions == dimensions for v in vectors):
            raise ValueError("All vectors must have same dimensions")

        if weights is None:
            weights = [1.0 / len(vectors)] * len(vectors)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        # Weighted sum of spatial components
        spatial_sum = sum(v.spatial * w for v, w in zip(vectors, weights))
        bundled_spatial = self.normalize(spatial_sum)

        # Weighted sum of temporal components
        temporal_sum = sum(v.temporal * w for v, w in zip(vectors, weights))
        bundled_temporal = self.normalize(temporal_sum)

        # Average phase (circular mean)
        phase_sum = sum(v.phase * w for v, w in zip(vectors, weights))
        bundled_phase = (
            phase_sum / abs(phase_sum) if abs(phase_sum) > 1e-8 else complex(1.0, 0.0)
        )

        # Entropy is max of inputs (superposition doesn't lose info)
        bundled_entropy = max(v.entropy for v in vectors)

        return DiracVector(
            spatial=bundled_spatial,
            temporal=bundled_temporal,
            phase=bundled_phase,
            entropy=bundled_entropy,
        )

    def negate(self, v: DiracVector) -> DiracVector:
        """
        Negate via π phase rotation.

        NOT(x) preserves spatial similarity while rotating phase by π.
        This is reversible: NOT(NOT(x)) = x.

        Args:
            v: DiracVector to negate

        Returns:
            Negated DiracVector
        """
        return v.negate()

    def similarity(self, a: DiracVector, b: DiracVector) -> float:
        """
        Compute similarity between two DiracVectors.

        Args:
            a: First DiracVector
            b: Second DiracVector

        Returns:
            Similarity score in [-1, 1]
        """
        return a.similarity(b)

    def temporal_asymmetry_score(
        self,
        a: DiracVector,
        b: DiracVector,
    ) -> float:
        """
        Measure asymmetry between bind(a,b) and bind(b,a).

        Should be significantly above noise floor (~0.01) to validate
        that temporal binding is working correctly.

        Target: > 0.1 (21σ above noise floor of 0.01)

        Args:
            a: First DiracVector
            b: Second DiracVector

        Returns:
            Asymmetry score (higher = more asymmetric)
        """
        ab = self.bind(a, b, symmetric=False)
        ba = self.bind(b, a, symmetric=False)

        # Measure dissimilarity in temporal components
        ab_temporal_norm = self.normalize(ab.temporal)
        ba_temporal_norm = self.normalize(ba.temporal)

        # 1 - similarity gives asymmetry
        sim = torch.real(torch.sum(ab_temporal_norm * ba_temporal_norm.conj())).item()
        return 1.0 - abs(sim)


# =============================================================================
# ANALYSIS CONTEXT
# =============================================================================


@dataclass
class AnalysisContext:
    """
    Request-scoped context for VSA analysis.

    Carries all mutable state for a single analysis request:
    - codebook: Dict mapping entity names to hypervectors
    - primitives: Dict mapping primitive names to hypervectors
    - Statistics for adaptive thresholds
    - Resonator hyperparameters

    Thread-safe: Each request creates its own context instance.

    GLM Extensions:
    - validate_claim(): Geometric validation gate
    - grounded_query(): Query with confidence thresholds
    - similarity(): Cosine similarity for complex vectors
    - permute(): Circular permutation for asymmetric binding
    """

    # Core configuration
    dimensions: int = 16384
    dtype: torch.dtype = field(default=torch.complex64)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # Resonator hyperparameters
    alpha: float = 0.85
    power: float = 0.64
    top_k: int = 50
    multi_steps: int = 2
    iters: int = 450
    convergence_threshold: float = 1e-4

    # State (mutated during analysis)
    codebook: dict[str, torch.Tensor] = field(default_factory=dict)
    sku_set: set[str] = field(default_factory=set)  # Track which entries are SKUs
    _primitives: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    _codebook_tensor: torch.Tensor | None = field(default=None, repr=False)
    _codebook_keys: list[str] | None = field(default=None, repr=False)
    _codebook_dirty: bool = field(default=True, repr=False)

    # FAISS index for fast similarity search (v3.5 optimization)
    _faiss_index: Any = field(default=None, repr=False)
    _faiss_dirty: bool = field(default=True, repr=False)
    use_faiss: bool = field(default=True)  # Enable FAISS by default

    # Relation store for chain inference (v3.7)
    _relations: dict[str, torch.Tensor] | None = field(default=None, repr=False)

    # Statistics (updated during bundling)
    avg_qty: float = 20.0
    avg_margin: float = 0.3
    avg_sold: float = 10.0
    rows_processed: int = 0
    leak_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize device and dtype after dataclass initialization."""
        # Ensure dtype is a torch dtype
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

        # Ensure device is a torch device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    # =========================================================================
    # CORE VSA OPERATIONS
    # =========================================================================

    def seed_hash(self, string: str) -> torch.Tensor:
        """
        Generate deterministic hypervector from string via SHA256 → phases.

        The vector is complex with unit magnitude elements (phasor representation).
        This ensures binding operations are well-defined and invertible.

        Args:
            string: Input string to hash

        Returns:
            Complex hypervector of shape (dimensions,) with unit magnitude elements
        """
        # Create deterministic hash
        hash_bytes = hashlib.sha256(string.encode()).digest()

        # Expand hash to required dimensions using SHAKE256
        shake = hashlib.shake_256(hash_bytes)
        expanded = shake.digest(self.dimensions)

        # Convert bytes to phases in [0, 2π)
        # Each byte gives us one element (0-255 → 0-2π)
        byte_values = torch.tensor(
            [b for b in expanded], dtype=torch.float32, device=self.device
        )
        phases = byte_values / 255.0 * 2 * torch.pi

        # Create unit complex vector: e^(i*phase) = cos(phase) + i*sin(phase)
        vec = torch.complex(torch.cos(phases), torch.sin(phases))
        return vec.to(dtype=self.dtype)

    def batch_seed_hash(
        self, strings: list[str], prefix: str = "entity_"
    ) -> torch.Tensor:
        """
        BATCHED seed_hash - generates multiple hypervectors in chunked operations.

        v3.6 OPTIMIZATION: Instead of calling seed_hash() N times (88s for 36K SKUs),
        this generates vectors in chunks to balance speed vs memory.

        Memory calculation for 36K SKUs x 16384 dims:
        - Full batch: ~4.8 GB (too large for 16GB Fargate)
        - Chunk of 1000: ~130 MB per chunk (safe)

        Args:
            strings: List of strings to hash
            prefix: Prefix added to each string before hashing (default: "entity_")

        Returns:
            Tensor of shape (N, dimensions) with complex phasor vectors
        """
        n = len(strings)
        if n == 0:
            return torch.zeros(0, self.dimensions, dtype=self.dtype, device=self.device)

        # Process in chunks to avoid memory issues
        # Each chunk: 1000 x 16384 x 8 bytes (complex64) = ~130 MB
        CHUNK_SIZE = 1000
        chunks = []

        for chunk_start in range(0, n, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n)
            chunk_strings = strings[chunk_start:chunk_end]
            chunk_n = len(chunk_strings)

            # Pre-allocate numpy array for chunk bytes
            chunk_bytes = np.zeros((chunk_n, self.dimensions), dtype=np.uint8)

            # Generate hashes for this chunk
            for i, s in enumerate(chunk_strings):
                full_string = f"{prefix}{s}"
                hash_bytes = hashlib.sha256(full_string.encode()).digest()
                shake = hashlib.shake_256(hash_bytes)
                expanded = shake.digest(self.dimensions)
                chunk_bytes[i] = np.frombuffer(expanded, dtype=np.uint8)

            # Vectorized phase computation
            phases = chunk_bytes.astype(np.float32) / 255.0 * (2 * np.pi)

            # Vectorized complex phasor creation
            cos_phases = np.cos(phases)
            sin_phases = np.sin(phases)
            complex_array = cos_phases + 1j * sin_phases

            # Convert chunk to torch tensor
            chunk_tensor = torch.from_numpy(complex_array.astype(np.complex64)).to(
                device=self.device, dtype=self.dtype
            )
            chunks.append(chunk_tensor)

            # Free memory
            del chunk_bytes, phases, cos_phases, sin_phases, complex_array

        # Concatenate all chunks
        return torch.cat(chunks, dim=0)

    def normalize(self, v: torch.Tensor) -> torch.Tensor:
        """
        L2 normalize a tensor along the last dimension.

        For complex vectors, this uses the complex norm.

        Args:
            v: Input tensor

        Returns:
            Normalized tensor with unit L2 norm
        """
        norm = torch.norm(v, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)
        return v / norm

    def zeros(self) -> torch.Tensor:
        """
        Create a zero vector of the correct shape and dtype.

        Returns:
            Zero tensor of shape (dimensions,)
        """
        return torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Compute cosine similarity between two vectors.

        For complex vectors: Re(⟨a, b*⟩) / (||a|| * ||b||)

        This is the core geometric measurement that determines
        whether two vectors are "close" in the hyperdimensional space.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        a_norm = self.normalize(a)
        b_norm = self.normalize(b)
        return torch.real(torch.sum(a_norm * b_norm.conj())).item()

    def permute(self, v: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """
        Circular permutation - used for asymmetric binding.

        This breaks commutativity: permute(a) ⊗ b ≠ a ⊗ permute(b)
        Essential for encoding directional relationships like cause → effect.

        Args:
            v: Input vector
            shifts: Number of positions to shift (positive = right)

        Returns:
            Permuted vector
        """
        return torch.roll(v, shifts=shifts, dims=-1)

    def inverse_permute(self, v: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """
        Inverse of permute operation.

        Args:
            v: Input vector
            shifts: Original shift amount

        Returns:
            Inverse permuted vector
        """
        return torch.roll(v, shifts=-shifts, dims=-1)

    # =========================================================================
    # CODEBOOK MANAGEMENT
    # =========================================================================

    def add_to_codebook(self, entity: str, is_sku: bool = False) -> None:
        """
        Add an entity to the codebook.

        Lazily creates the hypervector when first accessed via get_or_create().
        Marks codebook as dirty to invalidate cached tensor.

        Args:
            entity: Entity name (SKU, description, vendor, category, etc.)
            is_sku: Whether this entity is a SKU (for filtering in queries)
        """
        if entity not in self.codebook:
            self.codebook[entity] = None  # Lazy creation
            self._codebook_dirty = True

        if is_sku:
            self.sku_set.add(entity)

    def get_or_create(self, entity: str) -> torch.Tensor:
        """
        Get or create hypervector for an entity.

        If the entity doesn't exist in the codebook, creates it.
        Uses deterministic hashing for consistency.

        Args:
            entity: Entity name

        Returns:
            Hypervector for the entity
        """
        if entity not in self.codebook or self.codebook[entity] is None:
            self.codebook[entity] = self.seed_hash(f"entity_{entity}")
            self._codebook_dirty = True

        return self.codebook[entity]

    def batch_populate_codebook(
        self, entities: list[str], is_sku: bool = False
    ) -> None:
        """
        BATCHED codebook population - creates all vectors in a single operation.

        v3.6 OPTIMIZATION: Instead of calling get_or_create() N times,
        this uses batch_seed_hash() to create all vectors at once.

        For 36K SKUs: ~88s → ~2s speedup.

        Args:
            entities: List of entity names to add to codebook
            is_sku: Whether these entities are SKUs (for filtering)
        """
        import time

        start_time = time.time()

        # Filter out entities that already exist
        new_entities = [
            e for e in entities if e not in self.codebook or self.codebook[e] is None
        ]

        if not new_entities:
            logger.debug("batch_populate_codebook: all entities already exist")
            return

        logger.info(f"v3.6: Batch creating {len(new_entities)} vectors...")

        # Batch create all vectors
        vectors = self.batch_seed_hash(new_entities, prefix="entity_")

        # Populate codebook
        for i, entity in enumerate(new_entities):
            self.codebook[entity] = vectors[i]
            if is_sku:
                self.sku_set.add(entity)

        self._codebook_dirty = True
        self._faiss_dirty = True

        elapsed = time.time() - start_time
        logger.info(
            f"v3.6: Batch populated {len(new_entities)} vectors in {elapsed:.2f}s"
        )

    def get_codebook_tensor(self) -> torch.Tensor | None:
        """
        Get codebook as a stacked tensor for batch operations.

        Caches the result until codebook is modified.
        Also builds FAISS index if enabled.

        Returns:
            Tensor of shape (num_entities, dimensions), or None if empty
        """
        if not self.codebook:
            return None

        if self._codebook_dirty or self._codebook_tensor is None:
            # Ensure all vectors are created
            for key in self.codebook:
                if self.codebook[key] is None:
                    self.codebook[key] = self.seed_hash(f"entity_{key}")

            self._codebook_keys = list(self.codebook.keys())
            vectors = [self.codebook[k] for k in self._codebook_keys]
            self._codebook_tensor = torch.stack(vectors)
            self._codebook_dirty = False
            self._faiss_dirty = True  # Rebuild FAISS index when codebook changes

        return self._codebook_tensor

    def _build_faiss_index(self) -> None:
        """
        Build FAISS index from codebook for fast similarity search.

        v3.5 OPTIMIZATION: Uses FAISS IndexFlatIP for inner product similarity.
        For complex phasor vectors, we convert to real 2D representation
        (real, imag interleaved) and use inner product which approximates
        the real part of the complex dot product.
        """
        if not FAISS_AVAILABLE or not self.use_faiss:
            return

        codebook_tensor = self.get_codebook_tensor()
        if codebook_tensor is None:
            return

        if not self._faiss_dirty and self._faiss_index is not None:
            return

        start_time = __import__("time").time()

        # Convert complex to real by interleaving real and imaginary parts
        # Shape: (N, D) complex -> (N, 2*D) real
        real_part = torch.real(codebook_tensor).cpu().numpy()
        imag_part = torch.imag(codebook_tensor).cpu().numpy()

        # Interleave: [r0, i0, r1, i1, ...]
        n_vectors, d = real_part.shape
        interleaved = np.empty((n_vectors, d * 2), dtype=np.float32)
        interleaved[:, 0::2] = real_part.astype(np.float32)
        interleaved[:, 1::2] = imag_part.astype(np.float32)

        # Normalize for inner product similarity
        norms = np.linalg.norm(interleaved, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        interleaved = interleaved / norms

        # Build FAISS index - use flat index for exact search
        # Inner product gives us the cosine similarity for normalized vectors
        index = faiss.IndexFlatIP(d * 2)
        index.add(interleaved)

        self._faiss_index = index
        self._faiss_dirty = False

        elapsed = __import__("time").time() - start_time
        logger.info(
            f"FAISS index built: {n_vectors} vectors, {d * 2} dims in {elapsed:.3f}s"
        )

    def faiss_search(
        self,
        query_vector: torch.Tensor,
        top_k: int = 50,
    ) -> tuple[list[str], list[float]]:
        """
        Fast similarity search using FAISS.

        v3.5 OPTIMIZATION: O(log n) search instead of O(n) matrix multiplication.

        Args:
            query_vector: Complex query vector of shape (dimensions,)
            top_k: Number of top results to return

        Returns:
            Tuple of (entity_names, similarity_scores)
        """
        if not FAISS_AVAILABLE or not self.use_faiss:
            return self._fallback_search(query_vector, top_k)

        # Build index if needed
        self._build_faiss_index()

        if self._faiss_index is None:
            return self._fallback_search(query_vector, top_k)

        # Convert query to interleaved real format
        query_real = torch.real(query_vector).cpu().numpy().astype(np.float32)
        query_imag = torch.imag(query_vector).cpu().numpy().astype(np.float32)

        d = len(query_real)
        query_interleaved = np.empty((1, d * 2), dtype=np.float32)
        query_interleaved[0, 0::2] = query_real
        query_interleaved[0, 1::2] = query_imag

        # Normalize query
        norm = np.linalg.norm(query_interleaved)
        if norm > 1e-8:
            query_interleaved = query_interleaved / norm

        # Search
        k = min(top_k, len(self._codebook_keys or []))
        if k == 0:
            return [], []

        distances, indices = self._faiss_index.search(query_interleaved, k)

        # Convert to results
        results = []
        scores = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0 and idx < len(self._codebook_keys):
                results.append(self._codebook_keys[idx])
                scores.append(float(score))

        return results, scores

    def _fallback_search(
        self,
        query_vector: torch.Tensor,
        top_k: int = 50,
    ) -> tuple[list[str], list[float]]:
        """
        Fallback similarity search using matrix multiplication.

        Used when FAISS is not available.
        """
        codebook_tensor = self.get_codebook_tensor()
        if codebook_tensor is None:
            return [], []

        # Normalize query
        query_norm = self.normalize(query_vector)

        # Matrix multiplication: (N, D) @ (D,) -> (N,)
        F_mat = codebook_tensor.T.to(self.dtype)
        raw_sims = torch.real(torch.conj(F_mat).T @ query_norm)

        # Get top matches
        k = min(top_k, len(self.codebook))
        topk_vals, topk_idx = torch.topk(raw_sims, k)

        codebook_keys = self.get_codebook_keys()
        results = [codebook_keys[i] for i in topk_idx.tolist()]
        scores = topk_vals.tolist()

        return results, scores

    def get_codebook_keys(self) -> list[str]:
        """
        Get ordered list of codebook keys matching the tensor.

        Returns:
            List of entity names in same order as codebook tensor
        """
        if self._codebook_dirty or self._codebook_keys is None:
            self.get_codebook_tensor()  # This updates _codebook_keys

        return self._codebook_keys or []

    # =========================================================================
    # PRIMITIVES
    # =========================================================================

    def get_primitives(self) -> dict[str, torch.Tensor]:
        """
        Get or create all primitives including domain and logical operators.

        Primitives are deterministically generated and cached per-context.

        Domain Primitives (11 profit leak types):
        - low_stock, high_margin_leak, dead_item, negative_inventory
        - overstock, price_discrepancy, shrinkage_pattern, margin_erosion
        - zero_cost_anomaly, negative_profit, severe_inventory_deficit

        Utility Primitives:
        - high_velocity, seasonal

        Logical Primitives (GLM reasoning):
        - and, or, implies, not, forall, exists, equals

        Temporal/Causal Primitives:
        - causes, before, after, during, trend_up, trend_down

        Returns:
            Dict mapping primitive names to hypervectors
        """
        if self._primitives is not None:
            return self._primitives

        # Domain primitives (profit leak detection)
        domain_primitives = [
            "low_stock",
            "high_margin_leak",
            "dead_item",
            "negative_inventory",
            "overstock",
            "price_discrepancy",
            "shrinkage_pattern",
            "margin_erosion",
            "zero_cost_anomaly",
            "negative_profit",
            "severe_inventory_deficit",
            # Utility primitives
            "high_velocity",
            "seasonal",
        ]

        # Logical primitives (symbolic reasoning)
        logical_primitives = [
            "and",  # Conjunction: bind(and, bind(A, B))
            "or",  # Disjunction: bundle(bind(or, A), bind(or, B))
            "implies",  # Implication: bind(implies, bind(antecedent, consequent))
            "not",  # Negation: bind(not, A) - approximate
            "forall",  # Universal: bind(forall, bind(variable, predicate))
            "exists",  # Existential: bind(exists, bind(variable, predicate))
            "equals",  # Identity: bind(equals, bind(A, B))
        ]

        # Temporal/Causal primitives
        temporal_primitives = [
            "causes",  # Causal (asymmetric): permute(bind(causes, A)) ⊗ B
            "before",  # Temporal precedence
            "after",  # Temporal succession
            "during",  # Temporal overlap
            "trend_up",  # Directional change
            "trend_down",  # Directional change
        ]

        self._primitives = {}

        all_primitives = domain_primitives + logical_primitives + temporal_primitives
        for name in all_primitives:
            self._primitives[name] = self.seed_hash(f"primitive_{name}")

        return self._primitives

    def get_primitive(self, key: str) -> torch.Tensor | None:
        """
        Get a single primitive by name.

        Args:
            key: Primitive name

        Returns:
            Hypervector for the primitive, or None if not found
        """
        primitives = self.get_primitives()
        return primitives.get(key)

    # =========================================================================
    # STATISTICS MANAGEMENT
    # =========================================================================

    def update_stats(self, avg_qty: float, avg_margin: float, avg_sold: float) -> None:
        """
        Update running statistics for adaptive thresholds.

        Called during bundling after sampling first N rows.

        Args:
            avg_qty: Average quantity on hand
            avg_margin: Average margin percentage
            avg_sold: Average units sold
        """
        self.avg_qty = avg_qty
        self.avg_margin = avg_margin
        self.avg_sold = avg_sold

    def increment_leak_count(self, primitive_name: str) -> None:
        """
        Increment the count for a specific leak type.

        Args:
            primitive_name: Name of the leak primitive
        """
        if primitive_name not in self.leak_counts:
            self.leak_counts[primitive_name] = 0
        self.leak_counts[primitive_name] += 1

    # =========================================================================
    # GLM VALIDATION METHODS (Geometric Language Model)
    # =========================================================================

    def validate_claim(
        self,
        bundle: torch.Tensor,
        claim_vector: torch.Tensor,
        threshold: float = 0.40,
    ) -> tuple[bool, float, str]:
        """
        Geometric validation gate - the core GLM constraint mechanism.

        This is where "invalid reasoning becomes geometrically impossible."
        Claims that don't resonate with the encoded knowledge are rejected
        based on quantifiable similarity bounds.

        The validation gate provides 60x separation between valid claims
        (similarity ~ 0.6-1.0) and invalid claims (similarity ~ 0.01),
        enabling reliable rejection of hallucinated or unsupported claims.

        Args:
            bundle: The bundled knowledge hypervector
            claim_vector: The proposed claim encoded as a vector
            threshold: Minimum similarity for acceptance (default 0.40)

        Returns:
            Tuple of (is_valid, similarity_score, confidence_level)
            - is_valid: True if claim passes the threshold
            - similarity_score: Raw cosine similarity
            - confidence_level: One of "rejected", "noise_floor", "low_confidence",
              "moderate_confidence", "high_confidence", "very_high_confidence"

        Example:
            >>> is_valid, sim, conf = ctx.validate_claim(bundle, sku_vec)
            >>> if is_valid:
            ...     print(f"Claim validated with {conf} confidence: {sim:.3f}")
            ... else:
            ...     print(f"Claim rejected: similarity {sim:.3f} at {conf}")
        """
        sim = self.similarity(bundle, claim_vector)

        if sim < VALIDATION_THRESHOLDS["rejection_threshold"]:
            return (False, sim, "rejected")
        elif sim < VALIDATION_THRESHOLDS["low_confidence"]:
            return (False, sim, "noise_floor")
        elif sim < VALIDATION_THRESHOLDS["moderate_confidence"]:
            return (True, sim, "low_confidence")
        elif sim < VALIDATION_THRESHOLDS["high_confidence"]:
            return (True, sim, "moderate_confidence")
        elif sim < VALIDATION_THRESHOLDS["very_high_confidence"]:
            return (True, sim, "high_confidence")
        else:
            return (True, sim, "very_high_confidence")

    def grounded_query(
        self,
        bundle: torch.Tensor,
        primitive_key: str,
        min_confidence: float = 0.40,
        top_k: int = 20,
    ) -> list[dict] | None:
        """
        Query bundle with geometric grounding - only returns results above threshold.

        Unlike raw query_bundle(), this enforces the GLM constraint:
        results below min_confidence are NOT returned, preventing
        the system from claiming relationships that don't exist in the geometry.

        This is the key interface for LLM grounding - the LLM proposes queries,
        and this method validates that the results are geometrically grounded.

        Args:
            bundle: The bundled knowledge hypervector
            primitive_key: Key for the primitive to query
            min_confidence: Minimum similarity for acceptance (default 0.40)
            top_k: Maximum number of results to return

        Returns:
            List of {entity, similarity, confidence_level} dicts, or None if
            no grounded results found.

        Note:
            This method imports query_bundle from core to avoid circular imports.
            The actual resonator cleanup happens in query_bundle.
        """
        # Import here to avoid circular dependency
        from core import query_bundle

        results, scores = query_bundle(self, bundle, primitive_key, top_k * 2)

        grounded_results = []
        for entity, score in zip(results, scores):
            entity_vec = self.get_or_create(entity)
            is_valid, sim, confidence = self.validate_claim(
                bundle, entity_vec, min_confidence
            )
            if is_valid:
                grounded_results.append(
                    {
                        "entity": entity,
                        "similarity": sim,
                        "confidence_level": confidence,
                    }
                )

            if len(grounded_results) >= top_k:
                break

        return grounded_results if grounded_results else None

    # =========================================================================
    # BATCH OPERATIONS (Performance Optimization)
    # =========================================================================

    def batch_similarity(
        self,
        vectors: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity of multiple vectors against a single target.

        Optimized for batch processing - much faster than calling
        similarity() in a loop.

        Args:
            vectors: Tensor of shape (N, dimensions) - vectors to compare
            target: Tensor of shape (dimensions,) - target vector

        Returns:
            Tensor of shape (N,) containing similarity scores
        """
        # Normalize all vectors
        vectors_norm = self.normalize(vectors)
        target_norm = self.normalize(target)

        # Batch dot product: (N, D) @ (D,) -> (N,)
        similarities = torch.real(torch.sum(vectors_norm * target_norm.conj(), dim=-1))
        return similarities

    def batch_validate_claims(
        self,
        bundle: torch.Tensor,
        claim_vectors: torch.Tensor,
        threshold: float = 0.40,
    ) -> list[tuple[bool, float, str]]:
        """
        Validate multiple claims in a single batch operation.

        Much faster than calling validate_claim() in a loop.

        Args:
            bundle: The bundled knowledge hypervector
            claim_vectors: Tensor of shape (N, dimensions) - claims to validate
            threshold: Minimum similarity for acceptance

        Returns:
            List of (is_valid, similarity, confidence_level) tuples
        """
        similarities = self.batch_similarity(claim_vectors, bundle)

        results = []
        for sim in similarities.tolist():
            if sim < VALIDATION_THRESHOLDS["rejection_threshold"]:
                results.append((False, sim, "rejected"))
            elif sim < VALIDATION_THRESHOLDS["low_confidence"]:
                results.append((False, sim, "noise_floor"))
            elif sim < VALIDATION_THRESHOLDS["moderate_confidence"]:
                results.append((True, sim, "low_confidence"))
            elif sim < VALIDATION_THRESHOLDS["high_confidence"]:
                results.append((True, sim, "moderate_confidence"))
            elif sim < VALIDATION_THRESHOLDS["very_high_confidence"]:
                results.append((True, sim, "high_confidence"))
            else:
                results.append((True, sim, "very_high_confidence"))

        return results

    def resonator_cleanup(
        self,
        noisy_vector: torch.Tensor,
        max_iters: int | None = None,
    ) -> torch.Tensor:
        """
        Clean up a noisy query vector using the resonator.

        This is the core operation for multi-hop reasoning - after each
        unbinding operation, the result is noisy. The resonator "cleans up"
        the vector by projecting it onto the nearest codebook entries.

        Args:
            noisy_vector: Noisy vector to clean up, shape (dimensions,)
            max_iters: Override max iterations (default: use ctx.iters)

        Returns:
            Cleaned vector, shape (dimensions,)

        Note:
            This method imports convergence_lock_resonator from core
            to avoid circular imports.
        """
        from core import convergence_lock_resonator

        # Add batch dimension
        batch = noisy_vector.unsqueeze(0)

        # Run resonator
        cleaned_batch = convergence_lock_resonator(self, batch)

        # Remove batch dimension
        return cleaned_batch[0]

    def bind(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Bind two vectors using element-wise complex multiplication.

        For phasor vectors: (a ⊗ b)[i] = a[i] * b[i]
        This adds phases: e^(iθ₁) * e^(iθ₂) = e^(i(θ₁+θ₂))

        Binding is:
        - Associative: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
        - Commutative: a ⊗ b = b ⊗ a
        - Invertible: a ⊗ b ⊗ b* ≈ a (where b* is conjugate)
        - Preserves norm: ||a ⊗ b|| ≈ ||a|| * ||b||

        Args:
            a: First vector, shape (dimensions,)
            b: Second vector, shape (dimensions,)

        Returns:
            Bound vector, shape (dimensions,)
        """
        return self.normalize(a * b)

    def unbind(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Unbind a key from a bound vector.

        For phasor vectors: unbind(bound, key) = bound * key.conj()
        This subtracts phases: e^(i(θ₁+θ₂)) * e^(-iθ₂) = e^(iθ₁)

        Used to recover one component of a binding:
        unbind(a ⊗ b, b) ≈ a

        Args:
            bound: The bound vector
            key: The key to unbind

        Returns:
            Unbound vector (approximate original)
        """
        return self.normalize(bound * key.conj())

    def bundle(
        self,
        vectors: list[torch.Tensor],
        weights: list[float] | None = None,
    ) -> torch.Tensor:
        """
        Bundle multiple vectors into a superposition.

        Bundling creates a vector that is similar to all input vectors.
        With weights, some vectors contribute more than others.

        Args:
            vectors: List of vectors to bundle
            weights: Optional weights (default: uniform)

        Returns:
            Bundled vector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty list")

        if weights is None:
            weights = [1.0] * len(vectors)

        weighted_sum = sum(v * w for v, w in zip(vectors, weights))
        return self.normalize(weighted_sum)

    # =========================================================================
    # DIRAC VSA INTEGRATION
    # =========================================================================

    def get_dirac_vsa(self) -> DiracVSA:
        """
        Get a DiracVSA instance configured for this context.

        Creates a DiracVSA with matching dimensions, device, and dtype.
        This is the recommended way to access Dirac algebra operations
        from within an AnalysisContext.

        Returns:
            DiracVSA instance matching context configuration

        Example:
            >>> ctx = create_analysis_context()
            >>> dvsa = ctx.get_dirac_vsa()
            >>> cause = dvsa.seed_hash("price_drop")
            >>> effect = dvsa.seed_hash("margin_erosion")
            >>> bound = dvsa.bind(cause, effect)
        """
        return DiracVSA(
            dimensions=self.dimensions,
            device=self.device,
            dtype=self.dtype,
        )

    def to_dirac(self, v: torch.Tensor, entropy: float = 0.0) -> DiracVector:
        """
        Convert a standard hypervector to a DiracVector.

        Useful for upgrading existing dense vectors (from codebook or
        primitives) to the full Dirac representation.

        The input vector is used as both spatial and temporal components
        (they start identical but diverge through asymmetric operations).

        Args:
            v: Dense complex tensor of shape (dimensions,)
            entropy: Initial entropy value (default: 0.0)

        Returns:
            DiracVector with spatial and temporal components from v

        Example:
            >>> ctx = create_analysis_context()
            >>> sku_vec = ctx.get_or_create("SKU123")
            >>> sku_dirac = ctx.to_dirac(sku_vec)
        """
        if v.shape[0] != self.dimensions:
            raise ValueError(
                f"Vector dimensions {v.shape[0]} != context dimensions {self.dimensions}"
            )

        # Ensure complex type
        if not torch.is_complex(v):
            v = v.to(self.dtype)

        return DiracVector(
            spatial=v.clone(),
            temporal=v.clone(),  # Start identical, diverge through operations
            phase=complex(1.0, 0.0),
            entropy=entropy,
        )

    def dirac_similarity(
        self,
        a: DiracVector,
        b: DiracVector,
    ) -> float:
        """
        Compute similarity between two DiracVectors.

        Convenience method that delegates to DiracVector.similarity().

        Args:
            a: First DiracVector
            b: Second DiracVector

        Returns:
            Similarity score in [-1, 1]
        """
        return a.similarity(b)

    # =========================================================================
    # CHAIN INFERENCE (Causal Reasoning)
    # =========================================================================

    def _get_relations(self) -> dict[str, torch.Tensor]:
        """Get or create the relations store."""
        if self._relations is None:
            self._relations = {}
        return self._relations

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
    ) -> str:
        """
        Add a causal relation using asymmetric phasor binding.

        Encodes: subject --predicate--> object
        Uses permutation for asymmetry: bind(permute(predicate), bind(subject, object))

        This creates a relation vector where:
        - subject and object are bound together
        - predicate is permuted then bound, making relation directional
        - unbinding predicate recovers the subject-object binding
        - unbinding subject from that recovers object (and vice versa)

        Args:
            subject: The cause entity (e.g., "pricing_error")
            predicate: The relation type (e.g., "causes")
            obj: The effect entity (e.g., "margin_erosion")
            confidence: Optional confidence weight (default 1.0)

        Returns:
            Relation key for later lookup

        Example:
            >>> ctx = create_analysis_context()
            >>> ctx.add_relation("pricing_error", "causes", "margin_erosion")
            >>> ctx.add_relation("margin_erosion", "causes", "profit_leak")
            >>> path = ctx.find_causal_path("pricing_error", "profit_leak")
            >>> # Returns: ["pricing_error", "margin_erosion", "profit_leak"]
        """
        relations = self._get_relations()

        # Get or create vectors for each component
        subject_vec = self.get_or_create(subject)
        predicate_vec = self.get_or_create(f"rel_{predicate}")
        object_vec = self.get_or_create(obj)

        # Create asymmetric relation encoding:
        # 1. Bind subject and object: subject ⊗ object
        subject_object = self.bind(subject_vec, object_vec)

        # 2. Permute predicate for asymmetry: permute(predicate)
        permuted_pred = self.permute(predicate_vec)

        # 3. Bind permuted predicate with subject-object binding
        relation_vec = self.bind(permuted_pred, subject_object)

        # 4. Apply confidence weighting
        if confidence != 1.0:
            relation_vec = relation_vec * confidence

        # Store with normalized key
        relation_key = f"{subject.lower()}|{predicate.lower()}|{obj.lower()}"
        relations[relation_key] = self.normalize(relation_vec)

        # Also store in codebook for similarity search
        self.codebook[relation_key] = relations[relation_key]
        self._codebook_dirty = True
        self._faiss_dirty = True

        return relation_key

    def chain_inference(
        self,
        query_subject: str,
        query_predicate: str,
        max_hops: int = 3,
        min_similarity: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        Perform multi-hop inference following causal chains.

        Given a subject and predicate, finds all entities reachable
        through chains of the predicate relation.

        Algorithm:
        1. Start with query_subject
        2. Find all direct relations (query_subject, predicate, X)
        3. For each X found, recursively find (X, predicate, Y)
        4. Continue until max_hops or no more relations found

        Args:
            query_subject: Starting entity
            query_predicate: Relation to follow (e.g., "causes")
            max_hops: Maximum chain length (default 3)
            min_similarity: Minimum similarity threshold (default 0.3)

        Returns:
            List of (entity, cumulative_similarity) tuples ordered by similarity

        Example:
            >>> ctx = create_analysis_context()
            >>> ctx.add_relation("overstock", "causes", "dead_inventory")
            >>> ctx.add_relation("dead_inventory", "causes", "margin_erosion")
            >>> ctx.add_relation("margin_erosion", "causes", "profit_leak")
            >>> results = ctx.chain_inference("overstock", "causes", max_hops=3)
            >>> # Returns: [("dead_inventory", 0.9), ("margin_erosion", 0.7), ("profit_leak", 0.5)]
        """
        relations = self._get_relations()
        if not relations:
            return []

        # Track visited to avoid cycles
        visited = {query_subject.lower()}
        results = []

        # BFS with similarity tracking
        frontier = [(query_subject, 1.0, 0)]  # (entity, cumulative_sim, hops)

        while frontier:
            current_entity, current_sim, hops = frontier.pop(0)

            if hops >= max_hops:
                continue

            # Find all relations starting from current_entity with query_predicate
            predicate_lower = query_predicate.lower()

            for rel_key, rel_vec in relations.items():
                parts = rel_key.split("|")
                if len(parts) != 3:
                    continue

                subj, pred, obj = parts

                # Check if this relation matches our query
                if subj == current_entity.lower() and pred == predicate_lower:
                    if obj not in visited:
                        # Compute similarity to verify the relation is valid
                        obj_vec = self.get_or_create(obj)
                        sim = self.similarity(rel_vec, obj_vec)

                        if sim >= min_similarity:
                            cumulative_sim = current_sim * sim
                            results.append((obj, cumulative_sim))
                            visited.add(obj)
                            frontier.append((obj, cumulative_sim, hops + 1))

        # Sort by cumulative similarity descending
        results.sort(key=lambda x: -x[1])
        return results

    def find_causal_path(
        self,
        start: str,
        end: str,
        predicate: str = "causes",
        max_hops: int = 5,
    ) -> list[str] | None:
        """
        Find the causal path from start to end entity.

        Uses BFS to find the shortest path through the relation graph.

        Args:
            start: Starting entity
            end: Target entity
            predicate: Relation type to follow (default "causes")
            max_hops: Maximum path length (default 5)

        Returns:
            List of entities forming the path, or None if no path found

        Example:
            >>> ctx = create_analysis_context()
            >>> ctx.add_relation("pricing_error", "causes", "margin_erosion")
            >>> ctx.add_relation("margin_erosion", "causes", "profit_leak")
            >>> path = ctx.find_causal_path("pricing_error", "profit_leak")
            >>> # Returns: ["pricing_error", "margin_erosion", "profit_leak"]
        """
        relations = self._get_relations()
        if not relations:
            return None

        start_lower = start.lower()
        end_lower = end.lower()
        predicate_lower = predicate.lower()

        if start_lower == end_lower:
            return [start]

        # Build adjacency from relations
        adjacency: dict[str, list[str]] = {}
        for rel_key in relations.keys():
            parts = rel_key.split("|")
            if len(parts) != 3:
                continue
            subj, pred, obj = parts
            if pred == predicate_lower:
                if subj not in adjacency:
                    adjacency[subj] = []
                adjacency[subj].append(obj)

        # BFS for shortest path
        visited = {start_lower}
        queue = [(start_lower, [start])]  # (current, path)

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            for neighbor in adjacency.get(current, []):
                if neighbor == end_lower:
                    return path + [end]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_all_effects(
        self,
        cause: str,
        predicate: str = "causes",
        max_hops: int = 3,
    ) -> dict[str, list[str]]:
        """
        Get all effects of a cause, organized by hop distance.

        Args:
            cause: The cause entity
            predicate: Relation type (default "causes")
            max_hops: Maximum chain length (default 3)

        Returns:
            Dict mapping hop_distance -> list of effect entities

        Example:
            >>> effects = ctx.get_all_effects("pricing_error")
            >>> # Returns: {1: ["margin_erosion"], 2: ["profit_leak"]}
        """
        relations = self._get_relations()
        if not relations:
            return {}

        cause_lower = cause.lower()
        predicate_lower = predicate.lower()

        result: dict[int, list[str]] = {}
        visited = {cause_lower}

        # BFS with hop tracking
        frontier = [(cause_lower, 0)]

        while frontier:
            current, hops = frontier.pop(0)

            if hops >= max_hops:
                continue

            for rel_key in relations.keys():
                parts = rel_key.split("|")
                if len(parts) != 3:
                    continue

                subj, pred, obj = parts
                if subj == current and pred == predicate_lower:
                    if obj not in visited:
                        visited.add(obj)
                        hop_distance = hops + 1
                        if hop_distance not in result:
                            result[hop_distance] = []
                        result[hop_distance].append(obj)
                        frontier.append((obj, hop_distance))

        return result

    def get_all_causes(
        self,
        effect: str,
        predicate: str = "causes",
        max_hops: int = 3,
    ) -> dict[str, list[str]]:
        """
        Get all causes of an effect, organized by hop distance.

        This traces the causal chain backwards.

        Args:
            effect: The effect entity
            predicate: Relation type (default "causes")
            max_hops: Maximum chain length (default 3)

        Returns:
            Dict mapping hop_distance -> list of cause entities

        Example:
            >>> causes = ctx.get_all_causes("profit_leak")
            >>> # Returns: {1: ["margin_erosion"], 2: ["pricing_error"]}
        """
        relations = self._get_relations()
        if not relations:
            return {}

        effect_lower = effect.lower()
        predicate_lower = predicate.lower()

        # Build reverse adjacency
        reverse_adj: dict[str, list[str]] = {}
        for rel_key in relations.keys():
            parts = rel_key.split("|")
            if len(parts) != 3:
                continue
            subj, pred, obj = parts
            if pred == predicate_lower:
                if obj not in reverse_adj:
                    reverse_adj[obj] = []
                reverse_adj[obj].append(subj)

        result: dict[int, list[str]] = {}
        visited = {effect_lower}

        # BFS backwards
        frontier = [(effect_lower, 0)]

        while frontier:
            current, hops = frontier.pop(0)

            if hops >= max_hops:
                continue

            for cause_entity in reverse_adj.get(current, []):
                if cause_entity not in visited:
                    visited.add(cause_entity)
                    hop_distance = hops + 1
                    if hop_distance not in result:
                        result[hop_distance] = []
                    result[hop_distance].append(cause_entity)
                    frontier.append((cause_entity, hop_distance))

        return result

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    @property
    def dataset_stats(self) -> dict[str, float]:
        """
        Get current dataset statistics.

        Returns:
            Dict with avg_margin, avg_quantity, avg_sold
        """
        return {
            "avg_margin": self.avg_margin,
            "avg_quantity": self.avg_qty,
            "avg_sold": self.avg_sold,
        }

    def reset(self) -> None:
        """
        Reset context state for reuse.

        Clears codebook, leak counts, statistics, and relations.
        Primitives are kept (they are deterministic).
        """
        self.codebook.clear()
        self.sku_set.clear()
        self._codebook_tensor = None
        self._codebook_keys = None
        self._codebook_dirty = True
        self._relations = None  # Clear relations

        self.avg_qty = 20.0
        self.avg_margin = 0.3
        self.avg_sold = 10.0
        self.rows_processed = 0
        self.leak_counts.clear()

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the context state for debugging/logging.

        Returns:
            Dict with codebook_size, sku_count, leak_counts, stats
        """
        return {
            "codebook_size": len(self.codebook),
            "sku_count": len(self.sku_set),
            "leak_counts": dict(self.leak_counts),
            "rows_processed": self.rows_processed,
            "stats": {
                "avg_qty": self.avg_qty,
                "avg_margin": self.avg_margin,
                "avg_sold": self.avg_sold,
            },
            "dimensions": self.dimensions,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def confidence_to_label(similarity: float) -> str:
    """
    Convert similarity score to human-readable confidence label.

    Uses the standard VALIDATION_THRESHOLDS to determine the label.

    Args:
        similarity: Cosine similarity score

    Returns:
        One of: "rejected", "noise_floor", "low_confidence",
                "moderate_confidence", "high_confidence", "very_high_confidence"
    """
    if similarity < VALIDATION_THRESHOLDS["rejection_threshold"]:
        return "rejected"
    elif similarity < VALIDATION_THRESHOLDS["low_confidence"]:
        return "noise_floor"
    elif similarity < VALIDATION_THRESHOLDS["moderate_confidence"]:
        return "low_confidence"
    elif similarity < VALIDATION_THRESHOLDS["high_confidence"]:
        return "moderate_confidence"
    elif similarity < VALIDATION_THRESHOLDS["very_high_confidence"]:
        return "high_confidence"
    else:
        return "very_high_confidence"


def create_sparse_vsa(
    dimensions: int = 16384,
    sparsity_ratio: float = 0.1,
) -> SparseVSA:
    """
    Create a SparseVSA instance for sparse vector operations.

    NOTE: This creates the sparse OPTIMIZATION version of VSA.
    For true Dirac-inspired semantic extension, use create_dirac_vsa().

    Args:
        dimensions: Total dimensionality D
        sparsity_ratio: Fraction of non-zero elements (K/D)

    Returns:
        SparseVSA instance
    """
    default_sparsity = int(dimensions * sparsity_ratio)
    return SparseVSA(dimensions=dimensions, default_sparsity=default_sparsity)


def create_dirac_vsa(
    dimensions: int = 16384,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> DiracVSA:
    """
    Create a DiracVSA instance for true Dirac-inspired semantic operations.

    This creates the full Dirac algebra with:
    - Temporal asymmetry: bind(A,B) ≠ bind(B,A)
    - Entropy tracking: Monotonically increasing for irreversible ops
    - Negation: Phase rotation by π
    - Causal binding: Clean cause→effect recovery

    Args:
        dimensions: Vector dimensionality d
        device: Target device (default: CPU)
        dtype: Complex dtype (default: complex64)

    Returns:
        DiracVSA instance
    """
    return DiracVSA(dimensions=dimensions, device=device, dtype=dtype)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_analysis_context(
    dimensions: int = 16384,
    use_gpu: bool = True,
    dtype: torch.dtype | str = torch.complex64,
    alpha: float = 0.85,
    power: float = 0.64,
    top_k: int = 50,
    multi_steps: int = 2,
    iters: int = 450,
    convergence_threshold: float = 1e-4,
) -> AnalysisContext:
    """
    Create a new analysis context with the given parameters.

    This is the recommended way to create contexts for analysis.
    Each analysis request should get its own context instance.

    Args:
        dimensions: VSA dimensionality (default 16384 = 2^14)
        use_gpu: Whether to use GPU if available
        dtype: Torch data type (default complex64 for phasor VSA)
        alpha: Resonator blending factor
        power: Resonator exponential sharpening
        top_k: Resonator sparse projection
        multi_steps: Outer iteration loops
        iters: Total inner iterations
        convergence_threshold: Early-stop delta

    Returns:
        New AnalysisContext instance

    Example:
        >>> ctx = create_analysis_context(use_gpu=True)
        >>> bundle = bundle_pos_facts(ctx, rows)
        >>> results, scores = query_bundle(ctx, bundle, "low_stock")
    """
    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Handle string dtype
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    return AnalysisContext(
        dimensions=dimensions,
        dtype=dtype,
        device=device,
        alpha=alpha,
        power=power,
        top_k=top_k,
        multi_steps=multi_steps,
        iters=iters,
        convergence_threshold=convergence_threshold,
    )
