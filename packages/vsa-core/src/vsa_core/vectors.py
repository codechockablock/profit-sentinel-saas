"""
vsa_core/vectors.py - Hyperdimensional Vector Generation and Operations

Mathematical Foundation:
    Complex phasor vectors v ∈ ℂ^d where each component lies on the unit circle:
    v[i] = e^(iθ[i]) where θ[i] ∈ [0, 2π)

    This embeds all vectors on the complex hypersphere S^(2d-1) ⊂ ℂ^d,
    giving the space rich geometric structure with geodesic distances
    and Fréchet means.

Key Properties:
    - All vectors have unit magnitude: |v[i]| = 1 for all i
    - Binding (element-wise multiplication) is a group operation
    - Bundling (sum + normalize) computes circular mean of phases
    - Similarity measured via normalized real inner product
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from .types import VectorConfig

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

_config = VectorConfig()
_device: torch.device | None = None


def configure(
    dimensions: int | None = None,
    dtype: str | None = None,
    device: str | None = None,
) -> VectorConfig:
    """Configure global VSA settings.

    Args:
        dimensions: Vector dimensionality (power of 2 recommended)
        dtype: "complex64" or "complex128"
        device: "cuda", "cpu", or "auto"

    Returns:
        Updated configuration
    """
    global _config, _device

    updates = {}
    if dimensions is not None:
        updates["dimensions"] = dimensions
    if dtype is not None:
        updates["dtype"] = dtype
    if device is not None:
        updates["device"] = device

    if updates:
        _config = VectorConfig(**{**_config.model_dump(), **updates})
        _device = None  # Reset cached device

    return _config


def get_config() -> VectorConfig:
    """Get current configuration."""
    return _config


def get_device() -> torch.device:
    """Get current torch device (cached)."""
    global _device
    if _device is None:
        _device = _config.get_device()
    return _device


def get_dtype() -> torch.dtype:
    """Get current torch dtype."""
    return _config.get_dtype()


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize(v: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Project vector onto complex unit hypersphere.

    Each component is normalized to unit magnitude while preserving phase:
        v[i] → v[i] / |v[i]|

    This is the projection onto the constraint manifold where all
    components lie on the unit circle in ℂ.

    Args:
        v: Input tensor of shape (..., d)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with |v[i]| = 1 for all i

    Mathematical Note:
        This is NOT L2 normalization (which would put the whole vector
        on a sphere). This normalizes each component independently,
        keeping us on the product manifold (S^1)^d.
    """
    magnitudes = torch.abs(v)
    # Handle zero/near-zero magnitudes by replacing with random phase
    mask = magnitudes < eps
    if mask.any():
        random_phases = torch.rand_like(v.real) * 2 * math.pi
        v = torch.where(
            mask,
            torch.exp(1j * random_phases.to(v.dtype)),
            v
        )
        magnitudes = torch.abs(v)

    return v / magnitudes


def normalize_global(v: torch.Tensor) -> torch.Tensor:
    """L2 normalize entire vector (for similarity computation).

    This is standard L2 normalization: v → v / ||v||_2
    Used after bundling to maintain comparable magnitudes.

    Args:
        v: Input tensor

    Returns:
        L2-normalized tensor
    """
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / (norm + 1e-10)


# =============================================================================
# VECTOR GENERATION
# =============================================================================

def seed_hash(
    seed: str,
    dimensions: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Generate deterministic phasor vector from seed string.

    Uses SHA-256 expanded hash to generate phases θ ∈ [0, 2π),
    then constructs v[i] = e^(iθ[i]).

    Args:
        seed: Deterministic seed string
        dimensions: Override dimension (default: config)
        device: Override device (default: config)
        dtype: Override dtype (default: config)

    Returns:
        Complex unit phasor vector of shape (d,)

    Properties:
        - Deterministic: same seed always produces same vector
        - Quasi-orthogonal: different seeds produce nearly orthogonal vectors
          (expected similarity ≈ 1/√d for random seeds)
    """
    dim = dimensions or _config.dimensions
    dev = device or get_device()
    dt = dtype or get_dtype()

    # Expand hash to cover all dimensions
    hash_bytes = b""
    block_idx = 0
    bytes_needed = dim * 4  # 4 bytes per float32 phase

    while len(hash_bytes) < bytes_needed:
        block_seed = f"{seed}:{block_idx}"
        hash_bytes += hashlib.sha256(block_seed.encode()).digest()
        block_idx += 1

    # Convert to phases in [0, 2π)
    import numpy as np
    phase_ints = np.frombuffer(hash_bytes[:bytes_needed], dtype=np.uint32)
    phases = (phase_ints / np.iinfo(np.uint32).max) * 2 * np.pi

    # Construct phasor vector
    phases_tensor = torch.tensor(phases[:dim], dtype=torch.float32, device=dev)
    vector = torch.exp(1j * phases_tensor)

    if dt == torch.complex128:
        vector = vector.to(torch.complex128)

    return vector


def random_vector(
    dimensions: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Generate random phasor vector.

    Samples phases uniformly from [0, 2π) and constructs e^(iθ).

    Args:
        dimensions: Override dimension (default: config)
        device: Override device (default: config)
        dtype: Override dtype (default: config)

    Returns:
        Random complex unit phasor vector
    """
    dim = dimensions or _config.dimensions
    dev = device or get_device()
    dt = dtype or get_dtype()

    phases = torch.rand(dim, device=dev) * 2 * math.pi
    vector = torch.exp(1j * phases)

    if dt == torch.complex128:
        vector = vector.to(torch.complex128)

    return vector


def identity_vector(
    dimensions: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Generate multiplicative identity vector (all ones).

    This is the identity element for binding: bind(v, identity) = v

    Args:
        dimensions: Override dimension (default: config)
        device: Override device (default: config)
        dtype: Override dtype (default: config)

    Returns:
        Vector of all 1+0j
    """
    dim = dimensions or _config.dimensions
    dev = device or get_device()
    dt = dtype or get_dtype()

    return torch.ones(dim, dtype=dt, device=dev)


def zero_vector(
    dimensions: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Generate additive identity vector (all zeros).

    This is the identity element for bundling: bundle(v, zero) = v
    Note: This is NOT on the manifold (components have zero magnitude).

    Args:
        dimensions: Override dimension (default: config)
        device: Override device (default: config)
        dtype: Override dtype (default: config)

    Returns:
        Vector of all 0+0j
    """
    dim = dimensions or _config.dimensions
    dev = device or get_device()
    dt = dtype or get_dtype()

    return torch.zeros(dim, dtype=dt, device=dev)


# =============================================================================
# SIMILARITY
# =============================================================================

def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute normalized cosine similarity between two vectors.

    For phasor vectors, this measures phase alignment:
        sim(a, b) = Re(⟨a, b*⟩) / (||a|| · ||b||)

    Args:
        a: First vector
        b: Second vector

    Returns:
        Similarity in [-1, 1], where 1 means identical phases

    Mathematical Note:
        For unit phasors, this equals the mean of cos(θ_a[i] - θ_b[i]).
    """
    # Ensure same device
    if a.device != b.device:
        b = b.to(a.device)

    dot = torch.sum(a * torch.conj(b))
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)

    sim = (dot.real / (norm_a * norm_b + 1e-10)).item()
    return float(sim)


def batch_similarity(
    query: torch.Tensor,
    codebook: torch.Tensor
) -> torch.Tensor:
    """Compute similarity between query and all codebook vectors.

    Efficiently computes sim(query, codebook[i]) for all i using
    batched matrix operations.

    Args:
        query: Query vector of shape (d,)
        codebook: Codebook matrix of shape (n, d)

    Returns:
        Similarity scores of shape (n,)
    """
    # Ensure same device
    if query.device != codebook.device:
        codebook = codebook.to(query.device)

    # Compute dot products: (n, d) @ (d,) -> (n,)
    dots = torch.sum(codebook * torch.conj(query), dim=-1)

    # Compute norms
    query_norm = torch.norm(query)
    codebook_norms = torch.norm(codebook, dim=-1)

    # Normalize
    similarities = dots.real / (query_norm * codebook_norms + 1e-10)

    return similarities


def top_k_similar(
    query: torch.Tensor,
    codebook: torch.Tensor,
    k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find top-k most similar vectors in codebook.

    Args:
        query: Query vector
        codebook: Codebook matrix
        k: Number of results

    Returns:
        (indices, similarities) of top-k matches
    """
    sims = batch_similarity(query, codebook)
    k = min(k, len(sims))
    values, indices = torch.topk(sims, k)
    return indices, values


# =============================================================================
# VECTOR INFO & DEBUGGING
# =============================================================================

def vector_info(v: torch.Tensor) -> dict[str, Any]:
    """Get diagnostic information about a vector.

    Args:
        v: Input vector

    Returns:
        Dictionary with shape, dtype, device, magnitude stats, phase stats
    """
    mags = torch.abs(v)
    phases = torch.angle(v)

    return {
        "shape": tuple(v.shape),
        "dtype": str(v.dtype),
        "device": str(v.device),
        "magnitude": {
            "min": float(mags.min()),
            "max": float(mags.max()),
            "mean": float(mags.mean()),
            "std": float(mags.std()),
        },
        "phase": {
            "min": float(phases.min()),
            "max": float(phases.max()),
            "mean": float(phases.mean()),
            "std": float(phases.std()),
        },
        "is_normalized": bool((torch.abs(mags - 1.0) < 0.01).all()),
        "l2_norm": float(torch.norm(v)),
    }


def angular_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean angular distance between phasor vectors.

    This is the geodesic distance on the product manifold (S^1)^d:
        d(a, b) = (1/d) Σ |angle(a[i]) - angle(b[i])|

    Args:
        a: First vector
        b: Second vector

    Returns:
        Mean angular distance in [0, π]
    """
    phase_diff = torch.angle(a * torch.conj(b))
    return float(torch.abs(phase_diff).mean())


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

def batch_seed_hash(
    seeds: list[str],
    dimensions: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Generate multiple vectors from seed strings.

    Args:
        seeds: List of seed strings
        dimensions: Override dimension (default: config)
        device: Override device (default: config)
        dtype: Override dtype (default: config)

    Returns:
        Tensor of shape (len(seeds), d)
    """
    vectors = [seed_hash(s, dimensions, device, dtype) for s in seeds]
    return torch.stack(vectors)


def ensure_device(v: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Ensure vector is on specified device.

    Args:
        v: Input vector
        device: Target device (default: config device)

    Returns:
        Vector on target device
    """
    target = device or get_device()
    if v.device != target:
        return v.to(target)
    return v
