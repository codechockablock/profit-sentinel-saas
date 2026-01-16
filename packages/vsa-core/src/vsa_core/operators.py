"""
vsa_core/operators.py - Algebraic Operations for Vector Symbolic Architectures

This module implements the core algebraic operations that make HDC/VSA powerful:

BINDING (⊗):
    Element-wise multiplication of phasor vectors.
    bind(a, b)[i] = a[i] · b[i] = e^(i(θ_a[i] + θ_b[i]))

    Properties:
    - Commutative: a ⊗ b = b ⊗ a
    - Associative: (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)
    - Self-inverse: a ⊗ a* = identity (where a* is conjugate)
    - Distributes over bundling: a ⊗ (b ⊕ c) ≈ (a ⊗ b) ⊕ (a ⊗ c)
    - Similarity-preserving: sim(a ⊗ c, b ⊗ c) = sim(a, b)

BUNDLING (⊕):
    Superposition of vectors (sum + normalize).
    bundle(a, b) = normalize(a + b)

    Properties:
    - Commutative: a ⊕ b = b ⊕ a
    - Approximate associativity: (a ⊕ b) ⊕ c ≈ a ⊕ (b ⊕ c)
    - Creates "set" representation: bundle is similar to all components
    - Capacity: ~√d items can be bundled before interference

PERMUTATION (π):
    Circular shift of vector indices (or phase rotation).
    permute(v, k)[i] = v[(i - k) mod d]

    Properties:
    - Invertible: permute(permute(v, k), -k) = v
    - Useful for: sequence encoding, role binding, temporal markers

UNBINDING:
    Inverse of binding using complex conjugate.
    unbind(bound, key) = bound ⊗ conj(key)

    If bound = a ⊗ b, then unbind(bound, a) ≈ b
"""
from __future__ import annotations
import torch
from typing import List, Optional, Tuple, Union
import math

from .vectors import normalize, normalize_global, get_device, get_dtype, similarity


# =============================================================================
# BINDING OPERATIONS
# =============================================================================

def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bind two vectors via element-wise multiplication.

    For phasor vectors, this adds phases:
        (a ⊗ b)[i] = e^(i(θ_a[i] + θ_b[i]))

    Use for: creating associations, role-filler pairs, relational facts.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Bound vector (normalized to unit phasors)

    Example:
        sku_anomaly = bind(sku_vector, anomaly_vector)
        # sku_anomaly is similar to neither sku nor anomaly alone,
        # but can be unbound to recover either component.
    """
    if a.device != b.device:
        b = b.to(a.device)
    return normalize(a * b)


def bind_many(*vectors: torch.Tensor) -> torch.Tensor:
    """Bind multiple vectors together.

    Computes: v1 ⊗ v2 ⊗ ... ⊗ vn

    Args:
        *vectors: Variable number of vectors to bind

    Returns:
        Bound result
    """
    if len(vectors) == 0:
        raise ValueError("At least one vector required")
    if len(vectors) == 1:
        return vectors[0]

    result = vectors[0]
    for v in vectors[1:]:
        result = bind(result, v)
    return result


def unbind(bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Unbind a vector using complex conjugate.

    If bound = a ⊗ b, then unbind(bound, a) ≈ b

    For phasor vectors, this subtracts phases:
        unbind(bound, key)[i] = e^(i(θ_bound[i] - θ_key[i]))

    Args:
        bound: Bound vector
        key: Key to unbind with

    Returns:
        Unbound result (approximate recovery of other component)
    """
    if bound.device != key.device:
        key = key.to(bound.device)
    return normalize(bound * torch.conj(key))


# =============================================================================
# BUNDLING OPERATIONS
# =============================================================================

def bundle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bundle two vectors via superposition.

    For phasor vectors, this computes the circular mean of phases
    at each dimension:
        bundle(a, b) = normalize(a + b)

    The result is similar to both inputs:
        sim(bundle(a, b), a) > 0 and sim(bundle(a, b), b) > 0

    Use for: creating sets, accumulating evidence, memory traces.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Bundled vector (normalized)
    """
    if a.device != b.device:
        b = b.to(a.device)
    return normalize(a + b)


def bundle_many(*vectors: torch.Tensor) -> torch.Tensor:
    """Bundle multiple vectors together.

    Computes: normalize(v1 + v2 + ... + vn)

    Args:
        *vectors: Variable number of vectors to bundle

    Returns:
        Bundled result

    Note:
        Capacity is approximately √d vectors before significant
        interference. For d=16384, this is ~128 vectors.
    """
    if len(vectors) == 0:
        raise ValueError("At least one vector required")
    if len(vectors) == 1:
        return vectors[0]

    result = vectors[0].clone()
    for v in vectors[1:]:
        if v.device != result.device:
            v = v.to(result.device)
        result = result + v
    return normalize(result)


def weighted_bundle(
    vectors: List[torch.Tensor],
    weights: List[float]
) -> torch.Tensor:
    """Bundle vectors with weights.

    Computes: normalize(Σ w_i · v_i)

    Useful for confidence-weighted evidence accumulation.

    Args:
        vectors: List of vectors to bundle
        weights: Corresponding weights

    Returns:
        Weighted bundle
    """
    if len(vectors) != len(weights):
        raise ValueError("vectors and weights must have same length")
    if len(vectors) == 0:
        raise ValueError("At least one vector required")

    result = weights[0] * vectors[0]
    for v, w in zip(vectors[1:], weights[1:]):
        if v.device != result.device:
            v = v.to(result.device)
        result = result + w * v
    return normalize(result)


def unbind_from_bundle(
    bundle_vec: torch.Tensor,
    key: torch.Tensor,
    codebook: torch.Tensor
) -> Tuple[int, float]:
    """Probe a bundle to find bound partner of key.

    If bundle contains (key ⊗ value), this finds value in codebook.

    Args:
        bundle_vec: The bundle to probe
        key: Known component to unbind with
        codebook: Matrix of candidate vectors to search

    Returns:
        (best_index, similarity) of most likely partner
    """
    unbound = unbind(bundle_vec, key)
    from .vectors import batch_similarity
    sims = batch_similarity(unbound, codebook)
    best_idx = int(torch.argmax(sims))
    return best_idx, float(sims[best_idx])


# =============================================================================
# PERMUTATION OPERATIONS
# =============================================================================

def permute(v: torch.Tensor, shift: int) -> torch.Tensor:
    """Permute vector by circular index shift.

    permute(v, k)[i] = v[(i - k) mod d]

    Positive shift moves elements forward (like shifting time).

    Use for: sequence encoding, temporal markers, structural roles.

    Args:
        v: Input vector
        shift: Number of positions to shift

    Returns:
        Permuted vector
    """
    return torch.roll(v, shifts=shift)


def inverse_permute(v: torch.Tensor, shift: int) -> torch.Tensor:
    """Inverse of permute operation.

    inverse_permute(permute(v, k), k) = v

    Args:
        v: Permuted vector
        shift: Original shift amount

    Returns:
        Original vector
    """
    return torch.roll(v, shifts=-shift)


def sequence_encode(
    vectors: List[torch.Tensor],
    position_shifts: Optional[List[int]] = None
) -> torch.Tensor:
    """Encode ordered sequence using permutation.

    Creates a single vector representing: v1 → v2 → v3 → ...

    Default uses power-of-2 shifts to create distinguishable positions.

    Args:
        vectors: Ordered list of vectors
        position_shifts: Custom shifts (default: [0, 1, 2, 4, 8, ...])

    Returns:
        Sequence encoding vector

    Example:
        # Encode "low_stock → high_velocity → margin_leak" pattern
        seq = sequence_encode([low_stock, high_velocity, margin_leak])
    """
    if len(vectors) == 0:
        raise ValueError("At least one vector required")

    if position_shifts is None:
        # Use power-of-2 shifts for better separation
        position_shifts = [0]
        shift = 1
        for _ in range(len(vectors) - 1):
            position_shifts.append(shift)
            shift *= 2

    if len(position_shifts) != len(vectors):
        raise ValueError("position_shifts must match vectors length")

    shifted = [permute(v, s) for v, s in zip(vectors, position_shifts)]
    return bundle_many(*shifted)


# =============================================================================
# ANALOGY & REASONING OPERATIONS
# =============================================================================

def solve_analogy(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor
) -> torch.Tensor:
    """Solve analogy: A is to B as C is to ?

    Computes: unbind(bind(b, unbind(c, a)), identity)
            = b ⊗ (c ⊗ a*) = b ⊗ c ⊗ a*

    For phasor vectors, this transfers the relationship.

    Args:
        a: First term
        b: Second term (related to a)
        c: Third term

    Returns:
        Fourth term d such that a:b :: c:d

    Example:
        # If low_stock : reorder_alert :: dead_item : ?
        result = solve_analogy(low_stock, reorder_alert, dead_item)
        # result should be similar to markdown_suggestion
    """
    # Compute: b ⊗ c ⊗ conj(a)
    return bind(bind(b, c), torch.conj(a))


def role_filler_bind(role: torch.Tensor, filler: torch.Tensor) -> torch.Tensor:
    """Create role-filler pair using binding.

    This is the standard VSA pattern for structured representations.

    Args:
        role: Role vector (e.g., "SUBJECT", "OBJECT", "LOCATION")
        filler: Filler vector (e.g., specific entity)

    Returns:
        Bound role-filler pair
    """
    return bind(role, filler)


def create_record(
    role_filler_pairs: List[Tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    """Create structured record from role-filler pairs.

    record = bundle(role1 ⊗ filler1, role2 ⊗ filler2, ...)

    Args:
        role_filler_pairs: List of (role, filler) tuples

    Returns:
        Record vector encoding all role-filler bindings

    Example:
        record = create_record([
            (ROLE_SKU, sku_vec),
            (ROLE_ANOMALY, anomaly_vec),
            (ROLE_SEVERITY, severity_vec)
        ])
    """
    bindings = [bind(role, filler) for role, filler in role_filler_pairs]
    return bundle_many(*bindings)


def query_record(
    record: torch.Tensor,
    role: torch.Tensor,
    codebook: torch.Tensor
) -> Tuple[int, float]:
    """Query a record for a role's filler.

    Args:
        record: Record vector
        role: Role to query
        codebook: Codebook of possible fillers

    Returns:
        (best_index, similarity) of most likely filler
    """
    return unbind_from_bundle(record, role, codebook)


# =============================================================================
# ADVANCED ALGEBRAIC OPERATIONS
# =============================================================================

def resonance_step(
    query: torch.Tensor,
    codebook: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Single step of resonator cleanup.

    Computes soft attention over codebook and reconstructs.

    Args:
        query: Current query vector
        codebook: Codebook matrix (n, d)
        temperature: Softmax temperature (lower = sharper)

    Returns:
        Cleaned query vector
    """
    from .vectors import batch_similarity

    # Compute similarities
    sims = batch_similarity(query, codebook)

    # Apply temperature and softmax
    weights = torch.softmax(sims / temperature, dim=0)

    # Weighted combination of codebook
    result = torch.einsum('n,nd->d', weights, codebook)

    return normalize(result)


def sparse_resonance_step(
    query: torch.Tensor,
    codebook: torch.Tensor,
    top_k: int = 64,
    power: float = 0.64
) -> torch.Tensor:
    """Sparse resonator step using only top-k similar vectors.

    More efficient than full resonance for large codebooks.

    Args:
        query: Current query vector
        codebook: Codebook matrix (n, d)
        top_k: Number of top matches to use
        power: Similarity amplification exponent

    Returns:
        Cleaned query vector
    """
    from .vectors import batch_similarity

    # Compute similarities
    sims = batch_similarity(query, codebook)

    # Get top-k
    k = min(top_k, len(sims))
    values, indices = torch.topk(sims, k)

    # Amplify with power
    weights = torch.pow(torch.clamp(values, min=0), power)
    weights = weights / (weights.sum() + 1e-10)

    # Weighted combination of top-k
    selected = codebook[indices]
    result = torch.einsum('n,nd->d', weights, selected)

    return normalize(result)


def bundle_capacity(dimensions: int) -> int:
    """Estimate bundle capacity for given dimensionality.

    Returns approximate number of vectors that can be bundled
    while maintaining reliable recovery (>50% similarity).

    Args:
        dimensions: Vector dimensionality

    Returns:
        Estimated capacity
    """
    # Theoretical capacity is O(√d)
    # Empirically, ~0.5√d is safe for reliable recovery
    return int(0.5 * math.sqrt(dimensions))


def orthogonality_check(
    vectors: List[torch.Tensor],
    threshold: float = 0.1
) -> bool:
    """Check if vectors are approximately orthogonal.

    Args:
        vectors: List of vectors to check
        threshold: Maximum acceptable pairwise similarity

    Returns:
        True if all pairs have |similarity| < threshold
    """
    for i, a in enumerate(vectors):
        for b in vectors[i + 1:]:
            if abs(similarity(a, b)) >= threshold:
                return False
    return True
