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

import math

import torch

from .vectors import normalize, similarity

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


def weighted_bundle(vectors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
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
    bundle_vec: torch.Tensor, key: torch.Tensor, codebook: torch.Tensor
) -> tuple[int, float]:
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
    vectors: list[torch.Tensor], position_shifts: list[int] | None = None
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


def solve_analogy(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
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
    role_filler_pairs: list[tuple[torch.Tensor, torch.Tensor]],
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
    record: torch.Tensor, role: torch.Tensor, codebook: torch.Tensor
) -> tuple[int, float]:
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
    query: torch.Tensor, codebook: torch.Tensor, temperature: float = 1.0
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
    result = torch.einsum("n,nd->d", weights, codebook)

    return normalize(result)


def sparse_resonance_step(
    query: torch.Tensor, codebook: torch.Tensor, top_k: int = 64, power: float = 0.64
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

    # Handle complex tensors: cast weights to match codebook dtype
    if selected.is_complex():
        weights = weights.to(selected.dtype)

    result = torch.einsum("n,nd->d", weights, selected)

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


def orthogonality_check(vectors: list[torch.Tensor], threshold: float = 0.1) -> bool:
    """Check if vectors are approximately orthogonal.

    Args:
        vectors: List of vectors to check
        threshold: Maximum acceptable pairwise similarity

    Returns:
        True if all pairs have |similarity| < threshold
    """
    for i, a in enumerate(vectors):
        for b in vectors[i + 1 :]:
            if abs(similarity(a, b)) >= threshold:
                return False
    return True


# =============================================================================
# NOVEL PRIMITIVES FOR ALWAYS-ON AGENT
# =============================================================================


def n_bind(v: torch.Tensor) -> torch.Tensor:
    """Negation binding: Create anti-vector that is maximally dissimilar.

    For FHRR (complex phasors): π phase shift
        n_bind(v)[i] = e^(i(θ_v[i] + π)) = -v[i]

    Properties:
        - sim(v, n_bind(v)) = -1 (maximally dissimilar)
        - n_bind(n_bind(v)) = v (involutory)
        - bundle(v, n_bind(v)) ≈ 0 (cancellation)

    Use for:
        - Exclusion queries: "Find X but NOT Y"
        - Contradiction detection
        - Negated facts in reasoning

    Args:
        v: Input vector

    Returns:
        Negated vector (π phase shift for phasors)

    Example:
        not_dead_item = n_bind(dead_item_vec)
        active_items = query_bundle(bundle, bind(in_stock, not_dead_item))
    """
    return -v  # For complex phasors, negation = π phase shift


def query_excluding(
    bundle_vec: torch.Tensor, include: torch.Tensor, exclude: torch.Tensor
) -> torch.Tensor:
    """Query bundle for items matching 'include' but not 'exclude'.

    Combines binding with negation for exclusion queries.

    Args:
        bundle_vec: The bundle to query
        include: Pattern to include
        exclude: Pattern to exclude

    Returns:
        Query result excluding unwanted pattern

    Example:
        # Find high-margin items that aren't dead stock
        result = query_excluding(inventory_bundle, high_margin, dead_item)
    """
    query = bind(include, n_bind(exclude))
    return unbind(bundle_vec, query)


def cw_bundle(
    vectors: list[torch.Tensor], confidences: list[float], temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence-weighted bundling with learned magnitude encoding.

    Instead of just weighting the sum, encode confidence in magnitude:
        result[i] = Σ (c_j^τ · v_j[i])

    Where τ (temperature) controls confidence sharpness:
        - τ < 1: Amplify high-confidence items (sharper)
        - τ = 1: Linear weighting
        - τ > 1: Smooth out confidence differences (softer)

    Args:
        vectors: List of vectors to bundle
        confidences: Confidence score [0, 1] for each vector
        temperature: Sharpness of confidence weighting

    Returns:
        (bundle_vector, confidence_vector) tuple
        - bundle_vector: Normalized weighted superposition
        - confidence_vector: Per-dimension average confidence

    Use for:
        - Evidence accumulation with varying certainty
        - Ensemble voting with confidence
        - Soft attention over memory

    Example:
        detections = [low_stock_vec, dead_item_vec]
        confidences = [0.95, 0.3]  # Very sure about low_stock, uncertain about dead_item
        bundle, conf = cw_bundle(detections, confidences, temperature=0.5)
    """
    if len(vectors) != len(confidences):
        raise ValueError("vectors and confidences must have same length")
    if len(vectors) == 0:
        raise ValueError("At least one vector required")

    # Normalize confidences to [0, 1]
    c = torch.tensor(confidences, dtype=torch.float32)
    c = torch.clamp(c, 0.01, 1.0)  # Avoid zero weights

    # Apply temperature (inverse for sharpening behavior)
    c_scaled = torch.pow(c, 1.0 / temperature)

    # Weight each vector
    result = torch.zeros_like(vectors[0])
    conf_accumulator = torch.zeros(
        vectors[0].shape[0], dtype=torch.float32, device=vectors[0].device
    )

    for v, conf in zip(vectors, c_scaled):
        if v.device != result.device:
            v = v.to(result.device)

        # Convert confidence to match vector dtype if complex
        if result.is_complex():
            conf_weight = conf.to(result.dtype)
        else:
            conf_weight = conf

        result = result + conf_weight * v
        conf_accumulator = conf_accumulator + float(conf)

    # Normalize bundle
    bundle_vec = normalize(result)

    # Average confidence per dimension
    conf_vec = conf_accumulator / len(vectors)

    return bundle_vec, conf_vec


def t_bind(
    v: torch.Tensor,
    timestamp: float,
    reference_time: float,
    decay_rate: float = 0.1,
    max_shift: int = 1000,
) -> torch.Tensor:
    """Temporal binding with exponential decay and position encoding.

    Combines:
    1. Temporal decay: Recent events have higher magnitude
    2. Position encoding: Events at different times are distinguishable

    Formula:
        t_bind(v, t) = decay(t) · permute(v, pos(t))

    Where:
        - decay(t) = exp(-λ · (t_ref - t))
        - pos(t) = hash(t) mod max_shift

    Args:
        v: Input vector
        timestamp: Event timestamp (seconds since epoch)
        reference_time: Current/reference time
        decay_rate: Decay constant λ (higher = faster decay)
        max_shift: Maximum permutation shift

    Returns:
        Temporally encoded vector

    Use for:
        - Temporal context in working memory
        - "What happened recently?" queries
        - Causal inference (did X precede Y?)

    Example:
        # Encode sales event from yesterday
        event_vec = t_bind(sale_vec, yesterday_ts, now_ts, decay_rate=0.01)
    """
    import hashlib

    # Compute decay factor (days-based for readability)
    time_delta_days = (reference_time - timestamp) / 86400.0  # Convert to days
    decay_factor = math.exp(-decay_rate * max(time_delta_days, 0))

    # Compute temporal position shift (deterministic hash)
    t_hash = int(hashlib.sha256(str(timestamp).encode()).hexdigest()[:8], 16)
    shift = t_hash % max_shift

    # Apply decay
    if v.is_complex():
        decayed = decay_factor * v
    else:
        decayed = decay_factor * v

    # Apply temporal shift
    shifted = permute(decayed, shift)

    return shifted


def t_unbind(
    bundle_vec: torch.Tensor,
    timestamp: float,
    reference_time: float,
    decay_rate: float = 0.1,
    max_shift: int = 1000,
) -> torch.Tensor:
    """Inverse of t_bind for temporal queries.

    Reverses the temporal encoding to query for events at a specific time.

    Args:
        bundle_vec: Temporally encoded bundle
        timestamp: Time to query
        reference_time: Reference time used in encoding
        decay_rate: Same decay rate used in encoding
        max_shift: Same max shift used in encoding

    Returns:
        Unbound vector for querying events at timestamp

    Example:
        # Query what happened yesterday
        yesterday_context = t_unbind(memory_bundle, yesterday_ts, now_ts)
        matches = resonator.resonate(yesterday_context)
    """
    import hashlib

    time_delta_days = (reference_time - timestamp) / 86400.0
    decay_factor = math.exp(-decay_rate * max(time_delta_days, 0))

    t_hash = int(hashlib.sha256(str(timestamp).encode()).hexdigest()[:8], 16)
    shift = t_hash % max_shift

    # Reverse operations
    unshifted = inverse_permute(bundle_vec, shift)

    # Undo decay (with epsilon to avoid division by zero)
    undecayed = unshifted / (decay_factor + 1e-10)

    return normalize(undecayed)
