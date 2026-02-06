"""
tests/vsa_core/test_operators.py - Algebraic Invariant Tests

These tests verify the fundamental algebraic properties of VSA operations.
Passing these tests ensures the mathematical foundations are sound.

Key Properties Tested:
    - Binding: commutative, associative, self-inverse, similarity-preserving
    - Bundling: commutative, creates set representation
    - Permutation: invertible
    - Unbinding: recovers bound components
"""

import os
import sys

import pytest
import torch

# Add parent to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from vsa_core import (
    configure,
    identity_vector,
    seed_hash,
    similarity,
)
from vsa_core.operators import (
    bind,
    bind_many,
    bundle,
    bundle_capacity,
    bundle_many,
    create_record,
    inverse_permute,
    orthogonality_check,
    permute,
    query_record,
    sequence_encode,
    solve_analogy,
    unbind,
    weighted_bundle,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def config():
    """Configure VSA for testing."""
    return configure(dimensions=4096, device="cpu")


@pytest.fixture
def vec_a():
    return seed_hash("test_vector_a")


@pytest.fixture
def vec_b():
    return seed_hash("test_vector_b")


@pytest.fixture
def vec_c():
    return seed_hash("test_vector_c")


# =============================================================================
# BINDING TESTS
# =============================================================================


class TestBinding:
    """Tests for binding operation algebraic properties."""

    def test_bind_commutativity(self, vec_a, vec_b):
        """bind(a, b) = bind(b, a)"""
        ab = bind(vec_a, vec_b)
        ba = bind(vec_b, vec_a)
        sim = similarity(ab, ba)
        assert sim > 0.999, f"Binding should be commutative, got sim={sim}"

    def test_bind_associativity(self, vec_a, vec_b, vec_c):
        """(a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)"""
        ab_c = bind(bind(vec_a, vec_b), vec_c)
        a_bc = bind(vec_a, bind(vec_b, vec_c))
        sim = similarity(ab_c, a_bc)
        assert sim > 0.999, f"Binding should be associative, got sim={sim}"

    def test_bind_self_inverse(self, vec_a):
        """a ⊗ conj(a) = identity"""
        identity = identity_vector()
        self_bound = bind(vec_a, torch.conj(vec_a))
        sim = similarity(self_bound, identity)
        assert (
            sim > 0.999
        ), f"Self-binding with conjugate should yield identity, got sim={sim}"

    def test_bind_identity(self, vec_a):
        """a ⊗ identity = a"""
        identity = identity_vector()
        result = bind(vec_a, identity)
        sim = similarity(result, vec_a)
        assert (
            sim > 0.999
        ), f"Binding with identity should preserve vector, got sim={sim}"

    def test_bind_similarity_preservation(self, vec_a, vec_b, vec_c):
        """sim(a ⊗ c, b ⊗ c) = sim(a, b)"""
        original_sim = similarity(vec_a, vec_b)
        ac = bind(vec_a, vec_c)
        bc = bind(vec_b, vec_c)
        bound_sim = similarity(ac, bc)

        # Should be very close
        assert (
            abs(original_sim - bound_sim) < 0.05
        ), f"Binding should preserve similarity: original={original_sim}, bound={bound_sim}"

    def test_bind_dissimilarity(self, vec_a, vec_b):
        """Bound vector should be dissimilar to both inputs."""
        bound = bind(vec_a, vec_b)
        sim_a = abs(similarity(bound, vec_a))
        sim_b = abs(similarity(bound, vec_b))

        # Should be nearly orthogonal (low similarity)
        assert sim_a < 0.15, f"Bound vector too similar to first input: {sim_a}"
        assert sim_b < 0.15, f"Bound vector too similar to second input: {sim_b}"

    def test_bind_many(self, vec_a, vec_b, vec_c):
        """bind_many should equal sequential binding."""
        sequential = bind(bind(vec_a, vec_b), vec_c)
        many = bind_many(vec_a, vec_b, vec_c)
        sim = similarity(sequential, many)
        assert sim > 0.999, f"bind_many should equal sequential bind, got sim={sim}"


# =============================================================================
# UNBINDING TESTS
# =============================================================================


class TestUnbinding:
    """Tests for unbinding operation."""

    def test_unbind_recovers_component(self, vec_a, vec_b):
        """unbind(a ⊗ b, a) ≈ b"""
        bound = bind(vec_a, vec_b)
        recovered = unbind(bound, vec_a)
        sim = similarity(recovered, vec_b)
        assert sim > 0.95, f"Unbinding should recover bound component, got sim={sim}"

    def test_unbind_symmetric(self, vec_a, vec_b):
        """unbind(a ⊗ b, b) ≈ a"""
        bound = bind(vec_a, vec_b)
        recovered = unbind(bound, vec_b)
        sim = similarity(recovered, vec_a)
        assert sim > 0.95, f"Unbinding should work symmetrically, got sim={sim}"

    def test_unbind_wrong_key(self, vec_a, vec_b, vec_c):
        """Unbinding with wrong key should not recover."""
        bound = bind(vec_a, vec_b)
        wrong_result = unbind(bound, vec_c)

        sim_a = abs(similarity(wrong_result, vec_a))
        sim_b = abs(similarity(wrong_result, vec_b))

        # Should be low similarity (near random)
        assert sim_a < 0.15, f"Wrong key should not recover: sim_a={sim_a}"
        assert sim_b < 0.15, f"Wrong key should not recover: sim_b={sim_b}"


# =============================================================================
# BUNDLING TESTS
# =============================================================================


class TestBundling:
    """Tests for bundling operation."""

    def test_bundle_commutativity(self, vec_a, vec_b):
        """bundle(a, b) = bundle(b, a)"""
        ab = bundle(vec_a, vec_b)
        ba = bundle(vec_b, vec_a)
        sim = similarity(ab, ba)
        assert sim > 0.999, f"Bundling should be commutative, got sim={sim}"

    def test_bundle_preserves_components(self, vec_a, vec_b):
        """Bundle should be similar to both components."""
        bundled = bundle(vec_a, vec_b)
        sim_a = similarity(bundled, vec_a)
        sim_b = similarity(bundled, vec_b)

        # Both should be positive and reasonably high
        assert sim_a > 0.4, f"Bundle should be similar to first component: {sim_a}"
        assert sim_b > 0.4, f"Bundle should be similar to second component: {sim_b}"

    def test_bundle_many_preserves_all(self, vec_a, vec_b, vec_c):
        """bundle_many should preserve similarity to all components."""
        bundled = bundle_many(vec_a, vec_b, vec_c)

        for i, v in enumerate([vec_a, vec_b, vec_c]):
            sim = similarity(bundled, v)
            assert sim > 0.3, f"Bundle should preserve component {i}, got sim={sim}"

    def test_weighted_bundle(self, vec_a, vec_b):
        """Weighted bundle should bias toward higher weight."""
        # Heavy weight on vec_a
        bundled = weighted_bundle([vec_a, vec_b], [0.9, 0.1])
        sim_a = similarity(bundled, vec_a)
        sim_b = similarity(bundled, vec_b)

        assert (
            sim_a > sim_b
        ), f"Higher weight should have higher similarity: {sim_a} vs {sim_b}"

    def test_bundle_capacity(self):
        """Test bundle capacity estimate."""
        cap = bundle_capacity(16384)
        assert 50 < cap < 200, f"Capacity estimate seems off: {cap}"


# =============================================================================
# PERMUTATION TESTS
# =============================================================================


class TestPermutation:
    """Tests for permutation operation."""

    def test_permute_invertible(self, vec_a):
        """permute and inverse_permute should cancel."""
        shift = 42
        permuted = permute(vec_a, shift)
        recovered = inverse_permute(permuted, shift)
        sim = similarity(recovered, vec_a)
        assert sim > 0.999, f"Permutation should be invertible, got sim={sim}"

    def test_permute_changes_vector(self, vec_a):
        """Permutation should produce dissimilar vector."""
        shift = 100
        permuted = permute(vec_a, shift)
        sim = abs(similarity(permuted, vec_a))
        assert sim < 0.2, f"Permutation should change vector significantly: {sim}"

    def test_different_shifts_different_results(self, vec_a):
        """Different shifts should produce dissimilar results."""
        p1 = permute(vec_a, 10)
        p2 = permute(vec_a, 100)
        sim = abs(similarity(p1, p2))
        assert sim < 0.2, f"Different shifts should be dissimilar: {sim}"


# =============================================================================
# SEQUENCE ENCODING TESTS
# =============================================================================


class TestSequenceEncoding:
    """Tests for sequence encoding."""

    def test_sequence_preserves_components(self, vec_a, vec_b, vec_c):
        """Sequence encoding should preserve component similarities."""
        seq = sequence_encode([vec_a, vec_b, vec_c])

        # Each component should be recoverable via inverse permute + similarity
        # First position (shift=0)
        sim_a = similarity(seq, vec_a)
        assert sim_a > 0.2, f"Sequence should contain first element: {sim_a}"

    def test_sequence_order_matters(self, vec_a, vec_b, vec_c):
        """Different orderings should produce different sequences."""
        seq1 = sequence_encode([vec_a, vec_b, vec_c])
        seq2 = sequence_encode([vec_c, vec_b, vec_a])
        sim = abs(similarity(seq1, seq2))
        assert sim < 0.5, f"Different orderings should be somewhat dissimilar: {sim}"


# =============================================================================
# ANALOGY TESTS
# =============================================================================


class TestAnalogy:
    """Tests for analogy solving."""

    def test_analogy_self_consistent(self, vec_a, vec_b):
        """a:b :: a:? should give b"""
        result = solve_analogy(vec_a, vec_b, vec_a)
        sim = similarity(result, vec_b)
        assert sim > 0.8, f"Self-consistent analogy should work: {sim}"


# =============================================================================
# RECORD OPERATIONS TESTS
# =============================================================================


class TestRecordOperations:
    """Tests for structured record operations."""

    def test_create_and_query_record(self):
        """Record creation and querying should work."""
        role_sku = seed_hash("ROLE:SKU")
        role_anomaly = seed_hash("ROLE:ANOMALY")
        filler_sku = seed_hash("SKU:12345")
        filler_anomaly = seed_hash("ANOMALY:low_stock")

        record = create_record([(role_sku, filler_sku), (role_anomaly, filler_anomaly)])

        # Create small codebook with fillers
        codebook = torch.stack([filler_sku, filler_anomaly])

        # Query for SKU filler
        idx, sim = query_record(record, role_sku, codebook)
        assert idx == 0, f"Should find SKU filler at index 0, got {idx}"
        assert sim > 0.3, f"Query similarity too low: {sim}"

        # Query for anomaly filler
        idx, sim = query_record(record, role_anomaly, codebook)
        assert idx == 1, f"Should find anomaly filler at index 1, got {idx}"
        assert sim > 0.3, f"Query similarity too low: {sim}"


# =============================================================================
# ORTHOGONALITY TESTS
# =============================================================================


class TestOrthogonality:
    """Tests for vector orthogonality."""

    def test_random_vectors_orthogonal(self):
        """Random seed_hash vectors should be nearly orthogonal."""
        vectors = [seed_hash(f"random_seed_{i}") for i in range(10)]
        assert orthogonality_check(
            vectors, threshold=0.15
        ), "Random vectors should be approximately orthogonal"

    def test_similar_seeds_not_orthogonal(self):
        """Same seed should produce identical (not orthogonal) vectors."""
        v1 = seed_hash("identical_seed")
        v2 = seed_hash("identical_seed")
        sim = similarity(v1, v2)
        assert sim > 0.999, "Same seed should produce identical vectors"


# =============================================================================
# DETERMINISM TESTS
# =============================================================================


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_seed_hash_deterministic(self):
        """seed_hash should be deterministic across calls."""
        v1 = seed_hash("determinism_test_seed")
        v2 = seed_hash("determinism_test_seed")
        sim = similarity(v1, v2)
        assert sim > 0.9999, f"seed_hash should be deterministic, got sim={sim}"

    def test_operations_deterministic(self, vec_a, vec_b):
        """All operations should be deterministic."""
        bound1 = bind(vec_a, vec_b)
        bound2 = bind(vec_a, vec_b)
        assert similarity(bound1, bound2) > 0.9999, "bind should be deterministic"

        bundled1 = bundle(vec_a, vec_b)
        bundled2 = bundle(vec_a, vec_b)
        assert similarity(bundled1, bundled2) > 0.9999, "bundle should be deterministic"


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_many_bindings_stable(self):
        """Many sequential bindings should not blow up."""
        v = seed_hash("stability_test")
        for i in range(100):
            v = bind(v, seed_hash(f"binding_{i}"))

        # Check still on manifold (unit magnitude components)
        mags = torch.abs(v)
        assert torch.allclose(
            mags, torch.ones_like(mags), atol=0.01
        ), "Many bindings should stay on unit hypersphere"

    def test_many_bundles_stable(self):
        """Many sequential bundles should not blow up."""
        v = seed_hash("stability_test_bundle")
        for i in range(100):
            v = bundle(v, seed_hash(f"bundle_{i}"))

        # Check still on manifold
        mags = torch.abs(v)
        assert torch.allclose(
            mags, torch.ones_like(mags), atol=0.01
        ), "Many bundles should stay on unit hypersphere"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
