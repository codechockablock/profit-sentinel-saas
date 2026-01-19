"""
tests/vsa_core/test_probabilistic.py - Tests for Probabilistic Superposition (P-Sup)

Tests the Bayesian-style hypothesis tracking using VSA superposition.
Verifies:
    - HypothesisBundle creation and properties
    - p_sup weighted superposition
    - p_sup_update Bayesian updates
    - p_sup_collapse threshold behavior
    - Adding/removing hypotheses
    - Merging bundles from multiple sources
"""
import pytest
import torch
import math
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from vsa_core import seed_hash, similarity, configure
from vsa_core.probabilistic import (
    HypothesisBundle,
    p_sup,
    p_sup_update,
    p_sup_collapse,
    p_sup_add_hypothesis,
    p_sup_remove_hypothesis,
    p_sup_merge,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def config():
    """Configure VSA for testing."""
    return configure(dimensions=4096, device="cpu")


@pytest.fixture
def shrinkage_vec():
    return seed_hash("hypothesis:shrinkage")


@pytest.fixture
def clerical_vec():
    return seed_hash("hypothesis:clerical_error")


@pytest.fixture
def vendor_vec():
    return seed_hash("hypothesis:vendor_issue")


@pytest.fixture
def margin_leak_vec():
    return seed_hash("hypothesis:margin_leak")


@pytest.fixture
def basic_hypotheses(shrinkage_vec, clerical_vec, vendor_vec):
    """Create a basic set of hypotheses with equal priors."""
    return [
        ("shrinkage", shrinkage_vec, 0.4),
        ("clerical_error", clerical_vec, 0.3),
        ("vendor_issue", vendor_vec, 0.3),
    ]


# =============================================================================
# HYPOTHESIS BUNDLE TESTS
# =============================================================================

class TestHypothesisBundle:
    """Tests for HypothesisBundle dataclass."""

    def test_bundle_creation(self, basic_hypotheses, config):
        """p_sup should create valid HypothesisBundle."""
        bundle = p_sup(basic_hypotheses)

        assert isinstance(bundle, HypothesisBundle)
        assert bundle.vector is not None
        assert len(bundle.hypotheses) == 3
        assert bundle.probabilities.shape == (3,)
        # Use configured dimensions (may be 4096 or higher depending on config)
        assert bundle.basis_vectors.shape[0] == 3
        assert bundle.basis_vectors.shape[1] > 0

    def test_bundle_repr(self, basic_hypotheses):
        """Bundle repr should show hypothesis probabilities."""
        bundle = p_sup(basic_hypotheses)
        repr_str = repr(bundle)

        assert "HypothesisBundle" in repr_str
        assert "shrinkage" in repr_str
        assert "clerical_error" in repr_str
        assert "%" in repr_str  # Probabilities formatted as percentages

    def test_top_hypothesis(self, basic_hypotheses):
        """top_hypothesis should return highest probability hypothesis."""
        bundle = p_sup(basic_hypotheses)
        label, prob = bundle.top_hypothesis()

        assert label == "shrinkage"  # Has highest prior (0.4)
        assert abs(prob - 0.4) < 0.01

    def test_entropy_uniform(self, shrinkage_vec, clerical_vec, vendor_vec):
        """Uniform distribution should have maximum entropy."""
        uniform = [
            ("a", shrinkage_vec, 1/3),
            ("b", clerical_vec, 1/3),
            ("c", vendor_vec, 1/3),
        ]
        bundle = p_sup(uniform)
        entropy = bundle.entropy()

        # Max entropy for 3 choices = log2(3) ≈ 1.585
        assert abs(entropy - math.log2(3)) < 0.01

    def test_entropy_concentrated(self, shrinkage_vec, clerical_vec, vendor_vec):
        """Concentrated distribution should have low entropy."""
        concentrated = [
            ("a", shrinkage_vec, 0.98),
            ("b", clerical_vec, 0.01),
            ("c", vendor_vec, 0.01),
        ]
        bundle = p_sup(concentrated)
        entropy = bundle.entropy()

        # Very concentrated = low entropy
        assert entropy < 0.3


# =============================================================================
# P_SUP CREATION TESTS
# =============================================================================

class TestPSupCreation:
    """Tests for p_sup creation function."""

    def test_p_sup_requires_hypotheses(self):
        """p_sup should raise on empty list."""
        with pytest.raises(ValueError, match="At least one hypothesis"):
            p_sup([])

    def test_p_sup_normalizes_probabilities(self, shrinkage_vec, clerical_vec):
        """p_sup should normalize probabilities to sum to 1."""
        unnormalized = [
            ("a", shrinkage_vec, 2.0),
            ("b", clerical_vec, 8.0),
        ]
        bundle = p_sup(unnormalized, normalize_probs=True)

        assert abs(bundle.probabilities.sum() - 1.0) < 1e-6
        assert abs(bundle.probabilities[0] - 0.2) < 1e-6
        assert abs(bundle.probabilities[1] - 0.8) < 1e-6

    def test_p_sup_skip_normalization(self, shrinkage_vec, clerical_vec):
        """p_sup can skip normalization if requested."""
        pre_normalized = [
            ("a", shrinkage_vec, 0.3),
            ("b", clerical_vec, 0.7),
        ]
        bundle = p_sup(pre_normalized, normalize_probs=False)

        assert abs(bundle.probabilities[0] - 0.3) < 1e-6
        assert abs(bundle.probabilities[1] - 0.7) < 1e-6

    def test_p_sup_preserves_labels(self, basic_hypotheses):
        """p_sup should preserve hypothesis labels in order."""
        bundle = p_sup(basic_hypotheses)

        assert bundle.hypotheses == ["shrinkage", "clerical_error", "vendor_issue"]

    def test_p_sup_vector_is_normalized(self, basic_hypotheses):
        """Superposition vector should be normalized."""
        bundle = p_sup(basic_hypotheses)

        # For FHRR, check each component has unit magnitude
        mags = torch.abs(bundle.vector)
        assert torch.allclose(mags, torch.ones_like(mags), atol=0.01)

    def test_p_sup_superposition_similar_to_components(self, basic_hypotheses):
        """Superposition should be somewhat similar to high-probability hypotheses."""
        bundle = p_sup(basic_hypotheses)

        # Similarity to highest-prior hypothesis
        sim_shrinkage = abs(float(similarity(bundle.vector, bundle.basis_vectors[0])))

        # Should have some similarity (due to weighted bundling)
        assert sim_shrinkage > 0.2, f"Expected some similarity to top hypothesis: {sim_shrinkage}"


# =============================================================================
# P_SUP_UPDATE TESTS (BAYESIAN UPDATES)
# =============================================================================

class TestPSupUpdate:
    """Tests for Bayesian update of hypothesis probabilities."""

    def test_update_increases_matching_probability(self, basic_hypotheses, shrinkage_vec):
        """Evidence similar to a hypothesis should increase its probability."""
        bundle = p_sup(basic_hypotheses)
        initial_prob = float(bundle.probabilities[0])  # shrinkage

        # Create evidence that's similar to shrinkage
        evidence = shrinkage_vec.clone()
        updated = p_sup_update(bundle, evidence)

        final_prob = float(updated.probabilities[0])  # shrinkage

        assert final_prob > initial_prob, \
            f"Matching evidence should increase probability: {initial_prob} -> {final_prob}"

    def test_update_decreases_non_matching_probability(self, basic_hypotheses, shrinkage_vec):
        """Evidence should decrease probability of non-matching hypotheses."""
        bundle = p_sup(basic_hypotheses)
        initial_prob = float(bundle.probabilities[2])  # vendor_issue

        # Evidence similar to shrinkage
        evidence = shrinkage_vec.clone()
        updated = p_sup_update(bundle, evidence)

        final_prob = float(updated.probabilities[2])  # vendor_issue

        assert final_prob < initial_prob, \
            f"Non-matching evidence should decrease probability: {initial_prob} -> {final_prob}"

    def test_update_preserves_normalization(self, basic_hypotheses, shrinkage_vec):
        """Updated probabilities should still sum to 1."""
        bundle = p_sup(basic_hypotheses)
        evidence = shrinkage_vec.clone()
        updated = p_sup_update(bundle, evidence)

        assert abs(updated.probabilities.sum() - 1.0) < 1e-5

    def test_update_preserves_labels(self, basic_hypotheses, shrinkage_vec):
        """Update should preserve hypothesis labels."""
        bundle = p_sup(basic_hypotheses)
        updated = p_sup_update(bundle, shrinkage_vec)

        assert updated.hypotheses == bundle.hypotheses

    def test_update_temperature_effect(self, basic_hypotheses, shrinkage_vec):
        """Lower temperature should make sharper updates."""
        bundle = p_sup(basic_hypotheses)
        evidence = shrinkage_vec.clone()

        # Low temperature = sharp update
        sharp = p_sup_update(bundle, evidence, temperature=0.1)

        # High temperature = soft update
        soft = p_sup_update(bundle, evidence, temperature=2.0)

        # Sharp update should concentrate probability more
        sharp_max = float(sharp.probabilities.max())
        soft_max = float(soft.probabilities.max())

        assert sharp_max > soft_max, \
            f"Lower temperature should concentrate probability: {sharp_max} vs {soft_max}"

    def test_multiple_updates_converge(self, basic_hypotheses, shrinkage_vec):
        """Multiple consistent updates should converge toward hypothesis."""
        bundle = p_sup(basic_hypotheses)
        evidence = shrinkage_vec.clone()

        # Apply same evidence multiple times
        for _ in range(5):
            bundle = p_sup_update(bundle, evidence)

        # Should strongly favor shrinkage
        label, prob = bundle.top_hypothesis()
        assert label == "shrinkage"
        assert prob > 0.8, f"Multiple updates should converge: {prob}"


# =============================================================================
# P_SUP_COLLAPSE TESTS
# =============================================================================

class TestPSupCollapse:
    """Tests for hypothesis collapse."""

    def test_collapse_returns_none_below_threshold(self, basic_hypotheses):
        """No collapse when no hypothesis exceeds threshold."""
        bundle = p_sup(basic_hypotheses)  # Max prob is 0.4
        result = p_sup_collapse(bundle, threshold=0.9)

        assert result is None

    def test_collapse_returns_winner_above_threshold(self, shrinkage_vec, clerical_vec, vendor_vec):
        """Collapse should return winner when threshold exceeded."""
        concentrated = [
            ("shrinkage", shrinkage_vec, 0.95),
            ("clerical_error", clerical_vec, 0.03),
            ("vendor_issue", vendor_vec, 0.02),
        ]
        bundle = p_sup(concentrated)
        result = p_sup_collapse(bundle, threshold=0.9)

        assert result == "shrinkage"

    def test_collapse_at_exact_threshold(self, shrinkage_vec, clerical_vec):
        """Collapse should occur at exactly the threshold."""
        hypotheses = [
            ("a", shrinkage_vec, 0.85),
            ("b", clerical_vec, 0.15),
        ]
        bundle = p_sup(hypotheses)

        # Should not collapse at 0.86
        assert p_sup_collapse(bundle, threshold=0.86) is None

        # Should collapse at 0.85
        assert p_sup_collapse(bundle, threshold=0.85) == "a"

    def test_collapse_default_threshold(self, shrinkage_vec, clerical_vec):
        """Default threshold should be 0.9."""
        high_conf = [
            ("winner", shrinkage_vec, 0.91),
            ("loser", clerical_vec, 0.09),
        ]
        bundle = p_sup(high_conf)

        # Should collapse with default threshold
        assert p_sup_collapse(bundle) == "winner"


# =============================================================================
# P_SUP_ADD_HYPOTHESIS TESTS
# =============================================================================

class TestPSupAddHypothesis:
    """Tests for adding hypotheses to existing bundle."""

    def test_add_hypothesis_increases_count(self, basic_hypotheses, margin_leak_vec):
        """Adding hypothesis should increase count."""
        bundle = p_sup(basic_hypotheses)
        updated = p_sup_add_hypothesis(bundle, "margin_leak", margin_leak_vec, prior=0.1)

        assert len(updated.hypotheses) == 4
        assert "margin_leak" in updated.hypotheses

    def test_add_hypothesis_rescales_probabilities(self, basic_hypotheses, margin_leak_vec):
        """Existing probabilities should be scaled down to make room."""
        bundle = p_sup(basic_hypotheses)
        new_prior = 0.1
        updated = p_sup_add_hypothesis(bundle, "margin_leak", margin_leak_vec, prior=new_prior)

        # New hypothesis should have its prior
        margin_idx = updated.hypotheses.index("margin_leak")
        assert abs(updated.probabilities[margin_idx] - new_prior) < 1e-5

        # Other probabilities scaled by (1 - new_prior)
        scale = 1.0 - new_prior
        for i, old_prob in enumerate(bundle.probabilities):
            new_prob = updated.probabilities[i]
            expected = old_prob * scale
            assert abs(new_prob - expected) < 1e-5

    def test_add_hypothesis_vector_included(self, basic_hypotheses, margin_leak_vec):
        """New hypothesis vector should be in basis_vectors."""
        bundle = p_sup(basic_hypotheses)
        updated = p_sup_add_hypothesis(bundle, "margin_leak", margin_leak_vec)

        margin_idx = updated.hypotheses.index("margin_leak")
        added_vec = updated.basis_vectors[margin_idx]

        # Should be the same vector
        sim = float(similarity(added_vec, margin_leak_vec))
        assert sim > 0.999


# =============================================================================
# P_SUP_REMOVE_HYPOTHESIS TESTS
# =============================================================================

class TestPSupRemoveHypothesis:
    """Tests for removing hypotheses from bundle."""

    def test_remove_hypothesis_decreases_count(self, basic_hypotheses):
        """Removing hypothesis should decrease count."""
        bundle = p_sup(basic_hypotheses)
        updated = p_sup_remove_hypothesis(bundle, "vendor_issue")

        assert len(updated.hypotheses) == 2
        assert "vendor_issue" not in updated.hypotheses

    def test_remove_hypothesis_renormalizes(self, basic_hypotheses):
        """Remaining probabilities should be renormalized to sum to 1."""
        bundle = p_sup(basic_hypotheses)
        updated = p_sup_remove_hypothesis(bundle, "vendor_issue")

        assert abs(updated.probabilities.sum() - 1.0) < 1e-5

    def test_remove_hypothesis_preserves_ratio(self, basic_hypotheses):
        """Remaining probabilities should preserve their ratio."""
        bundle = p_sup(basic_hypotheses)
        # Original: shrinkage=0.4, clerical=0.3, vendor=0.3
        # After removing vendor: shrinkage should be 0.4/(0.4+0.3) = 0.571...

        updated = p_sup_remove_hypothesis(bundle, "vendor_issue")

        expected_shrinkage = 0.4 / 0.7
        expected_clerical = 0.3 / 0.7

        shrinkage_idx = updated.hypotheses.index("shrinkage")
        clerical_idx = updated.hypotheses.index("clerical_error")

        assert abs(updated.probabilities[shrinkage_idx] - expected_shrinkage) < 1e-5
        assert abs(updated.probabilities[clerical_idx] - expected_clerical) < 1e-5

    def test_remove_unknown_raises(self, basic_hypotheses):
        """Removing unknown hypothesis should raise."""
        bundle = p_sup(basic_hypotheses)

        with pytest.raises(ValueError, match="not in bundle"):
            p_sup_remove_hypothesis(bundle, "nonexistent")


# =============================================================================
# P_SUP_MERGE TESTS
# =============================================================================

class TestPSupMerge:
    """Tests for merging multiple hypothesis bundles."""

    def test_merge_requires_bundles(self):
        """Merge should require at least one bundle."""
        with pytest.raises(ValueError, match="At least one bundle"):
            p_sup_merge([])

    def test_merge_single_bundle(self, basic_hypotheses):
        """Merging single bundle should return equivalent bundle."""
        bundle = p_sup(basic_hypotheses)
        merged = p_sup_merge([bundle])

        # Should have same hypotheses
        assert set(merged.hypotheses) == set(bundle.hypotheses)

        # Probabilities should be close
        for label in bundle.hypotheses:
            idx_orig = bundle.hypotheses.index(label)
            idx_merged = merged.hypotheses.index(label)
            assert abs(bundle.probabilities[idx_orig] - merged.probabilities[idx_merged]) < 0.01

    def test_merge_combines_hypotheses(self, shrinkage_vec, clerical_vec, vendor_vec, margin_leak_vec):
        """Merging bundles with different hypotheses should combine them."""
        bundle1 = p_sup([
            ("shrinkage", shrinkage_vec, 0.5),
            ("clerical", clerical_vec, 0.5),
        ])
        bundle2 = p_sup([
            ("vendor", vendor_vec, 0.5),
            ("margin", margin_leak_vec, 0.5),
        ])

        merged = p_sup_merge([bundle1, bundle2])

        # Should have all 4 hypotheses
        assert len(merged.hypotheses) == 4
        assert set(merged.hypotheses) == {"shrinkage", "clerical", "vendor", "margin"}

    def test_merge_overlapping_hypotheses(self, shrinkage_vec, clerical_vec, vendor_vec):
        """Overlapping hypotheses should have combined probabilities."""
        bundle1 = p_sup([
            ("shrinkage", shrinkage_vec, 0.6),
            ("clerical", clerical_vec, 0.4),
        ])
        bundle2 = p_sup([
            ("shrinkage", shrinkage_vec, 0.8),
            ("vendor", vendor_vec, 0.2),
        ])

        # Equal weights
        merged = p_sup_merge([bundle1, bundle2])

        # Shrinkage appears in both with high probability
        shrinkage_idx = merged.hypotheses.index("shrinkage")
        shrinkage_prob = float(merged.probabilities[shrinkage_idx])

        # Should be weighted combination
        assert shrinkage_prob > 0.3, f"Shrinkage should have high merged probability: {shrinkage_prob}"

    def test_merge_with_weights(self, shrinkage_vec, clerical_vec, vendor_vec):
        """Bundle weights should affect merged probabilities."""
        bundle1 = p_sup([
            ("shrinkage", shrinkage_vec, 0.9),
            ("clerical", clerical_vec, 0.1),
        ])
        bundle2 = p_sup([
            ("shrinkage", shrinkage_vec, 0.1),
            ("clerical", clerical_vec, 0.9),
        ])

        # Heavy weight on bundle1
        merged = p_sup_merge([bundle1, bundle2], weights=[0.9, 0.1])

        shrinkage_idx = merged.hypotheses.index("shrinkage")
        clerical_idx = merged.hypotheses.index("clerical")

        # Shrinkage should dominate due to bundle1's weight
        assert merged.probabilities[shrinkage_idx] > merged.probabilities[clerical_idx]

    def test_merge_normalizes(self, basic_hypotheses):
        """Merged bundle should have normalized probabilities."""
        bundle1 = p_sup(basic_hypotheses)
        bundle2 = p_sup(basic_hypotheses)

        merged = p_sup_merge([bundle1, bundle2])

        assert abs(merged.probabilities.sum() - 1.0) < 1e-5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPSupIntegration:
    """End-to-end integration tests."""

    def test_full_hypothesis_lifecycle(self, shrinkage_vec, clerical_vec, vendor_vec, margin_leak_vec):
        """Test complete hypothesis tracking workflow."""
        # Start with initial hypotheses
        hypotheses = [
            ("shrinkage", shrinkage_vec, 0.3),
            ("clerical_error", clerical_vec, 0.4),
            ("vendor_issue", vendor_vec, 0.3),
        ]
        bundle = p_sup(hypotheses)

        # Check initial state
        label, prob = bundle.top_hypothesis()
        assert label == "clerical_error"
        assert p_sup_collapse(bundle, threshold=0.9) is None

        # Add evidence pointing toward shrinkage
        for _ in range(3):
            bundle = p_sup_update(bundle, shrinkage_vec)

        # Now shrinkage should be top
        label, prob = bundle.top_hypothesis()
        assert label == "shrinkage"

        # Add new hypothesis
        bundle = p_sup_add_hypothesis(bundle, "margin_leak", margin_leak_vec, prior=0.1)
        assert len(bundle.hypotheses) == 4

        # Rule out vendor_issue
        bundle = p_sup_remove_hypothesis(bundle, "vendor_issue")
        assert len(bundle.hypotheses) == 3

        # More evidence for shrinkage to collapse
        for _ in range(10):
            bundle = p_sup_update(bundle, shrinkage_vec, temperature=0.5)

        # Should collapse
        winner = p_sup_collapse(bundle, threshold=0.9)
        assert winner == "shrinkage"

    def test_multi_source_fusion(self, shrinkage_vec, clerical_vec, vendor_vec):
        """Test fusing hypotheses from multiple detection sources."""
        # Source 1: Visual analysis (sees physical evidence)
        visual_bundle = p_sup([
            ("shrinkage", shrinkage_vec, 0.7),
            ("vendor_issue", vendor_vec, 0.3),
        ])

        # Source 2: Transaction analysis (sees data patterns)
        transaction_bundle = p_sup([
            ("shrinkage", shrinkage_vec, 0.5),
            ("clerical_error", clerical_vec, 0.5),
        ])

        # Merge with equal confidence in sources
        fused = p_sup_merge([visual_bundle, transaction_bundle])

        # Shrinkage should dominate (high in both sources)
        label, prob = fused.top_hypothesis()
        assert label == "shrinkage"

        # All three hypotheses should be present
        assert len(fused.hypotheses) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
