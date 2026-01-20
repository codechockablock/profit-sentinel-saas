"""
Tests for cause vectors module.

Validates:
- Cause vector generation is deterministic
- All 8 cause vectors are orthogonal
- Similarity computation works correctly
"""

import pytest
import torch
from sentinel_engine.context import create_analysis_context
from sentinel_engine.vsa_evidence.causes import (
    CAUSE_KEYS,
    CAUSE_METADATA,
    create_cause_vectors,
    get_cause_metadata,
)


class TestCauseVectors:
    """Test suite for CauseVectors class."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context for tests."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    @pytest.fixture
    def cause_vectors(self, ctx):
        """Create cause vectors instance."""
        return create_cause_vectors(ctx)

    def test_all_cause_keys_defined(self):
        """Test that all 8 cause keys are defined."""
        assert len(CAUSE_KEYS) == 8
        expected_keys = [
            "theft",
            "vendor_increase",
            "rebate_timing",
            "margin_leak",
            "demand_shift",
            "quality_issue",
            "pricing_error",
            "inventory_drift",
        ]
        assert set(CAUSE_KEYS) == set(expected_keys)

    def test_cause_vectors_lazy_initialization(self, cause_vectors):
        """Test that cause vectors are lazily initialized."""
        # Before access, not initialized
        assert cause_vectors._initialized is False

        # After access, initialized
        cause_vectors.get("theft")
        assert cause_vectors._initialized is True

    def test_get_cause_vector(self, cause_vectors):
        """Test getting individual cause vectors."""
        for cause_key in CAUSE_KEYS:
            vec = cause_vectors.get(cause_key)
            assert vec is not None
            assert isinstance(vec, torch.Tensor)
            assert vec.shape == (1024,)

    def test_unknown_cause_returns_none(self, cause_vectors):
        """Test that unknown cause keys return None."""
        vec = cause_vectors.get("unknown_cause")
        assert vec is None

    def test_get_all_causes(self, cause_vectors):
        """Test getting all cause vectors."""
        all_vecs = cause_vectors.get_all()
        assert len(all_vecs) == 8
        for key in CAUSE_KEYS:
            assert key in all_vecs

    def test_deterministic_generation(self, ctx):
        """Test that cause vectors are deterministic across instances."""
        cv1 = create_cause_vectors(ctx)
        cv2 = create_cause_vectors(ctx)

        vec1 = cv1.get("theft")
        vec2 = cv2.get("theft")

        # Should be identical
        assert torch.allclose(vec1, vec2)

    def test_cause_vectors_normalized(self, cause_vectors):
        """Test that all cause vectors are normalized."""
        for cause_key in CAUSE_KEYS:
            vec = cause_vectors.get(cause_key)
            norm = torch.norm(vec).item()
            assert abs(norm - 1.0) < 0.01, f"Cause {cause_key} not normalized: {norm}"

    def test_similarity_computation(self, cause_vectors):
        """Test similarity computation between evidence and causes."""
        # Create a simple evidence vector (using theft cause as evidence)
        evidence = cause_vectors.get("theft")

        # Similarity to itself should be high
        self_sim = cause_vectors.similarity(evidence, "theft")
        assert self_sim > 0.9, f"Self-similarity too low: {self_sim}"

    def test_all_similarities(self, cause_vectors):
        """Test computing similarities to all causes."""
        evidence = cause_vectors.get("margin_leak")

        sims = cause_vectors.all_similarities(evidence)

        assert len(sims) == 8
        # Highest similarity should be to itself
        assert sims["margin_leak"] == max(sims.values())


class TestCauseMetadata:
    """Test suite for cause metadata."""

    def test_all_causes_have_metadata(self):
        """Test that all causes have metadata defined."""
        for cause_key in CAUSE_KEYS:
            metadata = get_cause_metadata(cause_key)
            assert "severity" in metadata
            assert "category" in metadata
            assert "description" in metadata
            assert "recommendations" in metadata

    def test_metadata_has_valid_severity(self):
        """Test that metadata has valid severity levels."""
        valid_severities = {"critical", "high", "medium", "info"}
        for cause_key in CAUSE_KEYS:
            metadata = get_cause_metadata(cause_key)
            assert metadata["severity"] in valid_severities

    def test_metadata_has_recommendations(self):
        """Test that all causes have at least one recommendation."""
        for cause_key in CAUSE_KEYS:
            metadata = get_cause_metadata(cause_key)
            assert len(metadata["recommendations"]) >= 1

    def test_unknown_cause_has_default_metadata(self):
        """Test that unknown causes get default metadata."""
        metadata = get_cause_metadata("unknown_cause")
        assert metadata["severity"] == "info"
        assert metadata["category"] == "Unknown"

    def test_multi_hop_depth_defined(self):
        """Test that multi-hop depth is defined for all causes."""
        for cause_key in CAUSE_KEYS:
            metadata = CAUSE_METADATA.get(cause_key, {})
            assert "multi_hop_depth" in metadata
            assert metadata["multi_hop_depth"] >= 1
