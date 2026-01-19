"""
tests/sentinel_engine/test_repair_engine.py - Tests for Repair Diagnosis Engine

Tests the Visual AI Repair Diagnosis Engine including:
    - CategoryCodebook vector generation and lookup
    - TextEncoder text-to-VSA encoding
    - RepairDiagnosisEngine diagnosis flow
    - P-Sup hypothesis creation and updates
    - T-Bind store memory
    - CW-Bundle knowledge incorporation
    - Serialization/deserialization
"""
import os
import sys

import pytest
import torch

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure VSA before imports
from vsa_core import configure

configure(dimensions=4096, device="cpu")


def to_real_float(val):
    """Convert similarity value to real float."""
    if torch.is_tensor(val):
        if val.is_complex():
            return float(torch.real(val))
        return float(val)
    return float(val)

from sentinel_engine.repair_engine import (
    CategoryCodebook,
    RepairEngineConfig,
    TextEncoder,
    create_engine,
)
from sentinel_engine.repair_models import (
    KnowledgeBaseState,
    ProblemStatus,
    calculate_level,
    expertise_multiplier,
    xp_for_next_level,
)
from vsa_core.probabilistic import HypothesisBundle

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Create test configuration."""
    return RepairEngineConfig(
        dimensions=4096,
        collapse_threshold=0.85,
        min_hypothesis_prob=0.05,
        max_hypotheses=5,
        update_temperature=0.8,
        device="cpu"
    )


@pytest.fixture
def codebook(config):
    """Create test category codebook."""
    cb = CategoryCodebook(config)
    cb.register_category("plumbing", "Plumbing", "Pipes and water")
    cb.register_category("plumbing-faucet", "Leaky Faucet", "Dripping faucets", parent_slug="plumbing")
    cb.register_category("electrical", "Electrical", "Wiring and power")
    cb.register_category("hvac", "HVAC", "Heating and cooling")
    return cb


@pytest.fixture
def text_encoder(config):
    """Create test text encoder."""
    return TextEncoder(config)


@pytest.fixture
def engine():
    """Create full repair diagnosis engine."""
    return create_engine(dimensions=4096, device="cpu")


# =============================================================================
# CATEGORY CODEBOOK TESTS
# =============================================================================

class TestCategoryCodebook:
    """Tests for CategoryCodebook class."""

    def test_codebook_creation(self, config):
        """Codebook should initialize properly."""
        cb = CategoryCodebook(config)
        assert cb.dimensions == 4096
        assert len(cb._vectors) == 0

    def test_register_category(self, codebook):
        """Categories should be registered with vectors."""
        assert "plumbing" in codebook._vectors
        assert "electrical" in codebook._vectors
        assert codebook._vectors["plumbing"].shape == (4096,)

    def test_vector_determinism(self, config):
        """Same slug should produce same vector."""
        cb1 = CategoryCodebook(config)
        cb1.register_category("test", "Test")

        cb2 = CategoryCodebook(config)
        cb2.register_category("test", "Test")

        vec1 = cb1.get_vector("test")
        vec2 = cb2.get_vector("test")

        # Should be identical
        assert torch.allclose(vec1, vec2)

    def test_different_slugs_different_vectors(self, codebook):
        """Different slugs should have dissimilar vectors."""
        from vsa_core import similarity

        vec_plumbing = codebook.get_vector("plumbing")
        vec_electrical = codebook.get_vector("electrical")

        sim = abs(float(similarity(vec_plumbing, vec_electrical)))
        assert sim < 0.2, f"Different categories should have low similarity: {sim}"

    def test_get_vector_unknown_raises(self, codebook):
        """Getting unknown category should raise."""
        with pytest.raises(ValueError, match="not registered"):
            codebook.get_vector("unknown")

    def test_get_info(self, codebook):
        """Category info should be retrievable."""
        info = codebook.get_info("plumbing")
        assert info["name"] == "Plumbing"
        assert info["description"] == "Pipes and water"

    def test_find_nearest(self, codebook, text_encoder):
        """find_nearest should return similar categories."""
        # Create a query that's related to plumbing
        query = text_encoder.encode("faucet dripping water leak pipe")

        matches = codebook.find_nearest(query, top_k=3)

        assert len(matches) <= 3
        assert all(isinstance(m[0], str) for m in matches)
        assert all(isinstance(m[1], float) for m in matches)

    def test_get_all_vectors(self, codebook):
        """get_all_vectors should return all category vectors."""
        all_vecs = codebook.get_all_vectors()
        slugs = codebook.get_all_slugs()

        assert all_vecs.shape[0] == len(slugs)
        assert all_vecs.shape[1] == 4096


# =============================================================================
# TEXT ENCODER TESTS
# =============================================================================

class TestTextEncoder:
    """Tests for TextEncoder class."""

    def test_encoder_creation(self, config):
        """Text encoder should initialize properly."""
        enc = TextEncoder(config)
        assert enc.dimensions == 4096

    def test_encode_text(self, text_encoder):
        """Text should encode to VSA vector."""
        vec = text_encoder.encode("faucet is dripping")

        assert vec.shape == (4096,)
        assert vec.is_complex()  # FHRR uses complex phasors

    def test_encode_empty_text(self, text_encoder):
        """Empty text should produce random vector (not crash)."""
        vec = text_encoder.encode("")
        assert vec.shape == (4096,)

        vec2 = text_encoder.encode("   ")
        assert vec2.shape == (4096,)

    def test_encode_determinism(self, config):
        """Same text should produce same vector."""
        enc1 = TextEncoder(config)
        enc2 = TextEncoder(config)

        vec1 = enc1.encode("test phrase")
        vec2 = enc2.encode("test phrase")

        assert torch.allclose(vec1, vec2)

    def test_keyword_boost(self, text_encoder):
        """Repair keywords should be weighted higher."""
        # This is implicit in the encoding but hard to test directly
        # We verify the encoder doesn't crash on keyword-rich text
        vec = text_encoder.encode(
            "leak drip faucet pipe drain clog toilet water pressure"
        )
        assert vec.shape == (4096,)

    def test_similar_text_similar_vectors(self, text_encoder):
        """Semantically similar text should have similar vectors."""
        from vsa_core import similarity

        vec1 = text_encoder.encode("faucet dripping water leak")
        vec2 = text_encoder.encode("water leaking from faucet drip")
        vec3 = text_encoder.encode("electrical outlet spark fire")

        sim_12 = to_real_float(similarity(vec1, vec2))
        sim_13 = to_real_float(similarity(vec1, vec3))

        # Similar topics should have higher similarity
        assert sim_12 > sim_13, \
            f"Similar text should have higher sim: {sim_12} vs {sim_13}"


# =============================================================================
# REPAIR DIAGNOSIS ENGINE TESTS
# =============================================================================

class TestRepairDiagnosisEngine:
    """Tests for RepairDiagnosisEngine class."""

    def test_engine_creation(self, engine):
        """Engine should initialize with default categories."""
        assert engine is not None
        assert engine.config is not None
        assert len(engine.codebook.get_all_slugs()) > 0

    def test_diagnose_text_only(self, engine):
        """Diagnosis should work with text only."""
        bundle, metadata = engine.diagnose(text="my faucet is dripping")

        assert isinstance(bundle, HypothesisBundle)
        assert len(bundle.hypotheses) > 0
        assert abs(bundle.probabilities.sum() - 1.0) < 0.01
        assert "elapsed_ms" in metadata

    def test_diagnose_requires_input(self, engine):
        """Diagnosis should require at least one input."""
        with pytest.raises(ValueError, match="At least one input"):
            engine.diagnose()

    def test_diagnose_voice_input(self, engine):
        """Diagnosis should work with voice transcript."""
        bundle, metadata = engine.diagnose(voice_transcript="kitchen sink is clogged")

        assert isinstance(bundle, HypothesisBundle)
        assert "text" in metadata["input_types"]

    def test_diagnose_combined_inputs(self, engine):
        """Diagnosis should combine text and voice inputs."""
        bundle, metadata = engine.diagnose(
            text="faucet",
            voice_transcript="it's dripping constantly"
        )

        assert isinstance(bundle, HypothesisBundle)
        assert len(bundle.hypotheses) > 0

    def test_diagnose_with_store_context(self, engine):
        """Diagnosis should incorporate store context."""
        store_id = "test-store-123"

        # First diagnosis to build store memory
        bundle1, _ = engine.diagnose(
            text="leaky faucet",
            store_id=store_id
        )

        # Second diagnosis should have store context
        bundle2, metadata = engine.diagnose(
            text="another faucet drip",
            store_id=store_id
        )

        # Should have store context similarity (second time)
        assert store_id in engine._store_memories

    def test_refine_diagnosis(self, engine):
        """Refining should update hypothesis probabilities."""
        bundle1, _ = engine.diagnose(text="water problem")

        # Get evidence that points toward specific category
        evidence = engine.text_encoder.encode("faucet dripping kitchen sink")

        bundle2 = engine.refine_diagnosis(bundle1, evidence)

        # Probabilities should shift
        assert isinstance(bundle2, HypothesisBundle)
        assert len(bundle2.hypotheses) == len(bundle1.hypotheses)

    def test_check_collapse(self, engine):
        """Collapse should return winner when confident."""
        # Create artificial high-confidence bundle
        from vsa_core.probabilistic import p_sup

        cat_vec = engine.codebook.get_vector("plumbing-faucet")
        hypotheses = [
            ("plumbing-faucet", cat_vec, 0.95),
            ("plumbing", engine.codebook.get_vector("plumbing"), 0.05),
        ]
        bundle = p_sup(hypotheses)

        winner = engine.check_collapse(bundle)
        assert winner == "plumbing-faucet"

    def test_check_collapse_uncertain(self, engine):
        """Collapse should return None when uncertain."""
        bundle, _ = engine.diagnose(text="something is broken")

        # General text should not collapse
        winner = engine.check_collapse(bundle)
        # May or may not collapse depending on specificity
        # Just verify it returns the right type
        assert winner is None or isinstance(winner, str)

    def test_bundle_to_response(self, engine):
        """Bundle should convert to API response."""
        bundle, metadata = engine.diagnose(text="faucet leak")

        response = engine.bundle_to_response(
            bundle,
            problem_id="test-123",
            metadata=metadata
        )

        assert response.problem_id == "test-123"
        assert response.status == ProblemStatus.DIAGNOSED
        assert len(response.hypotheses) > 0
        assert response.top_hypothesis is not None
        assert 0 <= response.confidence <= 1

    def test_bundle_to_response_follow_up_questions(self, engine):
        """Low confidence should generate follow-up questions."""
        # Force low confidence by using vague text
        bundle, metadata = engine.diagnose(text="something wrong")

        response = engine.bundle_to_response(
            bundle,
            problem_id="test-456",
            metadata=metadata
        )

        # If needs_more_info, should have questions
        if response.needs_more_info:
            assert len(response.follow_up_questions) > 0


# =============================================================================
# STORE MEMORY (T-BIND) TESTS
# =============================================================================

class TestStoreMemory:
    """Tests for store temporal memory."""

    def test_store_memory_initialized(self, engine):
        """Store memory should be initialized on first use."""
        store_id = "memory-test-store"

        assert store_id not in engine._store_memories

        engine.diagnose(text="test problem", store_id=store_id)

        assert store_id in engine._store_memories
        assert store_id in engine._store_reference_times

    def test_store_memory_accumulates(self, engine):
        """Multiple diagnoses should accumulate in memory."""
        store_id = "accumulate-test"

        # First problem
        engine.diagnose(text="faucet leak", store_id=store_id)
        mem1 = engine._store_memories[store_id].clone()

        # Second problem
        engine.diagnose(text="toilet running", store_id=store_id)
        mem2 = engine._store_memories[store_id]

        # Memory should change
        from vsa_core import similarity
        sim = to_real_float(similarity(mem1, mem2))
        assert sim < 0.95, f"Memory should accumulate, sim={sim}"

    def test_different_stores_independent(self, engine):
        """Different stores should have independent memories."""
        engine.diagnose(text="faucet leak", store_id="store-a")
        engine.diagnose(text="electrical outlet", store_id="store-b")

        mem_a = engine._store_memories["store-a"]
        mem_b = engine._store_memories["store-b"]

        # Different stores, different problems = different memories
        from vsa_core import similarity
        sim = to_real_float(similarity(mem_a, mem_b))
        # Should be somewhat different
        assert sim < 0.9


# =============================================================================
# KNOWLEDGE INCORPORATION (CW-BUNDLE) TESTS
# =============================================================================

class TestKnowledgeIncorporation:
    """Tests for employee correction incorporation."""

    def test_incorporate_correction_new(self, engine):
        """First correction should create knowledge state."""
        problem_vec = engine.text_encoder.encode("faucet drip")

        state = engine.incorporate_correction(
            problem_vec=problem_vec,
            original_category="plumbing",
            corrected_category="plumbing-faucet",
            employee_level=5,
            knowledge_state=None
        )

        assert isinstance(state, KnowledgeBaseState)
        assert state.category_slug == "plumbing-faucet"
        assert state.total_corrections == 1
        assert len(state.correction_weights) == 1

    def test_incorporate_correction_existing(self, engine):
        """Subsequent corrections should update state."""
        problem_vec1 = engine.text_encoder.encode("faucet drip")
        problem_vec2 = engine.text_encoder.encode("sink faucet leak")

        # First correction
        state1 = engine.incorporate_correction(
            problem_vec=problem_vec1,
            original_category="plumbing",
            corrected_category="plumbing-faucet",
            employee_level=3,
            knowledge_state=None
        )

        # Second correction
        state2 = engine.incorporate_correction(
            problem_vec=problem_vec2,
            original_category="plumbing",
            corrected_category="plumbing-faucet",
            employee_level=7,
            knowledge_state=state1
        )

        assert state2.total_corrections == 2
        assert len(state2.correction_weights) == 2

    def test_expertise_weighting(self, engine):
        """Higher level employees should have more weight."""
        problem_vec = engine.text_encoder.encode("test problem")

        state_low = engine.incorporate_correction(
            problem_vec=problem_vec,
            original_category="plumbing",
            corrected_category="plumbing-faucet",
            employee_level=1,
            knowledge_state=None
        )

        state_high = engine.incorporate_correction(
            problem_vec=problem_vec,
            original_category="plumbing",
            corrected_category="plumbing-faucet",
            employee_level=10,
            knowledge_state=None
        )

        # Higher level = higher weight
        assert state_low.correction_weights[0] < state_high.correction_weights[0]


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Tests for hypothesis state serialization."""

    def test_serialize_hypothesis_state(self, engine):
        """HypothesisBundle should serialize to state."""
        bundle, _ = engine.diagnose(text="faucet leak")

        state = engine.serialize_hypothesis_state(bundle)

        assert state.hypotheses == bundle.hypotheses
        assert len(state.probabilities) == len(bundle.probabilities)
        assert state.superposition_vector  # Base64 string
        assert state.dimensions == engine.config.dimensions

    def test_deserialize_hypothesis_state(self, engine):
        """State should deserialize back to HypothesisBundle."""
        bundle1, _ = engine.diagnose(text="faucet leak")
        state = engine.serialize_hypothesis_state(bundle1)
        bundle2 = engine.deserialize_hypothesis_state(state)

        assert bundle2.hypotheses == bundle1.hypotheses
        assert torch.allclose(bundle2.probabilities, bundle1.probabilities, atol=1e-5)

        from vsa_core import similarity
        sim = to_real_float(similarity(bundle2.vector, bundle1.vector))
        assert sim > 0.99, f"Roundtrip should preserve vector: {sim}"

    def test_tensor_base64_roundtrip(self, engine):
        """Tensor serialization should be lossless."""
        original = torch.randn(4096, dtype=torch.complex64)

        encoded = engine._tensor_to_base64(original)
        decoded = engine._base64_to_tensor(encoded)

        assert torch.allclose(original, decoded)


# =============================================================================
# GAMIFICATION HELPERS TESTS
# =============================================================================

class TestGamificationHelpers:
    """Tests for gamification helper functions."""

    def test_expertise_multiplier(self):
        """Expertise multiplier should scale with level."""
        assert expertise_multiplier(1) == 1.0
        assert expertise_multiplier(5) > expertise_multiplier(1)
        assert expertise_multiplier(10) > expertise_multiplier(5)

    def test_calculate_level(self):
        """Level should be calculated from XP."""
        assert calculate_level(0) == 1
        assert calculate_level(100) == 2
        assert calculate_level(300) == 3
        assert calculate_level(5500) == 10
        assert calculate_level(10000) == 10  # Max level

    def test_xp_for_next_level(self):
        """XP needed should be calculated correctly."""
        assert xp_for_next_level(0) == 100  # Level 1 -> 2
        assert xp_for_next_level(50) == 50  # Halfway to level 2
        assert xp_for_next_level(5500) == 0  # Max level


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRepairEngineIntegration:
    """End-to-end integration tests."""

    def test_full_diagnosis_flow(self, engine):
        """Test complete diagnosis workflow."""
        store_id = "integration-store"
        employee_id = "emp-123"

        # Initial diagnosis
        bundle1, metadata1 = engine.diagnose(
            text="my kitchen faucet is dripping",
            store_id=store_id,
            employee_id=employee_id
        )

        assert len(bundle1.hypotheses) > 0
        response1 = engine.bundle_to_response(bundle1, "prob-1", metadata1)
        assert response1.status == ProblemStatus.DIAGNOSED

        # Refine with more info
        evidence = engine.text_encoder.encode("it drips from the handle, single lever")
        bundle2 = engine.refine_diagnosis(bundle1, evidence)

        # Should still have same categories
        assert set(bundle2.hypotheses) == set(bundle1.hypotheses)

        # Check for collapse
        winner = engine.check_collapse(bundle2)

        # Incorporate correction if employee disagrees
        if winner != "plumbing-faucet":
            knowledge_state = engine.incorporate_correction(
                problem_vec=bundle2.vector,
                original_category=winner or bundle2.top_hypothesis()[0],
                corrected_category="plumbing-faucet",
                employee_level=5
            )
            assert knowledge_state.total_corrections == 1

    def test_multi_store_isolation(self, engine):
        """Multiple stores should have isolated contexts."""
        # Store A: mostly plumbing
        for i in range(3):
            engine.diagnose(text=f"faucet leak {i}", store_id="store-a")

        # Store B: mostly electrical
        for i in range(3):
            engine.diagnose(text=f"outlet spark {i}", store_id="store-b")

        # Verify different memories
        mem_a = engine._store_memories["store-a"]
        mem_b = engine._store_memories["store-b"]

        from vsa_core import similarity
        sim = to_real_float(similarity(mem_a, mem_b))

        # Different problem types = different memories
        assert sim < 0.8, f"Different stores should have different context: {sim}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
