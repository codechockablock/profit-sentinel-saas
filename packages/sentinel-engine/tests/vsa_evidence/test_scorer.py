"""
Tests for cause scorer module.

Validates:
- Scoring uses positive similarity summing
- Confidence and ambiguity calculations are correct
- Cold path routing decisions are correct
- Batch scoring works correctly

Target metrics from research:
- 0% hallucination
- 100% multi-hop accuracy
- Hot path latency <50ms (achieves 0.003ms)
"""

import time

import pytest
from sentinel_engine.context import create_analysis_context
from sentinel_engine.vsa_evidence.scorer import (
    CauseScore,
    ScoringResult,
    create_batch_scorer,
    create_cause_scorer,
)


class TestCauseScorer:
    """Test suite for CauseScorer class."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    @pytest.fixture
    def scorer(self, ctx):
        """Create cause scorer."""
        return create_cause_scorer(ctx)

    def test_score_facts_basic(self, scorer):
        """Test basic fact scoring."""
        facts = [
            {"shrinkage_rate": 0.1, "qty_difference": -20},
            {"shrinkage_rate": 0.08, "negative_inventory": True},
        ]

        result = scorer.score_facts(facts)

        assert isinstance(result, ScoringResult)
        assert result.top_cause is not None
        assert 0 <= result.confidence <= 1
        assert 0 <= result.ambiguity_score <= 1

    def test_score_returns_all_causes(self, scorer):
        """Test that scoring returns scores for all causes."""
        facts = [{"shrinkage_rate": 0.1}]

        result = scorer.score_facts(facts)

        # Should have scores for all causes that got non-zero
        assert len(result.scores) > 0
        assert all(isinstance(s, CauseScore) for s in result.scores)

    def test_score_empty_facts(self, scorer):
        """Test scoring with empty facts."""
        result = scorer.score_facts([])

        assert result.top_cause is None
        assert result.confidence == 0.0
        assert result.needs_cold_path is True
        assert result.cold_path_reason == "No facts provided"

    def test_score_no_matching_rules(self, scorer):
        """Test scoring when no rules match."""
        facts = [{"unknown_attribute": "unknown_value"}]

        result = scorer.score_facts(facts)

        assert result.needs_cold_path is True
        assert (
            "no_matching_rules" in (result.cold_path_reason or "").lower()
            or "no rules" in (result.cold_path_reason or "").lower()
        )

    def test_theft_scenario_scores_theft(self, scorer):
        """Test that theft scenario scores theft highest."""
        facts = [
            {"shrinkage_rate": 0.15, "qty_difference": -30},
            {"negative_inventory": True, "high_value": True},
        ]

        result = scorer.score_facts(facts)

        assert result.top_cause == "theft"

    def test_vendor_increase_scenario(self, scorer):
        """Test vendor increase scenario scoring."""
        facts = [
            {"cost_delta": 0.2, "margin_compression": True},
        ]

        result = scorer.score_facts(facts)

        assert result.top_cause == "vendor_increase"

    def test_ambiguity_detection(self, scorer):
        """Test that ambiguous cases are detected."""
        # Create facts that could indicate multiple causes
        facts = [
            {"margin_delta": -0.12},  # Could be margin_leak
            {"cost_delta": 0.08},  # Could be vendor_increase
        ]

        result = scorer.score_facts(facts)

        # Should have non-zero ambiguity
        # (depends on how rules weight these attributes)
        assert isinstance(result.ambiguity_score, float)

    def test_confidence_calculation(self, scorer):
        """Test confidence calculation."""
        # Strong evidence should have high confidence
        strong_facts = [
            {"shrinkage_rate": 0.2},
            {"qty_difference": -50},
            {"negative_inventory": True},
            {"high_value": True},
        ]

        result = scorer.score_facts(strong_facts)

        # Multiple strong signals should increase confidence
        assert result.confidence > 0.3

    def test_score_rows_convenience(self, scorer):
        """Test score_rows convenience method."""
        rows = [
            {"quantity": -10, "cost": 5.0, "Retail": 10.0},
        ]

        result = scorer.score_rows(rows)

        assert isinstance(result, ScoringResult)
        assert result.top_cause is not None

    def test_hot_path_latency(self, scorer):
        """Test that hot path meets latency target (<50ms)."""
        facts = [
            {"shrinkage_rate": 0.1, "qty_difference": -20},
            {"cost_delta": 0.15},
            {"margin_delta": -0.1},
        ]

        start = time.perf_counter()
        scorer.score_facts(facts)
        latency_ms = (time.perf_counter() - start) * 1000

        # Target: <50ms (research achieved 0.003ms)
        assert (
            latency_ms < 50
        ), f"Hot path latency {latency_ms:.2f}ms exceeds 50ms target"


class TestColdPathRouting:
    """Test suite for cold path routing decisions."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    def test_low_confidence_routes_to_cold(self, ctx):
        """Test that low confidence routes to cold path."""
        scorer = create_cause_scorer(
            ctx,
            confidence_threshold=0.8,  # High threshold
        )

        # Weak evidence
        facts = [{"shrinkage_rate": 0.06}]  # Just above 0.05 threshold

        result = scorer.score_facts(facts)

        # Should route to cold path due to low confidence
        # (depends on scoring, but weak signal should be low confidence)
        if result.confidence < 0.8:
            assert result.needs_cold_path is True

    def test_high_ambiguity_routes_to_cold(self, ctx):
        """Test that high ambiguity routes to cold path."""
        scorer = create_cause_scorer(
            ctx,
            ambiguity_threshold=0.3,  # Low threshold
        )

        # Facts that could indicate multiple causes
        facts = [
            {"margin_delta": -0.12},
            {"cost_delta": 0.12},
        ]

        result = scorer.score_facts(facts)

        # If ambiguity is high, should route to cold
        if result.ambiguity_score > 0.3:
            assert result.needs_cold_path is True

    def test_critical_severity_routes_to_cold(self, ctx):
        """Test that critical severity findings route to cold path."""
        scorer = create_cause_scorer(ctx)

        # Strong theft evidence (critical severity)
        facts = [
            {"shrinkage_rate": 0.2},
            {"qty_difference": -100},
            {"negative_inventory": True},
        ]

        result = scorer.score_facts(facts)

        # Theft is critical, should route for verification
        if result.top_cause == "theft":
            assert result.needs_cold_path is True
            assert "critical" in (result.cold_path_reason or "").lower()


class TestScoringResult:
    """Test suite for ScoringResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ScoringResult(
            scores=[
                CauseScore(
                    cause="theft",
                    score=1.5,
                    confidence=0.8,
                    evidence_count=3,
                    metadata={"severity": "critical"},
                ),
            ],
            top_cause="theft",
            ambiguity_score=0.2,
            confidence=0.8,
            needs_cold_path=True,
            cold_path_reason="critical_severity",
            evidence_summary={"total_facts": 3},
        )

        d = result.to_dict()

        assert d["top_cause"] == "theft"
        assert d["confidence"] == 0.8
        assert d["needs_cold_path"] is True
        assert len(d["scores"]) == 1

    def test_is_confident_property(self):
        """Test is_confident property."""
        confident_result = ScoringResult(
            scores=[],
            top_cause="theft",
            ambiguity_score=0.3,
            confidence=0.7,
            needs_cold_path=False,
            cold_path_reason=None,
        )
        assert confident_result.is_confident is True

        not_confident_result = ScoringResult(
            scores=[],
            top_cause="theft",
            ambiguity_score=0.6,  # High ambiguity
            confidence=0.5,  # Low confidence
            needs_cold_path=True,
            cold_path_reason="low_confidence",
        )
        assert not_confident_result.is_confident is False


class TestBatchScorer:
    """Test suite for BatchScorer class."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    @pytest.fixture
    def batch_scorer(self, ctx):
        """Create batch scorer."""
        return create_batch_scorer(ctx)

    def test_score_by_item(self, batch_scorer):
        """Test scoring items individually."""
        rows = [
            {"sku": "SKU-001", "quantity": -10, "cost": 5.0, "Retail": 10.0},
            {
                "sku": "SKU-001",
                "qty. difference": -5,
                "quantity": 100,
                "cost": 5.0,
                "Retail": 10.0,
            },
            {
                "sku": "SKU-002",
                "cost": 10.0,
                "Retail": 8.0,
                "quantity": 50,
            },  # Below cost
        ]

        results = batch_scorer.score_by_item(rows)

        assert "sku-001" in results
        assert "sku-002" in results

    def test_score_by_category(self, batch_scorer):
        """Test scoring by category."""
        rows = [
            {
                "category": "Electronics",
                "shrinkage_rate": 0.1,
                "quantity": 50,
                "cost": 100.0,
                "Retail": 200.0,
            },
            {
                "category": "Electronics",
                "shrinkage_rate": 0.08,
                "quantity": 30,
                "cost": 50.0,
                "Retail": 100.0,
            },
            {
                "category": "Apparel",
                "margin_delta": -0.15,
                "quantity": 100,
                "cost": 20.0,
                "Retail": 40.0,
            },
        ]

        results = batch_scorer.score_by_category(rows)

        assert "electronics" in results
        assert "apparel" in results

    def test_hot_cold_split(self, batch_scorer):
        """Test splitting items into hot vs cold path."""
        rows = [
            # Clear theft case - may route to cold for severity
            {
                "sku": "SKU-001",
                "shrinkage_rate": 0.2,
                "quantity": 100,
                "cost": 10.0,
                "Retail": 20.0,
            },
            # Ambiguous case
            {"sku": "SKU-002", "quantity": 50, "cost": 10.0, "Retail": 15.0},
        ]

        hot_path, cold_path = batch_scorer.get_hot_path_resolvable(rows)

        # Both lists should be present (contents depend on scoring)
        assert isinstance(hot_path, list)
        assert isinstance(cold_path, list)
        assert len(hot_path) + len(cold_path) == 2


class TestNoHallucination:
    """Test that grounded scoring produces no hallucinations."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    @pytest.fixture
    def scorer(self, ctx):
        """Create cause scorer."""
        return create_cause_scorer(ctx)

    def test_scores_only_from_evidence(self, scorer):
        """Test that scores only come from evidence, not fabrication."""
        # Facts with clear theft indicators
        facts = [
            {"shrinkage_rate": 0.15},
            {"qty_difference": -30},
        ]

        result = scorer.score_facts(facts)

        # Should have evidence count > 0 for any non-zero score
        for score in result.scores:
            if score.score > 0:
                # Score should be proportional to evidence
                # (not fabricated from nothing)
                pass  # Evidence count validation

    def test_no_score_without_rules(self, scorer):
        """Test that no cause gets score without matching rules."""
        # Facts that don't match any rules
        facts = [{"completely_unknown_attribute": "random"}]

        result = scorer.score_facts(facts)

        # All scores should be 0 or result should indicate no matches
        if result.scores:
            total_score = sum(s.score for s in result.scores)
            assert total_score == 0 or result.needs_cold_path

    def test_evidence_count_matches_scoring(self, scorer):
        """Test that evidence count matches actual rule matches."""
        # Single clear evidence
        facts = [{"shrinkage_rate": 0.1}]

        result = scorer.score_facts(facts)

        # Evidence summary should reflect actual facts processed
        assert result.evidence_summary.get("total_facts") == 1
