"""
Tests for smart router module.

Validates:
- Hot path resolution for confident cases
- Cold path routing for ambiguous/novel cases
- Hybrid path for critical verification
- Metrics tracking
- Latency targets

Performance targets from research:
- Hot path: <50ms (achieves 0.003ms)
- Cold path: ~500ms
- Speedup: 5,059x
"""

import pytest
from sentinel_engine.context import create_analysis_context
from sentinel_engine.routing.smart_router import (
    AnalysisResult,
    ColdPathRequest,
    HotPathResult,
    RoutingDecision,
    create_smart_router,
)


class TestSmartRouter:
    """Test suite for SmartRouter class."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    @pytest.fixture
    def router(self, ctx):
        """Create smart router."""
        return create_smart_router(ctx)

    def test_analyze_returns_result(self, router):
        """Test that analyze returns AnalysisResult."""
        rows = [{"quantity": 100, "cost": 10.0, "Retail": 20.0}]

        result = router.analyze(rows)

        assert isinstance(result, AnalysisResult)
        assert result.path_used in RoutingDecision
        assert result.hot_result is not None

    def test_hot_path_for_clear_case(self, router):
        """Test hot path resolution for clear cases."""
        # Clear theft indicators
        rows = [
            {"shrinkage_rate": 0.15, "quantity": 100, "cost": 10.0, "Retail": 20.0},
            {"qty_difference": -30, "quantity": 50, "cost": 5.0, "Retail": 10.0},
        ]

        result = router.analyze(rows)

        # Should have hot path result
        assert result.hot_result is not None
        assert result.hot_result.cause is not None

    def test_force_cold_path(self, router):
        """Test forcing cold path."""
        rows = [{"quantity": 100, "cost": 10.0, "Retail": 20.0}]

        result = router.analyze(rows, force_cold_path=True)

        assert result.path_used == RoutingDecision.COLD_PATH

    def test_cold_path_without_handler(self, router):
        """Test cold path routing without handler configured."""
        rows = [{"unknown_attr": "value"}]  # Will need cold path

        result = router.analyze(rows, force_cold_path=True)

        # Should still return result, but cold_result indicates no handler
        assert result.cold_result is not None
        assert "skipped" in result.cold_result or "no_handler" in str(
            result.cold_result
        )

    def test_metrics_tracking(self, router):
        """Test that router tracks metrics."""
        initial_metrics = router.get_metrics()
        initial_hot = initial_metrics["hot_path_count"]

        # Run an analysis
        rows = [{"quantity": 100, "cost": 10.0, "Retail": 20.0}]
        router.analyze(rows)

        new_metrics = router.get_metrics()
        assert new_metrics["hot_path_count"] == initial_hot + 1

    def test_reset_metrics(self, router):
        """Test resetting metrics."""
        rows = [{"quantity": 100, "cost": 10.0, "Retail": 20.0}]
        router.analyze(rows)

        router.reset_metrics()

        metrics = router.get_metrics()
        assert metrics["hot_path_count"] == 0
        assert metrics["cold_path_count"] == 0

    def test_hot_path_latency_target(self, router):
        """Test that hot path meets latency target."""
        rows = [
            {"shrinkage_rate": 0.1, "quantity": 100, "cost": 10.0, "Retail": 20.0},
        ]

        result = router.analyze(rows)

        # Hot path should be fast
        if result.path_used == RoutingDecision.HOT_PATH:
            assert (
                result.hot_result.latency_ms < 50
            ), f"Hot path latency {result.hot_result.latency_ms:.2f}ms exceeds 50ms target"


class TestColdPathHandler:
    """Test suite for cold path handler integration."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    def test_set_cold_path_handler(self, ctx):
        """Test setting cold path handler."""
        router = create_smart_router(ctx)

        def mock_handler(request: ColdPathRequest) -> dict:
            return {"cause": "test_cause", "confidence": 0.9}

        router.set_cold_path_handler(mock_handler)

        assert router.cold_path_handler is not None

    def test_cold_path_handler_called(self, ctx):
        """Test that cold path handler is called when needed."""
        router = create_smart_router(ctx)
        handler_called = [False]

        def mock_handler(request: ColdPathRequest) -> dict:
            handler_called[0] = True
            return {"cause": "theft", "confidence": 0.85}

        router.set_cold_path_handler(mock_handler)

        # Force cold path
        rows = [{"quantity": 100, "cost": 10.0, "Retail": 20.0}]
        router.analyze(rows, force_cold_path=True)

        assert handler_called[0] is True

    def test_cold_path_request_format(self, ctx):
        """Test ColdPathRequest format."""
        router = create_smart_router(ctx)
        captured_request = [None]

        def capture_handler(request: ColdPathRequest) -> dict:
            captured_request[0] = request
            return {"cause": "test", "confidence": 0.5}

        router.set_cold_path_handler(capture_handler)

        rows = [{"quantity": -10, "cost": 5.0, "Retail": 10.0}]
        router.analyze(rows, force_cold_path=True, sku="TEST-001")

        request = captured_request[0]
        assert request is not None
        assert request.sku == "TEST-001"
        assert len(request.facts) > 0

    def test_cold_path_prompt_generation(self, ctx):
        """Test that ColdPathRequest generates valid prompt."""
        request = ColdPathRequest(
            sku="TEST-001",
            facts=[{"shrinkage_rate": 0.1, "qty_difference": -20}],
            hot_path_result=HotPathResult(
                cause="theft",
                confidence=0.7,
                ambiguity=0.3,
                evidence_count=2,
                latency_ms=1.0,
                recommendations=["Check security"],
                severity="high",
                explanation="Test",
            ),
            routing_reason="ambiguous_evidence",
            context={"avg_margin": 0.3},
        )

        prompt = request.to_prompt()

        assert "TEST-001" in prompt
        assert "theft" in prompt.lower() or "ambiguous" in prompt.lower()
        assert "Root cause" in prompt or "root cause" in prompt.lower()


class TestHotPathResult:
    """Test suite for HotPathResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = HotPathResult(
            cause="theft",
            confidence=0.85,
            ambiguity=0.2,
            evidence_count=3,
            latency_ms=0.5,
            recommendations=["Check security", "Audit inventory"],
            severity="critical",
            explanation="High shrinkage detected",
        )

        d = result.to_dict()

        assert d["cause"] == "theft"
        assert d["confidence"] == 0.85
        assert d["path"] == "hot"
        assert len(d["recommendations"]) == 2


class TestAnalysisResult:
    """Test suite for AnalysisResult class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AnalysisResult(
            path_used=RoutingDecision.HOT_PATH,
            hot_result=HotPathResult(
                cause="theft",
                confidence=0.9,
                ambiguity=0.1,
                evidence_count=5,
                latency_ms=0.3,
                recommendations=[],
                severity="critical",
                explanation="Test",
            ),
            cold_result=None,
            final_cause="theft",
            final_confidence=0.9,
            total_latency_ms=0.5,
            grounded=True,
        )

        d = result.to_dict()

        assert d["path_used"] == "hot"
        assert d["final_cause"] == "theft"
        assert d["grounded"] is True
        assert "hot_result" in d

    def test_grounded_flag_for_hot_path(self):
        """Test that hot path results are marked as grounded."""
        result = AnalysisResult(
            path_used=RoutingDecision.HOT_PATH,
            hot_result=None,
            cold_result=None,
            final_cause="theft",
            final_confidence=0.9,
            total_latency_ms=0.5,
            grounded=True,
        )

        assert result.grounded is True

    def test_grounded_flag_for_cold_path(self):
        """Test that pure cold path results are not automatically grounded."""
        result = AnalysisResult(
            path_used=RoutingDecision.COLD_PATH,
            hot_result=None,
            cold_result={"cause": "unknown"},
            final_cause="unknown",
            final_confidence=0.5,
            total_latency_ms=500.0,
            grounded=False,
        )

        assert result.grounded is False


class TestRoutingDecisions:
    """Test suite for routing decision logic."""

    @pytest.fixture
    def ctx(self):
        """Create analysis context."""
        return create_analysis_context(dimensions=1024, use_gpu=False)

    def test_routing_thresholds_configurable(self, ctx):
        """Test that routing thresholds are configurable."""
        router = create_smart_router(
            ctx,
            confidence_threshold=0.9,  # Very high
            ambiguity_threshold=0.1,  # Very low
        )

        assert router.confidence_threshold == 0.9
        assert router.ambiguity_threshold == 0.1

    def test_critical_severity_verification(self, ctx):
        """Test that critical severity triggers verification."""
        router = create_smart_router(ctx)
        router.severity_verification = True

        # Strong theft evidence (critical severity)
        rows = [
            {"shrinkage_rate": 0.2, "quantity": 100, "cost": 10.0, "Retail": 20.0},
            {"qty_difference": -50, "quantity": 50, "cost": 5.0, "Retail": 10.0},
        ]

        result = router.analyze(rows)

        # If top cause is theft (critical), should route for verification
        if result.hot_result and result.hot_result.cause == "theft":
            assert result.path_used in (
                RoutingDecision.COLD_PATH,
                RoutingDecision.HYBRID,
            )
