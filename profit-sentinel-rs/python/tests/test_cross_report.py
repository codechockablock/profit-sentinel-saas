"""Tests for cross-report comparison engine.

Covers:
    - Trend detection: new, resolved, worsening, improving, stable
    - Impact trajectory
    - Edge cases: empty analyses, identical results
    - Compare API endpoint
"""

import pytest

from sentinel_agent.cross_report import compare_analyses, CrossReportResult, LeakTrend
from sentinel_agent.analysis_store import (
    InMemoryAnalysisStore,
    init_analysis_store,
    save_analysis,
    get_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_store():
    """Reset analysis store before each test."""
    store = InMemoryAnalysisStore()
    init_analysis_store(store)
    yield store


def _make_analysis_record(
    *,
    detection_counts: dict | None = None,
    total_impact_low: float = 1000,
    total_impact_high: float = 5000,
    file_row_count: int = 100,
    label: str = "Test",
    analysis_id: str = "test-1",
    full_result: dict | None = None,
):
    """Build a minimal analysis record for comparison tests."""
    if detection_counts is None:
        detection_counts = {"low_stock": 5, "dead_stock": 3}

    if full_result is None:
        leaks = {}
        for key, count in detection_counts.items():
            leaks[key] = {"count": count, "items": []}

        full_result = {
            "summary": {
                "total_items_flagged": sum(detection_counts.values()),
                "estimated_impact": {
                    "low": total_impact_low,
                    "high": total_impact_high,
                },
                "dataset_stats": {},
            },
            "leaks": leaks,
        }

    return {
        "id": analysis_id,
        "detection_counts": detection_counts,
        "total_impact_estimate_low": total_impact_low,
        "total_impact_estimate_high": total_impact_high,
        "file_row_count": file_row_count,
        "analysis_label": label,
        "full_result": full_result,
        "created_at": "2026-02-06T10:00:00Z",
    }


# ---------------------------------------------------------------------------
# LeakTrend data class
# ---------------------------------------------------------------------------


class TestLeakTrend:
    def test_severity_new(self):
        t = LeakTrend("x", 5, 0, 5, 1000, 0, 1000, "new")
        assert t.severity == "critical"

    def test_severity_resolved(self):
        t = LeakTrend("x", 0, 5, -5, 0, 1000, -1000, "resolved")
        assert t.severity == "success"

    def test_severity_worsening(self):
        t = LeakTrend("x", 10, 5, 5, 2000, 1000, 1000, "worsening")
        assert t.severity == "warning"

    def test_severity_improving(self):
        t = LeakTrend("x", 3, 5, -2, 500, 1000, -500, "improving")
        assert t.severity == "info"

    def test_severity_stable(self):
        t = LeakTrend("x", 5, 5, 0, 1000, 1000, 0, "stable")
        assert t.severity == "neutral"


# ---------------------------------------------------------------------------
# compare_analyses tests
# ---------------------------------------------------------------------------


class TestCompareAnalyses:
    def test_basic_comparison(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 10, "dead_stock": 5},
            total_impact_high=8000,
            analysis_id="c1",
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 7, "dead_stock": 3},
            total_impact_high=5000,
            analysis_id="p1",
        )

        result = compare_analyses(current, previous)
        assert result.current_total_issues == 15
        assert result.previous_total_issues == 10
        assert result.issues_delta == 5

    def test_new_leak_detected(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 5, "margin_erosion": 3},
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 5},
        )

        result = compare_analyses(current, previous)
        assert len(result.new_leaks) == 1
        assert result.new_leaks[0].leak_key == "margin_erosion"
        assert result.new_leaks[0].status == "new"

    def test_resolved_leak(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 5},
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 5, "dead_stock": 8},
        )

        result = compare_analyses(current, previous)
        assert len(result.resolved_leaks) == 1
        assert result.resolved_leaks[0].leak_key == "dead_stock"
        assert result.resolved_leaks[0].status == "resolved"

    def test_worsening_leak(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 10},
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 3},
        )

        result = compare_analyses(current, previous)
        assert len(result.worsening_leaks) == 1
        assert result.worsening_leaks[0].count_delta == 7

    def test_improving_leak(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 2},
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 8},
        )

        result = compare_analyses(current, previous)
        assert len(result.improving_leaks) == 1
        assert result.improving_leaks[0].count_delta == -6

    def test_stable_leak(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 5},
            total_impact_high=1000,
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 5},
            total_impact_high=1000,
        )

        result = compare_analyses(current, previous)
        assert len(result.stable_leaks) == 1

    def test_overall_trend_improving(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 2},
            total_impact_high=1000,
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 10, "dead_stock": 5},
            total_impact_high=8000,
        )

        result = compare_analyses(current, previous)
        assert result.overall_trend == "improving"

    def test_overall_trend_worsening(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 10, "dead_stock": 8, "margin_erosion": 5},
            total_impact_high=10000,
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 3},
            total_impact_high=2000,
        )

        result = compare_analyses(current, previous)
        assert result.overall_trend == "worsening"

    def test_overall_trend_stable(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 5},
            total_impact_high=5000,
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 5},
            total_impact_high=5000,
        )

        result = compare_analyses(current, previous)
        assert result.overall_trend == "stable"

    def test_empty_current(self):
        current = _make_analysis_record(
            detection_counts={},
            total_impact_high=0,
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 5},
        )

        result = compare_analyses(current, previous)
        assert result.current_total_issues == 0
        assert len(result.resolved_leaks) == 1

    def test_empty_previous(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 5},
        )
        previous = _make_analysis_record(
            detection_counts={},
            total_impact_high=0,
        )

        result = compare_analyses(current, previous)
        assert len(result.new_leaks) == 1

    def test_both_empty(self):
        current = _make_analysis_record(detection_counts={}, total_impact_high=0)
        previous = _make_analysis_record(detection_counts={}, total_impact_high=0)

        result = compare_analyses(current, previous)
        assert result.overall_trend == "stable"
        assert len(result.leak_trends) == 0

    def test_metadata_preserved(self):
        current = _make_analysis_record(analysis_id="cur-1", label="Current")
        previous = _make_analysis_record(analysis_id="prev-1", label="Previous")

        result = compare_analyses(current, previous)
        assert result.current_analysis_id == "cur-1"
        assert result.previous_analysis_id == "prev-1"
        assert result.current_label == "Current"
        assert result.previous_label == "Previous"

    def test_to_dict_serializable(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 10, "margin_erosion": 3},
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 5, "dead_stock": 2},
        )

        result = compare_analyses(current, previous)
        d = result.to_dict()

        # Verify structure
        assert "summary" in d
        assert "leak_trends" in d
        assert "new_leaks" in d
        assert "resolved_leaks" in d
        assert "metadata" in d

        # Verify it's JSON-serializable
        import json
        json_str = json.dumps(d)
        assert len(json_str) > 10

    def test_summary_dict(self):
        current = _make_analysis_record(
            detection_counts={"low_stock": 10},
            total_impact_high=8000,
        )
        previous = _make_analysis_record(
            detection_counts={"low_stock": 5},
            total_impact_high=5000,
        )

        result = compare_analyses(current, previous)
        s = result.summary

        assert s["overall_trend"] in ("improving", "worsening", "stable")
        assert s["issues_delta"] == 5
        assert s["current_total_issues"] == 10
        assert s["previous_total_issues"] == 5


# ---------------------------------------------------------------------------
# Compare endpoint integration test
# ---------------------------------------------------------------------------


class TestCompareEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from sentinel_agent.sidecar import create_app
        from sentinel_agent.sidecar_config import SidecarSettings

        settings = SidecarSettings(
            sidecar_dev_mode=True,
            csv_path="fixtures/sample_inventory.csv",
            supabase_url="",
            supabase_service_key="",
        )
        app = create_app(settings)
        return TestClient(app)

    def test_compare_two_analyses(self, client):
        saved1 = save_analysis(
            user_id="dev-user",
            result={
                "summary": {
                    "total_items_flagged": 5,
                    "estimated_impact": {"low": 1000, "high": 5000},
                    "dataset_stats": {},
                },
                "leaks": {"low_stock": {"count": 5, "items": []}},
            },
            file_hash="a" * 64,
            file_row_count=100,
        )
        saved2 = save_analysis(
            user_id="dev-user",
            result={
                "summary": {
                    "total_items_flagged": 10,
                    "estimated_impact": {"low": 2000, "high": 10000},
                    "dataset_stats": {},
                },
                "leaks": {
                    "low_stock": {"count": 8, "items": []},
                    "dead_stock": {"count": 2, "items": []},
                },
            },
            file_hash="b" * 64,
            file_row_count=200,
        )

        resp = client.post(
            "/api/v1/analyses/compare",
            json={
                "current_id": saved2["id"],
                "previous_id": saved1["id"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["summary"]["issues_delta"] == 5
        assert "dead_stock" in data["new_leaks"]
        assert len(data["leak_trends"]) >= 1

    def test_compare_not_found(self, client):
        saved = save_analysis(
            user_id="dev-user",
            result={
                "summary": {"total_items_flagged": 1, "estimated_impact": {}, "dataset_stats": {}},
                "leaks": {},
            },
            file_hash="x" * 64,
            file_row_count=10,
        )

        resp = client.post(
            "/api/v1/analyses/compare",
            json={
                "current_id": saved["id"],
                "previous_id": "nonexistent",
            },
        )
        assert resp.status_code == 404

    def test_compare_route_ordering(self, client):
        """Ensure /analyses/compare is not captured by /{analysis_id}."""
        # This would fail if compare was registered after {analysis_id}
        resp = client.post(
            "/api/v1/analyses/compare",
            json={"current_id": "a", "previous_id": "b"},
        )
        # Should get 404 (analysis not found) NOT 405 (method not allowed)
        # or 422 (would happen if matched against GET /{analysis_id})
        assert resp.status_code == 404
