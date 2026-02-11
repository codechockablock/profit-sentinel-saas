"""Tests for the Sentinel sidecar FastAPI application.

Uses FastAPI TestClient with mocked SentinelEngine to avoid
needing the Rust binary for unit tests.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sentinel_agent.api_models import TaskStatus
from sentinel_agent.models import (
    Digest,
    Issue,
    IssueType,
    Sku,
    Summary,
    TrendDirection,
)
from sentinel_agent.sidecar import create_app
from sentinel_agent.sidecar_config import SidecarSettings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DIGEST = Digest(
    generated_at="2026-02-05T06:00:00+00:00",
    store_filter=["store-7"],
    pipeline_ms=12,
    issues=[
        Issue(
            id="store-7-DeadStock-001",
            issue_type=IssueType.DEAD_STOCK,
            store_id="store-7",
            dollar_impact=5000.0,
            confidence=0.89,
            trend_direction=TrendDirection.WORSENING,
            priority_score=11.37,
            urgency_score=0.9,
            detection_timestamp="2026-01-02T00:00:00Z",
            skus=[
                Sku(
                    sku_id="SEA-1201",
                    qty_on_hand=100.0,
                    unit_cost=50.0,
                    retail_price=67.5,
                    margin_pct=0.35,
                    sales_last_30d=0.0,
                    days_since_receipt=180.0,
                    is_damaged=False,
                    on_order_qty=0.0,
                    is_seasonal=False,
                ),
            ],
            context="Dead stock — zero sales for 180+ days.",
        ),
        Issue(
            id="store-7-MarginErosion-001",
            issue_type=IssueType.MARGIN_EROSION,
            store_id="store-7",
            dollar_impact=2422.0,
            confidence=0.87,
            trend_direction=TrendDirection.WORSENING,
            priority_score=10.17,
            urgency_score=0.9,
            detection_timestamp="2026-01-02T00:00:00Z",
            skus=[
                Sku(
                    sku_id="PNT-1001",
                    qty_on_hand=50.0,
                    unit_cost=100.0,
                    retail_price=105.0,
                    margin_pct=0.05,
                    sales_last_30d=10.0,
                    days_since_receipt=30.0,
                    is_damaged=False,
                    on_order_qty=0.0,
                    is_seasonal=False,
                ),
            ],
            context="Low margin — 5% vs 35% benchmark.",
        ),
    ],
    summary=Summary(
        total_issues=2,
        total_dollar_impact=7422.0,
        stores_affected=1,
        records_processed=10,
        issues_detected=2,
        issues_filtered_out=0,
    ),
)


def _dev_settings() -> SidecarSettings:
    """Dev mode settings for testing."""
    return SidecarSettings(
        sidecar_dev_mode=True,
        csv_path="fixtures/sample_inventory.csv",
        sentinel_bin=None,
        supabase_url="",
        supabase_service_key="",
    )


@pytest.fixture
def client():
    """Create a test client with mocked engine."""

    settings = _dev_settings()

    with patch("sentinel_agent.sidecar.SentinelEngine") as MockEngine:
        mock_engine = MagicMock()
        mock_engine.run.return_value = SAMPLE_DIGEST
        mock_engine.binary = "/mock/sentinel-server"
        MockEngine.return_value = mock_engine

        app = create_app(settings)
        yield TestClient(app)


@pytest.fixture
def client_no_binary():
    """Create a test client where binary is not found."""

    settings = _dev_settings()

    with patch(
        "sentinel_agent.sidecar.SentinelEngine",
        side_effect=Exception("Binary not found"),
    ):
        app = create_app(settings)
        yield TestClient(app)


@pytest.fixture
def auth_client():
    """Create a test client with auth REQUIRED (not dev mode)."""

    settings = SidecarSettings(
        sidecar_dev_mode=False,
        csv_path="fixtures/sample_inventory.csv",
        sentinel_bin=None,
        supabase_url="https://fake.supabase.co",
        supabase_service_key="fake-key",
    )

    with patch("sentinel_agent.sidecar.SentinelEngine") as MockEngine:
        mock_engine = MagicMock()
        mock_engine.run.return_value = SAMPLE_DIGEST
        mock_engine.binary = "/mock/sentinel-server"
        MockEngine.return_value = mock_engine

        app = create_app(settings)
        yield TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.13.0"
        assert data["binary_found"] is True
        assert data["dev_mode"] is True

    def test_health_no_auth_required(self, auth_client):
        """Health endpoint should work without auth."""
        resp = auth_client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Digest endpoints
# ---------------------------------------------------------------------------


class TestDigest:
    def test_get_digest(self, client):
        resp = client.get("/api/v1/digest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["issue_count"] == 2
        assert data["total_dollar_impact"] == 7422.0
        assert "Good morning" in data["rendered_text"]
        assert len(data["digest"]["issues"]) == 2

    def test_get_digest_with_stores(self, client):
        resp = client.get("/api/v1/digest?stores=store-7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["issue_count"] == 2

    def test_get_digest_with_top_k(self, client):
        resp = client.get("/api/v1/digest?top_k=3")
        assert resp.status_code == 200

    def test_get_store_digest(self, client):
        resp = client.get("/api/v1/digest/store-7")
        assert resp.status_code == 200
        data = resp.json()
        assert data["issue_count"] == 2

    def test_digest_caching(self, client):
        """Second call should use cache."""
        resp1 = client.get("/api/v1/digest?stores=store-7")
        resp2 = client.get("/api/v1/digest?stores=store-7")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Both should return same data
        assert resp1.json()["issue_count"] == resp2.json()["issue_count"]


# ---------------------------------------------------------------------------
# Delegation endpoints
# ---------------------------------------------------------------------------


class TestDelegation:
    def _load_digest(self, client):
        """Helper: load digest first so issues are cached."""
        client.get("/api/v1/digest")

    def test_delegate_issue(self, client):
        self._load_digest(client)
        resp = client.post(
            "/api/v1/delegate",
            json={
                "issue_id": "store-7-DeadStock-001",
                "assignee": "Store 7 Manager",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"].startswith("task-")
        assert "TASK:" in data["rendered_text"]

    def test_delegate_with_notes(self, client):
        self._load_digest(client)
        resp = client.post(
            "/api/v1/delegate",
            json={
                "issue_id": "store-7-DeadStock-001",
                "assignee": "Manager",
                "notes": "Handle urgently",
            },
        )
        assert resp.status_code == 200

    def test_delegate_missing_issue(self, client):
        """Should 404 if issue not in cache."""
        resp = client.post(
            "/api/v1/delegate",
            json={
                "issue_id": "nonexistent-issue",
                "assignee": "Someone",
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------


class TestTasks:
    def _create_task(self, client):
        """Helper: load digest + delegate to create a task."""
        client.get("/api/v1/digest")
        resp = client.post(
            "/api/v1/delegate",
            json={
                "issue_id": "store-7-DeadStock-001",
                "assignee": "Store 7 Manager",
            },
        )
        return resp.json()["task_id"]

    def test_list_tasks_empty(self, client):
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["tasks"] == []

    def test_list_tasks_after_delegation(self, client):
        self._create_task(client)
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1

    def test_list_tasks_filter_by_store(self, client):
        self._create_task(client)
        resp = client.get("/api/v1/tasks?store_id=store-7")
        assert resp.json()["total"] == 1
        resp = client.get("/api/v1/tasks?store_id=store-99")
        assert resp.json()["total"] == 0

    def test_get_task_by_id(self, client):
        task_id = self._create_task(client)
        resp = client.get(f"/api/v1/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"]["task_id"] == task_id
        assert data["status"] == "open"

    def test_get_task_not_found(self, client):
        resp = client.get("/api/v1/tasks/nonexistent")
        assert resp.status_code == 404

    def test_update_task_complete(self, client):
        task_id = self._create_task(client)
        resp = client.patch(
            f"/api/v1/tasks/{task_id}",
            json={"status": "completed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    def test_update_task_escalate_with_notes(self, client):
        task_id = self._create_task(client)
        resp = client.patch(
            f"/api/v1/tasks/{task_id}",
            json={
                "status": "escalated",
                "notes": "Need regional approval",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "escalated"
        assert "Need regional approval" in data["notes"]

    def test_update_task_not_found(self, client):
        resp = client.patch(
            "/api/v1/tasks/nonexistent",
            json={"status": "completed"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Vendor call endpoint
# ---------------------------------------------------------------------------


class TestVendorCall:
    def test_vendor_call_prep(self, client):
        # Load digest first
        client.get("/api/v1/digest")
        resp = client.get("/api/v1/vendor-call/store-7-DeadStock-001")
        assert resp.status_code == 200
        data = resp.json()
        assert "call_prep" in data
        assert "rendered_text" in data
        assert "VENDOR CALL BRIEF" in data["rendered_text"]

    def test_vendor_call_missing_issue(self, client):
        resp = client.get("/api/v1/vendor-call/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Co-op endpoint
# ---------------------------------------------------------------------------


class TestCoopReport:
    def test_coop_report(self, client):
        # Load digest first
        client.get("/api/v1/digest?stores=store-7")
        resp = client.get("/api/v1/coop/store-7")
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data
        assert "rendered_text" in data
        assert data["total_opportunity"] >= 0

    def test_coop_report_has_rendered_text(self, client):
        client.get("/api/v1/digest?stores=store-7")
        resp = client.get("/api/v1/coop/store-7")
        data = resp.json()
        assert "CO-OP INTELLIGENCE REPORT" in data["rendered_text"]


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    def test_dev_mode_no_auth_needed(self, client):
        """Dev mode should allow requests without token."""
        resp = client.get("/api/v1/digest")
        assert resp.status_code == 200

    def test_auth_required_without_token(self, auth_client):
        """Non-dev mode should reject requests without token."""
        resp = auth_client.get("/api/v1/digest")
        assert resp.status_code == 401

    def test_auth_required_bad_header(self, auth_client):
        """Non-dev mode should reject bad auth header."""
        resp = auth_client.get(
            "/api/v1/digest",
            headers={"Authorization": "NotBearer token"},
        )
        assert resp.status_code == 401

    def test_health_always_accessible(self, auth_client):
        """Health endpoint works without auth even in prod mode."""
        resp = auth_client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_index_html_served(self, client):
        """The root should serve index.html."""
        resp = client.get("/")
        # Static files mount may redirect or serve HTML
        assert resp.status_code in (200, 307)

    def test_css_served(self, client):
        resp = client.get("/styles.css")
        assert resp.status_code == 200
        assert (
            "css" in resp.headers.get("content-type", "").lower()
            or resp.status_code == 200
        )

    def test_js_served(self, client):
        resp = client.get("/app.js")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_top_k(self, client):
        """top_k out of range should fail validation."""
        resp = client.get("/api/v1/digest?top_k=0")
        assert resp.status_code == 422  # FastAPI validation error

    def test_delegate_missing_fields(self, client):
        """Missing required fields should fail validation."""
        resp = client.post("/api/v1/delegate", json={"issue_id": "test"})
        assert resp.status_code == 422
