"""Tests for analysis store and history endpoints.

Covers:
    - InMemoryAnalysisStore CRUD operations
    - Analysis persistence from upload route
    - History API endpoints (list, get, rename, delete)
"""

from unittest.mock import MagicMock, patch

import pytest
from sentinel_agent.analysis_store import (
    InMemoryAnalysisStore,
    compute_file_hash,
    delete_analysis,
    get_analysis,
    get_comparison_pair,
    init_analysis_store,
    list_user_analyses,
    rename_analysis,
    save_analysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_store():
    """Reset to a fresh InMemoryAnalysisStore before each test."""
    store = InMemoryAnalysisStore()
    init_analysis_store(store)
    yield store


def _make_result(total_items_flagged=5, leaks=None):
    """Helper to build a minimal analysis result dict."""
    if leaks is None:
        leaks = {
            "low_stock": {"count": 3, "items": []},
            "dead_stock": {"count": 2, "items": []},
        }
    return {
        "summary": {
            "total_items_flagged": total_items_flagged,
            "estimated_impact": {"low": 1000, "high": 5000},
            "dataset_stats": {"avg_margin": 0.32, "avg_qty": 45},
        },
        "leaks": leaks,
        "engine_version": "sidecar-test",
    }


# ---------------------------------------------------------------------------
# InMemoryAnalysisStore unit tests
# ---------------------------------------------------------------------------


class TestInMemoryAnalysisStore:
    def test_save_and_get(self, fresh_store):
        result = _make_result()
        saved = fresh_store.save(
            user_id="user-1",
            result=result,
            file_hash="abc123" * 10 + "abcd",
            file_row_count=100,
            original_filename="test.csv",
        )

        assert saved["id"] is not None
        assert saved["user_id"] == "user-1"
        assert saved["file_row_count"] == 100
        assert saved["original_filename"] == "test.csv"
        assert saved["full_result"] == result

        # Retrieve by ID
        fetched = fresh_store.get_by_id(saved["id"], "user-1")
        assert fetched is not None
        assert fetched["id"] == saved["id"]

    def test_auto_generated_label(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(total_items_flagged=7),
            file_hash="x" * 64,
            file_row_count=250,
            original_filename="inventory_export.csv",
        )
        assert "inventory_export" in saved["analysis_label"]
        assert "250 rows" in saved["analysis_label"]
        assert "7 issues" in saved["analysis_label"]

    def test_custom_label(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
            analysis_label="My Custom Analysis",
        )
        assert saved["analysis_label"] == "My Custom Analysis"

    def test_list_for_user(self, fresh_store):
        for i in range(5):
            fresh_store.save(
                user_id="user-1",
                result=_make_result(),
                file_hash=f"hash{i}" + "x" * 59,
                file_row_count=100 + i,
            )
        # Different user
        fresh_store.save(
            user_id="user-2",
            result=_make_result(),
            file_hash="other" + "x" * 59,
            file_row_count=200,
        )

        user1_list = fresh_store.list_for_user("user-1")
        assert len(user1_list) == 5
        # Should NOT include full_result in list view
        assert "full_result" not in user1_list[0]
        assert user1_list[0]["has_full_result"] is True

        user2_list = fresh_store.list_for_user("user-2")
        assert len(user2_list) == 1

    def test_list_pagination(self, fresh_store):
        for i in range(10):
            fresh_store.save(
                user_id="user-1",
                result=_make_result(),
                file_hash=f"hash{i}" + "x" * 59,
                file_row_count=i,
            )

        page1 = fresh_store.list_for_user("user-1", limit=3, offset=0)
        page2 = fresh_store.list_for_user("user-1", limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 3
        # Pages should not overlap
        ids1 = {r["id"] for r in page1}
        ids2 = {r["id"] for r in page2}
        assert ids1.isdisjoint(ids2)

    def test_list_ordered_newest_first(self, fresh_store):
        for i in range(3):
            fresh_store.save(
                user_id="user-1",
                result=_make_result(),
                file_hash=f"h{i}" + "x" * 62,
                file_row_count=i,
            )

        analyses = fresh_store.list_for_user("user-1")
        # Newest (highest counter) should be first
        assert analyses[0]["id"] == "analysis-0003"
        assert analyses[2]["id"] == "analysis-0001"

    def test_get_scoped_to_user(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )

        # Same user can see it
        assert fresh_store.get_by_id(saved["id"], "user-1") is not None
        # Different user cannot
        assert fresh_store.get_by_id(saved["id"], "user-2") is None

    def test_delete(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )

        assert fresh_store.delete(saved["id"], "user-1") is True
        assert fresh_store.get_by_id(saved["id"], "user-1") is None
        # Double delete returns False
        assert fresh_store.delete(saved["id"], "user-1") is False

    def test_delete_scoped_to_user(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )
        # Different user cannot delete
        assert fresh_store.delete(saved["id"], "user-2") is False
        # Original user can still see it
        assert fresh_store.get_by_id(saved["id"], "user-1") is not None

    def test_update_label(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )

        assert fresh_store.update_label(saved["id"], "user-1", "New Label") is True
        fetched = fresh_store.get_by_id(saved["id"], "user-1")
        assert fetched["analysis_label"] == "New Label"

    def test_update_label_wrong_user(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )
        assert fresh_store.update_label(saved["id"], "user-2", "Hacked") is False

    def test_get_recent_pair(self, fresh_store):
        fresh_store.save(
            user_id="user-1",
            result=_make_result(total_items_flagged=3),
            file_hash="a" * 64,
            file_row_count=100,
        )
        fresh_store.save(
            user_id="user-1",
            result=_make_result(total_items_flagged=7),
            file_hash="b" * 64,
            file_row_count=200,
        )

        current, previous = fresh_store.get_recent_pair("user-1")
        assert current is not None
        assert previous is not None
        assert current["file_row_count"] == 200  # Newest
        assert previous["file_row_count"] == 100

    def test_get_recent_pair_single_analysis(self, fresh_store):
        fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="a" * 64,
            file_row_count=100,
        )

        current, previous = fresh_store.get_recent_pair("user-1")
        assert current is not None
        assert previous is None

    def test_get_recent_pair_no_analyses(self, fresh_store):
        current, previous = fresh_store.get_recent_pair("user-1")
        assert current is None
        assert previous is None

    def test_detection_counts_extracted(self, fresh_store):
        saved = fresh_store.save(
            user_id="user-1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )
        assert saved["detection_counts"]["low_stock"] == 3
        assert saved["detection_counts"]["dead_stock"] == 2


# ---------------------------------------------------------------------------
# Module-level function tests (delegate to global store)
# ---------------------------------------------------------------------------


class TestModuleFunctions:
    def test_save_and_list(self):
        save_analysis(
            user_id="u1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=50,
        )
        analyses = list_user_analyses("u1")
        assert len(analyses) == 1

    def test_get_and_delete(self):
        saved = save_analysis(
            user_id="u1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=50,
        )
        assert get_analysis(saved["id"], "u1") is not None
        assert delete_analysis(saved["id"], "u1") is True
        assert get_analysis(saved["id"], "u1") is None

    def test_rename(self):
        saved = save_analysis(
            user_id="u1",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=50,
        )
        rename_analysis(saved["id"], "u1", "Renamed")
        fetched = get_analysis(saved["id"], "u1")
        assert fetched["analysis_label"] == "Renamed"

    def test_comparison_pair(self):
        save_analysis(
            user_id="u1", result=_make_result(), file_hash="a" * 64, file_row_count=10
        )
        save_analysis(
            user_id="u1", result=_make_result(), file_hash="b" * 64, file_row_count=20
        )
        current, previous = get_comparison_pair("u1")
        assert current is not None
        assert previous is not None


class TestComputeFileHash:
    def test_deterministic(self):
        content = b"hello world"
        h1 = compute_file_hash(content)
        h2 = compute_file_hash(content)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_content(self):
        h1 = compute_file_hash(b"file1")
        h2 = compute_file_hash(b"file2")
        assert h1 != h2


# ---------------------------------------------------------------------------
# API endpoint integration tests (via TestClient)
# ---------------------------------------------------------------------------


class TestAnalysisHistoryEndpoints:
    @pytest.fixture
    def client(self):
        """Create a test client with dev mode enabled."""
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

    def test_list_analyses_empty(self, client):
        resp = client.get("/api/v1/analyses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["analyses"] == []
        assert data["total"] == 0

    def test_list_analyses_after_save(self, client):
        # Save directly via the store
        save_analysis(
            user_id="dev-user",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
            original_filename="test.csv",
        )

        resp = client.get("/api/v1/analyses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["analyses"][0]["original_filename"] == "test.csv"

    def test_get_analysis_detail(self, client):
        saved = save_analysis(
            user_id="dev-user",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )

        resp = client.get(f"/api/v1/analyses/{saved['id']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == saved["id"]
        assert data["full_result"] is not None

    def test_get_analysis_not_found(self, client):
        resp = client.get("/api/v1/analyses/nonexistent-id")
        assert resp.status_code == 404

    def test_rename_analysis(self, client):
        saved = save_analysis(
            user_id="dev-user",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )

        resp = client.patch(
            f"/api/v1/analyses/{saved['id']}",
            json={"label": "My Renamed Analysis"},
        )
        assert resp.status_code == 200
        assert resp.json()["message"] == "Analysis renamed"

        # Verify the label changed
        detail = client.get(f"/api/v1/analyses/{saved['id']}").json()
        assert detail["analysis_label"] == "My Renamed Analysis"

    def test_rename_not_found(self, client):
        resp = client.patch(
            "/api/v1/analyses/nonexistent-id",
            json={"label": "Nope"},
        )
        assert resp.status_code == 404

    def test_delete_analysis(self, client):
        saved = save_analysis(
            user_id="dev-user",
            result=_make_result(),
            file_hash="x" * 64,
            file_row_count=100,
        )

        resp = client.delete(f"/api/v1/analyses/{saved['id']}")
        assert resp.status_code == 200
        assert resp.json()["message"] == "Analysis deleted"

        # Verify it's gone
        resp = client.get(f"/api/v1/analyses/{saved['id']}")
        assert resp.status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/api/v1/analyses/nonexistent-id")
        assert resp.status_code == 404

    def test_list_pagination(self, client):
        for i in range(5):
            save_analysis(
                user_id="dev-user",
                result=_make_result(),
                file_hash=f"h{i}" + "x" * 62,
                file_row_count=i,
            )

        resp = client.get("/api/v1/analyses?limit=2&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["analyses"]) == 2

        resp2 = client.get("/api/v1/analyses?limit=2&offset=2")
        data2 = resp2.json()
        assert len(data2["analyses"]) == 2

        # IDs should not overlap
        ids1 = {a["id"] for a in data["analyses"]}
        ids2 = {a["id"] for a in data2["analyses"]}
        assert ids1.isdisjoint(ids2)
