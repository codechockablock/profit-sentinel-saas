"""Tests for POS System Integrations.

Covers:
    - Supported systems listing
    - Connection CRUD (create, list, get, disconnect, delete)
    - Sync triggering
    - Connection status lifecycle
    - API endpoint integration
"""

import pytest

from sentinel_agent.pos_integrations import (
    ConnectionStatus,
    InMemoryPosConnectionStore,
    PosConnection,
    PosSystem,
    SyncFrequency,
    SyncResult,
    create_pos_connection,
    delete_pos_connection,
    disconnect_pos,
    get_pos_connection,
    get_supported_systems,
    init_pos_store,
    list_pos_connections,
    trigger_sync,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_store():
    """Reset POS connection store before each test."""
    store = InMemoryPosConnectionStore()
    init_pos_store(store)
    yield store


# ---------------------------------------------------------------------------
# Supported systems tests
# ---------------------------------------------------------------------------


class TestSupportedSystems:
    def test_all_four_systems(self):
        systems = get_supported_systems()
        assert len(systems) == 4
        names = {s.system for s in systems}
        assert names == {PosSystem.SQUARE, PosSystem.LIGHTSPEED, PosSystem.CLOVER, PosSystem.SHOPIFY}

    def test_system_has_setup_steps(self):
        systems = get_supported_systems()
        for system in systems:
            assert len(system.setup_steps) >= 3

    def test_system_has_inventory_fields(self):
        systems = get_supported_systems()
        for system in systems:
            assert "sku" in system.inventory_fields
            assert "qty_on_hand" in system.inventory_fields

    def test_system_to_dict(self):
        import json
        systems = get_supported_systems()
        for system in systems:
            data = system.to_dict()
            json_str = json.dumps(data)
            assert len(json_str) > 20

    def test_pos_system_display_names(self):
        assert PosSystem.SQUARE.display_name == "Square POS"
        assert PosSystem.LIGHTSPEED.display_name == "Lightspeed Retail"
        assert PosSystem.CLOVER.display_name == "Clover POS"
        assert PosSystem.SHOPIFY.display_name == "Shopify POS"


# ---------------------------------------------------------------------------
# Connection CRUD tests
# ---------------------------------------------------------------------------


class TestConnectionCrud:
    def test_create_connection(self):
        conn = create_pos_connection(
            user_id="user-1",
            pos_system="square",
            store_name="My Hardware Store",
            sync_frequency="daily",
        )
        assert conn.connection_id.startswith("pos_")
        assert conn.user_id == "user-1"
        assert conn.pos_system == PosSystem.SQUARE
        assert conn.store_name == "My Hardware Store"
        assert conn.status == ConnectionStatus.CONNECTED
        assert conn.sync_frequency == SyncFrequency.DAILY

    def test_create_with_location(self):
        conn = create_pos_connection(
            user_id="user-1",
            pos_system="shopify",
            store_name="My Store",
            location_id="loc-123",
        )
        assert conn.location_id == "loc-123"

    def test_list_connections(self):
        create_pos_connection("user-1", "square", "Store A")
        create_pos_connection("user-1", "shopify", "Store B")
        create_pos_connection("user-2", "clover", "Other Store")

        conns = list_pos_connections("user-1")
        assert len(conns) == 2
        names = {c.store_name for c in conns}
        assert names == {"Store A", "Store B"}

    def test_list_empty(self):
        conns = list_pos_connections("nonexistent")
        assert conns == []

    def test_get_connection(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        retrieved = get_pos_connection(conn.connection_id, "user-1")
        assert retrieved is not None
        assert retrieved.store_name == "My Store"

    def test_get_connection_wrong_user(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        retrieved = get_pos_connection(conn.connection_id, "user-2")
        assert retrieved is None

    def test_get_connection_not_found(self):
        retrieved = get_pos_connection("pos_9999", "user-1")
        assert retrieved is None

    def test_disconnect(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        result = disconnect_pos(conn.connection_id, "user-1")
        assert result is True
        assert conn.status == ConnectionStatus.DISCONNECTED

    def test_disconnect_wrong_user(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        result = disconnect_pos(conn.connection_id, "user-2")
        assert result is False

    def test_delete_connection(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        result = delete_pos_connection(conn.connection_id, "user-1")
        assert result is True

        # Should be gone
        retrieved = get_pos_connection(conn.connection_id, "user-1")
        assert retrieved is None

    def test_delete_wrong_user(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        result = delete_pos_connection(conn.connection_id, "user-2")
        assert result is False

    def test_connection_to_dict(self):
        import json
        conn = create_pos_connection("user-1", "square", "My Store")
        data = conn.to_dict()
        assert data["pos_system"] == "square"
        assert data["pos_system_display"] == "Square POS"
        json_str = json.dumps(data)
        assert len(json_str) > 20


# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------


class TestSync:
    def test_trigger_sync_success(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        result = trigger_sync(conn.connection_id, "user-1")
        assert result.success is True
        assert result.connection_id == conn.connection_id

    def test_trigger_sync_not_found(self):
        result = trigger_sync("pos_9999", "user-1")
        assert result.success is False
        assert "not found" in result.errors[0].lower()

    def test_trigger_sync_disconnected(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        disconnect_pos(conn.connection_id, "user-1")
        result = trigger_sync(conn.connection_id, "user-1")
        assert result.success is False
        assert "disconnected" in result.errors[0].lower()

    def test_sync_result_to_dict(self):
        import json
        conn = create_pos_connection("user-1", "square", "My Store")
        result = trigger_sync(conn.connection_id, "user-1")
        data = result.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 20

    def test_sync_updates_last_sync(self):
        conn = create_pos_connection("user-1", "square", "My Store")
        assert conn.last_sync_at is None

        trigger_sync(conn.connection_id, "user-1")
        assert conn.last_sync_at is not None
        assert conn.last_sync_status == "success"


# ---------------------------------------------------------------------------
# API endpoint integration tests
# ---------------------------------------------------------------------------


class TestPosEndpoints:
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

    def test_supported_systems_endpoint(self, client):
        resp = client.get("/api/v1/pos/systems")
        assert resp.status_code == 200
        data = resp.json()
        assert "systems" in data
        assert len(data["systems"]) == 4

    def test_create_connection_endpoint(self, client):
        resp = client.post(
            "/api/v1/pos/connections",
            json={
                "pos_system": "square",
                "store_name": "My Store",
                "sync_frequency": "daily",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pos_system"] == "square"
        assert data["status"] == "connected"

    def test_list_connections_endpoint(self, client):
        # Create one first
        client.post(
            "/api/v1/pos/connections",
            json={"pos_system": "square", "store_name": "Test"},
        )
        resp = client.get("/api/v1/pos/connections")
        assert resp.status_code == 200
        data = resp.json()
        assert "connections" in data

    def test_trigger_sync_endpoint(self, client):
        create_resp = client.post(
            "/api/v1/pos/connections",
            json={"pos_system": "shopify", "store_name": "Shopify Store"},
        )
        conn_id = create_resp.json()["connection_id"]

        resp = client.post(f"/api/v1/pos/connections/{conn_id}/sync")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_disconnect_endpoint(self, client):
        create_resp = client.post(
            "/api/v1/pos/connections",
            json={"pos_system": "clover", "store_name": "Clover Store"},
        )
        conn_id = create_resp.json()["connection_id"]

        resp = client.post(f"/api/v1/pos/connections/{conn_id}/disconnect")
        assert resp.status_code == 200

    def test_delete_connection_endpoint(self, client):
        create_resp = client.post(
            "/api/v1/pos/connections",
            json={"pos_system": "lightspeed", "store_name": "LS Store"},
        )
        conn_id = create_resp.json()["connection_id"]

        resp = client.delete(f"/api/v1/pos/connections/{conn_id}")
        assert resp.status_code == 200
