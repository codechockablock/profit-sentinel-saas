"""Tests for Enterprise API Key Management.

Covers:
    - Key generation and format
    - Key validation (valid, invalid, revoked)
    - Rate limiting (hourly, daily)
    - Tier-based limits (free, pro, enterprise)
    - Key listing and revocation
    - Usage statistics
    - API endpoint integration
"""

import pytest
from sentinel_agent.api_keys import (
    TIER_LIMITS,
    ApiKeyRecord,
    ApiKeyValidation,
    ApiTier,
    InMemoryApiKeyStore,
    create_api_key,
    get_key_usage,
    init_api_key_store,
    list_api_keys,
    revoke_api_key,
    validate_api_key,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_store():
    """Reset API key store before each test."""
    store = InMemoryApiKeyStore()
    init_api_key_store(store)
    yield store


# ---------------------------------------------------------------------------
# Key generation tests
# ---------------------------------------------------------------------------


class TestKeyGeneration:
    def test_create_live_key(self):
        plaintext, record = create_api_key("user-1", name="My Key")
        assert plaintext.startswith("ps_live_")
        assert len(plaintext) > 20
        assert record.key_id.startswith("key_")
        assert record.user_id == "user-1"
        assert record.name == "My Key"
        assert record.is_active is True
        assert record.is_test is False

    def test_create_test_key(self):
        plaintext, record = create_api_key("user-1", test=True)
        assert plaintext.startswith("ps_test_")
        assert record.is_test is True

    def test_keys_are_unique(self):
        key1, _ = create_api_key("user-1")
        key2, _ = create_api_key("user-1")
        assert key1 != key2

    def test_default_tier_is_free(self):
        _, record = create_api_key("user-1")
        assert record.tier == ApiTier.FREE

    def test_enterprise_tier(self):
        _, record = create_api_key("user-1", tier=ApiTier.ENTERPRISE)
        assert record.tier == ApiTier.ENTERPRISE


# ---------------------------------------------------------------------------
# Key validation tests
# ---------------------------------------------------------------------------


class TestKeyValidation:
    def test_valid_key(self):
        plaintext, _ = create_api_key("user-1")
        result = validate_api_key(plaintext)
        assert result.is_valid is True
        assert result.user_id == "user-1"
        assert result.tier == ApiTier.FREE
        assert result.error is None

    def test_invalid_key(self):
        result = validate_api_key("ps_live_invalid_key_123")
        assert result.is_valid is False
        assert result.error == "Invalid API key"

    def test_revoked_key(self):
        plaintext, record = create_api_key("user-1")
        revoke_api_key(record.key_id, "user-1")
        result = validate_api_key(plaintext)
        assert result.is_valid is False
        assert "revoked" in result.error.lower()

    def test_validation_increments_usage(self):
        plaintext, record = create_api_key("user-1")
        assert record.usage_count == 0

        validate_api_key(plaintext)
        assert record.usage_count == 1

        validate_api_key(plaintext)
        assert record.usage_count == 2

    def test_validation_updates_last_used(self):
        plaintext, record = create_api_key("user-1")
        assert record.last_used_at is None

        validate_api_key(plaintext)
        assert record.last_used_at is not None

    def test_validation_returns_limits(self):
        plaintext, _ = create_api_key("user-1", tier=ApiTier.PRO)
        result = validate_api_key(plaintext)
        assert result.limits is not None
        assert result.limits.requests_per_hour == 100
        assert result.limits.requests_per_day == 2000


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_free_tier_hourly_limit(self):
        plaintext, _ = create_api_key("user-1", tier=ApiTier.FREE)
        limit = TIER_LIMITS[ApiTier.FREE].requests_per_hour

        # Make limit requests
        for _ in range(limit):
            result = validate_api_key(plaintext)
            assert result.is_valid is True

        # Next one should fail
        result = validate_api_key(plaintext)
        assert result.is_valid is False
        assert "rate limit" in result.error.lower()

    def test_pro_tier_higher_limit(self):
        plaintext, _ = create_api_key("user-1", tier=ApiTier.PRO)
        TIER_LIMITS[ApiTier.PRO].requests_per_hour

        # Make 50 requests (well under pro limit of 100)
        for _ in range(50):
            result = validate_api_key(plaintext)
            assert result.is_valid is True

    def test_enterprise_tier_highest_limit(self):
        plaintext, _ = create_api_key("user-1", tier=ApiTier.ENTERPRISE)
        # Enterprise has 1000/hour — just verify it works
        for _ in range(100):
            result = validate_api_key(plaintext)
            assert result.is_valid is True

    def test_rate_limit_error_shows_tier(self):
        plaintext, _ = create_api_key("user-1", tier=ApiTier.FREE)
        limit = TIER_LIMITS[ApiTier.FREE].requests_per_hour

        for _ in range(limit):
            validate_api_key(plaintext)

        result = validate_api_key(plaintext)
        assert "free" in result.error.lower()


# ---------------------------------------------------------------------------
# Key management tests
# ---------------------------------------------------------------------------


class TestKeyManagement:
    def test_list_keys(self):
        create_api_key("user-1", name="Key A")
        create_api_key("user-1", name="Key B")
        create_api_key("user-2", name="Other User")

        keys = list_api_keys("user-1")
        assert len(keys) == 2
        names = {k.name for k in keys}
        assert names == {"Key A", "Key B"}

    def test_list_keys_empty(self):
        keys = list_api_keys("nonexistent")
        assert keys == []

    def test_revoke_key(self):
        _, record = create_api_key("user-1")
        assert record.is_active is True

        result = revoke_api_key(record.key_id, "user-1")
        assert result is True
        assert record.is_active is False

    def test_revoke_nonexistent(self):
        result = revoke_api_key("key_nonexistent", "user-1")
        assert result is False

    def test_revoke_wrong_user(self):
        _, record = create_api_key("user-1")
        result = revoke_api_key(record.key_id, "user-2")
        assert result is False

    def test_usage_stats(self):
        plaintext, record = create_api_key("user-1")

        # Make some requests
        for _ in range(5):
            validate_api_key(plaintext)

        stats = get_key_usage(record.key_id, "user-1")
        assert stats is not None
        assert stats["usage_count_total"] == 5
        assert stats["usage_last_hour"] == 5
        assert (
            stats["remaining_hourly"] == TIER_LIMITS[ApiTier.FREE].requests_per_hour - 5
        )

    def test_usage_stats_not_found(self):
        stats = get_key_usage("key_nonexistent", "user-1")
        assert stats is None


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_record_to_dict(self):
        import json

        _, record = create_api_key("user-1", tier=ApiTier.PRO, name="Test")
        data = record.to_dict()
        assert data["tier"] == "pro"
        assert data["name"] == "Test"
        assert "limits" in data
        json_str = json.dumps(data)
        assert len(json_str) > 20

    def test_validation_to_dict(self):
        import json

        plaintext, _ = create_api_key("user-1")
        result = validate_api_key(plaintext)
        data = result.to_dict()
        assert data["is_valid"] is True
        json_str = json.dumps(data)
        assert len(json_str) > 10

    def test_tier_limits_to_dict(self):
        for tier in ApiTier:
            limits = TIER_LIMITS[tier]
            data = limits.to_dict()
            assert "requests_per_hour" in data
            assert "requests_per_day" in data


# ---------------------------------------------------------------------------
# API endpoint integration tests
# ---------------------------------------------------------------------------


class TestApiKeyEndpoints:
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

    def test_create_key_endpoint(self, client):
        resp = client.post(
            "/api/v1/api-keys",
            json={"name": "My Integration", "tier": "free"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "key" in data
        assert data["key"].startswith("ps_live_")
        assert "record" in data

    def test_list_keys_endpoint(self, client):
        # Create a key first
        client.post(
            "/api/v1/api-keys",
            json={"name": "Test Key"},
        )

        resp = client.get("/api/v1/api-keys")
        assert resp.status_code == 200
        data = resp.json()
        assert "keys" in data
        assert len(data["keys"]) >= 1

    def test_revoke_key_endpoint(self, client):
        # Create then revoke
        create_resp = client.post(
            "/api/v1/api-keys",
            json={"name": "To Revoke"},
        )
        key_id = create_resp.json()["record"]["key_id"]

        resp = client.delete(f"/api/v1/api-keys/{key_id}")
        assert resp.status_code == 200

    def test_usage_stats_endpoint(self, client):
        create_resp = client.post(
            "/api/v1/api-keys",
            json={"name": "Stats Key"},
        )
        key_id = create_resp.json()["record"]["key_id"]

        resp = client.get(f"/api/v1/api-keys/{key_id}/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert "usage_count_total" in data
