"""Tests for the dual authentication layer.

Covers:
    - Anonymous access to public endpoints (presign, suggest-mapping, analyze)
    - Authenticated access with higher limits
    - Rate limiting (5/hour anon, 100/hour auth)
    - File size limits (10MB anon, 50MB auth)
    - Upgrade prompt in anonymous responses
    - Protected endpoints reject anonymous users
    - UserContext properties
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sentinel_agent.dual_auth import (
    ANON_MAX_FILE_SIZE_MB,
    ANON_RATE_LIMIT,
    AUTH_MAX_FILE_SIZE_MB,
    AUTH_RATE_LIMIT,
    UserContext,
    _rate_limits,
    build_upgrade_prompt,
    check_rate_limit,
    get_client_ip,
)
from sentinel_agent.sidecar import create_app
from sentinel_agent.sidecar_config import SidecarSettings


@pytest.fixture
def dev_settings():
    """Settings with dev mode enabled (acts as authenticated user)."""
    return SidecarSettings(
        sidecar_dev_mode=True,
        s3_bucket_name="test-bucket",
        aws_region="us-east-1",
        sentinel_default_store="default",
        sentinel_top_k=20,
    )


@pytest.fixture
def prod_settings():
    """Settings with dev mode disabled (production auth)."""
    return SidecarSettings(
        sidecar_dev_mode=False,
        s3_bucket_name="test-bucket",
        aws_region="us-east-1",
        sentinel_default_store="default",
        sentinel_top_k=20,
        supabase_url="https://fake.supabase.co",
        supabase_service_key="fake-service-key",
    )


@pytest.fixture
def dev_client(dev_settings):
    """Test client with dev mode (authenticated user)."""
    app = create_app(settings=dev_settings)
    return TestClient(app)


@pytest.fixture
def prod_client(prod_settings):
    """Test client with production mode (no auth = anonymous)."""
    app = create_app(settings=prod_settings)
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_rate_limits():
    """Clear rate limit store between tests."""
    _rate_limits.clear()
    yield
    _rate_limits.clear()


# ---- UserContext unit tests ----


class TestUserContext:
    """Tests for UserContext properties."""

    def test_anonymous_user_properties(self):
        ctx = UserContext(
            user_id="anon_abc123",
            is_authenticated=False,
            ip_address="1.2.3.4",
        )
        assert ctx.rate_limit == ANON_RATE_LIMIT
        assert ctx.max_file_size_mb == ANON_MAX_FILE_SIZE_MB
        assert ctx.s3_prefix.startswith("uploads/anonymous/")
        assert "anon" in repr(ctx)

    def test_authenticated_user_properties(self):
        ctx = UserContext(
            user_id="user-123",
            is_authenticated=True,
            email="test@example.com",
            ip_address="1.2.3.4",
        )
        assert ctx.rate_limit == AUTH_RATE_LIMIT
        assert ctx.max_file_size_mb == AUTH_MAX_FILE_SIZE_MB
        assert ctx.s3_prefix == "uploads/user-123"
        assert "auth" in repr(ctx)

    def test_s3_prefix_anonymous_deterministic(self):
        """Same IP should produce same s3_prefix."""
        ctx1 = UserContext(
            user_id="anon_1", is_authenticated=False, ip_address="10.0.0.1"
        )
        ctx2 = UserContext(
            user_id="anon_2", is_authenticated=False, ip_address="10.0.0.1"
        )
        assert ctx1.s3_prefix == ctx2.s3_prefix

    def test_s3_prefix_anonymous_different_ips(self):
        """Different IPs should produce different s3_prefix."""
        ctx1 = UserContext(
            user_id="anon_1", is_authenticated=False, ip_address="10.0.0.1"
        )
        ctx2 = UserContext(
            user_id="anon_2", is_authenticated=False, ip_address="10.0.0.2"
        )
        assert ctx1.s3_prefix != ctx2.s3_prefix


# ---- Rate limiting unit tests ----


class TestRateLimit:
    """Tests for check_rate_limit."""

    @pytest.mark.asyncio
    async def test_anonymous_rate_limit(self):
        ctx = UserContext(
            user_id="anon_test", is_authenticated=False, ip_address="1.2.3.4"
        )

        # First 5 should succeed
        for _ in range(ANON_RATE_LIMIT):
            await check_rate_limit(ctx)

        # 6th should fail
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(ctx)
        assert exc_info.value.status_code == 429
        assert "Sign up" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_authenticated_rate_limit(self):
        ctx = UserContext(
            user_id="auth_test",
            is_authenticated=True,
            email="test@example.com",
        )

        # 100 should all succeed
        for _ in range(AUTH_RATE_LIMIT):
            await check_rate_limit(ctx)

        # 101st should fail
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await check_rate_limit(ctx)
        assert exc_info.value.status_code == 429
        assert "Sign up" not in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_rate_limit_window_expiry(self):
        """Old entries outside the 1-hour window should be pruned."""
        ctx = UserContext(
            user_id="expiry_test", is_authenticated=False, ip_address="1.2.3.4"
        )

        # Inject old entries (2 hours ago)
        old_time = datetime.now(UTC) - timedelta(hours=2)
        _rate_limits[ctx.user_id] = [old_time] * 10

        # Should succeed because old entries are pruned
        await check_rate_limit(ctx)


# ---- Upgrade prompt ----


class TestUpgradePrompt:
    def test_upgrade_prompt_structure(self):
        prompt = build_upgrade_prompt()
        assert "message" in prompt
        assert "cta" in prompt
        assert "url" in prompt
        assert prompt["url"] == "/signup"


# ---- HTTP integration: anonymous access to public endpoints ----


class TestAnonymousAccess:
    """Anonymous users can access public upload/analysis endpoints."""

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_anonymous(self, mock_s3_factory, prod_client):
        """Anonymous users can get presigned URLs."""
        mock_client = MagicMock()
        mock_client.generate_presigned_post.return_value = {
            "url": "https://s3.amazonaws.com/test-bucket",
            "fields": {"key": "anon/uuid-inventory.csv"},
        }
        mock_s3_factory.return_value = mock_client

        response = prod_client.post(
            "/uploads/presign",
            data={"filenames": ["inventory.csv"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "presigned_urls" in data
        # Should use anonymous prefix
        key = data["presigned_urls"][0]["key"]
        assert key.startswith("uploads/anonymous/")
        # File size limit should be 10MB for anonymous
        assert data["limits"]["max_file_size_mb"] == ANON_MAX_FILE_SIZE_MB

    @patch("sentinel_agent.upload_routes.get_s3_client")
    def test_presign_dev_mode_authenticated(self, mock_s3_factory, dev_client):
        """Dev mode users are treated as authenticated."""
        mock_client = MagicMock()
        mock_client.generate_presigned_post.return_value = {
            "url": "https://s3.example.com",
            "fields": {"key": "url"},
        }
        mock_s3_factory.return_value = mock_client

        response = dev_client.post(
            "/uploads/presign",
            data={"filenames": ["inventory.csv"]},
        )
        assert response.status_code == 200
        data = response.json()
        key = data["presigned_urls"][0]["key"]
        # Dev mode uses "uploads/dev-user" prefix
        assert key.startswith("uploads/dev-user/")
        assert data["limits"]["max_file_size_mb"] == AUTH_MAX_FILE_SIZE_MB


# ---- HTTP integration: protected endpoints reject anonymous ----


class TestProtectedEndpoints:
    """Endpoints requiring auth should return 401 for anonymous users."""

    def test_diagnostic_requires_auth(self, prod_client):
        response = prod_client.post(
            "/api/v1/diagnostic/start",
            json={"items": [], "store_name": "Test"},
        )
        assert response.status_code == 401
        assert "Authentication required" in response.json()["detail"]

    def test_explain_requires_auth(self, prod_client):
        response = prod_client.get("/api/v1/explain/issue-123")
        assert response.status_code == 401

    def test_delegate_requires_auth(self, prod_client):
        response = prod_client.post(
            "/api/v1/delegate",
            json={"issue_id": "x", "assignee": "someone"},
        )
        assert response.status_code == 401

    def test_tasks_requires_auth(self, prod_client):
        response = prod_client.get("/api/v1/tasks")
        assert response.status_code == 401

    def test_coop_requires_auth(self, prod_client):
        response = prod_client.get("/api/v1/coop/store-7")
        assert response.status_code == 401

    def test_vendor_call_requires_auth(self, prod_client):
        response = prod_client.get("/api/v1/vendor-call/issue-123")
        assert response.status_code == 401

    def test_digest_requires_auth(self, prod_client):
        response = prod_client.get("/api/v1/digest")
        assert response.status_code == 401

    def test_backward_chain_requires_auth(self, prod_client):
        response = prod_client.post(
            "/api/v1/explain/issue-123/why",
            json={"goal": "root_cause(Theft)"},
        )
        assert response.status_code == 401


# ---- Public endpoints remain accessible ----


class TestPublicEndpoints:
    """Public info endpoints don't require auth."""

    def test_health_no_auth(self, prod_client):
        response = prod_client.get("/health")
        assert response.status_code == 200

    def test_primitives_no_auth(self, prod_client):
        response = prod_client.get("/analysis/primitives")
        assert response.status_code == 200
        assert response.json()["count"] == 11

    def test_supported_pos_no_auth(self, prod_client):
        response = prod_client.get("/analysis/supported-pos")
        assert response.status_code == 200
        assert response.json()["count"] == 16


# ---- get_client_ip ----


class TestGetClientIP:
    def test_x_forwarded_for(self):
        """Trusts the rightmost IP (ALB-appended), not the spoofable leftmost."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "1.2.3.4, 10.0.0.1"}
        assert get_client_ip(request) == "10.0.0.1"

    def test_direct_client(self):
        request = MagicMock()
        request.headers = {}
        request.client.host = "192.168.1.1"
        assert get_client_ip(request) == "192.168.1.1"

    def test_no_client(self):
        request = MagicMock()
        request.headers = {}
        request.client = None
        assert get_client_ip(request) == "unknown"
