"""Tests for the digest scheduler module.

Covers subscription CRUD, scheduler lifecycle, and timezone handling.
"""

import asyncio
from datetime import UTC, datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sentinel_agent.digest_scheduler import (
    DigestScheduler,
    add_subscription,
    get_subscription,
    init_subscription_store,
    list_subscriptions,
    pause_subscription,
    remove_subscription,
    resume_subscription,
)
from sentinel_agent.subscription_store import InMemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_store():
    """Reset to a fresh InMemoryStore before each test."""
    store = InMemoryStore()
    init_subscription_store(store)
    yield
    init_subscription_store(InMemoryStore())


# ---------------------------------------------------------------------------
# Subscription CRUD
# ---------------------------------------------------------------------------


class TestSubscriptionCRUD:
    def test_add_subscription(self):
        sub = add_subscription("owner@store.com")
        assert sub["email"] == "owner@store.com"
        assert sub["enabled"] is True
        assert sub["send_hour"] == 6
        assert sub["timezone"] == "America/New_York"
        assert sub["stores"] == []
        assert "created_at" in sub

    def test_add_subscription_with_options(self):
        sub = add_subscription(
            "mgr@store.com",
            stores=["store-7", "store-12"],
            send_hour=7,
            tz="America/Chicago",
        )
        assert sub["stores"] == ["store-7", "store-12"]
        assert sub["send_hour"] == 7
        assert sub["timezone"] == "America/Chicago"

    def test_update_subscription(self):
        add_subscription("test@example.com", send_hour=6)
        updated = add_subscription("test@example.com", send_hour=8)
        assert updated["send_hour"] == 8
        # Should still be one subscription, not two
        all_subs = list_subscriptions()
        assert len(all_subs) == 1

    def test_get_subscription(self):
        add_subscription("get@test.com")
        sub = get_subscription("get@test.com")
        assert sub is not None
        assert sub["email"] == "get@test.com"

    def test_get_nonexistent(self):
        sub = get_subscription("missing@test.com")
        assert sub is None

    def test_remove_subscription(self):
        add_subscription("remove@test.com")
        assert remove_subscription("remove@test.com") is True
        assert get_subscription("remove@test.com") is None

    def test_remove_nonexistent(self):
        assert remove_subscription("missing@test.com") is False

    def test_list_subscriptions_only_enabled(self):
        add_subscription("a@test.com")
        add_subscription("b@test.com")
        add_subscription("c@test.com")
        pause_subscription("b@test.com")

        active = list_subscriptions()
        assert len(active) == 2
        emails = {s["email"] for s in active}
        assert "a@test.com" in emails
        assert "c@test.com" in emails
        assert "b@test.com" not in emails

    def test_pause_subscription(self):
        add_subscription("pause@test.com")
        assert pause_subscription("pause@test.com") is True
        sub = get_subscription("pause@test.com")
        assert sub["enabled"] is False

    def test_pause_nonexistent(self):
        assert pause_subscription("missing@test.com") is False

    def test_resume_subscription(self):
        add_subscription("resume@test.com")
        pause_subscription("resume@test.com")
        assert resume_subscription("resume@test.com") is True
        sub = get_subscription("resume@test.com")
        assert sub["enabled"] is True

    def test_resume_nonexistent(self):
        assert resume_subscription("missing@test.com") is False

    def test_list_empty(self):
        assert list_subscriptions() == []


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    def test_init(self):
        scheduler = DigestScheduler(
            resend_api_key="re_test",
            generator=MagicMock(),
            csv_path="test.csv",
        )
        assert scheduler.resend_api_key == "re_test"
        assert scheduler.csv_path == "test.csv"
        assert scheduler.is_running is False

    def test_start_without_api_key(self):
        scheduler = DigestScheduler(
            resend_api_key="",
            generator=MagicMock(),
            csv_path="test.csv",
        )
        scheduler.start()
        assert scheduler.is_running is False  # Should not start without key

    def test_start_with_api_key(self):
        scheduler = DigestScheduler(
            resend_api_key="re_test",
            generator=MagicMock(),
            csv_path="test.csv",
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._start_and_stop(scheduler))
        finally:
            loop.close()

    async def _start_and_stop(self, scheduler):
        scheduler.start()
        assert scheduler.is_running is True
        scheduler.stop()
        # Give the task a moment to cancel
        await asyncio.sleep(0.1)

    def test_stop_when_not_running(self):
        scheduler = DigestScheduler(
            resend_api_key="re_test",
            generator=MagicMock(),
            csv_path="test.csv",
        )
        # Should not raise
        scheduler.stop()

    def test_day_rollover_clears_sent_set(self):
        scheduler = DigestScheduler(
            resend_api_key="re_test",
            generator=MagicMock(),
            csv_path="test.csv",
        )
        scheduler._sent_today.add("test@example.com")
        scheduler._last_date = "2026-01-01"

        # Simulate a new day
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._check_rollover(scheduler))
        finally:
            loop.close()

    async def _check_rollover(self, scheduler):
        # _check_and_send will see today != _last_date and clear
        await scheduler._check_and_send()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert scheduler._last_date == today
        # sent_today was cleared (old entry removed)
        assert "test@example.com" not in scheduler._sent_today


# ---------------------------------------------------------------------------
# Scheduler send_now
# ---------------------------------------------------------------------------


class TestSchedulerSendNow:
    @pytest.mark.asyncio
    async def test_send_now_with_subscription(self):
        add_subscription("now@test.com", stores=["store-7"])

        mock_generator = MagicMock()
        mock_digest = MagicMock()
        mock_digest.issues = []
        mock_digest.summary.total_issues = 0
        mock_digest.summary.total_dollar_impact = 0
        mock_digest.summary.stores_affected = 0
        mock_digest.pipeline_ms = 10
        mock_digest.generated_at = "2026-02-06"
        mock_generator.generate.return_value = mock_digest

        scheduler = DigestScheduler(
            resend_api_key="re_test",
            generator=mock_generator,
            csv_path="test.csv",
        )

        with patch(
            "sentinel_agent.digest_scheduler.send_digest_email",
            new_callable=AsyncMock,
        ) as mock_send:
            mock_send.return_value = {"id": "email-now"}
            result = await scheduler.send_now("now@test.com")

            assert result["id"] == "email-now"
            mock_send.assert_called_once()
            # Verify it used the subscriber's stores
            mock_generator.generate.assert_called_once_with(
                csv_path="test.csv",
                stores=["store-7"],
                top_k=10,
            )

    @pytest.mark.asyncio
    async def test_send_now_without_subscription(self):
        """Send now for an email not subscribed — should still work."""
        mock_generator = MagicMock()
        mock_digest = MagicMock()
        mock_digest.issues = []
        mock_digest.summary.total_issues = 0
        mock_digest.summary.total_dollar_impact = 0
        mock_digest.summary.stores_affected = 0
        mock_digest.pipeline_ms = 5
        mock_digest.generated_at = "2026-02-06"
        mock_generator.generate.return_value = mock_digest

        scheduler = DigestScheduler(
            resend_api_key="re_test",
            generator=mock_generator,
            csv_path="test.csv",
        )

        with patch(
            "sentinel_agent.digest_scheduler.send_digest_email",
            new_callable=AsyncMock,
        ) as mock_send:
            mock_send.return_value = {"id": "email-adhoc"}
            result = await scheduler.send_now("adhoc@test.com")

            assert result["id"] == "email-adhoc"
            # Should generate with no store filter
            mock_generator.generate.assert_called_once_with(
                csv_path="test.csv",
                stores=None,
                top_k=10,
            )


# ---------------------------------------------------------------------------
# Sidecar endpoint integration
# ---------------------------------------------------------------------------


class TestDigestEndpoints:
    """Integration tests for digest email endpoints via TestClient."""

    @pytest.fixture
    def client(self):
        """Create a test client with dev mode enabled (no auth)."""
        from fastapi.testclient import TestClient
        from sentinel_agent.sidecar import create_app
        from sentinel_agent.sidecar_config import SidecarSettings

        settings = SidecarSettings(
            sidecar_dev_mode=True,
            resend_api_key="re_test_key",
            digest_email_enabled=False,
            supabase_url="",
            supabase_service_key="",
        )
        app = create_app(settings)
        return TestClient(app)

    def test_subscribe(self, client):
        resp = client.post(
            "/api/v1/digest/subscribe",
            json={
                "email": "test@store.com",
                "stores": ["store-7"],
                "send_hour": 7,
                "timezone": "America/Chicago",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["subscription"]["email"] == "test@store.com"
        assert data["message"] == "Subscription created"

    def test_list_subscriptions(self, client):
        # Subscribe first
        client.post(
            "/api/v1/digest/subscribe",
            json={"email": "list@test.com"},
        )
        resp = client.get("/api/v1/digest/subscriptions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_unsubscribe(self, client):
        client.post(
            "/api/v1/digest/subscribe",
            json={"email": "unsub@test.com"},
        )
        resp = client.delete("/api/v1/digest/subscribe/unsub@test.com")
        assert resp.status_code == 200
        assert "Unsubscribed" in resp.json()["message"]

    def test_unsubscribe_nonexistent(self, client):
        resp = client.delete("/api/v1/digest/subscribe/missing@test.com")
        assert resp.status_code == 404

    def test_scheduler_status(self, client):
        resp = client.get("/api/v1/digest/scheduler-status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False  # We set it to False in fixture
        assert data["running"] is False
        assert "subscribers" in data
        assert "send_hour" in data

    def test_send_digest_now_without_engine(self, client):
        """Send now will fail because no engine is available in tests."""
        resp = client.post(
            "/api/v1/digest/send",
            json={"email": "send@test.com"},
        )
        # Should get 502 because the pipeline can't run without the binary
        assert resp.status_code == 502
