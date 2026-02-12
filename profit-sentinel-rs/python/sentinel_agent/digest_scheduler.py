"""Digest Email Scheduler.

In-process scheduler that sends morning digest emails at configured times.
Uses asyncio tasks — no external dependencies (no Celery, no Redis).

Subscription storage is pluggable:
    - InMemoryStore  for dev/testing (default)
    - SupabaseStore  for production (when SUPABASE_URL + key are set)

Usage:
    store = create_store(supabase_url, service_key)
    init_subscription_store(store)
    scheduler = DigestScheduler(resend_api_key=..., generator=..., csv_path=...)
    scheduler.start()  # Non-blocking — spawns background task
    scheduler.stop()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .email_service import send_digest_email
from .llm_layer import narrate_digest, render_digest
from .subscription_store import InMemoryStore, SubscriptionStore, create_store

logger = logging.getLogger("sentinel.scheduler")

# ---------------------------------------------------------------------------
# Global store — initialized at app startup via init_subscription_store()
# ---------------------------------------------------------------------------

_store: SubscriptionStore = InMemoryStore()


def init_subscription_store(store: SubscriptionStore) -> None:
    """Set the global subscription store (called once at app startup)."""
    global _store
    _store = store
    logger.info("Subscription store initialized: %s", type(store).__name__)


# ---------------------------------------------------------------------------
# Module-level CRUD functions (delegate to the global store)
# ---------------------------------------------------------------------------
# Kept as thin wrappers for backward compatibility — sidecar.py imports these.


def add_subscription(
    email: str,
    *,
    stores: list[str] | None = None,
    send_hour: int = 6,
    tz: str = "America/New_York",
) -> dict:
    """Add or update a digest subscription."""
    return _store.add(email, stores=stores, send_hour=send_hour, tz=tz)


def remove_subscription(email: str) -> bool:
    """Remove a digest subscription."""
    return _store.remove(email)


def get_subscription(email: str) -> dict | None:
    """Get a single subscription."""
    return _store.get(email)


def list_subscriptions() -> list[dict]:
    """List all active subscriptions."""
    return _store.list_active()


def pause_subscription(email: str) -> bool:
    """Pause a subscription without deleting it."""
    return _store.pause(email)


def resume_subscription(email: str) -> bool:
    """Resume a paused subscription."""
    return _store.resume(email)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class DigestScheduler:
    """Background scheduler for morning digest emails.

    Checks every 60 seconds if it's time to send digests.
    Sends once per day per subscriber at their configured hour.
    """

    def __init__(
        self,
        *,
        resend_api_key: str,
        generator: Any,  # MorningDigestGenerator
        csv_path: str,
        check_interval: int = 60,
        anthropic_api_key: str = "",
    ):
        self.resend_api_key = resend_api_key
        self.anthropic_api_key = anthropic_api_key
        self.generator = generator
        self.csv_path = csv_path
        self.check_interval = check_interval
        self._task: asyncio.Task | None = None
        self._sent_today: set[str] = set()  # emails already sent today
        self._last_date: str = ""  # track day rollover

    def start(self) -> None:
        """Start the scheduler as a background task."""
        if not self.resend_api_key:
            logger.warning("RESEND_API_KEY not set — digest scheduler disabled")
            return
        if self._task and not self._task.done():
            logger.warning("Scheduler already running")
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Digest scheduler started (interval=%ds)", self.check_interval)

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._task and not self._task.done():
            self._task.cancel()
            logger.info("Digest scheduler stopped")

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while True:
            try:
                await self._check_and_send()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Scheduler error")
            await asyncio.sleep(self.check_interval)

    async def _check_and_send(self) -> None:
        """Check if any subscriptions need sending."""
        now = datetime.now(UTC)
        today = now.strftime("%Y-%m-%d")

        # Reset sent tracker on day change
        if today != self._last_date:
            self._sent_today.clear()
            self._last_date = today

        for sub in list_subscriptions():
            email = sub["email"]
            if email in self._sent_today:
                continue

            # Check if it's the right hour in the subscriber's timezone
            send_hour = sub.get("send_hour", 6)
            try:
                tz_name = sub.get("timezone", "America/New_York")
                local_now = now.astimezone(ZoneInfo(tz_name))
                local_hour = local_now.hour
            except (KeyError, Exception):
                # Fallback to UTC if timezone is invalid
                local_hour = now.hour

            if local_hour == send_hour:
                await self._send_digest_to(sub)
                self._sent_today.add(email)

    async def _send_digest_to(self, sub: dict) -> None:
        """Generate and send a digest to one subscriber."""
        email = sub["email"]
        stores = sub.get("stores") or None

        try:
            logger.info("Generating digest for %s (stores=%s)", email, stores)

            # Generate digest
            digest = self.generator.generate(
                csv_path=self.csv_path,
                stores=stores,
                top_k=10,
            )

            # Render plain text (template — always works)
            plain_text = render_digest(digest)

            # Optional: Claude narration (falls back to template on failure)
            if self.anthropic_api_key:
                try:
                    plain_text = await narrate_digest(
                        digest,
                        plain_text,
                        self.anthropic_api_key,
                    )
                except Exception as e:
                    logger.warning(f"Digest narration failed, using template: {e}")

            # Build issue dicts for HTML template
            issues = []
            for issue in digest.issues[:10]:
                issues.append(
                    {
                        "issue_type": (
                            issue.issue_type.value
                            if hasattr(issue.issue_type, "value")
                            else str(issue.issue_type)
                        ),
                        "store_id": issue.store_id,
                        "dollar_impact": issue.dollar_impact,
                        "priority_score": issue.priority_score,
                        "trend_direction": (
                            issue.trend_direction.value
                            if hasattr(issue.trend_direction, "value")
                            else str(issue.trend_direction)
                        ),
                        "root_cause": (
                            issue.root_cause.value
                            if issue.root_cause and hasattr(issue.root_cause, "value")
                            else str(issue.root_cause or "")
                        ),
                        "skus": [{"sku_id": s.sku_id} for s in issue.skus],
                    }
                )

            # Send
            await send_digest_email(
                api_key=self.resend_api_key,
                to=email,
                plain_text=plain_text,
                issue_count=digest.summary.total_issues,
                total_impact=digest.summary.total_dollar_impact,
                store_count=digest.summary.stores_affected,
                pipeline_ms=digest.pipeline_ms,
                issues=issues,
                generated_at=digest.generated_at,
            )

            logger.info(
                "Digest sent to %s (%d issues)", email, digest.summary.total_issues
            )

        except Exception:
            logger.exception("Failed to send digest to %s", email)

    async def send_now(self, email: str) -> dict:
        """Send a digest immediately to one email (on-demand).

        Returns the Resend response dict.
        """
        sub = get_subscription(email)
        if not sub:
            # Create a temporary subscription for one-off send
            sub = {"email": email, "stores": [], "enabled": False}

        stores = sub.get("stores") or None

        digest = self.generator.generate(
            csv_path=self.csv_path,
            stores=stores,
            top_k=10,
        )

        # Render plain text (template — always works)
        plain_text = render_digest(digest)

        # Optional: Claude narration (falls back to template on failure)
        if self.anthropic_api_key:
            try:
                plain_text = await narrate_digest(
                    digest,
                    plain_text,
                    self.anthropic_api_key,
                )
            except Exception as e:
                logger.warning(f"Digest narration failed, using template: {e}")

        issues = []
        for issue in digest.issues[:10]:
            issues.append(
                {
                    "issue_type": (
                        issue.issue_type.value
                        if hasattr(issue.issue_type, "value")
                        else str(issue.issue_type)
                    ),
                    "store_id": issue.store_id,
                    "dollar_impact": issue.dollar_impact,
                    "priority_score": issue.priority_score,
                    "trend_direction": (
                        issue.trend_direction.value
                        if hasattr(issue.trend_direction, "value")
                        else str(issue.trend_direction)
                    ),
                    "root_cause": (
                        issue.root_cause.value
                        if issue.root_cause and hasattr(issue.root_cause, "value")
                        else str(issue.root_cause or "")
                    ),
                    "skus": [{"sku_id": s.sku_id} for s in issue.skus],
                }
            )

        return await send_digest_email(
            api_key=self.resend_api_key,
            to=email,
            plain_text=plain_text,
            issue_count=digest.summary.total_issues,
            total_impact=digest.summary.total_dollar_impact,
            store_count=digest.summary.stores_affected,
            pipeline_ms=digest.pipeline_ms,
            issues=issues,
            generated_at=digest.generated_at,
        )
