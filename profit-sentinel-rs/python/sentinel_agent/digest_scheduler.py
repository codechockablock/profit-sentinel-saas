"""Digest Email Scheduler.

In-process scheduler that sends morning digest emails at configured times.
Uses asyncio tasks — no external dependencies (no Celery, no Redis).

Usage:
    scheduler = DigestScheduler(settings, generator)
    scheduler.start()  # Non-blocking — spawns background task
    scheduler.stop()
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timezone
from typing import Any

from .email_service import send_digest_email
from .llm_layer import render_digest

logger = logging.getLogger("sentinel.scheduler")

# In-memory subscription store (production would use Supabase)
_subscriptions: dict[str, dict[str, Any]] = {}
# Key: email, Value: {email, stores, enabled, send_hour, timezone}


def add_subscription(
    email: str,
    *,
    stores: list[str] | None = None,
    send_hour: int = 6,
    tz: str = "America/New_York",
) -> dict:
    """Add or update a digest subscription."""
    sub = {
        "email": email,
        "stores": stores or [],
        "enabled": True,
        "send_hour": send_hour,
        "timezone": tz,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _subscriptions[email] = sub
    logger.info("Subscription added/updated: %s", email)
    return sub


def remove_subscription(email: str) -> bool:
    """Remove a digest subscription."""
    if email in _subscriptions:
        del _subscriptions[email]
        logger.info("Subscription removed: %s", email)
        return True
    return False


def get_subscription(email: str) -> dict | None:
    """Get a single subscription."""
    return _subscriptions.get(email)


def list_subscriptions() -> list[dict]:
    """List all active subscriptions."""
    return [s for s in _subscriptions.values() if s.get("enabled")]


def pause_subscription(email: str) -> bool:
    """Pause a subscription without deleting it."""
    sub = _subscriptions.get(email)
    if sub:
        sub["enabled"] = False
        return True
    return False


def resume_subscription(email: str) -> bool:
    """Resume a paused subscription."""
    sub = _subscriptions.get(email)
    if sub:
        sub["enabled"] = True
        return True
    return False


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
    ):
        self.resend_api_key = resend_api_key
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
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # Reset sent tracker on day change
        if today != self._last_date:
            self._sent_today.clear()
            self._last_date = today

        for sub in list_subscriptions():
            email = sub["email"]
            if email in self._sent_today:
                continue

            # Check if it's the right hour (UTC-based for simplicity)
            send_hour = sub.get("send_hour", 6)
            # Simple offset: Eastern = UTC-5 (approximate)
            tz_offsets = {
                "America/New_York": -5,
                "America/Chicago": -6,
                "America/Denver": -7,
                "America/Los_Angeles": -8,
                "UTC": 0,
            }
            offset = tz_offsets.get(sub.get("timezone", "America/New_York"), -5)
            local_hour = (now.hour + offset) % 24

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

            # Render plain text
            plain_text = render_digest(digest)

            # Build issue dicts for HTML template
            issues = []
            for issue in digest.issues[:10]:
                issues.append({
                    "issue_type": issue.issue_type.value
                    if hasattr(issue.issue_type, "value")
                    else str(issue.issue_type),
                    "store_id": issue.store_id,
                    "dollar_impact": issue.dollar_impact,
                    "priority_score": issue.priority_score,
                    "trend_direction": issue.trend_direction.value
                    if hasattr(issue.trend_direction, "value")
                    else str(issue.trend_direction),
                    "root_cause": (
                        issue.root_cause.value
                        if issue.root_cause and hasattr(issue.root_cause, "value")
                        else str(issue.root_cause or "")
                    ),
                    "skus": [{"sku_id": s.sku_id} for s in issue.skus],
                })

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

            logger.info("Digest sent to %s (%d issues)", email, digest.summary.total_issues)

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

        plain_text = render_digest(digest)

        issues = []
        for issue in digest.issues[:10]:
            issues.append({
                "issue_type": issue.issue_type.value
                if hasattr(issue.issue_type, "value")
                else str(issue.issue_type),
                "store_id": issue.store_id,
                "dollar_impact": issue.dollar_impact,
                "priority_score": issue.priority_score,
                "trend_direction": issue.trend_direction.value
                if hasattr(issue.trend_direction, "value")
                else str(issue.trend_direction),
                "root_cause": (
                    issue.root_cause.value
                    if issue.root_cause and hasattr(issue.root_cause, "value")
                    else str(issue.root_cause or "")
                ),
                "skus": [{"sku_id": s.sku_id} for s in issue.skus],
            })

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
