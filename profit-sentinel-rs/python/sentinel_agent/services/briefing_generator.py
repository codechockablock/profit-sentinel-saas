"""Agent briefing generation service.

Calls Anthropic Claude to generate executive briefings from structured
business data. Briefings are cached in the agent_briefings table
with a 4-hour TTL and data-hash invalidation.

Cache invalidation triggers:
  - Cache expires (4 hours)
  - New analysis results arrive (data_hash changes)
  - User explicitly refreshes
  - User approves/defers/rejects an action (action queue changed)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

logger = logging.getLogger("sentinel.services.briefing_generator")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

EXECUTIVE_BRIEFING_PROMPT = """You are Profit Sentinel, an AI operations partner for \
multi-store hardware retailers. You generate concise, actionable executive briefings \
from structured inventory analysis data.

Your role: {role} for {org_name}
Scope: Entire business ({total_stores} stores)

Guidelines:
- Lead with the single most important change since the last briefing
- Be direct and specific — this executive checks this multiple times a day
- Include specific dollar amounts for every finding (from the data provided)
- Group findings by theme (vendor issues, dead stock, margin erosion) not by store
- Flag stores in critical status that need immediate attention
- Suggest 3-5 concrete actions with estimated financial impact and confidence
- Keep the narrative under 300 words
- Use plain business language — you're a colleague, not a report generator
- Reference specific store names when recommending actions

After the briefing text, output a JSON block with structured action items:
```json
[{{"type": "transfer|clearance|reorder|price_adjustment|vendor_contact|threshold_change",
   "store_id": "uuid or null for network-wide",
   "description": "what to do",
   "reasoning": "why — include the dollar math",
   "financial_impact": dollar_amount,
   "confidence": 0.0_to_1.0}}]
```"""


# ---------------------------------------------------------------------------
# Briefing generator
# ---------------------------------------------------------------------------


class BriefingGenerator:
    """Generates executive briefings using Anthropic Claude."""

    CACHE_TTL_HOURS = 4

    def __init__(
        self,
        anthropic_api_key: str = "",
        supabase_url: str = "",
        service_key: str = "",
    ) -> None:
        self._api_key = anthropic_api_key
        self._has_api_key = bool(anthropic_api_key)

        # PostgREST client for briefing cache
        self._use_supabase = bool(supabase_url and service_key)
        if self._use_supabase:
            self._base_url = f"{supabase_url}/rest/v1/agent_briefings"
            self._headers = {
                "apikey": service_key,
                "Authorization": f"Bearer {service_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            }

        logger.info(
            "BriefingGenerator initialized (anthropic=%s, supabase=%s)",
            "configured" if self._has_api_key else "not configured",
            "configured" if self._use_supabase else "not configured",
        )

    # -- Cache --

    def compute_data_hash(self, business_data: dict) -> str:
        """Hash the input data to detect when briefing needs refresh."""
        canonical = json.dumps(business_data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def get_cached(
        self, org_id: str, user_id: str, data_hash: str | None = None
    ) -> dict | None:
        """Check for non-expired briefing matching the data hash."""
        if not self._use_supabase:
            return None

        try:
            now = datetime.now(UTC).isoformat()
            params: dict[str, str] = {
                "org_id": f"eq.{org_id}",
                "user_id": f"eq.{user_id}",
                "expires_at": f"gt.{now}",
                "order": "generated_at.desc",
                "limit": "1",
            }
            if data_hash:
                params["data_hash"] = f"eq.{data_hash}"

            resp = httpx.get(
                self._base_url, headers=self._headers, params=params, timeout=10.0
            )
            if resp.status_code == 200 and resp.json():
                row = resp.json()[0]
                return {
                    "briefing": row.get("briefing_text", ""),
                    "action_items": row.get("action_items", []),
                    "generated_at": row.get("generated_at"),
                    "expires_at": row.get("expires_at"),
                }
        except Exception as e:
            logger.warning("Briefing cache lookup failed: %s", e)

        return None

    def _save_cache(
        self,
        org_id: str,
        user_id: str,
        briefing_text: str,
        action_items: list[dict],
        data_hash: str,
    ) -> dict:
        """Save briefing to cache with TTL."""
        now = datetime.now(UTC)
        expires = now + timedelta(hours=self.CACHE_TTL_HOURS)
        record = {
            "org_id": org_id,
            "user_id": user_id,
            "scope_type": "business",
            "briefing_text": briefing_text,
            "action_items": json.dumps(action_items),
            "data_hash": data_hash,
            "generated_at": now.isoformat(),
            "expires_at": expires.isoformat(),
        }

        if self._use_supabase:
            try:
                resp = httpx.post(
                    self._base_url,
                    headers=self._headers,
                    json=record,
                    timeout=10.0,
                )
                if resp.status_code in (200, 201) and resp.json():
                    return resp.json()[0]
            except Exception as e:
                logger.warning("Briefing cache save failed: %s", e)

        return record

    # -- Generation --

    def generate(
        self,
        org_id: str,
        user_id: str,
        role: str,
        org_name: str,
        business_data: dict,
    ) -> dict:
        """Generate a new briefing using Claude.

        Returns:
            {
                "briefing": str,
                "action_items": list[dict],
                "generated_at": str,
                "expires_at": str,
            }
        """
        if not self._has_api_key:
            return {
                "briefing": (
                    "Briefing generation is not configured. "
                    "Set the ANTHROPIC_API_KEY environment variable to enable "
                    "AI-powered executive briefings."
                ),
                "action_items": [],
                "generated_at": datetime.now(UTC).isoformat(),
                "expires_at": (
                    datetime.now(UTC) + timedelta(hours=self.CACHE_TTL_HOURS)
                ).isoformat(),
            }

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=self._api_key)

            system = EXECUTIVE_BRIEFING_PROMPT.format(
                role=role,
                org_name=org_name,
                total_stores=business_data.get("org", {}).get("total_stores", 0),
            )

            context = self._build_context(business_data)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": context}],
            )

            raw_text = response.content[0].text.strip()
            briefing_text, action_items = self._parse_response(raw_text)

            # Cache the result
            data_hash = self.compute_data_hash(business_data)
            cached = self._save_cache(
                org_id, user_id, briefing_text, action_items, data_hash
            )

            return {
                "briefing": briefing_text,
                "action_items": action_items,
                "generated_at": cached.get(
                    "generated_at", datetime.now(UTC).isoformat()
                ),
                "expires_at": cached.get(
                    "expires_at",
                    (
                        datetime.now(UTC) + timedelta(hours=self.CACHE_TTL_HOURS)
                    ).isoformat(),
                ),
            }

        except ImportError:
            logger.error("anthropic package not installed")
            return {
                "briefing": "Briefing generation unavailable — anthropic package not installed.",
                "action_items": [],
                "generated_at": datetime.now(UTC).isoformat(),
                "expires_at": (
                    datetime.now(UTC) + timedelta(hours=self.CACHE_TTL_HOURS)
                ).isoformat(),
            }
        except Exception as e:
            logger.error("Briefing generation failed: %s", e)
            return {
                "briefing": f"Briefing generation encountered an error: {e}",
                "action_items": [],
                "generated_at": datetime.now(UTC).isoformat(),
                "expires_at": (
                    datetime.now(UTC) + timedelta(hours=self.CACHE_TTL_HOURS)
                ).isoformat(),
            }

    def _build_context(self, business_data: dict) -> str:
        """Build the structured context that Claude sees as the user message."""
        org = business_data.get("org", {})
        regions = business_data.get("regions", [])
        unassigned = business_data.get("unassigned_stores", [])
        alerts = business_data.get("network_alerts", [])

        parts = [
            "Generate an executive briefing from this business data:",
            "",
            "NETWORK OVERVIEW:",
            f"  Total stores: {org.get('total_stores', 0)}",
            f"  Total exposure: ${org.get('total_exposure', 0):,.0f}",
            f"  Exposure trend: {org.get('exposure_trend', 0):+.1%}",
            f"  Pending actions: {org.get('total_pending_actions', 0)}",
            f"  Completed actions (30d): {org.get('total_completed_actions_30d', 0)}",
            "",
        ]

        for region in regions:
            parts.append(
                f"REGION: {region['name']} "
                f"({region.get('store_count', 0)} stores, "
                f"${region.get('total_exposure', 0):,.0f} exposure, "
                f"{region.get('exposure_trend', 0):+.1%} trend)"
            )
            for store in region.get("stores", []):
                status_icon = {
                    "healthy": "GREEN",
                    "attention": "YELLOW",
                    "critical": "RED",
                }.get(store.get("status", ""), "?")
                parts.append(
                    f"  [{status_icon}] {store['name']}: "
                    f"${store.get('total_impact', 0):,.0f} exposure, "
                    f"{store.get('flagged_count', 0)} flagged items, "
                    f"{store.get('pending_actions', 0)} pending actions"
                    + (f" — {store['top_issue']}" if store.get("top_issue") else "")
                )
            parts.append("")

        if unassigned:
            parts.append("UNASSIGNED STORES:")
            for store in unassigned:
                parts.append(
                    f"  {store['name']}: ${store.get('total_impact', 0):,.0f} exposure"
                )
            parts.append("")

        if alerts:
            parts.append("NETWORK ALERTS:")
            for alert in alerts:
                parts.append(
                    f"  {alert.get('type', 'alert')}: {alert.get('description', '')}"
                )
            parts.append("")

        return "\n".join(parts)

    def _parse_response(self, text: str) -> tuple[str, list[dict]]:
        """Split Claude's response into narrative text and JSON action items."""
        # Look for ```json ... ``` block
        json_pattern = r"```json\s*\n?(.*?)```"
        match = re.search(json_pattern, text, re.DOTALL)

        action_items: list[dict] = []
        briefing_text = text

        if match:
            json_str = match.group(1).strip()
            # Remove the JSON block from the narrative
            briefing_text = text[: match.start()].strip()

            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    action_items = parsed
                elif isinstance(parsed, dict) and "actions" in parsed:
                    action_items = parsed["actions"]
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse action items JSON: %s", e)

        # Validate action items have required fields
        validated_items = []
        for item in action_items:
            if isinstance(item, dict) and item.get("description"):
                validated_items.append(
                    {
                        "type": item.get("type", "custom"),
                        "store_id": item.get("store_id"),
                        "description": item["description"],
                        "reasoning": item.get("reasoning", ""),
                        "financial_impact": float(item.get("financial_impact", 0)),
                        "confidence": float(item.get("confidence", 0.5)),
                    }
                )

        return briefing_text, validated_items
