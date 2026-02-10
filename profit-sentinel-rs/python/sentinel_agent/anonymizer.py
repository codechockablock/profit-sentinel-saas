"""Anonymization pipeline for Profit Sentinel.

Takes a full analysis result and produces anonymized, aggregated facts
suitable for permanent storage in the dorian_facts table.

Privacy rules:
  - NO product names, SKU numbers, or store identifiers
  - NO raw quantities or exact dollar amounts per item
  - ONLY category-level patterns, statistical summaries, anomaly counts,
    and impact estimates
  - Each analysis produces ~5-15 facts (one per detected leak type)

This is the knowledge moat: every guest analysis makes the system smarter
while keeping customer data private.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger("sentinel.anonymizer")


# ---------------------------------------------------------------------------
# Category inference from descriptions/SKU patterns
# ---------------------------------------------------------------------------

# Common retail categories inferred from product descriptions
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "lumber": ["lumber", "2x4", "2x6", "4x4", "plywood", "osb", "deck board", "stud"],
    "electrical": ["wire", "outlet", "switch", "breaker", "conduit", "romex", "led", "bulb"],
    "plumbing": ["pipe", "pvc", "fitting", "faucet", "valve", "copper", "drain", "toilet"],
    "hardware": ["screw", "nail", "bolt", "nut", "hinge", "lock", "latch", "anchor"],
    "paint": ["paint", "stain", "primer", "brush", "roller", "caulk", "sealant"],
    "tools": ["drill", "saw", "hammer", "wrench", "socket", "level", "tape measure"],
    "outdoor": ["mower", "trimmer", "hose", "sprinkler", "fertilizer", "mulch", "soil"],
    "safety": ["glove", "safety glass", "mask", "respirator", "ear plug", "hard hat"],
    "concrete": ["concrete", "mortar", "cement", "rebar", "block"],
    "hvac": ["filter", "duct", "thermostat", "furnace", "ac", "refrigerant"],
    "general_merchandise": [],  # Fallback
}


def _infer_category(description: str) -> str:
    """Infer a retail category from a product description."""
    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in desc_lower:
                return category
    return "general_merchandise"


def _infer_industry(descriptions: list[str]) -> str:
    """Infer the store's industry from all product descriptions."""
    category_counts: dict[str, int] = {}
    for desc in descriptions[:200]:  # Sample first 200
        cat = _infer_category(desc)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if not category_counts:
        return "retail"

    top = max(category_counts, key=category_counts.get)  # type: ignore[arg-type]
    # Map top category to industry
    hardware_cats = {"lumber", "electrical", "plumbing", "hardware", "tools", "concrete", "hvac"}
    if top in hardware_cats:
        return "hardware"
    if top in {"outdoor"}:
        return "garden_center"
    if top in {"paint"}:
        return "hardware"
    return "retail"


# ---------------------------------------------------------------------------
# Anonymization: analysis result → dorian_facts rows
# ---------------------------------------------------------------------------

def anonymize_analysis(analysis_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract anonymized facts from a full analysis result.

    Parameters
    ----------
    analysis_result : dict
        The full analysis result from /analysis/analyze.

    Returns
    -------
    list[dict]
        List of fact dicts ready for insertion into dorian_facts.
        Each has: subject, predicate, object, confidence, industry,
        pattern_type, sku_category, metadata (JSONB).
    """
    facts: list[dict[str, Any]] = []
    leaks: dict = analysis_result.get("leaks", {})
    summary: dict = analysis_result.get("summary", {})
    impact: dict = summary.get("estimated_impact", {})
    cause: dict = analysis_result.get("cause_diagnosis", {})

    total_rows = summary.get("total_rows_analyzed", 0)
    total_flagged = summary.get("total_items_flagged", 0)

    # Collect all descriptions for industry inference
    all_descriptions: list[str] = []
    for leak_data in leaks.values():
        for item in leak_data.get("item_details", []):
            desc = item.get("description", "")
            if desc:
                all_descriptions.append(desc)

    industry = _infer_industry(all_descriptions)
    now_iso = datetime.now(timezone.utc).isoformat()

    # ── Per-leak-type facts ──────────────────────────────────────
    for leak_key, leak_data in leaks.items():
        count = leak_data.get("count", 0)
        if count == 0:
            continue

        items = leak_data.get("item_details", [])
        breakdown_impact = impact.get("breakdown", {}).get(leak_key, 0)

        # Compute aggregate stats from items (NO individual item data)
        costs = [it.get("cost", 0) for it in items if it.get("cost", 0) > 0]
        margins = [it.get("margin", 0) for it in items if it.get("margin") is not None]
        scores = [it.get("score", 0) for it in items if it.get("score", 0) > 0]

        # Infer dominant category from item descriptions
        item_categories: dict[str, int] = {}
        for it in items:
            cat = _infer_category(it.get("description", ""))
            item_categories[cat] = item_categories.get(cat, 0) + 1
        dominant_category = (
            max(item_categories, key=item_categories.get)  # type: ignore[arg-type]
            if item_categories else "general_merchandise"
        )

        fact = {
            "subject": f"{industry}_store",
            "predicate": "has_leak_pattern",
            "object": leak_key,
            "confidence": 1.0,
            "industry": industry,
            "pattern_type": leak_key,
            "sku_category": dominant_category,
            "metadata": {
                "item_count": count,
                "flag_rate_pct": round(count / total_rows * 100, 2) if total_rows > 0 else 0,
                "severity": leak_data.get("severity", "low"),
                "impact_estimate_usd": round(breakdown_impact, 2) if breakdown_impact else None,
                "avg_cost": round(sum(costs) / len(costs), 2) if costs else None,
                "avg_margin_pct": round(sum(margins) / len(margins), 2) if margins else None,
                "avg_score": round(sum(scores) / len(scores), 3) if scores else None,
                "category_distribution": {
                    k: v for k, v in sorted(
                        item_categories.items(), key=lambda x: -x[1]
                    )[:5]
                },
                "analyzed_at": now_iso,
            },
        }
        facts.append(fact)

    # ── Overall analysis fact ────────────────────────────────────
    active_count = sum(1 for v in leaks.values() if v.get("count", 0) > 0)
    facts.append({
        "subject": f"{industry}_store",
        "predicate": "analysis_summary",
        "object": f"{active_count}_of_11_leaks_detected",
        "confidence": 1.0,
        "industry": industry,
        "pattern_type": "analysis_summary",
        "sku_category": None,
        "metadata": {
            "total_items_analyzed": total_rows,
            "total_items_flagged": total_flagged,
            "flag_rate_pct": round(total_flagged / total_rows * 100, 2) if total_rows > 0 else 0,
            "active_leak_types": active_count,
            "impact_low_usd": round(impact.get("low_estimate", 0), 2),
            "impact_high_usd": round(impact.get("high_estimate", 0), 2),
            "analysis_time_seconds": summary.get("analysis_time_seconds", 0),
            "analyzed_at": now_iso,
        },
    })

    # ── Root cause fact (if available) ───────────────────────────
    if cause and cause.get("top_cause"):
        facts.append({
            "subject": f"{industry}_store",
            "predicate": "root_cause_detected",
            "object": cause["top_cause"],
            "confidence": cause.get("confidence", 0.5),
            "industry": industry,
            "pattern_type": "root_cause",
            "sku_category": None,
            "metadata": {
                "hypotheses_count": len(cause.get("hypotheses", [])),
                "analyzed_at": now_iso,
            },
        })

    logger.info(
        "Anonymized analysis: %d facts from %d items (%s industry)",
        len(facts), total_rows, industry,
    )
    return facts


# ---------------------------------------------------------------------------
# Persistence: insert anonymized facts into Supabase dorian_facts
# ---------------------------------------------------------------------------

async def store_anonymized_facts(
    facts: list[dict[str, Any]],
    supabase_url: str,
    supabase_service_key: str,
) -> int:
    """Insert anonymized facts into the dorian_facts table via Supabase REST API.

    Uses the service_role key for direct insertion (bypasses RLS).

    Parameters
    ----------
    facts : list[dict]
        Anonymized fact dicts from anonymize_analysis().
    supabase_url : str
        Supabase project URL.
    supabase_service_key : str
        Supabase service_role key.

    Returns
    -------
    int
        Number of facts successfully inserted.
    """
    if not facts:
        return 0

    # Build rows for insertion (without vector — application can backfill later)
    rows = []
    for fact in facts:
        rows.append({
            "subject": fact["subject"],
            "predicate": fact["predicate"],
            "object": fact["object"],
            "confidence": fact.get("confidence", 1.0),
            "industry": fact.get("industry"),
            "pattern_type": fact.get("pattern_type"),
            "sku_category": fact.get("sku_category"),
            "metadata": fact.get("metadata", {}),
            "agent_id": "guest_analysis",
            "domain": "retail",
            "source": "auto_anonymization",
            "status": "active",
        })

    rest_url = f"{supabase_url}/rest/v1/dorian_facts"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                rest_url,
                json=rows,
                headers={
                    "apikey": supabase_service_key,
                    "Authorization": f"Bearer {supabase_service_key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
            resp.raise_for_status()
            logger.info("Stored %d anonymized facts in dorian_facts", len(rows))
            return len(rows)

    except httpx.HTTPStatusError as e:
        logger.error(
            "Failed to store anonymized facts: %s %s",
            e.response.status_code, e.response.text,
        )
        return 0
    except Exception as e:
        logger.error("Failed to store anonymized facts: %s", e, exc_info=True)
        return 0
