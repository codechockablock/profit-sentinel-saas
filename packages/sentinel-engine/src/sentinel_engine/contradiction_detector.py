"""
Contradiction Detector - Identifies conflicting anomaly classifications.

Detects logical contradictions such as:
- SKU flagged as both low_stock AND overstock
- SKU flagged as both dead_item AND high_velocity
- SKU flagged as both negative_inventory AND overstock

These contradictions indicate either:
1. Data quality issues
2. Detection threshold misconfiguration
3. Edge cases requiring manual review
"""

from typing import Dict, Set, List, Tuple
from dataclasses import dataclass


@dataclass
class Contradiction:
    """A detected contradiction between two primitives."""
    sku: str
    primitive_a: str
    primitive_b: str
    reason: str


# Define contradictory primitive pairs
CONTRADICTORY_PAIRS = [
    ("low_stock", "overstock", "Cannot have both low stock and overstock"),
    ("dead_item", "high_velocity", "Cannot be dead and high velocity"),
    ("negative_inventory", "overstock", "Cannot have negative inventory and overstock"),
    ("negative_inventory", "low_stock", "Negative inventory is more severe than low stock"),
]


def detect_contradictions(
    detections: Dict[str, Set[str]]
) -> Tuple[List[Contradiction], Dict[str, int]]:
    """
    Detect contradictory anomaly classifications.

    Args:
        detections: Dict mapping primitive name to set of detected SKUs

    Returns:
        Tuple of (list of contradictions, summary counts by type)
    """
    contradictions = []
    summary = {}

    for prim_a, prim_b, reason in CONTRADICTORY_PAIRS:
        skus_a = detections.get(prim_a, set())
        skus_b = detections.get(prim_b, set())

        # Find SKUs in both sets
        overlapping = skus_a & skus_b

        if overlapping:
            key = f"{prim_a}_vs_{prim_b}"
            summary[key] = len(overlapping)

            for sku in overlapping:
                contradictions.append(Contradiction(
                    sku=sku,
                    primitive_a=prim_a,
                    primitive_b=prim_b,
                    reason=reason,
                ))

    return contradictions, summary


def resolve_contradictions(
    detections: Dict[str, Set[str]],
    priority_order: List[str] = None
) -> Dict[str, Set[str]]:
    """
    Resolve contradictions by keeping higher-priority primitive.

    Default priority (highest to lowest):
    1. negative_inventory (data integrity)
    2. high_margin_leak (profitability)
    3. low_stock (lost sales)
    4. dead_item (tied capital)
    5. overstock (cash flow)
    6. Other primitives

    Args:
        detections: Dict mapping primitive name to set of detected SKUs
        priority_order: Custom priority list (highest first)

    Returns:
        Resolved detections with contradictions removed
    """
    if priority_order is None:
        priority_order = [
            "negative_inventory",
            "high_margin_leak",
            "shrinkage_pattern",
            "low_stock",
            "margin_erosion",
            "dead_item",
            "price_discrepancy",
            "overstock",
        ]

    # Create priority map
    priority = {p: i for i, p in enumerate(priority_order)}

    # Work on copy
    resolved = {p: skus.copy() for p, skus in detections.items()}

    for prim_a, prim_b, reason in CONTRADICTORY_PAIRS:
        skus_a = resolved.get(prim_a, set())
        skus_b = resolved.get(prim_b, set())

        overlapping = skus_a & skus_b

        if overlapping:
            # Determine which primitive has higher priority
            prio_a = priority.get(prim_a, 999)
            prio_b = priority.get(prim_b, 999)

            if prio_a <= prio_b:
                # Keep prim_a, remove from prim_b
                resolved[prim_b] = skus_b - overlapping
            else:
                # Keep prim_b, remove from prim_a
                resolved[prim_a] = skus_a - overlapping

    return resolved


def generate_contradiction_report(
    contradictions: List[Contradiction],
    summary: Dict[str, int]
) -> str:
    """Generate Markdown report of contradictions."""

    if not contradictions:
        return "## Contradiction Analysis\n\nNo contradictions detected. ✅\n"

    report = f"""## Contradiction Analysis

### Summary

| Contradiction Type | Count |
|-------------------|-------|
"""

    for key, count in summary.items():
        report += f"| {key} | {count} |\n"

    report += f"""
**Total Contradictions:** {len(contradictions)}

### Sample Contradictions

| SKU | Primitive A | Primitive B | Reason |
|-----|-------------|-------------|--------|
"""

    for c in contradictions[:10]:
        report += f"| `{c.sku}` | {c.primitive_a} | {c.primitive_b} | {c.reason} |\n"

    if len(contradictions) > 10:
        report += f"\n*... and {len(contradictions) - 10} more*\n"

    report += """
### Resolution Strategy

Contradictions are resolved by priority:
1. `negative_inventory` > all (data integrity)
2. `high_margin_leak` > stock issues (profitability)
3. `low_stock` > `overstock` (lost sales worse than excess)

"""

    return report
