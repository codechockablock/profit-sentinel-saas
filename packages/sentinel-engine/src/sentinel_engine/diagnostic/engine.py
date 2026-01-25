"""
PROFIT SENTINEL v2 - CONVERSATIONAL DIAGNOSTIC
===============================================

Key change: Instead of auto-classifying items, the system ASKS about
every pattern it detects. The user sees each finding and confirms/denies.

This gives users:
1. VISIBILITY - They see exactly what patterns exist
2. CONTROL - They decide what's a process issue vs real shrinkage
3. LEARNING - They understand their own inventory gaps
4. TRUST - No black box magic, everything is transparent

Flow:
1. Detect ALL patterns in negative stock
2. Present patterns one by one, largest first
3. User confirms: "Yes that's a process issue" or "No, investigate those"
4. Running total updates with each answer
5. Final report shows the journey from $726K → $178K

Author: Joseph + Claude
Date: 2026-01-25
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

sys.path.insert(0, "/home/claude")


# =============================================================================
# PATTERN DETECTION
# =============================================================================

# Patterns we look for - these become QUESTIONS, not auto-classifications
DETECTION_PATTERNS = {
    # Lumber & boards
    "lumber_2x": {
        "keywords": ["2X4", "2X6", "2X8", "2X10", "2X12", "2X3"],
        "question": "I found {count} items matching **2X lumber** (2x4, 2x6, etc.) with {value} in negative stock.\n\nAt many stores, lumber is sold at the register but not received into inventory. Is that how it works here?",
        "suggested_answers": [
            ("Yes - sold at POS but not received", "receiving_gap"),
            ("No - these are fully tracked", "investigate"),
            ("It varies by item", "partial"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "lumber_4x": {
        "keywords": ["4X4", "6X6", "4X6"],
        "question": "Found {count} **post lumber items** (4x4, 6x6) with {value} negative.\n\nAre these handled the same as dimensional lumber - sold at POS but not received?",
        "suggested_answers": [
            ("Yes - same as other lumber", "receiving_gap"),
            ("No - these are tracked differently", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "boards_1x": {
        "keywords": ["1X4", "1X6", "1X8", "1X10", "1X12", "1X3"],
        "question": "Found {count} **1X boards** (1x4, 1x6, pine boards, etc.) with {value} negative.\n\nAre these sold at the register without being received into inventory?",
        "suggested_answers": [
            ("Yes - POS only, not received", "receiving_gap"),
            ("No - fully tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "deck_boards": {
        "keywords": ["5/4X", "DECK BOARD", "DECK BRD", "DECKING"],
        "question": "Found {count} **deck boards** with {value} negative.\n\nDeck boards are typically sold at the register. Are these received into inventory or POS-only?",
        "suggested_answers": [
            ("POS only - not received", "receiving_gap"),
            ("Fully tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    # Sheet goods
    "plywood": {
        "keywords": ["PLYWOOD", "PLY ", "BIRCH", "LAUAN", "HARDWOOD PLY"],
        "question": "Found {count} **plywood/sheet goods** with {value} negative.\n\nSheet goods are often sold at the register. How does your store handle plywood receiving?",
        "suggested_answers": [
            ("Sold at POS, not received into system", "receiving_gap"),
            ("Fully received and tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "osb_sheathing": {
        "keywords": ["OSB", "SHEATHING", "ORIENTED STRAND"],
        "question": "Found {count} **OSB/sheathing** items with {value} negative.\n\nIs OSB handled like plywood - sold but not received?",
        "suggested_answers": [
            ("Yes - same as plywood", "receiving_gap"),
            ("No - tracked differently", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "drywall": {
        "keywords": ["DRYWALL", "SHEETROCK", "GYPSUM"],
        "question": "Found {count} **drywall** items with {value} negative.\n\nDrywall is heavy and often sold at the register. Is it received into inventory?",
        "suggested_answers": [
            ("No - sold at POS only", "receiving_gap"),
            ("Yes - fully tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    # Moulding & trim
    "moulding": {
        "keywords": [
            "MOULDING",
            "MOLDING",
            "TRIM",
            "BASE",
            "CASING",
            "CROWN",
            "COLONIAL",
            "QUARTERROUND",
            "PRIMED",
        ],
        "question": "Found {count} **moulding/trim** items with {value} negative.\n\nMoulding is often sold at the register in lengths. How is it tracked here?",
        "suggested_answers": [
            ("Sold at POS, not received", "receiving_gap"),
            ("Fully tracked", "investigate"),
            ("Some tracked, some not", "partial"),
        ],
        "typical_behavior": "receiving_gap",
    },
    # Concrete & heavy bags
    "concrete": {
        "keywords": ["CONCRETE", "QUIKRETE", "SAKRETE", "MORTAR", "FAST SET", "CEMENT"],
        "question": "Found {count} **concrete/mortar** items with {value} negative.\n\nBagged concrete is heavy and often sold at the register. Is it received into inventory?",
        "suggested_answers": [
            ("No - sold at POS only", "receiving_gap"),
            ("Yes - fully tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "soil_mulch": {
        "keywords": ["MULCH", "TOPSOIL", "POTTING", "SOIL", "COMPOST", "MANURE"],
        "question": "Found {count} **soil/mulch** items with {value} negative.\n\nBagged soil and mulch - sold at register or fully tracked?",
        "suggested_answers": [
            ("Sold at POS, not received", "receiving_gap"),
            ("Fully tracked", "investigate"),
            ("Seasonal - tracking varies", "partial"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "pellets": {
        "keywords": ["PELLET", "WOOD PEL", "FUEL PEL"],
        "question": "Found {count} **pellet** items with {value} negative.\n\nPellets are heavy and seasonal. How are they tracked?",
        "suggested_answers": [
            ("Sold at POS only", "receiving_gap"),
            ("Fully tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "salt_ice_melt": {
        "keywords": ["SALT", "ICE MELT", "MELT ", "ROCK SALT", "PET SAFE"],
        "question": "Found {count} **salt/ice melt** items with {value} negative.\n\nIce melt is seasonal and heavy. Is it received into inventory?",
        "suggested_answers": [
            ("No - sold at POS only", "receiving_gap"),
            ("Yes - tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    "stone_sand": {
        "keywords": ["STONE", "PAVER", "SAND", "GRAVEL", "PEBBLE"],
        "question": "Found {count} **stone/sand** items with {value} negative.\n\nBagged stone and sand - tracked or POS only?",
        "suggested_answers": [
            ("POS only", "receiving_gap"),
            ("Fully tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    # Cut-to-length / Non-tracked by design
    "wire_by_foot": {
        "keywords": ["WIRE BY", "BY THE FOOT", "BY FOOT", "/FT", "PER FOOT"],
        "question": "Found {count} **by-the-foot** items (wire, rope, etc.) with {value} negative.\n\nThese are typically cut-to-length and not tracked. Is that correct?",
        "suggested_answers": [
            ("Correct - not tracked by design", "non_tracked"),
            ("We do track these", "investigate"),
        ],
        "typical_behavior": "non_tracked",
    },
    "rope_chain": {
        "keywords": ["ROPE", "CHAIN", "CORD ", "TWINE"],
        "question": "Found {count} **rope/chain** items with {value} negative.\n\nRope and chain are usually sold by the foot. Are these tracked?",
        "suggested_answers": [
            ("No - sold by the foot, not tracked", "non_tracked"),
            ("Yes - tracked by the roll", "investigate"),
        ],
        "typical_behavior": "non_tracked",
    },
    "tubing": {
        "keywords": ["TUBING", "TUBE ", "PIPE BY"],
        "question": "Found {count} **tubing** items with {value} negative.\n\nIs tubing sold by the foot / cut to length?",
        "suggested_answers": [
            ("Yes - cut to order, not tracked", "non_tracked"),
            ("No - sold in fixed lengths", "investigate"),
        ],
        "typical_behavior": "non_tracked",
    },
    "bin_fasteners": {
        "keywords": ["NUT ", "BOLT ", "WASHER", "SCREW "],
        "question": "Found {count} **loose fasteners** (nuts, bolts, washers) with {value} negative.\n\nAre these sold from bins without individual tracking?",
        "suggested_answers": [
            ("Yes - bin items, not tracked", "non_tracked"),
            ("No - individually tracked", "investigate"),
        ],
        "typical_behavior": "non_tracked",
    },
    "keys": {
        "keywords": ["KEY ", "KEYS", "KEY BLANK"],
        "question": "Found {count} **key** items with {value} negative.\n\nKeys are typically cut at the counter. Are they tracked?",
        "suggested_answers": [
            ("No - not tracked", "non_tracked"),
            ("Yes - tracked by blank", "investigate"),
        ],
        "typical_behavior": "non_tracked",
    },
    # Vendor managed
    "lawn_chemicals": {
        "keywords": [
            "SCOTTS",
            "MIRACLE",
            "ROUNDUP",
            "ORTHO",
            "VIGORO",
            "WEED",
            "FERTILIZER",
            "TURF BUILDER",
        ],
        "question": "Found {count} **lawn chemical** items (Scotts, Miracle-Gro, etc.) with {value} negative.\n\nLawn chemicals are often vendor-managed (they ship and track their own inventory). Is that the case here?",
        "suggested_answers": [
            ("Yes - vendor managed", "vendor_managed"),
            ("No - we receive these ourselves", "investigate"),
        ],
        "typical_behavior": "vendor_managed",
    },
    "landscape_staples": {
        "keywords": ["STAPLE", "LANDSCAPE STAPLE"],
        "question": "Found {count} **landscape staples** with {value} negative.\n\nThese are sometimes vendor-managed with lawn products. How are they handled?",
        "suggested_answers": [
            ("Vendor managed", "vendor_managed"),
            ("We track them", "investigate"),
        ],
        "typical_behavior": "vendor_managed",
    },
    # Expiration / damage prone
    "beverages": {
        "keywords": [
            "COKE",
            "PEPSI",
            "DIET",
            "WATER",
            "MONSTER",
            "GATORADE",
            "SODA",
            "DRINK",
            "DR PEPPER",
            "SPRITE",
            "DEW",
        ],
        "question": "Found {count} **beverage** items with {value} negative.\n\nBeverages can expire or get damaged. Are expired drinks always written off properly?",
        "suggested_answers": [
            ("No - often tossed without write-off", "expiration"),
            ("Yes - always written off", "investigate"),
            ("High theft category", "theft"),
        ],
        "typical_behavior": "expiration",
    },
    "snacks": {
        "keywords": ["CANDY", "SNACK", "CHIP", "GUM", "SLIM JIM"],
        "question": "Found {count} **snack** items with {value} negative.\n\nSnacks can expire and are also theft-prone. What's most likely here?",
        "suggested_answers": [
            ("Expiration without write-off", "expiration"),
            ("Likely theft", "theft"),
            ("Fully tracked - investigate", "investigate"),
        ],
        "typical_behavior": "expiration",
    },
    # Animal / outdoor
    "deer_corn": {
        "keywords": ["DEER CORN", "DEER FEED", "BIRD SEED", "WILDLIFE"],
        "question": "Found {count} **animal feed** items with {value} negative.\n\nBagged feed is heavy. Is it received into inventory?",
        "suggested_answers": [
            ("No - sold at POS only", "receiving_gap"),
            ("Yes - tracked", "investigate"),
        ],
        "typical_behavior": "receiving_gap",
    },
    # High-value / theft prone
    "batteries": {
        "keywords": ["BATTERY", "BATT ", "ENERGIZER", "DURACELL"],
        "question": "Found {count} **battery** items with {value} negative.\n\nBatteries are small and high-value - a common theft target. What do you think is happening?",
        "suggested_answers": [
            ("Likely theft", "theft"),
            ("Receiving/returns issue", "investigate"),
            ("Fully tracked - investigate", "investigate"),
        ],
        "typical_behavior": "theft",
    },
    "power_equipment": {
        "keywords": [
            "GENERATOR",
            "CHAINSAW",
            "BLOWER",
            "TRIMMER",
            "MOWER",
            "PRESSURE WASH",
        ],
        "question": "Found {count} **power equipment** items with {value} negative.\n\nThese are high-value items. This could be theft, returns, or count errors. What's most likely?",
        "suggested_answers": [
            ("Count error - need to verify", "investigate"),
            ("Returns not processed", "investigate"),
            ("Possible theft", "theft"),
        ],
        "typical_behavior": "theft",
    },
    "filters": {
        "keywords": ["FILTER", "FURNACE FILTER", "AIR FILTER", "HVAC"],
        "question": "Found {count} **filter** items with {value} negative.\n\nFilters sometimes have seasonal damage or returns issues. What's the situation here?",
        "suggested_answers": [
            ("Damage not written off", "expiration"),
            ("Returns issue", "investigate"),
            ("Fully tracked - investigate", "investigate"),
        ],
        "typical_behavior": "investigate",
    },
}


# =============================================================================
# DIAGNOSTIC STATE
# =============================================================================


class Classification(Enum):
    """How an item pattern was classified."""

    RECEIVING_GAP = "receiving_gap"  # Sold but not received
    NON_TRACKED = "non_tracked"  # By design (bins, cut-to-length)
    VENDOR_MANAGED = "vendor_managed"  # Direct ship
    EXPIRATION = "expiration"  # Expires without write-off
    THEFT = "theft"  # Actual theft
    INVESTIGATE = "investigate"  # Needs investigation
    PENDING = "pending"  # Not yet asked


@dataclass
class DetectedPattern:
    """A pattern detected in the negative stock."""

    pattern_id: str
    name: str
    keywords: list[str]
    items: list[dict]
    total_value: float
    question: str
    suggested_answers: list[tuple[str, str]]
    classification: Classification = Classification.PENDING
    user_answer: str = ""

    @property
    def item_count(self) -> int:
        return len(self.items)


@dataclass
class DiagnosticSession:
    """State of a diagnostic session."""

    patterns: list[DetectedPattern] = field(default_factory=list)
    current_index: int = 0
    total_shrinkage: float = 0
    items_analyzed: int = 0
    negative_items: int = 0

    @property
    def current_pattern(self) -> DetectedPattern | None:
        if 0 <= self.current_index < len(self.patterns):
            return self.patterns[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.patterns)

    @property
    def explained_value(self) -> float:
        """Value explained as process issues (not theft)."""
        explained = 0
        for p in self.patterns:
            if p.classification in [
                Classification.RECEIVING_GAP,
                Classification.NON_TRACKED,
                Classification.VENDOR_MANAGED,
                Classification.EXPIRATION,
            ]:
                explained += p.total_value
        return explained

    @property
    def unexplained_value(self) -> float:
        """Value still needing investigation."""
        return self.total_shrinkage - self.explained_value

    @property
    def reduction_percent(self) -> float:
        if self.total_shrinkage == 0:
            return 0
        return (self.explained_value / self.total_shrinkage) * 100

    def get_summary(self) -> dict:
        """Get current summary statistics."""
        by_classification = defaultdict(lambda: {"items": 0, "value": 0})

        for p in self.patterns:
            by_classification[p.classification.value]["items"] += p.item_count
            by_classification[p.classification.value]["value"] += p.total_value

        return {
            "total_shrinkage": self.total_shrinkage,
            "explained_value": self.explained_value,
            "unexplained_value": self.unexplained_value,
            "reduction_percent": self.reduction_percent,
            "patterns_total": len(self.patterns),
            "patterns_answered": self.current_index,
            "by_classification": dict(by_classification),
        }


# =============================================================================
# CONVERSATIONAL DIAGNOSTIC ENGINE
# =============================================================================


class ConversationalDiagnostic:
    """
    Diagnostic engine that asks about EVERY pattern.
    No auto-classification - user confirms everything.
    """

    def __init__(self):
        self.session: DiagnosticSession | None = None

    def start_session(self, items: list[dict]) -> DiagnosticSession:
        """
        Start a new diagnostic session.

        Args:
            items: List of inventory items

        Returns:
            New DiagnosticSession
        """
        self.session = DiagnosticSession()

        # Find negative stock items
        negative_items = [i for i in items if i.get("stock", 0) < 0]

        self.session.items_analyzed = len(items)
        self.session.negative_items = len(negative_items)
        self.session.total_shrinkage = sum(
            abs(i.get("stock", 0)) * i.get("cost", 0) for i in negative_items
        )

        # Detect all patterns
        self._detect_patterns(negative_items)

        return self.session

    def _detect_patterns(self, items: list[dict]):
        """Detect patterns in negative stock items."""
        matched_items = set()  # Track which items have been matched
        patterns = []

        for pattern_id, config in DETECTION_PATTERNS.items():
            matching_items = []

            for item in items:
                # Skip if already matched to a pattern
                item_key = item.get("sku", "")
                if item_key in matched_items:
                    continue

                sku = item.get("sku", "").upper()
                desc = item.get("description", "").upper()

                # Check if any keyword matches
                for keyword in config["keywords"]:
                    if keyword in sku or keyword in desc:
                        matching_items.append(item)
                        matched_items.add(item_key)
                        break

            if matching_items:
                total_value = sum(
                    abs(i.get("stock", 0)) * i.get("cost", 0) for i in matching_items
                )

                patterns.append(
                    DetectedPattern(
                        pattern_id=pattern_id,
                        name=pattern_id.replace("_", " ").title(),
                        keywords=config["keywords"],
                        items=matching_items,
                        total_value=total_value,
                        question=config["question"].format(
                            count=len(matching_items), value=f"${total_value:,.0f}"
                        ),
                        suggested_answers=config["suggested_answers"],
                    )
                )

        # Collect unmatched items as "Other"
        unmatched = [i for i in items if i.get("sku", "") not in matched_items]
        if unmatched:
            total_value = sum(
                abs(i.get("stock", 0)) * i.get("cost", 0) for i in unmatched
            )

            patterns.append(
                DetectedPattern(
                    pattern_id="other",
                    name="Other Items",
                    keywords=[],
                    items=unmatched,
                    total_value=total_value,
                    question=f"Found {len(unmatched)} **other items** that don't match common patterns, with {f'${total_value:,.0f}'} in negative stock.\n\nThese need individual review. How would you like to handle them?",
                    suggested_answers=[
                        ("Mark for investigation", "investigate"),
                        ("Likely misc process issues", "receiving_gap"),
                    ],
                )
            )

        # Sort by value (highest first)
        patterns.sort(key=lambda p: p.total_value, reverse=True)

        self.session.patterns = patterns

    def get_current_question(self) -> dict | None:
        """Get the current question to ask."""
        if not self.session or self.session.is_complete:
            return None

        pattern = self.session.current_pattern
        if not pattern:
            return None

        return {
            "pattern_id": pattern.pattern_id,
            "pattern_name": pattern.name,
            "question": pattern.question,
            "suggested_answers": pattern.suggested_answers,
            "item_count": pattern.item_count,
            "total_value": pattern.total_value,
            "sample_items": pattern.items[:5],
            "progress": {
                "current": self.session.current_index + 1,
                "total": len(self.session.patterns),
            },
            "running_totals": self.session.get_summary(),
        }

    def answer_question(self, classification: str, user_note: str = "") -> dict:
        """
        Process user's answer to current question.

        Args:
            classification: One of: receiving_gap, non_tracked, vendor_managed,
                          expiration, theft, investigate
            user_note: Optional note from user

        Returns:
            Updated session state
        """
        if not self.session or self.session.is_complete:
            return {"error": "No active question"}

        pattern = self.session.current_pattern
        if not pattern:
            return {"error": "No current pattern"}

        # Record the answer
        try:
            pattern.classification = Classification(classification)
        except ValueError:
            pattern.classification = Classification.INVESTIGATE

        pattern.user_answer = user_note

        # Move to next question
        self.session.current_index += 1

        # Return updated state
        return {
            "answered": {
                "pattern": pattern.pattern_id,
                "classification": pattern.classification.value,
                "value": pattern.total_value,
            },
            "progress": {
                "current": self.session.current_index,
                "total": len(self.session.patterns),
            },
            "running_totals": self.session.get_summary(),
            "is_complete": self.session.is_complete,
            "next_question": self.get_current_question(),
        }

    def get_final_report(self) -> dict:
        """Get the final diagnostic report."""
        if not self.session:
            return {"error": "No session"}

        summary = self.session.get_summary()

        # Group patterns by classification
        by_class = defaultdict(list)
        for p in self.session.patterns:
            by_class[p.classification.value].append(
                {
                    "name": p.name,
                    "items": p.item_count,
                    "value": p.total_value,
                }
            )

        # Top items still needing investigation
        investigate_items = []
        for p in self.session.patterns:
            if p.classification in [Classification.INVESTIGATE, Classification.THEFT]:
                for item in p.items:
                    value = abs(item.get("stock", 0)) * item.get("cost", 0)
                    investigate_items.append(
                        {
                            "sku": item.get("sku"),
                            "description": item.get("description"),
                            "stock": item.get("stock"),
                            "value": value,
                            "pattern": p.name,
                            "classification": p.classification.value,
                        }
                    )

        investigate_items.sort(key=lambda x: x["value"], reverse=True)

        return {
            "summary": summary,
            "by_classification": dict(by_class),
            "items_to_investigate": investigate_items[:50],
            "journey": [
                {
                    "pattern": p.name,
                    "value": p.total_value,
                    "classification": p.classification.value,
                    "items": p.item_count,
                }
                for p in self.session.patterns
            ],
        }


# =============================================================================
# DEMO
# =============================================================================


def demo():
    """Demo the conversational diagnostic."""
    import csv

    print("=" * 70)
    print("PROFIT SENTINEL - CONVERSATIONAL DIAGNOSTIC")
    print("=" * 70)

    # Load inventory
    print("\nLoading inventory...")
    items = []
    with open(
        "/mnt/user-data/uploads/Inventory_Report_Audit_Adjust.csv",
        encoding="utf-8",
        errors="replace",
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stock = float(row.get("In Stock Qty.", "0").replace(",", "") or 0)
                cost = float(
                    row.get("Cost", "0").replace(",", "").replace("$", "") or 0
                )
                items.append(
                    {
                        "sku": row.get("SKU", "").strip(),
                        "description": row.get("Description ", "").strip(),
                        "stock": stock,
                        "cost": cost,
                    }
                )
            except:
                pass

    print(f"Loaded {len(items):,} items")

    # Start diagnostic
    engine = ConversationalDiagnostic()
    session = engine.start_session(items)

    print(f"\nTotal apparent shrinkage: ${session.total_shrinkage:,.0f}")
    print(f"Detected {len(session.patterns)} patterns to review")

    # Show all questions
    print("\n" + "=" * 70)
    print("PATTERNS DETECTED (would be asked one-by-one)")
    print("=" * 70)

    for i, pattern in enumerate(session.patterns, 1):
        print(f"\n--- Pattern {i}/{len(session.patterns)} ---")
        print(f"Name: {pattern.name}")
        print(f"Items: {pattern.item_count}")
        print(f"Value: ${pattern.total_value:,.0f}")
        print(f"\nQuestion: {pattern.question[:200]}...")
        print(f"Answers: {[a[0][:40] for a in pattern.suggested_answers]}")

    # Simulate answering all with typical behaviors
    print("\n" + "=" * 70)
    print("SIMULATING USER ANSWERS (using typical behaviors)")
    print("=" * 70)

    while not session.is_complete:
        q = engine.get_current_question()
        if not q:
            break

        # Get typical behavior for this pattern
        pattern = session.current_pattern
        config = DETECTION_PATTERNS.get(pattern.pattern_id, {})
        typical = config.get("typical_behavior", "investigate")

        result = engine.answer_question(typical)

        print(f"\n✓ {pattern.name}: {typical} (${pattern.total_value:,.0f})")
        print(
            f"  Running total - Explained: ${result['running_totals']['explained_value']:,.0f} "
            + f"({result['running_totals']['reduction_percent']:.1f}%)"
        )

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    report = engine.get_final_report()
    summary = report["summary"]

    print(
        f"""
┌────────────────────────────────────────────────────────────────┐
│  DIAGNOSTIC COMPLETE                                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Total Apparent Shrinkage:   ${summary['total_shrinkage']:>10,.0f}                 │
│                                                                │
│  Process Issues Found:       ${summary['explained_value']:>10,.0f}                 │
│  Still to Investigate:       ${summary['unexplained_value']:>10,.0f}                 │
│                                                                │
│  ════════════════════════════════════════════════════════════ │
│  REDUCTION:                      {summary['reduction_percent']:>6.1f}%                   │
│  ════════════════════════════════════════════════════════════ │
│                                                                │
│  {summary['patterns_total']} patterns reviewed                                      │
└────────────────────────────────────────────────────────────────┘
"""
    )

    print("\nTop 10 items to investigate:")
    for item in report["items_to_investigate"][:10]:
        print(
            f"  {item['sku']:<20} ${item['value']:>8,.0f}  ({item['classification']})"
        )


if __name__ == "__main__":
    demo()
