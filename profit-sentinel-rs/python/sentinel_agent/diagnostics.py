"""Conversational Diagnostic Engine.

Interactive shrinkage classification system that presents detected patterns
one-by-one to the user for confirmation. Each negative-stock pattern becomes
a QUESTION, not an auto-classification. The user sees every finding and
decides what's a process issue vs. real shrinkage.

Key properties:
    1. VISIBILITY  - users see exactly what patterns exist
    2. CONTROL     - they decide the classification
    3. LEARNING    - they understand their own inventory gaps
    4. TRUST       - no black box, everything is transparent

Typical flow:
    1. Detect ALL patterns in negative stock
    2. Present patterns one-by-one, largest value first
    3. User confirms: "Yes that's a process issue" or "No, investigate"
    4. Running total updates with each answer
    5. Final report shows the journey (e.g. $726K -> $178K)

Integration:
    This module is standalone (no external dependencies beyond our models).
    It plugs into sidecar.py via POST /api/v1/diagnostic/start, GET .../question,
    POST .../answer, and GET .../summary endpoints.

Author: Joseph + Claude
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

logger = logging.getLogger("sentinel.diagnostics")


# =============================================================================
# Classification
# =============================================================================


class Classification(str, Enum):
    """How a detected pattern was classified by the user."""

    RECEIVING_GAP = "receiving_gap"  # Sold but not received into system
    NON_TRACKED = "non_tracked"  # Not tracked by design (bins, cut-to-length)
    VENDOR_MANAGED = "vendor_managed"  # Vendor manages inventory (direct ship)
    EXPIRATION = "expiration"  # Expires/damaged without write-off
    THEFT = "theft"  # Actual theft / shrinkage
    INVESTIGATE = "investigate"  # Needs further investigation
    PENDING = "pending"  # Not yet asked

    @property
    def is_explained(self) -> bool:
        """Whether this classification counts as an 'explained' process issue."""
        return self in (
            Classification.RECEIVING_GAP,
            Classification.NON_TRACKED,
            Classification.VENDOR_MANAGED,
            Classification.EXPIRATION,
        )

    @property
    def display_name(self) -> str:
        return {
            Classification.RECEIVING_GAP: "Receiving Gap",
            Classification.NON_TRACKED: "Not Tracked by Design",
            Classification.VENDOR_MANAGED: "Vendor Managed",
            Classification.EXPIRATION: "Expiration / Damage",
            Classification.THEFT: "Theft / Shrinkage",
            Classification.INVESTIGATE: "Needs Investigation",
            Classification.PENDING: "Pending",
        }[self]


# =============================================================================
# Detection Patterns (26 patterns)
# =============================================================================

# Each pattern becomes a QUESTION. The suggested answers give the user
# pre-built choices, but they can also classify manually.
# "typical_behavior" guides the default suggestion ordering.

DETECTION_PATTERNS: dict[str, dict] = {
    # -----------------------------------------------------------------------
    # Lumber & boards (4)
    # -----------------------------------------------------------------------
    "lumber_2x": {
        "keywords": ["2X4", "2X6", "2X8", "2X10", "2X12", "2X3"],
        "question": (
            "I found {count} items matching **2X lumber** (2x4, 2x6, etc.) "
            "with {value} in negative stock.\n\n"
            "At many stores, lumber is sold at the register but not received "
            "into inventory. Is that how it works here?"
        ),
        "suggested_answers": [
            ["Yes - sold at POS but not received", "receiving_gap"],
            ["No - these are fully tracked", "investigate"],
            ["It varies by item", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "lumber_4x": {
        "keywords": ["4X4", "6X6", "4X6"],
        "question": (
            "Found {count} **post lumber items** (4x4, 6x6) with {value} negative.\n\n"
            "Are these handled the same as dimensional lumber - sold at POS "
            "but not received?"
        ),
        "suggested_answers": [
            ["Yes - same as other lumber", "receiving_gap"],
            ["No - these are tracked differently", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "boards_1x": {
        "keywords": ["1X4", "1X6", "1X8", "1X10", "1X12", "1X3"],
        "question": (
            "Found {count} **1X boards** (1x4, 1x6, pine boards, etc.) "
            "with {value} negative.\n\n"
            "Are these sold at the register without being received into inventory?"
        ),
        "suggested_answers": [
            ["Yes - POS only, not received", "receiving_gap"],
            ["No - fully tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "deck_boards": {
        "keywords": ["5/4X", "DECK BOARD", "DECK BRD", "DECKING"],
        "question": (
            "Found {count} **deck boards** with {value} negative.\n\n"
            "Deck boards are typically sold at the register. Are these "
            "received into inventory or POS-only?"
        ),
        "suggested_answers": [
            ["POS only - not received", "receiving_gap"],
            ["Fully tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    # -----------------------------------------------------------------------
    # Sheet goods (3)
    # -----------------------------------------------------------------------
    "plywood": {
        "keywords": ["PLYWOOD", "PLY ", "BIRCH", "LAUAN", "HARDWOOD PLY"],
        "question": (
            "Found {count} **plywood/sheet goods** with {value} negative.\n\n"
            "Sheet goods are often sold at the register. How does your "
            "store handle plywood receiving?"
        ),
        "suggested_answers": [
            ["Sold at POS, not received into system", "receiving_gap"],
            ["Fully received and tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "osb_sheathing": {
        "keywords": ["OSB", "SHEATHING", "ORIENTED STRAND"],
        "question": (
            "Found {count} **OSB/sheathing** items with {value} negative.\n\n"
            "Is OSB handled like plywood - sold but not received?"
        ),
        "suggested_answers": [
            ["Yes - same as plywood", "receiving_gap"],
            ["No - tracked differently", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "drywall": {
        "keywords": ["DRYWALL", "SHEETROCK", "GYPSUM"],
        "question": (
            "Found {count} **drywall** items with {value} negative.\n\n"
            "Drywall is heavy and often sold at the register. "
            "Is it received into inventory?"
        ),
        "suggested_answers": [
            ["No - sold at POS only", "receiving_gap"],
            ["Yes - fully tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    # -----------------------------------------------------------------------
    # Moulding & trim (1)
    # -----------------------------------------------------------------------
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
        "question": (
            "Found {count} **moulding/trim** items with {value} negative.\n\n"
            "Moulding is often sold at the register in lengths. "
            "How is it tracked here?"
        ),
        "suggested_answers": [
            ["Sold at POS, not received", "receiving_gap"],
            ["Fully tracked", "investigate"],
            ["Some tracked, some not", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    # -----------------------------------------------------------------------
    # Concrete & heavy bags (5)
    # -----------------------------------------------------------------------
    "concrete": {
        "keywords": ["CONCRETE", "QUIKRETE", "SAKRETE", "MORTAR", "FAST SET", "CEMENT"],
        "question": (
            "Found {count} **concrete/mortar** items with {value} negative.\n\n"
            "Bagged concrete is heavy and often sold at the register. "
            "Is it received into inventory?"
        ),
        "suggested_answers": [
            ["No - sold at POS only", "receiving_gap"],
            ["Yes - fully tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "soil_mulch": {
        "keywords": ["MULCH", "TOPSOIL", "POTTING", "SOIL", "COMPOST", "MANURE"],
        "question": (
            "Found {count} **soil/mulch** items with {value} negative.\n\n"
            "Bagged soil and mulch - sold at register or fully tracked?"
        ),
        "suggested_answers": [
            ["Sold at POS, not received", "receiving_gap"],
            ["Fully tracked", "investigate"],
            ["Seasonal - tracking varies", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "pellets": {
        "keywords": ["PELLET", "WOOD PEL", "FUEL PEL"],
        "question": (
            "Found {count} **pellet** items with {value} negative.\n\n"
            "Pellets are heavy and seasonal. How are they tracked?"
        ),
        "suggested_answers": [
            ["Sold at POS only", "receiving_gap"],
            ["Fully tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "salt_ice_melt": {
        "keywords": ["SALT", "ICE MELT", "MELT ", "ROCK SALT", "PET SAFE"],
        "question": (
            "Found {count} **salt/ice melt** items with {value} negative.\n\n"
            "Ice melt is seasonal and heavy. Is it received into inventory?"
        ),
        "suggested_answers": [
            ["No - sold at POS only", "receiving_gap"],
            ["Yes - tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    "stone_sand": {
        "keywords": ["STONE", "PAVER", "SAND", "GRAVEL", "PEBBLE"],
        "question": (
            "Found {count} **stone/sand** items with {value} negative.\n\n"
            "Bagged stone and sand - tracked or POS only?"
        ),
        "suggested_answers": [
            ["POS only", "receiving_gap"],
            ["Fully tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    # -----------------------------------------------------------------------
    # Cut-to-length / Non-tracked by design (5)
    # -----------------------------------------------------------------------
    "wire_by_foot": {
        "keywords": ["WIRE BY", "BY THE FOOT", "BY FOOT", "/FT", "PER FOOT"],
        "question": (
            "Found {count} **by-the-foot** items (wire, rope, etc.) "
            "with {value} negative.\n\n"
            "These are typically cut-to-length and not tracked. Is that correct?"
        ),
        "suggested_answers": [
            ["Correct - not tracked by design", "non_tracked"],
            ["We do track these", "investigate"],
        ],
        "typical_behavior": "non_tracked",
    },
    "rope_chain": {
        "keywords": ["ROPE", "CHAIN", "CORD ", "TWINE"],
        "question": (
            "Found {count} **rope/chain** items with {value} negative.\n\n"
            "Rope and chain are usually sold by the foot. Are these tracked?"
        ),
        "suggested_answers": [
            ["No - sold by the foot, not tracked", "non_tracked"],
            ["Yes - tracked by the roll", "investigate"],
        ],
        "typical_behavior": "non_tracked",
    },
    "tubing": {
        "keywords": ["TUBING", "TUBE ", "PIPE BY"],
        "question": (
            "Found {count} **tubing** items with {value} negative.\n\n"
            "Is tubing sold by the foot / cut to length?"
        ),
        "suggested_answers": [
            ["Yes - cut to order, not tracked", "non_tracked"],
            ["No - sold in fixed lengths", "investigate"],
        ],
        "typical_behavior": "non_tracked",
    },
    "bin_fasteners": {
        "keywords": ["NUT ", "BOLT ", "WASHER", "SCREW "],
        "question": (
            "Found {count} **loose fasteners** (nuts, bolts, washers) "
            "with {value} negative.\n\n"
            "Are these sold from bins without individual tracking?"
        ),
        "suggested_answers": [
            ["Yes - bin items, not tracked", "non_tracked"],
            ["No - individually tracked", "investigate"],
        ],
        "typical_behavior": "non_tracked",
    },
    "keys": {
        "keywords": ["KEY ", "KEYS", "KEY BLANK"],
        "question": (
            "Found {count} **key** items with {value} negative.\n\n"
            "Keys are typically cut at the counter. Are they tracked?"
        ),
        "suggested_answers": [
            ["No - not tracked", "non_tracked"],
            ["Yes - tracked by blank", "investigate"],
        ],
        "typical_behavior": "non_tracked",
    },
    # -----------------------------------------------------------------------
    # Vendor managed (2)
    # -----------------------------------------------------------------------
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
        "question": (
            "Found {count} **lawn chemical** items (Scotts, Miracle-Gro, etc.) "
            "with {value} negative.\n\n"
            "Lawn chemicals are often vendor-managed (they ship and track "
            "their own inventory). Is that the case here?"
        ),
        "suggested_answers": [
            ["Yes - vendor managed", "vendor_managed"],
            ["No - we receive these ourselves", "investigate"],
        ],
        "typical_behavior": "vendor_managed",
    },
    "landscape_staples": {
        "keywords": ["STAPLE", "LANDSCAPE STAPLE"],
        "question": (
            "Found {count} **landscape staples** with {value} negative.\n\n"
            "These are sometimes vendor-managed with lawn products. "
            "How are they handled?"
        ),
        "suggested_answers": [
            ["Vendor managed", "vendor_managed"],
            ["We track them", "investigate"],
        ],
        "typical_behavior": "vendor_managed",
    },
    # -----------------------------------------------------------------------
    # Expiration / damage prone (2)
    # -----------------------------------------------------------------------
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
        "question": (
            "Found {count} **beverage** items with {value} negative.\n\n"
            "Beverages can expire or get damaged. Are expired drinks "
            "always written off properly?"
        ),
        "suggested_answers": [
            ["No - often tossed without write-off", "expiration"],
            ["Yes - always written off", "investigate"],
            ["High theft category", "theft"],
        ],
        "typical_behavior": "expiration",
    },
    "snacks": {
        "keywords": ["CANDY", "SNACK", "CHIP", "GUM", "SLIM JIM"],
        "question": (
            "Found {count} **snack** items with {value} negative.\n\n"
            "Snacks can expire and are also theft-prone. "
            "What's most likely here?"
        ),
        "suggested_answers": [
            ["Expiration without write-off", "expiration"],
            ["Likely theft", "theft"],
            ["Fully tracked - investigate", "investigate"],
        ],
        "typical_behavior": "expiration",
    },
    # -----------------------------------------------------------------------
    # Animal / outdoor (1)
    # -----------------------------------------------------------------------
    "deer_corn": {
        "keywords": ["DEER CORN", "DEER FEED", "BIRD SEED", "WILDLIFE"],
        "question": (
            "Found {count} **animal feed** items with {value} negative.\n\n"
            "Bagged feed is heavy. Is it received into inventory?"
        ),
        "suggested_answers": [
            ["No - sold at POS only", "receiving_gap"],
            ["Yes - tracked", "investigate"],
        ],
        "typical_behavior": "receiving_gap",
    },
    # -----------------------------------------------------------------------
    # High-value / theft prone (3)
    # -----------------------------------------------------------------------
    "batteries": {
        "keywords": ["BATTERY", "BATT ", "ENERGIZER", "DURACELL"],
        "question": (
            "Found {count} **battery** items with {value} negative.\n\n"
            "Batteries are small and high-value - a common theft target. "
            "What do you think is happening?"
        ),
        "suggested_answers": [
            ["Likely theft", "theft"],
            ["Receiving/returns issue", "investigate"],
            ["Fully tracked - investigate", "investigate"],
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
        "question": (
            "Found {count} **power equipment** items with {value} negative.\n\n"
            "These are high-value items. This could be theft, returns, "
            "or count errors. What's most likely?"
        ),
        "suggested_answers": [
            ["Count error - need to verify", "investigate"],
            ["Returns not processed", "investigate"],
            ["Possible theft", "theft"],
        ],
        "typical_behavior": "theft",
    },
    "filters": {
        "keywords": ["FILTER", "FURNACE FILTER", "AIR FILTER", "HVAC"],
        "question": (
            "Found {count} **filter** items with {value} negative.\n\n"
            "Filters sometimes have seasonal damage or returns issues. "
            "What's the situation here?"
        ),
        "suggested_answers": [
            ["Damage not written off", "expiration"],
            ["Returns issue", "investigate"],
            ["Fully tracked - investigate", "investigate"],
        ],
        "typical_behavior": "investigate",
    },
}


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class InventoryItem:
    """Simplified inventory item for diagnostic analysis."""

    sku: str
    description: str
    stock: float
    cost: float

    @property
    def shrinkage_value(self) -> float:
        """Absolute dollar value of shrinkage (cost * |stock|)."""
        return abs(self.stock) * self.cost


@dataclass
class DetectedPattern:
    """A keyword pattern detected in the negative stock."""

    pattern_id: str
    name: str
    keywords: list[str]
    items: list[InventoryItem]
    total_value: float
    question: str
    suggested_answers: list[list[str]]
    classification: Classification = Classification.PENDING
    user_note: str = ""

    @property
    def item_count(self) -> int:
        return len(self.items)

    def to_dict(self) -> dict:
        """Serialize for API responses."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_name": self.name,
            "item_count": self.item_count,
            "total_value": self.total_value,
            "classification": self.classification.value,
        }


@dataclass
class DiagnosticSession:
    """State of a diagnostic session."""

    session_id: str
    patterns: list[DetectedPattern] = field(default_factory=list)
    current_index: int = 0
    total_shrinkage: float = 0.0
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
        """Value explained as process issues (not theft or unknown)."""
        return sum(
            p.total_value for p in self.patterns if p.classification.is_explained
        )

    @property
    def unexplained_value(self) -> float:
        """Value still needing investigation."""
        return self.total_shrinkage - self.explained_value

    @property
    def reduction_percent(self) -> float:
        if self.total_shrinkage == 0:
            return 0.0
        return (self.explained_value / self.total_shrinkage) * 100.0

    def get_summary(self) -> dict:
        """Get current summary statistics."""
        by_classification: dict[str, dict] = defaultdict(
            lambda: {"items": 0, "value": 0.0}
        )

        for p in self.patterns:
            by_classification[p.classification.value]["items"] += p.item_count
            by_classification[p.classification.value]["value"] += p.total_value

        return {
            "total_shrinkage": self.total_shrinkage,
            "explained_value": self.explained_value,
            "unexplained_value": self.unexplained_value,
            "reduction_percent": self.reduction_percent,
            "patterns_total": len(self.patterns),
            "patterns_answered": min(self.current_index, len(self.patterns)),
            "by_classification": dict(by_classification),
        }


# =============================================================================
# Diagnostic Engine
# =============================================================================


class DiagnosticEngine:
    """Conversational diagnostic that asks about EVERY detected pattern.

    No auto-classification -- user confirms everything.

    Usage::

        engine = DiagnosticEngine()
        session = engine.start_session(items)
        # Loop:
        q = engine.get_current_question(session)
        result = engine.answer_question(session, "receiving_gap")
        # ...until result["is_complete"]
        report = engine.get_final_report(session)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self, items: list[dict]) -> DiagnosticSession:
        """Start a new diagnostic session.

        Args:
            items: List of inventory dicts with keys:
                sku, description, stock (float), cost (float)

        Returns:
            A new DiagnosticSession with detected patterns.
        """
        session = DiagnosticSession(
            session_id=f"diag-{uuid4().hex[:8]}",
        )

        # Parse into InventoryItems
        parsed = [
            InventoryItem(
                sku=str(item.get("sku", "")),
                description=str(item.get("description", "")),
                stock=float(item.get("stock", 0)),
                cost=float(item.get("cost", 0)),
            )
            for item in items
            if item.get("sku", "")
        ]

        # Filter to negative stock
        negative = [i for i in parsed if i.stock < 0]

        session.items_analyzed = len(parsed)
        session.negative_items = len(negative)
        session.total_shrinkage = sum(i.shrinkage_value for i in negative)

        # Detect patterns
        session.patterns = self._detect_patterns(negative)

        logger.info(
            "Diagnostic session %s: %d items, %d negative, "
            "$%.0f shrinkage, %d patterns",
            session.session_id,
            session.items_analyzed,
            session.negative_items,
            session.total_shrinkage,
            len(session.patterns),
        )

        return session

    def get_current_question(self, session: DiagnosticSession) -> dict | None:
        """Get the current question to present to the user.

        Returns:
            Dict with question data, or None if session is complete.
        """
        if session.is_complete:
            return None

        pattern = session.current_pattern
        if not pattern:
            return None

        # Build sample items (first 5)
        sample_items = [
            {
                "sku": item.sku,
                "description": item.description,
                "stock": item.stock,
                "value": item.shrinkage_value,
            }
            for item in pattern.items[:5]
        ]

        return {
            "pattern_id": pattern.pattern_id,
            "pattern_name": pattern.name,
            "question": pattern.question,
            "suggested_answers": pattern.suggested_answers,
            "item_count": pattern.item_count,
            "total_value": pattern.total_value,
            "sample_items": sample_items,
            "progress": {
                "current": session.current_index + 1,
                "total": len(session.patterns),
            },
            "running_totals": session.get_summary(),
        }

    def answer_question(
        self,
        session: DiagnosticSession,
        classification: str,
        note: str = "",
    ) -> dict:
        """Process user's answer to the current question.

        Args:
            session: Active diagnostic session.
            classification: One of: receiving_gap, non_tracked, vendor_managed,
                expiration, theft, investigate
            note: Optional user note.

        Returns:
            Dict with answered pattern info, progress, running totals,
            is_complete flag, and next_question (or None).
        """
        if session.is_complete:
            return {"error": "No active question"}

        pattern = session.current_pattern
        if not pattern:
            return {"error": "No current pattern"}

        # Record classification
        try:
            pattern.classification = Classification(classification)
        except ValueError:
            pattern.classification = Classification.INVESTIGATE

        pattern.user_note = note

        # Advance to next pattern
        session.current_index += 1

        # Build response
        return {
            "answered": {
                "pattern": pattern.pattern_id,
                "classification": pattern.classification.value,
                "value": pattern.total_value,
            },
            "progress": {
                "current": session.current_index,
                "total": len(session.patterns),
            },
            "running_totals": session.get_summary(),
            "is_complete": session.is_complete,
            "next_question": self.get_current_question(session),
        }

    def get_final_report(self, session: DiagnosticSession) -> dict:
        """Generate the final diagnostic report.

        Returns:
            Dict with summary, patterns grouped by classification,
            items to investigate, and the classification journey.
        """
        summary = session.get_summary()

        # Group patterns by classification
        by_class: dict[str, list[dict]] = defaultdict(list)
        for p in session.patterns:
            by_class[p.classification.value].append(
                {
                    "name": p.name,
                    "items": p.item_count,
                    "value": p.total_value,
                }
            )

        # Top items still needing investigation
        investigate_items: list[dict] = []
        for p in session.patterns:
            if p.classification in (
                Classification.INVESTIGATE,
                Classification.THEFT,
            ):
                for item in p.items:
                    investigate_items.append(
                        {
                            "sku": item.sku,
                            "description": item.description,
                            "stock": item.stock,
                            "value": item.shrinkage_value,
                            "pattern": p.name,
                            "classification": p.classification.value,
                        }
                    )

        investigate_items.sort(key=lambda x: x["value"], reverse=True)

        return {
            "session_id": session.session_id,
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
                for p in session.patterns
            ],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_patterns(
        self,
        items: list[InventoryItem],
    ) -> list[DetectedPattern]:
        """Detect keyword patterns in negative stock items.

        Each item can only match ONE pattern (first match wins).
        Unmatched items are collected into an "other" bucket.
        Results sorted by total_value (highest first).
        """
        matched_skus: set[str] = set()
        patterns: list[DetectedPattern] = []

        for pattern_id, config in DETECTION_PATTERNS.items():
            matching: list[InventoryItem] = []

            for item in items:
                if item.sku in matched_skus:
                    continue

                sku_upper = item.sku.upper()
                desc_upper = item.description.upper()

                for keyword in config["keywords"]:
                    if keyword in sku_upper or keyword in desc_upper:
                        matching.append(item)
                        matched_skus.add(item.sku)
                        break

            if matching:
                total_value = sum(i.shrinkage_value for i in matching)
                question = config["question"].format(
                    count=len(matching),
                    value=f"${total_value:,.0f}",
                )

                patterns.append(
                    DetectedPattern(
                        pattern_id=pattern_id,
                        name=pattern_id.replace("_", " ").title(),
                        keywords=config["keywords"],
                        items=matching,
                        total_value=total_value,
                        question=question,
                        suggested_answers=config["suggested_answers"],
                    )
                )

        # Collect unmatched items as "Other"
        unmatched = [i for i in items if i.sku not in matched_skus]
        if unmatched:
            total_value = sum(i.shrinkage_value for i in unmatched)
            patterns.append(
                DetectedPattern(
                    pattern_id="other",
                    name="Other Items",
                    keywords=[],
                    items=unmatched,
                    total_value=total_value,
                    question=(
                        f"Found {len(unmatched)} **other items** that don't match "
                        f"common patterns, with ${total_value:,.0f} in negative "
                        f"stock.\n\nThese need individual review. "
                        f"How would you like to handle them?"
                    ),
                    suggested_answers=[
                        ["Mark for investigation", "investigate"],
                        ["Likely misc process issues", "receiving_gap"],
                    ],
                )
            )

        # Sort by value (highest first) so most impactful is asked first
        patterns.sort(key=lambda p: p.total_value, reverse=True)

        return patterns


# =============================================================================
# Render helpers for text output
# =============================================================================


async def enhance_question_with_llm(
    question: dict,
    anthropic_api_key: str,
) -> dict:
    """Optionally enhance a diagnostic question with Claude narration.

    Takes the structured question dict from get_current_question()
    and adds a 'narrative_question' field with Claude's conversational
    version. Falls back silently — the original 'question' field
    is always preserved.

    Args:
        question: The question dict from get_current_question()
        anthropic_api_key: Anthropic API key. If empty, returns question unchanged.

    Returns:
        The question dict, possibly with an added 'narrative_question' field.
    """
    if not anthropic_api_key or not question:
        return question

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_api_key)

        # Build context from the question data
        pattern_name = question.get("pattern_name", "Unknown pattern")
        item_count = question.get("item_count", 0)
        total_value = question.get("total_value", 0)
        sample_items = question.get("sample_items", [])
        suggested_answers = question.get("suggested_answers", [])
        original_question = question.get("question", "")

        sample_descriptions = [
            item.get("description", item.get("sku", "")) for item in sample_items[:3]
        ]

        prompt = (
            "You are Profit Sentinel's diagnostic assistant helping a hardware "
            "store owner understand inventory anomalies. Rewrite this diagnostic "
            "question in a natural, conversational tone. Reference specific items "
            "from the data to make it concrete.\n\n"
            f"Pattern detected: {pattern_name}\n"
            f"Items affected: {item_count}\n"
            f"Total value: ${total_value:,.0f}\n"
            f"Sample items: {', '.join(sample_descriptions)}\n"
            f"Template question: {original_question}\n"
            f"Possible explanations: {', '.join(str(a) for a in suggested_answers)}\n\n"
            "RULES:\n"
            "- Keep it under 3 sentences\n"
            "- Reference 1-2 specific items by name to make it concrete\n"
            "- Suggest the most likely explanation first based on the pattern\n"
            "- End with an open question so the owner can correct you\n"
            "- Use plain language, not technical jargon\n"
            "- Do NOT use bullet points or lists\n\n"
            "Return ONLY the rewritten question text, nothing else."
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        narrative = response.content[0].text.strip()

        # Preserve the original, add the narrative version
        question["narrative_question"] = narrative

    except Exception as e:
        logger.warning(f"Diagnostic LLM enhancement failed (non-fatal): {e}")
        # Silently fall back — original question is untouched

    return question


async def narrate_diagnostic_report(
    report: dict,
    anthropic_api_key: str,
) -> dict:
    """Add a narrative summary to the diagnostic report.

    Falls back silently — original report is always preserved.
    """
    if not anthropic_api_key or not report:
        return report

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_api_key)

        summary = report.get("summary", {})
        classifications = report.get("by_classification", {})

        prompt = (
            "You are Profit Sentinel's diagnostic assistant. Write a brief "
            "closing summary of this inventory diagnostic session for a "
            "hardware store owner.\n\n"
            "Session results:\n"
            f"- Items analyzed: {summary.get('items_analyzed', 0)}\n"
            f"- Negative inventory items: {summary.get('negative_items', 0)}\n"
            f"- Total shrinkage value: ${summary.get('total_shrinkage', 0):,.0f}\n"
            f"- Classifications: {classifications}\n\n"
            "RULES:\n"
            "- 2-3 sentences maximum\n"
            "- Lead with the most important finding\n"
            "- If theft/shrinkage was identified, mention it directly but "
            "constructively\n"
            "- End with a concrete next step\n"
            "- Plain language, no jargon\n\n"
            "Return ONLY the summary text."
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        report["narrative_summary"] = response.content[0].text.strip()

    except Exception as e:
        logger.warning(f"Diagnostic report narration failed (non-fatal): {e}")

    return report


def render_diagnostic_summary(session: DiagnosticSession) -> str:
    """Render a text summary of the diagnostic session."""
    summary = session.get_summary()
    lines = [
        "DIAGNOSTIC SUMMARY",
        "=" * 50,
        f"Total Apparent Shrinkage:  ${summary['total_shrinkage']:>10,.0f}",
        f"Process Issues Found:      ${summary['explained_value']:>10,.0f}",
        f"Still to Investigate:      ${summary['unexplained_value']:>10,.0f}",
        f"Reduction:                     {summary['reduction_percent']:>6.1f}%",
        "",
        f"Patterns reviewed: {summary['patterns_answered']}/{summary['patterns_total']}",
    ]

    # Breakdown by classification
    if summary["by_classification"]:
        lines.append("")
        lines.append("By Classification:")
        for cls_value, data in sorted(summary["by_classification"].items()):
            try:
                cls = Classification(cls_value)
                label = cls.display_name
            except ValueError:
                label = cls_value
            lines.append(
                f"  {label:<25} {data['items']:>4} items  ${data['value']:>10,.0f}"
            )

    return "\n".join(lines)


def render_diagnostic_report(report: dict) -> str:
    """Render the final report as text."""
    summary = report["summary"]
    lines = [
        "SHRINKAGE DIAGNOSTIC REPORT",
        "=" * 60,
        "",
        f"Total Apparent Shrinkage:  ${summary['total_shrinkage']:>10,.0f}",
        f"Process Issues Identified: ${summary['explained_value']:>10,.0f}",
        f"Remaining to Investigate:  ${summary['unexplained_value']:>10,.0f}",
        "",
        f"{'=' * 40}",
        f"REDUCTION: {summary['reduction_percent']:.1f}%",
        f"{'=' * 40}",
        "",
    ]

    # Journey
    lines.append("Classification Journey:")
    for step in report.get("journey", []):
        try:
            cls = Classification(step["classification"])
            label = cls.display_name
        except ValueError:
            label = step["classification"]
        lines.append(f"  {step['pattern']:<25} ${step['value']:>10,.0f}  -> {label}")

    # Top items to investigate
    investigate = report.get("items_to_investigate", [])
    if investigate:
        lines.append("")
        lines.append(f"Top {min(10, len(investigate))} Items to Investigate:")
        for item in investigate[:10]:
            lines.append(
                f"  {item['sku']:<20} ${item['value']:>8,.0f}  "
                f"({item['classification']})"
            )

    return "\n".join(lines)
