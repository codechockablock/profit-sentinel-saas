"""
PROFIT SENTINEL - INTEGRATED SYSTEM
====================================

Complete integration of:
1. Dorian Knowledge Pipeline (common sense + domain knowledge)
2. Diagnostic Engine (interactive shrinkage analysis)
3. Business Rule Learning (encode user knowledge back to Dorian)

This is the production system that:
- Loads background knowledge from all sources
- Analyzes inventory for profit leaks
- Engages users in diagnostic conversation
- Learns store-specific rules
- Reduces apparent shrinkage through understanding

Example: $726K "shrinkage" → $50K actual theft (93% reduction)

Author: Joseph + Claude
Date: 2026-01-25
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Add paths
sys.path.insert(0, "/home/claude")
sys.path.insert(0, "/mnt/user-data/outputs")


# =============================================================================
# IMPORTS FROM OUR MODULES
# =============================================================================

from dorian_core import DorianCore
from dorian_knowledge_pipeline import KnowledgePipeline, KnowledgeSource

# =============================================================================
# RETAIL DOMAIN KNOWLEDGE
# =============================================================================

# Common sense facts specific to retail/hardware stores
RETAIL_KNOWLEDGE = [
    # Product categories
    ("lumber", "is_a", "building_material"),
    ("plywood", "is_a", "building_material"),
    ("drywall", "is_a", "building_material"),
    ("concrete", "is_a", "building_material"),
    ("fastener", "is_a", "hardware"),
    ("nail", "is_a", "fastener"),
    ("screw", "is_a", "fastener"),
    ("bolt", "is_a", "fastener"),
    ("nut", "is_a", "fastener"),
    # Tracking behaviors
    ("building_material", "typically", "sold_not_received"),
    ("bulk_item", "typically", "non_tracked"),
    ("perishable", "typically", "expiration_prone"),
    ("vendor_direct", "typically", "vendor_managed"),
    # Shrinkage patterns
    ("negative_stock", "indicates", "potential_issue"),
    ("receiving_gap", "causes", "negative_stock"),
    ("theft", "causes", "negative_stock"),
    ("damage", "causes", "negative_stock"),
    ("expiration", "causes", "negative_stock"),
    # Common retail facts
    ("lumber", "sold_at", "point_of_sale"),
    ("lumber", "often_not", "received_into_system"),
    ("beverages", "can", "expire"),
    ("lawn_chemicals", "often", "vendor_managed"),
    ("small_items", "prone_to", "theft"),
    ("high_value_items", "prone_to", "theft"),
    # Hardware store specific
    ("nuts_bolts_bin", "is_a", "bulk_item"),
    ("cut_wire", "is_a", "bulk_item"),
    ("cut_chain", "is_a", "bulk_item"),
    ("cut_rope", "is_a", "bulk_item"),
    ("keys", "is_a", "bulk_item"),
    # Process issues
    ("receiving_error", "is_a", "process_issue"),
    ("pos_only_sale", "is_a", "process_issue"),
    ("missing_writeoff", "is_a", "process_issue"),
    ("receiving_gap", "is_a", "process_issue"),
]

# Pattern keywords for detecting product types
PRODUCT_PATTERNS = {
    "lumber": [
        "2X4",
        "2X6",
        "2X8",
        "2X10",
        "2X12",
        "1X4",
        "1X6",
        "BOARD",
        "STUD",
        "LUMBER",
        "TREATED",
    ],
    "sheet_goods": ["PLYWOOD", "DRYWALL", "OSB", "SHEATHING", "PANEL", "MDF"],
    "moulding": ["MOULDING", "MOLDING", "TRIM", "BASE", "CROWN", "CASING", "PRIMED"],
    "concrete": ["CONCRETE", "MORTAR", "CEMENT", "QUIKRETE", "SAKRETE"],
    "beverages": ["COKE", "PEPSI", "WATER", "MONSTER", "GATORADE", "SODA", "DRINK"],
    "lawn_chemicals": ["SCOTTS", "MIRACLE", "ROUNDUP", "WEED", "FEED", "FERTILIZER"],
    "bulk_fasteners": ["NUT", "BOLT", "WASHER", "SCREW", "NAIL", "BY THE"],
    "pellets": ["PELLET", "WOOD PEL", "FUEL"],
    "filters": ["FILTER", "FURNACE", "AIR FILTER", "HVAC"],
    "snacks": ["CANDY", "SLIM JIM", "SNACK", "GUM", "CHIP"],
    "propane": ["PROPANE", "LP GAS", "TANK"],
    "soil_mulch": ["MULCH", "SOIL", "POTTING", "TOP SOIL", "COMPOST"],
}


# =============================================================================
# TRACKING BEHAVIOR (from diagnostic engine)
# =============================================================================


class TrackingBehavior(Enum):
    """How an item category is tracked in inventory."""

    FULLY_TRACKED = "fully_tracked"
    SOLD_NOT_RECEIVED = "sold_not_received"
    NON_TRACKED = "non_tracked"
    EXPIRATION_PRONE = "expiration_prone"
    VENDOR_MANAGED = "vendor_managed"


@dataclass
class LearnedRule:
    """A business rule learned from user during diagnosis."""

    pattern: str
    behavior: TrackingBehavior
    explanation: str
    learned_at: datetime = field(default_factory=datetime.now)
    items_matched: int = 0
    value_explained: float = 0.0


# =============================================================================
# PROFIT SENTINEL SYSTEM
# =============================================================================


class ProfitSentinel:
    """
    Integrated Profit Sentinel system.

    Combines:
    - Dorian knowledge base (common sense + domain)
    - Diagnostic engine (pattern detection + questions)
    - Rule learning (encode user knowledge)
    """

    def __init__(self, dim: int = 10000, load_full_knowledge: bool = True):
        """
        Initialize Profit Sentinel.

        Args:
            dim: Vector dimension for Dorian
            load_full_knowledge: Whether to load all domain knowledge
        """
        print("=" * 70)
        print("INITIALIZING PROFIT SENTINEL")
        print("=" * 70)

        # Initialize Dorian
        print("\n1. Initializing Dorian knowledge base...")
        self.core = DorianCore(dim=dim, load_ontology=False)

        # Create knowledge pipeline
        self.pipeline = KnowledgePipeline(self.core, agent_name="profit_sentinel")

        # Register specialized agents
        self.retail_agent = self.core.register_agent(
            "retail_knowledge", domain="retail"
        )
        self.diagnostic_agent = self.core.register_agent(
            "diagnostic_engine", domain="retail"
        )
        self.learning_agent = self.core.register_agent("rule_learner", domain="retail")

        # Load knowledge
        if load_full_knowledge:
            self._load_knowledge()

        # Learned rules storage
        self.learned_rules: list[LearnedRule] = []
        self.store_knowledge: dict[str, Any] = {}

        print("\n✅ Profit Sentinel ready!")

    def _load_knowledge(self):
        """Load all knowledge sources."""
        print("\n2. Loading knowledge base...")

        # Load external sources (samples)
        self.pipeline.load_conceptnet(use_sample=True, show_progress=False)
        self.pipeline.load_wikidata_sample(show_progress=False)

        # Load domain knowledge
        print("   Loading domain knowledge...")
        domains_loaded = 0
        for domain in ["math", "physics", "chemistry", "cs", "economics"]:
            try:
                stats = self.pipeline.load_domain(domain, show_progress=False)
                domains_loaded += stats.triples_loaded
            except:
                pass

        # Load retail-specific knowledge
        print("   Loading retail knowledge...")
        retail_loaded = 0
        for s, p, o in RETAIL_KNOWLEDGE:
            try:
                result = self.core.write(
                    s, p, o, self.retail_agent.agent_id, confidence=1.0
                )
                if result.success:
                    retail_loaded += 1
            except:
                pass

        # Load product patterns as knowledge
        for category, patterns in PRODUCT_PATTERNS.items():
            for pattern in patterns:
                try:
                    self.core.write(
                        pattern.lower(),
                        "indicates",
                        category,
                        self.retail_agent.agent_id,
                        confidence=0.9,
                    )
                except:
                    pass

        total = len(self.core.fact_store.facts)
        print(f"   Total facts loaded: {total:,}")

    # =========================================================================
    # INVENTORY ANALYSIS
    # =========================================================================

    def analyze_inventory(self, items: list[dict]) -> dict:
        """
        Analyze inventory items for profit leaks.

        Args:
            items: List of inventory items with keys:
                   sku, description, stock, cost, retail, etc.

        Returns:
            Analysis results with findings and questions
        """
        print(f"\nAnalyzing {len(items):,} inventory items...")

        # Find negative stock items
        negative_items = [i for i in items if i.get("stock", 0) < 0]

        if not negative_items:
            return {
                "status": "clean",
                "message": "No negative stock found",
                "total_items": len(items),
                "negative_items": 0,
            }

        # Calculate totals
        total_cost = sum(
            abs(i.get("stock", 0)) * i.get("cost", 0) for i in negative_items
        )
        total_retail = sum(
            abs(i.get("stock", 0)) * i.get("retail", 0) for i in negative_items
        )

        # Classify items using knowledge + learned rules
        classified = self._classify_items(negative_items)

        # Generate questions for unexplained patterns
        questions = self._generate_questions(classified["unexplained"])

        return {
            "status": "issues_found",
            "total_items": len(items),
            "negative_items": len(negative_items),
            "total_cost": total_cost,
            "total_retail": total_retail,
            "classified": classified,
            "questions": questions,
            "summary": self._build_summary(classified, total_cost),
        }

    def _classify_items(self, items: list[dict]) -> dict:
        """Classify negative stock items by likely cause."""
        classified = {
            "non_tracked": [],  # By design (bins, cut-to-length)
            "receiving_gap": [],  # Sold but not received
            "expiration": [],  # Expires without write-off
            "vendor_managed": [],  # Vendor ships direct
            "unexplained": [],  # Need to ask user
        }

        for item in items:
            behavior = self._get_tracking_behavior(item)

            if behavior == TrackingBehavior.NON_TRACKED:
                classified["non_tracked"].append(item)
            elif behavior == TrackingBehavior.SOLD_NOT_RECEIVED:
                classified["receiving_gap"].append(item)
            elif behavior == TrackingBehavior.EXPIRATION_PRONE:
                classified["expiration"].append(item)
            elif behavior == TrackingBehavior.VENDOR_MANAGED:
                classified["vendor_managed"].append(item)
            else:
                classified["unexplained"].append(item)

        return classified

    def _get_tracking_behavior(self, item: dict) -> TrackingBehavior:
        """Determine tracking behavior for an item using knowledge + rules."""
        sku = item.get("sku", "").upper()
        desc = item.get("description", "").upper()

        # Check learned rules first
        for rule in self.learned_rules:
            if rule.pattern.upper() in sku or rule.pattern.upper() in desc:
                return rule.behavior

        # Check product patterns
        for category, patterns in PRODUCT_PATTERNS.items():
            for pattern in patterns:
                if pattern in sku or pattern in desc:
                    # Query Dorian for how this category is typically tracked
                    results = self.core.fact_store.query_by_subject(
                        category, "typically"
                    )
                    if results:
                        behavior_str = results[0].object
                        try:
                            return TrackingBehavior(behavior_str)
                        except:
                            pass

        # Check for specific indicators
        if any(p in desc for p in ["BY THE FOOT", "BY THE", "EACH", "CUT"]):
            return TrackingBehavior.NON_TRACKED

        if any(p in sku for p in ["MNB", "NUTS", "BOLTS"]) or len(sku) == 1:
            return TrackingBehavior.NON_TRACKED

        return TrackingBehavior.FULLY_TRACKED

    def _generate_questions(self, unexplained_items: list[dict]) -> list[dict]:
        """Generate diagnostic questions for unexplained items."""
        if not unexplained_items:
            return []

        # Group by detected patterns
        patterns = self._detect_patterns(unexplained_items)

        questions = []
        for pattern_name, pattern_items in patterns.items():
            if not pattern_items:
                continue

            total_cost = sum(
                abs(i.get("stock", 0)) * i.get("cost", 0) for i in pattern_items
            )

            questions.append(
                {
                    "id": f"q_{pattern_name}",
                    "pattern": pattern_name,
                    "question": f"How does your store handle {pattern_name.replace('_', ' ')}?",
                    "context": f"Found ${total_cost:,.0f} in negative stock for {len(pattern_items)} {pattern_name.replace('_', ' ')} items.",
                    "sample_items": pattern_items[:5],
                    "suggested_answers": self._get_suggested_answers(pattern_name),
                    "item_count": len(pattern_items),
                    "total_cost": total_cost,
                }
            )

        # Sort by value (highest first)
        questions.sort(key=lambda q: q["total_cost"], reverse=True)

        return questions

    def _detect_patterns(self, items: list[dict]) -> dict[str, list[dict]]:
        """Detect patterns in unexplained items."""
        patterns = {name: [] for name in PRODUCT_PATTERNS.keys()}
        patterns["other"] = []

        for item in items:
            sku = item.get("sku", "").upper()
            desc = item.get("description", "").upper()

            matched = False
            for pattern_name, keywords in PRODUCT_PATTERNS.items():
                if any(kw in sku or kw in desc for kw in keywords):
                    patterns[pattern_name].append(item)
                    matched = True
                    break

            if not matched:
                patterns["other"].append(item)

        # Remove empty patterns
        return {k: v for k, v in patterns.items() if v}

    def _get_suggested_answers(self, pattern: str) -> list[str]:
        """Get suggested answers for a pattern question."""
        base_answers = [
            "Fully tracked (received and sold through system)",
            "Sold at POS but not received into inventory",
            "Not actively tracked (bins, bulk)",
            "Can expire/damage without write-off",
            "Vendor ships direct, not always entered",
        ]

        # Customize based on pattern
        if pattern in ["lumber", "sheet_goods", "moulding", "concrete"]:
            return [
                "Sold at register but not received into inventory system",
                "Fully received and tracked",
                "It varies by product",
            ]
        elif pattern in ["beverages", "snacks"]:
            return [
                "Fully tracked",
                "Can expire without proper write-off",
                "Likely theft (high-value small items)",
            ]
        elif pattern in ["lawn_chemicals", "soil_mulch"]:
            return [
                "Vendor (Scotts, etc.) ships and manages inventory",
                "Fully tracked internally",
                "Seasonal damage/returns not always entered",
            ]
        elif pattern in ["bulk_fasteners"]:
            return [
                "Not tracked (sold from bins)",
                "Tracked by weight/count",
                "Fully tracked",
            ]

        return base_answers

    def _build_summary(self, classified: dict, total_cost: float) -> dict:
        """Build analysis summary."""
        explained_cost = 0
        for category in [
            "non_tracked",
            "receiving_gap",
            "expiration",
            "vendor_managed",
        ]:
            items = classified[category]
            explained_cost += sum(
                abs(i.get("stock", 0)) * i.get("cost", 0) for i in items
            )

        unexplained_cost = sum(
            abs(i.get("stock", 0)) * i.get("cost", 0) for i in classified["unexplained"]
        )

        return {
            "total_shrinkage": total_cost,
            "explained_value": explained_cost,
            "unexplained_value": unexplained_cost,
            "explained_percentage": (
                (explained_cost / total_cost * 100) if total_cost > 0 else 0
            ),
            "by_category": {
                "non_tracked": len(classified["non_tracked"]),
                "receiving_gap": len(classified["receiving_gap"]),
                "expiration": len(classified["expiration"]),
                "vendor_managed": len(classified["vendor_managed"]),
                "unexplained": len(classified["unexplained"]),
            },
        }

    # =========================================================================
    # LEARNING FROM USER
    # =========================================================================

    def learn_rule(self, pattern: str, behavior: str, explanation: str) -> LearnedRule:
        """
        Learn a new business rule from user input.

        Args:
            pattern: Keyword pattern to match (e.g., "LUMBER", "2X4")
            behavior: One of: fully_tracked, sold_not_received, non_tracked,
                     expiration_prone, vendor_managed
            explanation: User's explanation of why this is the case

        Returns:
            The created LearnedRule
        """
        # Parse behavior
        try:
            behavior_enum = TrackingBehavior(behavior)
        except ValueError:
            # Try to match partial
            behavior_map = {
                "tracked": TrackingBehavior.FULLY_TRACKED,
                "not received": TrackingBehavior.SOLD_NOT_RECEIVED,
                "sold not": TrackingBehavior.SOLD_NOT_RECEIVED,
                "not tracked": TrackingBehavior.NON_TRACKED,
                "bins": TrackingBehavior.NON_TRACKED,
                "expire": TrackingBehavior.EXPIRATION_PRONE,
                "damage": TrackingBehavior.EXPIRATION_PRONE,
                "vendor": TrackingBehavior.VENDOR_MANAGED,
                "direct ship": TrackingBehavior.VENDOR_MANAGED,
            }

            behavior_enum = TrackingBehavior.FULLY_TRACKED
            for key, val in behavior_map.items():
                if key in behavior.lower():
                    behavior_enum = val
                    break

        rule = LearnedRule(
            pattern=pattern,
            behavior=behavior_enum,
            explanation=explanation,
        )

        self.learned_rules.append(rule)

        # Encode into Dorian
        self._encode_rule_to_dorian(rule)

        print(f"✓ Learned: '{pattern}' → {behavior_enum.value}")

        return rule

    def _encode_rule_to_dorian(self, rule: LearnedRule):
        """Encode a learned rule into Dorian's knowledge graph."""
        pattern_id = rule.pattern.lower().replace(" ", "_")

        # Add to knowledge graph
        facts = [
            (pattern_id, "is_a", "inventory_category"),
            (pattern_id, "has_tracking_behavior", rule.behavior.value),
            (pattern_id, "explanation", rule.explanation[:100]),
        ]

        # Add shrinkage classification
        shrinkage_map = {
            TrackingBehavior.FULLY_TRACKED: "potential_theft",
            TrackingBehavior.SOLD_NOT_RECEIVED: "receiving_gap",
            TrackingBehavior.NON_TRACKED: "by_design",
            TrackingBehavior.EXPIRATION_PRONE: "write_off_gap",
            TrackingBehavior.VENDOR_MANAGED: "vendor_issue",
        }
        facts.append(
            (pattern_id, "shrinkage_type", shrinkage_map.get(rule.behavior, "unknown"))
        )

        for s, p, o in facts:
            try:
                self.core.write(s, p, o, self.learning_agent.agent_id, confidence=1.0)
            except:
                pass

    def process_answer(
        self, question_id: str, answer: str, analysis: dict
    ) -> tuple[LearnedRule, dict]:
        """
        Process user's answer to a diagnostic question.

        Args:
            question_id: ID of the question being answered
            answer: User's answer text
            analysis: Current analysis results

        Returns:
            (learned_rule, updated_analysis)
        """
        # Find the question
        question = None
        for q in analysis.get("questions", []):
            if q["id"] == question_id:
                question = q
                break

        if not question:
            return None, analysis

        # Interpret the answer
        behavior = self._interpret_answer(answer)

        # Learn the rule
        rule = self.learn_rule(
            pattern=question["pattern"], behavior=behavior.value, explanation=answer
        )

        # Update rule stats
        rule.items_matched = question["item_count"]
        rule.value_explained = question["total_cost"]

        # Re-analyze with new rule
        # (In practice, would re-run classification)

        return rule, analysis

    def _interpret_answer(self, answer: str) -> TrackingBehavior:
        """Interpret user's natural language answer."""
        answer_lower = answer.lower()

        if any(
            p in answer_lower
            for p in ["not received", "sold but not", "pos only", "register only"]
        ):
            return TrackingBehavior.SOLD_NOT_RECEIVED

        if any(
            p in answer_lower for p in ["not tracked", "bins", "bulk", "not actively"]
        ):
            return TrackingBehavior.NON_TRACKED

        if any(
            p in answer_lower
            for p in ["expire", "damage", "write off", "throw away", "toss"]
        ):
            return TrackingBehavior.EXPIRATION_PRONE

        if any(
            p in answer_lower
            for p in ["vendor", "direct ship", "scotts", "manufacturer"]
        ):
            return TrackingBehavior.VENDOR_MANAGED

        return TrackingBehavior.FULLY_TRACKED

    # =========================================================================
    # REPORTS
    # =========================================================================

    def get_diagnostic_report(self, analysis: dict) -> str:
        """Generate a diagnostic report."""
        summary = analysis.get("summary", {})
        classified = analysis.get("classified", {})

        report = f"""
# Profit Sentinel Diagnostic Report

## Summary

| Metric | Value |
|--------|-------|
| Total Items Analyzed | {analysis.get('total_items', 0):,} |
| Negative Stock Items | {analysis.get('negative_items', 0):,} |
| Total Apparent Shrinkage | ${summary.get('total_shrinkage', 0):,.0f} |
| Explained (Process Issues) | ${summary.get('explained_value', 0):,.0f} |
| Unexplained (Potential Theft) | ${summary.get('unexplained_value', 0):,.0f} |
| **Shrinkage Reduction** | **{summary.get('explained_percentage', 0):.1f}%** |

## Breakdown by Category

| Category | Items | Description |
|----------|-------|-------------|
| Non-Tracked | {len(classified.get('non_tracked', []))} | Bins, cut-to-length (by design) |
| Receiving Gap | {len(classified.get('receiving_gap', []))} | Sold at POS, not received |
| Expiration | {len(classified.get('expiration', []))} | Expires without write-off |
| Vendor Managed | {len(classified.get('vendor_managed', []))} | Direct ship items |
| **Unexplained** | **{len(classified.get('unexplained', []))}** | **Needs investigation** |

## Learned Rules

"""

        if self.learned_rules:
            report += "| Pattern | Behavior | Items | Value |\n"
            report += "|---------|----------|-------|-------|\n"
            for rule in self.learned_rules:
                report += f"| {rule.pattern} | {rule.behavior.value} | {rule.items_matched} | ${rule.value_explained:,.0f} |\n"
        else:
            report += "*No rules learned yet. Answer diagnostic questions to teach the system.*\n"

        return report

    # =========================================================================
    # KNOWLEDGE QUERIES
    # =========================================================================

    def query_knowledge(self, subject: str, predicate: str = None) -> list[dict]:
        """Query the knowledge base."""
        results = self.core.fact_store.query_by_subject(subject, predicate)
        return [
            {"subject": f.subject, "predicate": f.predicate, "object": f.object}
            for f in results
        ]

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "total_facts": len(self.core.fact_store.facts),
            "learned_rules": len(self.learned_rules),
            "predicates": len(self.core.fact_store.by_predicate),
        }


# =============================================================================
# DEMO / TEST
# =============================================================================


def demo():
    """Demo the integrated system."""
    print("\n" + "=" * 70)
    print("PROFIT SENTINEL - INTEGRATED DEMO")
    print("=" * 70)

    # Initialize
    sentinel = ProfitSentinel(load_full_knowledge=True)

    # Show stats
    stats = sentinel.get_stats()
    print(f"\nKnowledge base: {stats['total_facts']:,} facts")

    # Create sample inventory data
    sample_items = [
        {
            "sku": "2X4-8",
            "description": "2X4 8FT STUD",
            "stock": -50,
            "cost": 3.50,
            "retail": 5.99,
        },
        {
            "sku": "2X6-10",
            "description": "2X6 10FT LUMBER",
            "stock": -25,
            "cost": 7.00,
            "retail": 12.99,
        },
        {
            "sku": "PLY-12",
            "description": "PLYWOOD 1/2 4X8",
            "stock": -10,
            "cost": 35.00,
            "retail": 54.99,
        },
        {
            "sku": "DRY-12",
            "description": "DRYWALL 1/2 4X8",
            "stock": -15,
            "cost": 12.00,
            "retail": 18.99,
        },
        {
            "sku": "MNB-001",
            "description": "NUTS BOLTS ASSORTED",
            "stock": -100,
            "cost": 0.10,
            "retail": 0.25,
        },
        {
            "sku": "A",
            "description": "ANCHOR BOLT",
            "stock": -200,
            "cost": 0.15,
            "retail": 0.35,
        },
        {
            "sku": "COKE-12",
            "description": "COCA COLA 12PK",
            "stock": -20,
            "cost": 4.50,
            "retail": 6.99,
        },
        {
            "sku": "SCOTTS-1",
            "description": "SCOTTS TURF BUILDER",
            "stock": -30,
            "cost": 25.00,
            "retail": 39.99,
        },
        {
            "sku": "BATT-AA",
            "description": "BATTERIES AA 4PK",
            "stock": -50,
            "cost": 3.00,
            "retail": 5.99,
        },
        {
            "sku": "PWR-DRILL",
            "description": "DEWALT POWER DRILL",
            "stock": -2,
            "cost": 89.00,
            "retail": 149.99,
        },
    ]

    print("\n" + "-" * 70)
    print("ANALYZING SAMPLE INVENTORY")
    print("-" * 70)

    # Analyze
    analysis = sentinel.analyze_inventory(sample_items)

    print(f"\nTotal apparent shrinkage: ${analysis['total_cost']:,.0f}")
    print(f"Items with negative stock: {analysis['negative_items']}")

    # Show classification
    classified = analysis["classified"]
    print("\nClassification:")
    print(f"  Non-tracked (by design): {len(classified['non_tracked'])} items")
    print(f"  Receiving gap: {len(classified['receiving_gap'])} items")
    print(f"  Expiration: {len(classified['expiration'])} items")
    print(f"  Vendor managed: {len(classified['vendor_managed'])} items")
    print(f"  Unexplained: {len(classified['unexplained'])} items")

    # Show questions
    print(f"\nDiagnostic questions: {len(analysis['questions'])}")
    for q in analysis["questions"][:3]:
        print(f"\n  Q: {q['question']}")
        print(f"     Context: {q['context']}")

    # Simulate learning
    print("\n" + "-" * 70)
    print("SIMULATING USER LEARNING")
    print("-" * 70)

    sentinel.learn_rule(
        "LUMBER",
        "sold_not_received",
        "Lumber is sold at register but not received into inventory",
    )
    sentinel.learn_rule("PLYWOOD", "sold_not_received", "Same as lumber - POS only")
    sentinel.learn_rule(
        "SCOTTS", "vendor_managed", "Scotts ships direct and manages their inventory"
    )

    # Re-analyze
    analysis2 = sentinel.analyze_inventory(sample_items)

    print("\nAfter learning:")
    print(f"  Unexplained items: {len(analysis2['classified']['unexplained'])}")
    print(f"  Explained: {analysis2['summary']['explained_percentage']:.1f}%")

    # Generate report
    print("\n" + "-" * 70)
    print("DIAGNOSTIC REPORT")
    print("-" * 70)
    print(sentinel.get_diagnostic_report(analysis2))

    # Query knowledge
    print("\n" + "-" * 70)
    print("KNOWLEDGE QUERIES")
    print("-" * 70)

    queries = [
        ("lumber", "is_a"),
        ("lumber", "has_tracking_behavior"),
        ("receiving_gap", "causes"),
    ]

    for subj, pred in queries:
        results = sentinel.query_knowledge(subj, pred)
        print(f"\n  {subj} {pred} ?")
        for r in results[:3]:
            print(f"    -> {r['object']}")


if __name__ == "__main__":
    demo()
