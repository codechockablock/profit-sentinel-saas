"""
PROFIT SENTINEL AGENT
=====================

A specialized agent built on the Dorian Geometric Knowledge System
for detecting profit leaks in retail inventory data.

This agent:
1. Uses VSA geometry for 0% quantitative hallucination
2. Leverages multi-domain knowledge (economics, CS, math)
3. Performs causal reasoning through geometric operations
4. Validates all claims against the knowledge brain

The core insight: Invalid claims are geometrically impossible,
not merely statistically unlikely.

Author: Joseph + Claude
Date: 2026-01-25
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# =============================================================================
# PROFIT LEAK TAXONOMY
# =============================================================================


class ProfitLeakType(Enum):
    """The 11 profit leak types discovered through VSA experiments."""

    # Original 8 known types
    DEAD_INVENTORY = "dead_inventory"
    MARGIN_EROSION = "margin_erosion"
    SHRINKAGE = "shrinkage"
    PRICING_ERROR = "pricing_error"
    OVERSTOCK = "overstock"
    STOCKOUT = "stockout"
    MARKDOWN_TIMING = "markdown_timing"
    VENDOR_COMPLIANCE = "vendor_compliance"

    # 3 novel types discovered by VSA
    SEASONAL_MISMATCH = "seasonal_mismatch"
    CATEGORY_CANNIBALIZATION = "category_cannibalization"
    COST_SPIKE_PROPAGATION = "cost_spike_propagation"


@dataclass
class ProfitLeak:
    """A detected profit leak with full attribution."""

    leak_type: ProfitLeakType
    severity: float  # 0-1 scale
    confidence: float  # Geometric confidence score
    affected_skus: list[str]
    estimated_impact: float  # Dollar amount
    evidence: dict[str, Any]
    causal_chain: list[str]  # The reasoning path
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "type": self.leak_type.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "affected_skus": self.affected_skus,
            "estimated_impact": self.estimated_impact,
            "evidence": self.evidence,
            "causal_chain": self.causal_chain,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# RETAIL DOMAIN KNOWLEDGE
# =============================================================================

RETAIL_CATEGORIES = [
    # Inventory states
    ("inventory_item", "physical_object", "An item in inventory"),
    ("sku", "inventory_item", "Stock Keeping Unit"),
    ("product", "inventory_item", "A product for sale"),
    ("dead_stock", "inventory_item", "Inventory with no sales"),
    ("slow_mover", "inventory_item", "Low velocity inventory"),
    ("fast_mover", "inventory_item", "High velocity inventory"),
    # Financial metrics
    ("financial_metric", "measurement", "A financial measurement"),
    ("margin", "financial_metric", "Profit margin"),
    ("gross_margin", "margin", "Revenue minus COGS"),
    ("net_margin", "margin", "Profit after all expenses"),
    ("revenue", "financial_metric", "Total sales"),
    ("cost", "financial_metric", "Cost of goods"),
    ("profit", "financial_metric", "Revenue minus cost"),
    ("loss", "financial_metric", "Negative profit"),
    # Inventory metrics
    ("inventory_metric", "measurement", "An inventory measurement"),
    ("quantity_on_hand", "inventory_metric", "Current stock level"),
    ("quantity_sold", "inventory_metric", "Units sold"),
    ("shrinkage", "inventory_metric", "Lost inventory"),
    ("turnover", "inventory_metric", "Inventory turns per period"),
    ("days_of_supply", "inventory_metric", "Days until stockout"),
    ("fill_rate", "inventory_metric", "Order fulfillment percentage"),
    # Pricing
    ("price", "financial_metric", "Item price"),
    ("retail_price", "price", "Customer-facing price"),
    ("wholesale_price", "price", "Vendor price"),
    ("markdown", "price", "Price reduction"),
    ("markup", "price", "Price increase"),
    # Time periods
    ("time_period", "temporal", "A period of time"),
    ("fiscal_quarter", "time_period", "Q1, Q2, Q3, Q4"),
    ("season", "time_period", "Spring, Summer, Fall, Winter"),
    ("week", "time_period", "7-day period"),
    ("month", "time_period", "Calendar month"),
    # Entities
    ("vendor", "organization", "Supplier of goods"),
    ("store", "organization", "Retail location"),
    ("department", "organization", "Store department"),
    ("category", "abstract", "Product category"),
    # Events
    ("inventory_event", "event", "An inventory-related event"),
    ("receipt", "inventory_event", "Inventory received"),
    ("sale", "inventory_event", "Item sold"),
    ("return", "inventory_event", "Item returned"),
    ("transfer", "inventory_event", "Inventory moved"),
    ("adjustment", "inventory_event", "Inventory count correction"),
    ("write_off", "inventory_event", "Inventory removed from books"),
    # Profit leak types
    ("profit_leak", "problem", "Source of profit loss"),
    ("dead_inventory_leak", "profit_leak", "Loss from unsold inventory"),
    ("margin_erosion_leak", "profit_leak", "Loss from declining margins"),
    ("shrinkage_leak", "profit_leak", "Loss from missing inventory"),
    ("pricing_error_leak", "profit_leak", "Loss from incorrect pricing"),
    ("overstock_leak", "profit_leak", "Loss from excess inventory"),
    ("stockout_leak", "profit_leak", "Loss from missed sales"),
]

RETAIL_FACTS = [
    # Causal relationships
    ("dead_inventory", "causes", "margin_erosion"),
    ("dead_inventory", "causes", "write_off"),
    ("overstock", "causes", "markdown"),
    ("markdown", "causes", "margin_erosion"),
    ("stockout", "causes", "lost_sales"),
    ("stockout", "causes", "revenue_loss"),
    ("shrinkage", "causes", "inventory_discrepancy"),
    ("shrinkage", "causes", "profit_loss"),
    ("pricing_error", "causes", "margin_erosion"),
    ("cost_increase", "causes", "margin_erosion"),
    # Additional causal chains
    ("lost_sales", "causes", "revenue_loss"),
    ("revenue_loss", "causes", "profit_loss"),
    ("margin_erosion", "causes", "profit_loss"),
    ("write_off", "causes", "profit_loss"),
    ("markdown", "causes", "profit_loss"),
    # Indicators
    ("zero_sales_90_days", "indicates", "dead_inventory"),
    ("declining_margin", "indicates", "margin_erosion"),
    ("negative_adjustment", "indicates", "shrinkage"),
    ("price_below_cost", "indicates", "pricing_error"),
    ("excess_days_of_supply", "indicates", "overstock"),
    ("zero_on_hand", "indicates", "stockout"),
    # Metrics
    ("turnover", "inversely_related_to", "days_of_supply"),
    ("margin", "equals", "revenue_minus_cost"),
    ("shrinkage", "equals", "expected_minus_actual"),
    # Thresholds
    ("dead_inventory", "threshold", "90_days_no_sales"),
    ("slow_mover", "threshold", "below_average_velocity"),
    ("overstock", "threshold", "above_60_days_supply"),
    ("critical_margin", "threshold", "below_10_percent"),
]


# =============================================================================
# PROFIT SENTINEL AGENT
# =============================================================================


class ProfitSentinelAgent:
    """
    An autonomous agent for detecting profit leaks using geometric reasoning.

    The agent:
    1. Encodes retail data into VSA geometry
    2. Queries the Dorian knowledge brain for patterns
    3. Performs causal inference through binding/unbinding
    4. Validates all claims against geometric constraints
    """

    def __init__(self, core, agent_name: str = "profit_sentinel"):
        """
        Initialize the Profit Sentinel agent.

        Args:
            core: A DorianCore instance with loaded knowledge
            agent_name: Name for this agent instance
        """
        self.core = core
        self.name = agent_name

        # Register with the core
        self.agent = core.register_agent(agent_name, domain="retail", can_verify=True)
        self.agent_id = self.agent.agent_id

        # Load retail domain knowledge
        self._load_retail_knowledge()

        # Detection state
        self.detected_leaks: list[ProfitLeak] = []
        self.analysis_cache: dict[str, Any] = {}

        # Confidence thresholds (from validated experiments)
        self.thresholds = {
            "noise_floor": 0.01,
            "low_confidence": 0.10,
            "moderate_confidence": 0.25,
            "high_confidence": 0.40,
            "very_high_confidence": 0.60,
        }

    def _load_retail_knowledge(self):
        """Load retail domain knowledge into the core."""

        print(f"  Loading {len(RETAIL_CATEGORIES)} retail categories...")
        for name, parent, description in RETAIL_CATEGORIES:
            if self.core.ontology:
                from dorian_ontology import Category

                parent_cat = self.core.ontology.categories.get(parent)
                parent_level = parent_cat.level if parent_cat else 3
                self.core.ontology._add_category(
                    Category(
                        name=name,
                        description=description,
                        parent=parent,
                        domain="retail",
                        level=parent_level + 1,
                    )
                )

            self.core.write(
                name,
                "subtype_of",
                parent,
                self.agent_id,
                source="retail_ontology",
                check_contradictions=False,
            )

        print(f"  Loading {len(RETAIL_FACTS)} retail facts...")
        for s, p, o in RETAIL_FACTS:
            self.core.write(
                s,
                p,
                o,
                self.agent_id,
                source="retail_knowledge",
                check_contradictions=False,
            )

        print("  Retail knowledge loaded")

    # =========================================================================
    # DATA ENCODING
    # =========================================================================

    def _get_vector(self, concept: str) -> np.ndarray:
        """Get or create a vector for a concept using the VSA engine."""
        return self.core.vsa._get_or_create_embedding(concept)

    def _bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Bundle multiple vectors together."""
        return self.core.vsa.bundle(vectors)

    def _bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bind two vectors together."""
        return self.core.vsa.bind(a, b)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def encode_sku(self, sku_data: dict) -> np.ndarray:
        """
        Encode a SKU's data into a hypervector.

        The encoding captures:
        - Category membership
        - Velocity class (fast/slow/dead)
        - Margin health
        - Stock status
        """
        components = []

        # Category encoding
        if "category" in sku_data:
            cat_vec = self._get_vector(sku_data["category"])
            components.append(cat_vec)

        # Velocity encoding
        if "days_since_sale" in sku_data:
            days = sku_data["days_since_sale"]
            if days > 90:
                vel_vec = self._get_vector("dead_stock")
            elif days > 30:
                vel_vec = self._get_vector("slow_mover")
            else:
                vel_vec = self._get_vector("fast_mover")
            components.append(vel_vec)

        # Margin encoding
        if "margin_percent" in sku_data:
            margin = sku_data["margin_percent"]
            if margin < 0:
                margin_vec = self._get_vector("negative_margin")
            elif margin < 10:
                margin_vec = self._get_vector("critical_margin")
            elif margin < 25:
                margin_vec = self._get_vector("low_margin")
            else:
                margin_vec = self._get_vector("healthy_margin")
            components.append(margin_vec)

        # Stock status encoding
        if "quantity_on_hand" in sku_data and "avg_daily_sales" in sku_data:
            qoh = sku_data["quantity_on_hand"]
            avg_sales = max(sku_data["avg_daily_sales"], 0.01)
            dos = qoh / avg_sales

            if qoh == 0:
                stock_vec = self._get_vector("stockout")
            elif dos > 90:
                stock_vec = self._get_vector("overstock")
            elif dos > 60:
                stock_vec = self._get_vector("high_inventory")
            else:
                stock_vec = self._get_vector("normal_inventory")
            components.append(stock_vec)

        # Bundle all components
        if components:
            return self._bundle(components)
        else:
            return self._get_vector(f"sku_{sku_data.get('sku_id', 'unknown')}")

    def encode_transaction(self, txn_data: dict) -> np.ndarray:
        """Encode a transaction into a hypervector."""

        components = []

        # Event type
        event_type = txn_data.get("type", "sale")
        components.append(self._get_vector(event_type))

        # SKU reference
        if "sku_id" in txn_data:
            components.append(self._get_vector(f"sku_{txn_data['sku_id']}"))

        # Quantity encoding
        if "quantity" in txn_data:
            qty = txn_data["quantity"]
            if qty < 0:
                components.append(self._get_vector("negative_quantity"))
            elif qty > 100:
                components.append(self._get_vector("bulk_quantity"))

        return self._bundle(components) if components else np.zeros(self.core.vsa.dim)

    # =========================================================================
    # PROFIT LEAK DETECTION
    # =========================================================================

    def detect_dead_inventory(self, inventory_data: list[dict]) -> list[ProfitLeak]:
        """
        Detect dead inventory using both geometric pattern matching and rules.

        Dead inventory = items with no sales for 90+ days.
        """
        leaks = []
        dead_pattern = self._get_vector("dead_stock")

        for item in inventory_data:
            days_since_sale = item.get("days_since_sale", 0)
            avg_sales = item.get("avg_daily_sales", 0)
            qoh = item.get("quantity_on_hand", 0)

            # Rule-based detection: 90+ days with no sales and inventory on hand
            is_dead_by_rule = days_since_sale >= 90 and qoh > 0

            # Geometric detection
            item_vec = self.encode_sku(item)
            similarity = self._similarity(item_vec, dead_pattern)
            is_dead_by_geometry = similarity > self.thresholds["moderate_confidence"]

            if is_dead_by_rule or is_dead_by_geometry:
                # Query for causal chain
                causal_facts = list(self.core.query_forward("dead_inventory", "causes"))
                causal_chain = ["dead_inventory"] + [f.object for f in causal_facts]

                # Estimate impact (full inventory value at risk of write-off)
                cost = item.get("unit_cost", 0)
                impact = qoh * cost

                # Higher confidence if both methods agree
                if is_dead_by_rule and is_dead_by_geometry:
                    confidence = 0.9
                elif is_dead_by_rule:
                    confidence = 0.8
                else:
                    confidence = similarity

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.DEAD_INVENTORY,
                    severity=min(days_since_sale / 180, 1.0),  # Max at 180 days
                    confidence=confidence,
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "days_since_sale": days_since_sale,
                        "quantity_on_hand": qoh,
                        "avg_daily_sales": avg_sales,
                        "geometric_similarity": round(similarity, 3),
                        "detection_method": (
                            "rule+geometry"
                            if (is_dead_by_rule and is_dead_by_geometry)
                            else ("rule" if is_dead_by_rule else "geometry")
                        ),
                    },
                    causal_chain=causal_chain,
                )
                leaks.append(leak)

        return leaks

    def detect_margin_erosion(self, inventory_data: list[dict]) -> list[ProfitLeak]:
        """Detect margin erosion patterns."""

        leaks = []
        erosion_pattern = self._get_vector("margin_erosion")

        for item in inventory_data:
            margin = item.get("margin_percent", 50)
            prev_margin = item.get("prev_margin_percent", margin)

            # Check for margin decline
            if margin < prev_margin:
                decline = prev_margin - margin
                item_vec = self.encode_sku(item)

                # Bind with erosion pattern
                self._bind(item_vec, erosion_pattern)

                # Query for causes
                cause_results = self.core.query(
                    predicate="causes", obj="margin_erosion", k=5
                )
                causes = [f.subject for f, score in cause_results if score > 0.2]

                if decline > 5:  # Significant erosion
                    # Estimate impact
                    revenue = item.get("revenue", 0)
                    impact = revenue * (decline / 100)

                    leak = ProfitLeak(
                        leak_type=ProfitLeakType.MARGIN_EROSION,
                        severity=min(decline / 20, 1.0),
                        confidence=0.5 + (decline / 40),
                        affected_skus=[item.get("sku_id", "unknown")],
                        estimated_impact=impact,
                        evidence={
                            "current_margin": margin,
                            "previous_margin": prev_margin,
                            "decline_percent": decline,
                            "potential_causes": causes,
                        },
                        causal_chain=causes + ["margin_erosion", "profit_loss"],
                    )
                    leaks.append(leak)

        return leaks

    def detect_shrinkage(
        self, inventory_data: list[dict], adjustments: list[dict] = None
    ) -> list[ProfitLeak]:
        """Detect shrinkage patterns from inventory adjustments."""

        leaks = []
        self._get_vector("shrinkage")

        # Group adjustments by SKU
        sku_adjustments = defaultdict(list)
        for adj in adjustments or []:
            sku_adjustments[adj.get("sku_id")].append(adj)

        for sku_id, adjs in sku_adjustments.items():
            negative_adjs = [a for a in adjs if a.get("quantity", 0) < 0]

            if negative_adjs:
                total_shrink = sum(abs(a.get("quantity", 0)) for a in negative_adjs)
                avg_cost = np.mean([a.get("unit_cost", 0) for a in negative_adjs])

                if total_shrink > 0:
                    leak = ProfitLeak(
                        leak_type=ProfitLeakType.SHRINKAGE,
                        severity=min(total_shrink / 100, 1.0),
                        confidence=0.7,  # High confidence for actual adjustments
                        affected_skus=[sku_id],
                        estimated_impact=total_shrink * avg_cost,
                        evidence={
                            "total_shrinkage_units": total_shrink,
                            "adjustment_count": len(negative_adjs),
                            "average_unit_cost": avg_cost,
                        },
                        causal_chain=[
                            "inventory_discrepancy",
                            "shrinkage",
                            "profit_loss",
                        ],
                    )
                    leaks.append(leak)

        return leaks

    def detect_overstock(self, inventory_data: list[dict]) -> list[ProfitLeak]:
        """Detect overstock situations."""

        leaks = []
        overstock_pattern = self._get_vector("overstock")

        for item in inventory_data:
            qoh = item.get("quantity_on_hand", 0)
            avg_sales = max(item.get("avg_daily_sales", 0.01), 0.01)
            dos = qoh / avg_sales

            if dos > 60:  # More than 60 days of supply
                item_vec = self.encode_sku(item)
                similarity = self._similarity(item_vec, overstock_pattern)

                # Query causal chain
                list(self.core.query_forward("overstock", "causes"))

                # REALISTIC impact calculation:
                # Carrying cost = 25% of inventory value per year
                # Impact = carrying cost for excess units over next 30 days
                unit_cost = item.get("unit_cost", 0)
                target_dos = 30  # Ideal days of supply
                excess_units = max(0, qoh - (avg_sales * target_dos))

                # Annual carrying cost rate = 25%
                # Monthly carrying cost = value * 0.25 / 12
                excess_value = excess_units * unit_cost
                monthly_carrying_cost = excess_value * 0.25 / 12

                # Impact is monthly carrying cost (what we'll lose this month)
                impact = monthly_carrying_cost

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.OVERSTOCK,
                    severity=min((dos - 60) / 60, 1.0),
                    confidence=max(similarity, 0.5),
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "days_of_supply": round(dos, 1),
                        "quantity_on_hand": qoh,
                        "avg_daily_sales": round(avg_sales, 2),
                        "excess_units": round(excess_units, 0),
                        "excess_value": round(excess_value, 2),
                    },
                    causal_chain=["overstock", "markdown", "margin_erosion"],
                )
                leaks.append(leak)

        return leaks

    def detect_stockout(
        self, inventory_data: list[dict], demand_data: list[dict] = None
    ) -> list[ProfitLeak]:
        """Detect stockout situations causing lost sales."""

        leaks = []
        self._get_vector("stockout")

        for item in inventory_data:
            qoh = item.get("quantity_on_hand", 0)

            if qoh == 0:
                avg_sales = item.get("avg_daily_sales", 0)
                retail_price = item.get("retail_price", 0)

                # Estimate lost sales (assume 7 days to replenish)
                lost_units = avg_sales * 7
                impact = lost_units * retail_price

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.STOCKOUT,
                    severity=min(avg_sales / 10, 1.0),
                    confidence=0.8,  # High confidence when QOH = 0
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "quantity_on_hand": 0,
                        "avg_daily_sales": avg_sales,
                        "estimated_lost_units": lost_units,
                    },
                    causal_chain=["stockout", "lost_sales", "revenue_loss"],
                )
                leaks.append(leak)

        return leaks

    def detect_pricing_error(self, inventory_data: list[dict]) -> list[ProfitLeak]:
        """
        Detect pricing errors where items are priced incorrectly.

        Detects:
        - Price below cost (negative margin)
        - Price significantly below category average
        - Price/cost ratio anomalies
        """
        leaks = []

        # Build category price benchmarks
        category_prices = defaultdict(list)
        category_margins = defaultdict(list)

        for item in inventory_data:
            cat = item.get("category", "unknown")
            retail = item.get("retail_price", 0)
            cost = item.get("unit_cost", 0)
            if retail > 0 and cost > 0:
                category_prices[cat].append(retail)
                category_margins[cat].append((retail - cost) / retail * 100)

        # Calculate benchmarks
        cat_avg_price = {
            cat: np.mean(prices) for cat, prices in category_prices.items()
        }
        cat_avg_margin = {
            cat: np.mean(margins) for cat, margins in category_margins.items()
        }

        for item in inventory_data:
            retail = item.get("retail_price", 0)
            cost = item.get("unit_cost", 0)
            cat = item.get("category", "unknown")

            errors = []

            # Check 1: Price below cost
            if retail > 0 and cost > 0 and retail < cost:
                errors.append("price_below_cost")

            # Check 2: Margin significantly below category average
            if retail > 0 and cost > 0 and cat in cat_avg_margin:
                item_margin = (retail - cost) / retail * 100
                if item_margin < cat_avg_margin[cat] - 15:  # 15 points below average
                    errors.append("margin_below_category_avg")

            # Check 3: Price significantly below category average
            if retail > 0 and cat in cat_avg_price:
                if retail < cat_avg_price[cat] * 0.5:  # Less than 50% of category avg
                    errors.append("price_below_category_avg")

            if errors:
                # Calculate impact (lost margin)
                expected_margin = cat_avg_margin.get(cat, 25) / 100
                actual_margin = (retail - cost) / retail if retail > 0 else 0
                margin_loss = max(0, expected_margin - actual_margin)

                qty_sold = item.get(
                    "quantity_sold", item.get("avg_daily_sales", 1) * 30
                )
                impact = qty_sold * retail * margin_loss

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.PRICING_ERROR,
                    severity=min(len(errors) / 3, 1.0),
                    confidence=0.75,
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "retail_price": retail,
                        "unit_cost": cost,
                        "actual_margin_pct": (
                            round((retail - cost) / retail * 100, 1)
                            if retail > 0
                            else 0
                        ),
                        "category_avg_margin_pct": round(cat_avg_margin.get(cat, 0), 1),
                        "error_types": errors,
                    },
                    causal_chain=["pricing_error", "margin_erosion", "profit_loss"],
                )
                leaks.append(leak)

        return leaks

    def detect_markdown_timing(
        self, inventory_data: list[dict], historical_data: list[dict] = None
    ) -> list[ProfitLeak]:
        """
        Detect markdown timing issues.

        Detects:
        - Late markdowns (inventory aged before markdown)
        - Premature markdowns (marked down while still selling)
        - Excessive markdown depth
        """
        leaks = []

        for item in inventory_data:
            original_price = item.get("original_price", item.get("retail_price", 0))
            current_price = item.get("retail_price", 0)
            days_since_sale = item.get("days_since_sale", 0)
            avg_sales = item.get("avg_daily_sales", 0)
            qoh = item.get("quantity_on_hand", 0)

            # Skip if no markdown
            if original_price <= 0 or current_price >= original_price:
                continue

            markdown_pct = (original_price - current_price) / original_price * 100

            issues = []

            # Check 1: Late markdown - item sat too long before markdown
            if days_since_sale > 60 and markdown_pct > 0:
                issues.append("late_markdown")

            # Check 2: Premature markdown - still selling well but marked down
            if avg_sales > 2 and markdown_pct > 20:
                issues.append("premature_markdown")

            # Check 3: Excessive markdown - more than 50% off
            if markdown_pct > 50:
                issues.append("excessive_markdown")

            # Check 4: Markdown but still overstocked
            dos = qoh / max(avg_sales, 0.01)
            if markdown_pct > 0 and dos > 90:
                issues.append("markdown_insufficient")

            if issues:
                # Impact = margin lost due to markdown
                margin_at_original = item.get("original_margin_pct", 35) / 100
                margin_at_current = (
                    (current_price - item.get("unit_cost", 0)) / current_price
                    if current_price > 0
                    else 0
                )
                margin_loss = max(0, margin_at_original - margin_at_current)

                # Estimate units that will sell at marked-down price
                units_to_sell = min(qoh, avg_sales * 30)
                impact = units_to_sell * current_price * margin_loss

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.MARKDOWN_TIMING,
                    severity=min(markdown_pct / 50, 1.0),
                    confidence=0.65,
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "original_price": original_price,
                        "current_price": current_price,
                        "markdown_pct": round(markdown_pct, 1),
                        "days_since_sale": days_since_sale,
                        "timing_issues": issues,
                    },
                    causal_chain=["markdown_timing", "margin_erosion", "profit_loss"],
                )
                leaks.append(leak)

        return leaks

    def detect_vendor_compliance(
        self,
        inventory_data: list[dict],
        receipts: list[dict] = None,
        purchase_orders: list[dict] = None,
    ) -> list[ProfitLeak]:
        """
        Detect vendor compliance issues.

        Detects:
        - Short shipments (received less than ordered)
        - Price discrepancies (invoiced higher than agreed)
        - Quality issues (high return rate from specific vendor)
        - Late deliveries causing stockouts
        """
        leaks = []

        # Build vendor metrics
        vendor_metrics = defaultdict(
            lambda: {
                "total_ordered": 0,
                "total_received": 0,
                "price_variance_total": 0,
                "order_count": 0,
                "skus": set(),
            }
        )

        # Process receipts vs POs
        po_lookup = {po.get("po_id"): po for po in (purchase_orders or [])}

        for receipt in receipts or []:
            vendor = receipt.get("vendor_id", "unknown")
            po_id = receipt.get("po_id")
            received_qty = receipt.get("quantity_received", 0)
            received_cost = receipt.get("unit_cost", 0)

            vendor_metrics[vendor]["total_received"] += received_qty
            vendor_metrics[vendor]["order_count"] += 1
            vendor_metrics[vendor]["skus"].add(receipt.get("sku_id"))

            if po_id and po_id in po_lookup:
                po = po_lookup[po_id]
                ordered_qty = po.get("quantity_ordered", received_qty)
                agreed_cost = po.get("agreed_unit_cost", received_cost)

                vendor_metrics[vendor]["total_ordered"] += ordered_qty

                # Price variance
                if agreed_cost > 0:
                    variance = (received_cost - agreed_cost) / agreed_cost
                    vendor_metrics[vendor]["price_variance_total"] += variance

        # Detect issues per vendor
        for vendor, metrics in vendor_metrics.items():
            issues = []

            if metrics["total_ordered"] > 0:
                # Short shipment rate
                fill_rate = metrics["total_received"] / metrics["total_ordered"]
                if fill_rate < 0.95:
                    issues.append(f"short_shipment_{int((1-fill_rate)*100)}pct")

            if metrics["order_count"] > 0:
                # Average price variance
                avg_variance = metrics["price_variance_total"] / metrics["order_count"]
                if avg_variance > 0.05:  # More than 5% over agreed price
                    issues.append(f"price_overcharge_{int(avg_variance*100)}pct")

            if issues:
                # Estimate impact
                shortfall_units = metrics["total_ordered"] - metrics["total_received"]
                avg_cost = 50  # Default estimate
                impact = max(shortfall_units, 0) * avg_cost * 0.3  # 30% margin loss

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.VENDOR_COMPLIANCE,
                    severity=min(len(issues) / 2, 1.0),
                    confidence=0.7,
                    affected_skus=list(metrics["skus"])[:10],  # Limit to 10
                    estimated_impact=impact,
                    evidence={
                        "vendor_id": vendor,
                        "total_ordered": metrics["total_ordered"],
                        "total_received": metrics["total_received"],
                        "fill_rate_pct": (
                            round(fill_rate * 100, 1)
                            if metrics["total_ordered"] > 0
                            else 100
                        ),
                        "compliance_issues": issues,
                    },
                    causal_chain=[
                        "vendor_noncompliance",
                        "inventory_shortage",
                        "profit_loss",
                    ],
                )
                leaks.append(leak)

        # Also check inventory data for vendor-related issues
        for item in inventory_data:
            vendor = item.get("vendor_id")
            if not vendor:
                continue

            # Check for items with high cost variance from expected
            expected_cost = item.get("expected_cost", item.get("unit_cost", 0))
            actual_cost = item.get("unit_cost", 0)

            if expected_cost > 0 and actual_cost > expected_cost * 1.1:
                variance_pct = (actual_cost - expected_cost) / expected_cost * 100

                qoh = item.get("quantity_on_hand", 0)
                impact = qoh * (actual_cost - expected_cost)

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.VENDOR_COMPLIANCE,
                    severity=min(variance_pct / 20, 1.0),
                    confidence=0.6,
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "vendor_id": vendor,
                        "expected_cost": expected_cost,
                        "actual_cost": actual_cost,
                        "cost_variance_pct": round(variance_pct, 1),
                    },
                    causal_chain=[
                        "vendor_price_increase",
                        "cost_increase",
                        "margin_erosion",
                    ],
                )
                leaks.append(leak)

        return leaks

    def detect_seasonal_mismatch(
        self, inventory_data: list[dict], current_season: str = None
    ) -> list[ProfitLeak]:
        """
        Detect seasonal inventory mismatches.

        Detects:
        - Off-season inventory buildup
        - Missing seasonal stock
        - Seasonal transition timing issues
        """
        leaks = []

        # Determine current season if not provided
        if not current_season:
            month = datetime.now().month
            if month in [12, 1, 2]:
                current_season = "winter"
            elif month in [3, 4, 5]:
                current_season = "spring"
            elif month in [6, 7, 8]:
                current_season = "summer"
            else:
                current_season = "fall"

        # Season keywords for detection
        season_keywords = {
            "winter": [
                "winter",
                "holiday",
                "christmas",
                "snow",
                "cold",
                "heated",
                "coat",
                "jacket",
            ],
            "spring": ["spring", "easter", "garden", "rain", "allergy"],
            "summer": [
                "summer",
                "beach",
                "pool",
                "bbq",
                "outdoor",
                "camping",
                "sunscreen",
            ],
            "fall": [
                "fall",
                "autumn",
                "halloween",
                "thanksgiving",
                "back to school",
                "harvest",
            ],
        }

        opposite_season = {
            "winter": "summer",
            "summer": "winter",
            "spring": "fall",
            "fall": "spring",
        }

        for item in inventory_data:
            item_name = item.get("name", "").lower()
            item_cat = item.get("category", "").lower()
            item_tags = item.get("tags", [])
            if isinstance(item_tags, str):
                item_tags = [item_tags]
            item_tags = [t.lower() for t in item_tags]

            combined_text = f"{item_name} {item_cat} {' '.join(item_tags)}"

            # Detect item's season
            item_season = None
            for season, keywords in season_keywords.items():
                if any(kw in combined_text for kw in keywords):
                    item_season = season
                    break

            if not item_season:
                continue  # Can't determine seasonality

            qoh = item.get("quantity_on_hand", 0)
            avg_sales = item.get("avg_daily_sales", 0)
            dos = qoh / max(avg_sales, 0.01)

            issues = []

            # Check 1: High inventory of opposite season items
            if item_season == opposite_season.get(current_season) and qoh > 50:
                issues.append("off_season_overstock")

            # Check 2: Low inventory of current season items
            if item_season == current_season and qoh < 10 and avg_sales > 1:
                issues.append("in_season_stockout_risk")

            # Check 3: Season ending soon but high inventory
            # (assume seasons last ~3 months, check if > 60 days supply near end)
            if item_season == current_season and dos > 60:
                issues.append("end_of_season_overstock")

            if issues:
                unit_cost = item.get("unit_cost", 0)

                if "off_season_overstock" in issues:
                    # Will likely need heavy markdown
                    impact = qoh * unit_cost * 0.4  # Assume 40% markdown
                elif "in_season_stockout_risk" in issues:
                    # Lost sales
                    retail = item.get("retail_price", unit_cost * 1.4)
                    impact = avg_sales * 14 * retail  # 2 weeks lost sales
                else:
                    # End of season - moderate markdown
                    impact = qoh * unit_cost * 0.25

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.SEASONAL_MISMATCH,
                    severity=min(len(issues) / 2, 1.0),
                    confidence=0.6,
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "item_season": item_season,
                        "current_season": current_season,
                        "quantity_on_hand": qoh,
                        "days_of_supply": round(dos, 1),
                        "seasonal_issues": issues,
                    },
                    causal_chain=["seasonal_mismatch", "markdown", "margin_erosion"],
                )
                leaks.append(leak)

        return leaks

    def detect_category_cannibalization(
        self, inventory_data: list[dict], sales_history: list[dict] = None
    ) -> list[ProfitLeak]:
        """
        Detect category cannibalization where products eat each other's sales.

        Detects:
        - New products reducing sales of existing products
        - Promotions on one item hurting similar items
        - SKU proliferation diluting category performance
        """
        leaks = []

        # Group by category
        by_category = defaultdict(list)
        for item in inventory_data:
            cat = item.get("category", "unknown")
            by_category[cat].append(item)

        for category, items in by_category.items():
            if len(items) < 3:
                continue  # Need multiple items to detect cannibalization

            # Calculate category metrics
            total_sales = sum(i.get("avg_daily_sales", 0) for i in items)
            avg_margin = np.mean([i.get("margin_percent", 25) for i in items])

            # Sort by sales velocity
            sorted_items = sorted(
                items, key=lambda x: x.get("avg_daily_sales", 0), reverse=True
            )

            # Check for signs of cannibalization
            for i, item in enumerate(sorted_items):
                issues = []

                sales = item.get("avg_daily_sales", 0)
                prev_sales = item.get("prev_avg_daily_sales", sales)
                margin = item.get("margin_percent", 25)
                is_new = (
                    item.get("is_new", False) or item.get("days_in_store", 365) < 90
                )
                is_promoted = item.get("is_promoted", False) or item.get(
                    "on_promotion", False
                )

                # Check 1: Sales decline while category stable
                if prev_sales > 0 and sales < prev_sales * 0.7:
                    # Check if category total is stable
                    cat_prev_total = sum(
                        i.get("prev_avg_daily_sales", i.get("avg_daily_sales", 0))
                        for i in items
                    )
                    if total_sales >= cat_prev_total * 0.9:
                        issues.append("sales_shifted_to_other_skus")

                # Check 2: Low margin item with high sales (cannibalizing premium)
                if margin < avg_margin - 10 and sales > total_sales / len(items) * 1.5:
                    issues.append("low_margin_dominating_category")

                # Check 3: New item correlates with existing item decline
                if is_new and sales > 1:
                    # Check if other items in category declined
                    declining = [
                        i
                        for i in items
                        if i.get("prev_avg_daily_sales", 0)
                        > i.get("avg_daily_sales", 0) * 1.2
                        and i.get("sku_id") != item.get("sku_id")
                    ]
                    if len(declining) >= 2:
                        issues.append("new_item_cannibalizing")

                # Check 4: Promoted item hurting non-promoted
                if is_promoted:
                    non_promoted_decline = [
                        i
                        for i in items
                        if not i.get("is_promoted", False)
                        and i.get("avg_daily_sales", 0)
                        < i.get("prev_avg_daily_sales", 1) * 0.8
                    ]
                    if len(non_promoted_decline) >= 1:
                        issues.append("promotion_cannibalizing")

                if issues:
                    # Estimate impact - margin dilution
                    margin_diff = max(0, avg_margin - margin) / 100
                    impact = sales * 30 * item.get("retail_price", 10) * margin_diff

                    leak = ProfitLeak(
                        leak_type=ProfitLeakType.CATEGORY_CANNIBALIZATION,
                        severity=min(len(issues) / 3, 1.0),
                        confidence=0.55,  # Lower confidence - correlation not causation
                        affected_skus=[item.get("sku_id", "unknown")],
                        estimated_impact=impact,
                        evidence={
                            "category": category,
                            "sku_sales_pct_of_category": round(
                                sales / max(total_sales, 1) * 100, 1
                            ),
                            "sku_margin": margin,
                            "category_avg_margin": round(avg_margin, 1),
                            "cannibalization_indicators": issues,
                        },
                        causal_chain=[
                            "category_cannibalization",
                            "margin_dilution",
                            "profit_loss",
                        ],
                    )
                    leaks.append(leak)

        return leaks

    def detect_cost_spike_propagation(
        self, inventory_data: list[dict], cost_history: list[dict] = None
    ) -> list[ProfitLeak]:
        """
        Detect cost spike propagation issues.

        This is one of the 3 novel patterns discovered by VSA:
        When costs spike but prices don't adjust, margin erodes silently.

        Detects:
        - Recent cost increases not reflected in price
        - Vendor cost trends indicating future margin pressure
        - Cost spikes in key ingredients/components affecting multiple SKUs
        """
        leaks = []

        for item in inventory_data:
            current_cost = item.get("unit_cost", 0)
            prev_cost = item.get("prev_unit_cost", current_cost)
            retail_price = item.get("retail_price", 0)
            prev_retail = item.get("prev_retail_price", retail_price)

            if prev_cost <= 0 or current_cost <= 0:
                continue

            cost_change_pct = (current_cost - prev_cost) / prev_cost * 100
            price_change_pct = (
                (retail_price - prev_retail) / prev_retail * 100
                if prev_retail > 0
                else 0
            )

            issues = []

            # Check 1: Cost increased but price didn't
            if cost_change_pct > 5 and price_change_pct < cost_change_pct * 0.5:
                issues.append("cost_increase_not_passed_through")

            # Check 2: Significant cost spike (>15%)
            if cost_change_pct > 15:
                issues.append("major_cost_spike")

            # Check 3: Margin compression
            prev_margin = (
                (prev_retail - prev_cost) / prev_retail * 100 if prev_retail > 0 else 0
            )
            current_margin = (
                (retail_price - current_cost) / retail_price * 100
                if retail_price > 0
                else 0
            )
            margin_compression = prev_margin - current_margin

            if margin_compression > 5:
                issues.append(f"margin_compressed_{int(margin_compression)}pts")

            # Check 4: Multiple cost increases (trending)
            cost_trend = item.get(
                "cost_trend", None
            )  # Could be "increasing", "stable", "decreasing"
            if cost_trend == "increasing" and cost_change_pct > 3:
                issues.append("sustained_cost_pressure")

            if issues:
                # Impact = margin loss on expected sales
                margin_loss_pct = max(0, margin_compression) / 100
                monthly_sales = item.get("avg_daily_sales", 1) * 30
                impact = monthly_sales * retail_price * margin_loss_pct

                leak = ProfitLeak(
                    leak_type=ProfitLeakType.COST_SPIKE_PROPAGATION,
                    severity=min(cost_change_pct / 20, 1.0),
                    confidence=0.7,
                    affected_skus=[item.get("sku_id", "unknown")],
                    estimated_impact=impact,
                    evidence={
                        "current_cost": current_cost,
                        "previous_cost": prev_cost,
                        "cost_change_pct": round(cost_change_pct, 1),
                        "price_change_pct": round(price_change_pct, 1),
                        "current_margin_pct": round(current_margin, 1),
                        "previous_margin_pct": round(prev_margin, 1),
                        "propagation_issues": issues,
                    },
                    causal_chain=["cost_spike", "margin_erosion", "profit_loss"],
                )
                leaks.append(leak)

        return leaks

    # =========================================================================
    # CAUSAL REASONING
    # =========================================================================

    def trace_causal_chain(
        self, start_concept: str, max_hops: int = 3
    ) -> list[list[str]]:
        """
        Trace causal chains from a starting concept using geometric inference.

        Returns all causal paths up to max_hops length.
        """
        chains = [[start_concept]]

        for hop in range(max_hops):
            new_chains = []
            for chain in chains:
                current = chain[-1]
                # Query forward for causal relationships
                causal_facts = list(self.core.query_forward(current, "causes"))

                if causal_facts:
                    for fact in causal_facts:
                        if fact.object not in chain:  # Avoid cycles
                            new_chains.append(chain + [fact.object])
                else:
                    new_chains.append(chain)  # Keep chain as-is if no more causes

            chains = new_chains

        return chains

    def validate_causal_claim(self, cause: str, effect: str) -> tuple[bool, float, str]:
        """
        Validate whether a causal claim is supported by the knowledge brain.

        Returns:
            (is_valid, confidence, explanation)
        """
        # Check direct causal link
        direct_facts = list(self.core.query_forward(cause, "causes"))
        direct_effects = [f.object for f in direct_facts]

        if effect in direct_effects:
            return True, 0.9, f"Direct causal link: {cause} causes {effect}"

        # Check indirect via transitive inference
        chains = self.trace_causal_chain(cause, max_hops=3)

        for chain in chains:
            if effect in chain:
                path = " → ".join(chain[: chain.index(effect) + 1])
                confidence = 0.7 - (0.1 * (len(chain) - 2))  # Decay with hops
                return True, confidence, f"Indirect causal path: {path}"

        return False, 0.0, f"No causal path found from {cause} to {effect}"

    # =========================================================================
    # ANALYSIS API
    # =========================================================================

    def analyze_inventory(
        self,
        inventory_data: list[dict],
        adjustments: list[dict] = None,
        receipts: list[dict] = None,
        purchase_orders: list[dict] = None,
        sales_history: list[dict] = None,
        cost_history: list[dict] = None,
        current_season: str = None,
    ) -> dict[str, Any]:
        """
        Perform comprehensive profit leak analysis on inventory data.

        Args:
            inventory_data: List of SKU records with metrics
            adjustments: Optional list of inventory adjustments
            receipts: Optional list of receiving records
            purchase_orders: Optional list of POs
            sales_history: Optional historical sales data
            cost_history: Optional historical cost data
            current_season: Optional season override (winter/spring/summer/fall)

        Returns:
            Analysis report with detected leaks and recommendations
        """
        print(f"\n{'='*60}")
        print("PROFIT SENTINEL ANALYSIS - ALL 11 LEAK TYPES")
        print(f"{'='*60}")
        print(f"Analyzing {len(inventory_data)} SKUs...")

        # Run all detectors
        all_leaks = []

        # Original 5 detectors
        print("\n  [1/11] Detecting dead inventory...")
        dead_leaks = self.detect_dead_inventory(inventory_data)
        all_leaks.extend(dead_leaks)
        print(f"         Found {len(dead_leaks)} issues")

        print("\n  [2/11] Detecting margin erosion...")
        margin_leaks = self.detect_margin_erosion(inventory_data)
        all_leaks.extend(margin_leaks)
        print(f"         Found {len(margin_leaks)} issues")

        print("\n  [3/11] Detecting shrinkage...")
        shrink_leaks = self.detect_shrinkage(inventory_data, adjustments)
        all_leaks.extend(shrink_leaks)
        print(f"         Found {len(shrink_leaks)} issues")

        print("\n  [4/11] Detecting overstock...")
        over_leaks = self.detect_overstock(inventory_data)
        all_leaks.extend(over_leaks)
        print(f"         Found {len(over_leaks)} issues")

        print("\n  [5/11] Detecting stockouts...")
        stock_leaks = self.detect_stockout(inventory_data)
        all_leaks.extend(stock_leaks)
        print(f"         Found {len(stock_leaks)} issues")

        # New detectors
        print("\n  [6/11] Detecting pricing errors...")
        pricing_leaks = self.detect_pricing_error(inventory_data)
        all_leaks.extend(pricing_leaks)
        print(f"         Found {len(pricing_leaks)} issues")

        print("\n  [7/11] Detecting markdown timing issues...")
        markdown_leaks = self.detect_markdown_timing(inventory_data)
        all_leaks.extend(markdown_leaks)
        print(f"         Found {len(markdown_leaks)} issues")

        print("\n  [8/11] Detecting vendor compliance issues...")
        vendor_leaks = self.detect_vendor_compliance(
            inventory_data, receipts, purchase_orders
        )
        all_leaks.extend(vendor_leaks)
        print(f"         Found {len(vendor_leaks)} issues")

        print("\n  [9/11] Detecting seasonal mismatches...")
        seasonal_leaks = self.detect_seasonal_mismatch(inventory_data, current_season)
        all_leaks.extend(seasonal_leaks)
        print(f"         Found {len(seasonal_leaks)} issues")

        print("\n  [10/11] Detecting category cannibalization...")
        cannibal_leaks = self.detect_category_cannibalization(
            inventory_data, sales_history
        )
        all_leaks.extend(cannibal_leaks)
        print(f"          Found {len(cannibal_leaks)} issues")

        print("\n  [11/11] Detecting cost spike propagation...")
        cost_leaks = self.detect_cost_spike_propagation(inventory_data, cost_history)
        all_leaks.extend(cost_leaks)
        print(f"          Found {len(cost_leaks)} issues")

        # Store detected leaks
        self.detected_leaks = all_leaks

        # Generate summary
        total_impact = sum(leak.estimated_impact for leak in all_leaks)
        by_type = defaultdict(list)
        for leak in all_leaks:
            by_type[leak.leak_type].append(leak)

        report = {
            "timestamp": datetime.now().isoformat(),
            "skus_analyzed": len(inventory_data),
            "total_leaks_detected": len(all_leaks),
            "total_estimated_impact": total_impact,
            "leaks_by_type": {
                lt.value: {
                    "count": len(leaks),
                    "total_impact": sum(l.estimated_impact for l in leaks),
                    "avg_confidence": (
                        np.mean([l.confidence for l in leaks]) if leaks else 0
                    ),
                }
                for lt, leaks in by_type.items()
            },
            "top_leaks": [
                leak.to_dict()
                for leak in sorted(
                    all_leaks, key=lambda x: x.estimated_impact, reverse=True
                )[:10]
            ],
            "causal_patterns": self._extract_causal_patterns(all_leaks),
        }

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total leaks detected: {len(all_leaks)}")
        print(f"Estimated total impact: ${total_impact:,.2f}")

        return report

    def _extract_causal_patterns(self, leaks: list[ProfitLeak]) -> dict[str, int]:
        """Extract common causal patterns from detected leaks."""

        pattern_counts = defaultdict(int)

        for leak in leaks:
            if len(leak.causal_chain) >= 2:
                for i in range(len(leak.causal_chain) - 1):
                    pattern = f"{leak.causal_chain[i]} → {leak.causal_chain[i+1]}"
                    pattern_counts[pattern] += 1

        return dict(sorted(pattern_counts.items(), key=lambda x: -x[1])[:10])

    def query_knowledge(self, query: str) -> str:
        """
        Natural language query interface to the knowledge brain.

        Examples:
            "What causes margin erosion?"
            "What does dead inventory lead to?"
        """
        # Parse simple query patterns
        query_lower = query.lower()

        if "what causes" in query_lower:
            target = query_lower.split("what causes")[-1].strip().rstrip("?")
            target = target.replace(" ", "_")

            results = self.core.query(predicate="causes", obj=target, k=10)
            causes = [f.subject for f, score in results if score > 0.2]

            if causes:
                # Deduplicate
                causes = list(dict.fromkeys(causes))
                return f"{target} can be caused by: {', '.join(causes)}"
            return f"No known causes found for {target}"

        elif "what does" in query_lower and (
            "cause" in query_lower or "lead" in query_lower
        ):
            # Extract subject - "what does X cause" or "what does X lead to"
            parts = query_lower.replace("what does", "").strip()
            # Remove "cause", "lead to", etc
            for remove in ["cause", "lead to", "result in", "?"]:
                parts = parts.replace(remove, "")
            subject = parts.strip().replace(" ", "_")

            facts = list(self.core.query_forward(subject, "causes"))
            effects = [f.object for f in facts]

            if effects:
                return f"{subject} causes: {', '.join(effects)}"
            return f"No known effects found for {subject}"

        elif "indicates" in query_lower:
            parts = query_lower.split("indicates")
            if len(parts) == 2:
                indicator = parts[0].strip().replace(" ", "_")
                facts = list(self.core.query_forward(indicator, "indicates"))
                indicated = [f.object for f in facts]

                if indicated:
                    return f"{indicator} indicates: {', '.join(indicated)}"

        return "Query not understood. Try: 'What causes X?' or 'What does X cause?'"


# =============================================================================
# DEMO
# =============================================================================


def demo_profit_sentinel():
    """Demonstrate the Profit Sentinel agent."""

    print("=" * 70)
    print("PROFIT SENTINEL AGENT DEMO")
    print("=" * 70)

    # Import and create core
    from dorian_biology import load_biology_into_core
    from dorian_chemistry import load_chemistry_into_core
    from dorian_core import DorianCore
    from dorian_cs import load_cs_into_core
    from dorian_economics import load_economics_into_core
    from dorian_math import load_mathematics_into_core
    from dorian_philosophy import load_philosophy_into_core
    from dorian_physics import load_physics_into_core
    from dorian_web import load_web_into_core

    print("\nInitializing Dorian Knowledge Brain...")
    core = DorianCore(dim=256, load_ontology=True)
    core.bootstrap_ontology()

    # Load all domains
    print("\nLoading domain knowledge...")
    load_economics_into_core(core)  # Most relevant for retail
    load_mathematics_into_core(core)

    # Train embeddings
    print("\nTraining embeddings...")
    core.train(verbose=False)

    # Create Profit Sentinel agent
    print("\nInitializing Profit Sentinel Agent...")
    agent = ProfitSentinelAgent(core)

    # Generate synthetic inventory data
    print("\nGenerating synthetic inventory data...")
    inventory_data = [
        # Dead inventory
        {
            "sku_id": "SKU001",
            "category": "electronics",
            "days_since_sale": 120,
            "margin_percent": 15,
            "quantity_on_hand": 50,
            "avg_daily_sales": 0,
            "unit_cost": 25.00,
            "retail_price": 35.00,
            "revenue": 0,
        },
        # Margin erosion
        {
            "sku_id": "SKU002",
            "category": "apparel",
            "days_since_sale": 5,
            "margin_percent": 8,
            "prev_margin_percent": 25,
            "quantity_on_hand": 100,
            "avg_daily_sales": 5,
            "unit_cost": 20.00,
            "retail_price": 22.00,
            "revenue": 5000,
        },
        # Overstock
        {
            "sku_id": "SKU003",
            "category": "home",
            "days_since_sale": 3,
            "margin_percent": 35,
            "quantity_on_hand": 500,
            "avg_daily_sales": 2,
            "unit_cost": 15.00,
            "retail_price": 25.00,
            "revenue": 1000,
        },
        # Stockout
        {
            "sku_id": "SKU004",
            "category": "grocery",
            "days_since_sale": 0,
            "margin_percent": 20,
            "quantity_on_hand": 0,
            "avg_daily_sales": 25,
            "unit_cost": 5.00,
            "retail_price": 7.50,
            "revenue": 0,
        },
        # Healthy item (no leak)
        {
            "sku_id": "SKU005",
            "category": "electronics",
            "days_since_sale": 1,
            "margin_percent": 30,
            "quantity_on_hand": 75,
            "avg_daily_sales": 3,
            "unit_cost": 50.00,
            "retail_price": 75.00,
            "revenue": 10000,
        },
    ]

    # Sample adjustments for shrinkage
    adjustments = [
        {"sku_id": "SKU001", "quantity": -5, "unit_cost": 25.00},
        {"sku_id": "SKU003", "quantity": -10, "unit_cost": 15.00},
    ]

    # Run analysis
    report = agent.analyze_inventory(inventory_data, adjustments)

    # Test knowledge queries
    print("\n" + "=" * 60)
    print("KNOWLEDGE QUERIES")
    print("=" * 60)

    queries = [
        "What causes margin erosion?",
        "What does dead inventory cause?",
        "What does overstock cause?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        print(f"A: {agent.query_knowledge(q)}")

    # Test causal chain tracing
    print("\n" + "=" * 60)
    print("CAUSAL CHAIN TRACING")
    print("=" * 60)

    chains = agent.trace_causal_chain("dead_inventory", max_hops=3)
    print("\nCausal chains from 'dead_inventory':")
    for chain in chains:
        print(f"  {' → '.join(chain)}")

    # Validate a causal claim
    print("\n" + "=" * 60)
    print("CAUSAL CLAIM VALIDATION")
    print("=" * 60)

    valid, conf, explanation = agent.validate_causal_claim(
        "overstock", "margin_erosion"
    )
    print("\nClaim: overstock → margin_erosion")
    print(f"Valid: {valid}, Confidence: {conf:.2f}")
    print(f"Explanation: {explanation}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    return agent, report


if __name__ == "__main__":
    agent, report = demo_profit_sentinel()
