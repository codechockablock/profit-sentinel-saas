"""
Retail Knowledge Graph for Causal Chain Tracing.

v3.7: Provides structured domain knowledge for profit leak diagnosis.

The RETAIL_KNOWLEDGE graph encodes causal relationships between:
- Operational conditions (what we observe)
- Root causes (why it happened)
- Effects (what results)

This enables:
1. Multi-hop causal inference (pricing_error → margin_erosion → profit_leak)
2. Root cause identification (trace back from symptoms)
3. Explainable recommendations

Graph Structure:
    RETAIL_KNOWLEDGE = [
        ("cause", "causes", "effect", confidence),
        ...
    ]

Each tuple represents: cause --causes--> effect with confidence weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..legacy.context import AnalysisContext

# =============================================================================
# RETAIL KNOWLEDGE GRAPH
# =============================================================================

# Causal relationships in retail profit leak domain
# Format: (cause, predicate, effect, confidence)
RETAIL_KNOWLEDGE: list[tuple[str, str, str, float]] = [
    # =========================================================================
    # PRICING CASCADE
    # =========================================================================
    # Pricing errors lead to margin problems
    ("pricing_error", "causes", "margin_erosion", 0.95),
    ("pricing_error", "causes", "negative_margin", 0.90),
    ("pricing_error", "causes", "below_cost_sale", 0.85),
    # Margin problems lead to profit leaks
    ("margin_erosion", "causes", "profit_leak", 0.95),
    ("negative_margin", "causes", "profit_leak", 1.0),
    ("below_cost_sale", "causes", "profit_leak", 1.0),
    # =========================================================================
    # INVENTORY CASCADE
    # =========================================================================
    # Demand forecasting errors
    ("demand_forecast_error", "causes", "overstock", 0.85),
    ("demand_forecast_error", "causes", "stockout", 0.85),
    ("seasonal_mismatch", "causes", "overstock", 0.80),
    ("seasonal_mismatch", "causes", "dead_inventory", 0.75),
    # Overstock cascade
    ("overstock", "causes", "dead_inventory", 0.80),
    ("overstock", "causes", "storage_cost", 0.90),
    ("overstock", "causes", "markdown_pressure", 0.85),
    # Dead inventory cascade
    ("dead_inventory", "causes", "markdown_pressure", 0.95),
    ("dead_inventory", "causes", "write_off", 0.70),
    ("dead_inventory", "causes", "tied_up_capital", 0.90),
    # Markdown cascade
    ("markdown_pressure", "causes", "margin_erosion", 0.90),
    ("markdown_pressure", "causes", "below_cost_sale", 0.60),
    # Stockout cascade
    ("stockout", "causes", "lost_sales", 0.95),
    ("stockout", "causes", "customer_churn", 0.70),
    ("lost_sales", "causes", "profit_leak", 0.95),
    # =========================================================================
    # SHRINKAGE CASCADE
    # =========================================================================
    # Theft and loss
    ("theft", "causes", "shrinkage", 0.95),
    ("damage", "causes", "shrinkage", 0.90),
    ("administrative_error", "causes", "shrinkage", 0.85),
    ("vendor_fraud", "causes", "shrinkage", 0.80),
    # Shrinkage impact
    ("shrinkage", "causes", "inventory_discrepancy", 0.95),
    ("shrinkage", "causes", "negative_inventory", 0.70),
    ("shrinkage", "causes", "profit_leak", 0.95),
    # Negative inventory cascade
    ("negative_inventory", "causes", "inventory_discrepancy", 1.0),
    ("negative_inventory", "causes", "reorder_error", 0.80),
    # =========================================================================
    # VENDOR CASCADE
    # =========================================================================
    # Vendor issues
    ("vendor_error", "causes", "cost_discrepancy", 0.90),
    ("vendor_error", "causes", "late_delivery", 0.75),
    ("vendor_pricing_change", "causes", "margin_erosion", 0.85),
    # Cost discrepancy cascade
    ("cost_discrepancy", "causes", "margin_erosion", 0.90),
    ("cost_discrepancy", "causes", "negative_margin", 0.60),
    ("zero_cost_item", "causes", "cost_discrepancy", 0.95),
    ("zero_cost_item", "causes", "data_quality_issue", 0.90),
    # =========================================================================
    # DATA QUALITY CASCADE
    # =========================================================================
    # Data issues
    ("data_entry_error", "causes", "data_quality_issue", 0.95),
    ("system_integration_error", "causes", "data_quality_issue", 0.85),
    # Data quality impact
    ("data_quality_issue", "causes", "pricing_error", 0.70),
    ("data_quality_issue", "causes", "inventory_discrepancy", 0.80),
    ("data_quality_issue", "causes", "reporting_error", 0.90),
    # =========================================================================
    # OPERATIONAL CASCADE
    # =========================================================================
    # Process failures
    ("receiving_error", "causes", "inventory_discrepancy", 0.85),
    ("cycle_count_error", "causes", "inventory_discrepancy", 0.80),
    ("transfer_error", "causes", "inventory_discrepancy", 0.85),
    # Inventory discrepancy impact
    ("inventory_discrepancy", "causes", "reorder_error", 0.75),
    ("inventory_discrepancy", "causes", "stockout", 0.60),
    ("inventory_discrepancy", "causes", "overstock", 0.50),
    # =========================================================================
    # FINAL IMPACTS (Terminal nodes)
    # =========================================================================
    # Everything flows to profit_leak
    ("storage_cost", "causes", "profit_leak", 0.80),
    ("write_off", "causes", "profit_leak", 1.0),
    ("tied_up_capital", "causes", "profit_leak", 0.70),
    ("customer_churn", "causes", "profit_leak", 0.85),
]

# =============================================================================
# CAUSE CATEGORIES (for grouping in reports)
# =============================================================================

CAUSE_CATEGORIES = {
    # Pricing issues
    "pricing": [
        "pricing_error",
        "margin_erosion",
        "negative_margin",
        "below_cost_sale",
        "markdown_pressure",
    ],
    # Inventory issues
    "inventory": [
        "overstock",
        "stockout",
        "dead_inventory",
        "negative_inventory",
        "inventory_discrepancy",
    ],
    # Shrinkage issues
    "shrinkage": [
        "theft",
        "damage",
        "administrative_error",
        "vendor_fraud",
        "shrinkage",
    ],
    # Vendor issues
    "vendor": [
        "vendor_error",
        "vendor_pricing_change",
        "cost_discrepancy",
        "zero_cost_item",
    ],
    # Data quality issues
    "data_quality": [
        "data_entry_error",
        "system_integration_error",
        "data_quality_issue",
    ],
    # Demand issues
    "demand": [
        "demand_forecast_error",
        "seasonal_mismatch",
        "lost_sales",
        "customer_churn",
    ],
    # Operational issues
    "operational": [
        "receiving_error",
        "cycle_count_error",
        "transfer_error",
        "reorder_error",
    ],
    # Final impacts
    "impact": [
        "profit_leak",
        "storage_cost",
        "write_off",
        "tied_up_capital",
    ],
}


# =============================================================================
# KNOWLEDGE GRAPH LOADER
# =============================================================================


@dataclass
class KnowledgeGraph:
    """
    Wrapper for the retail knowledge graph with utility methods.

    Provides:
    - Load knowledge into AnalysisContext
    - Query causal paths
    - Get root causes for an effect
    - Get all effects of a cause
    """

    ctx: AnalysisContext
    loaded: bool = False

    def load(self) -> int:
        """
        Load the retail knowledge graph into the context.

        Returns:
            Number of relations loaded
        """
        count = 0
        for cause, predicate, effect, confidence in RETAIL_KNOWLEDGE:
            self.ctx.add_relation(cause, predicate, effect, confidence)
            count += 1

        self.loaded = True
        return count

    def find_root_causes(
        self,
        effect: str,
        max_hops: int = 5,
    ) -> list[str]:
        """
        Find root causes for an observed effect.

        Root causes are entities with no incoming "causes" relations.

        Args:
            effect: The observed effect (e.g., "profit_leak")
            max_hops: Maximum chain length to search

        Returns:
            List of root cause entities
        """
        all_causes = self.ctx.get_all_causes(effect, "causes", max_hops)

        # Find which causes have no incoming relations (root causes)
        root_causes = []
        all_cause_entities = set()
        for hop_causes in all_causes.values():
            all_cause_entities.update(hop_causes)

        # Check each cause to see if it's a root cause
        for cause in all_cause_entities:
            incoming = self.ctx.get_all_causes(cause, "causes", max_hops=1)
            if not incoming:  # No incoming relations = root cause
                root_causes.append(cause)

        return root_causes

    def trace_causal_chain(
        self,
        symptom: str,
        root_cause: str,
    ) -> list[str] | None:
        """
        Trace the causal chain from root cause to symptom.

        Args:
            symptom: The observed symptom
            root_cause: The suspected root cause

        Returns:
            List of entities in the causal chain, or None if no path
        """
        return self.ctx.find_causal_path(root_cause, symptom, "causes")

    def get_category(self, cause: str) -> str | None:
        """
        Get the category for a cause.

        Args:
            cause: The cause entity

        Returns:
            Category name or None if not found
        """
        cause_lower = cause.lower()
        for category, causes in CAUSE_CATEGORIES.items():
            if cause_lower in causes:
                return category
        return None

    def explain_path(self, path: list[str]) -> str:
        """
        Generate human-readable explanation of a causal path.

        Args:
            path: List of entities in the causal chain

        Returns:
            Human-readable explanation
        """
        if not path or len(path) < 2:
            return "No causal path found"

        explanations = []
        for i in range(len(path) - 1):
            cause = path[i].replace("_", " ")
            effect = path[i + 1].replace("_", " ")
            explanations.append(f"{cause} → {effect}")

        return " → ".join(path).replace("_", " ")


def create_knowledge_graph(ctx: AnalysisContext) -> KnowledgeGraph:
    """
    Factory function to create and load a knowledge graph.

    Args:
        ctx: AnalysisContext to load knowledge into

    Returns:
        Loaded KnowledgeGraph instance
    """
    kg = KnowledgeGraph(ctx=ctx)
    kg.load()
    return kg


def get_retail_knowledge() -> list[tuple[str, str, str, float]]:
    """
    Get the raw retail knowledge graph data.

    Returns:
        List of (cause, predicate, effect, confidence) tuples
    """
    return RETAIL_KNOWLEDGE.copy()
