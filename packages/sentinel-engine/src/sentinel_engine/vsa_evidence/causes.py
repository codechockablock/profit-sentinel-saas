"""
Retail Cause Vectors - Defines the causal hypotheses for profit leaks.

Based on 16 validated hypotheses from VSA research:
- 0% quantitative hallucination (vs 39.6% ungrounded)
- 100% multi-hop reasoning accuracy
- +586% improvement over random baseline

These cause vectors represent the WHY behind profit leaks:
- Rules detect WHAT is wrong (anomaly)
- VSA evidence explains WHY (root cause)

Reference: RESEARCH_SUMMARY.md - Key validated components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..context import AnalysisContext

logger = logging.getLogger(__name__)


# =============================================================================
# CAUSE DEFINITIONS
# =============================================================================

# The 8 root causes for retail profit leaks
# These map to multi-hop causal chains: SKU -> Vendor -> Factory
CAUSE_KEYS = [
    "theft",              # Shrinkage due to theft (internal/external)
    "vendor_increase",    # Vendor raised costs without notice
    "rebate_timing",      # Rebate timing mismatch (requires vendor hop)
    "margin_leak",        # Margin erosion from pricing issues
    "demand_shift",       # Market demand changed (seasonal, trends)
    "quality_issue",      # Product quality problems (requires factory hop)
    "pricing_error",      # Incorrect pricing configuration
    "inventory_drift",    # Gradual inventory discrepancy
]


@dataclass
class CauseVectors:
    """
    Container for cause hypervectors used in evidence-based encoding.

    Each cause is a deterministic phasor hypervector generated from a seed.
    These are the "targets" that facts vote toward based on evidence rules.

    Architecture:
        - Causes are orthogonal high-dimensional vectors
        - Facts are encoded as weighted bundles of cause vectors
        - Similarity between fact bundle and cause vector = evidence strength

    Thread-safe: Each context gets its own CauseVectors instance.
    """

    ctx: "AnalysisContext"
    _vectors: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _initialized: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Lazily initialize cause vectors."""
        pass

    def _ensure_initialized(self):
        """Initialize cause vectors on first access."""
        if self._initialized:
            return

        logger.debug(f"Initializing {len(CAUSE_KEYS)} cause vectors")

        for cause_key in CAUSE_KEYS:
            # Use deterministic seed for reproducibility
            seed_string = f"cause_vector_{cause_key}_v1"
            self._vectors[cause_key] = self.ctx.seed_hash(seed_string)

        self._initialized = True
        logger.debug("Cause vectors initialized")

    def get(self, cause_key: str) -> torch.Tensor | None:
        """
        Get the hypervector for a specific cause.

        Args:
            cause_key: One of CAUSE_KEYS (e.g., "theft", "vendor_increase")

        Returns:
            Normalized phasor hypervector or None if unknown cause
        """
        self._ensure_initialized()
        return self._vectors.get(cause_key)

    def get_all(self) -> dict[str, torch.Tensor]:
        """
        Get all cause vectors.

        Returns:
            Dict mapping cause key to hypervector
        """
        self._ensure_initialized()
        return self._vectors.copy()

    def keys(self) -> list[str]:
        """Get list of all cause keys."""
        return CAUSE_KEYS.copy()

    def similarity(self, evidence_vec: torch.Tensor, cause_key: str) -> float:
        """
        Calculate similarity between evidence vector and a cause.

        Uses real part of complex inner product for phasor VSA.

        Args:
            evidence_vec: Encoded evidence hypervector
            cause_key: Target cause to compare against

        Returns:
            Similarity score (higher = stronger evidence)
        """
        cause_vec = self.get(cause_key)
        if cause_vec is None:
            return 0.0

        # Phasor similarity: real part of complex dot product
        sim = torch.real(torch.dot(torch.conj(cause_vec), evidence_vec)).item()
        return sim

    def all_similarities(self, evidence_vec: torch.Tensor) -> dict[str, float]:
        """
        Calculate similarity to all causes at once.

        Args:
            evidence_vec: Encoded evidence hypervector

        Returns:
            Dict mapping cause key to similarity score
        """
        self._ensure_initialized()

        # Stack all cause vectors for batch computation
        cause_tensor = torch.stack(list(self._vectors.values()))  # (n_causes, dim)

        # Batch similarity computation
        sims = torch.real(torch.conj(cause_tensor) @ evidence_vec)  # (n_causes,)

        return {
            cause_key: sims[i].item()
            for i, cause_key in enumerate(self._vectors.keys())
        }


# =============================================================================
# CAUSE METADATA
# =============================================================================

CAUSE_METADATA = {
    "theft": {
        "severity": "critical",
        "category": "Shrinkage",
        "description": "Inventory loss due to internal or external theft",
        "multi_hop_depth": 1,  # Direct evidence at SKU level
        "recommendations": [
            "Review security footage for high-value items",
            "Analyze void/return patterns by employee",
            "Conduct surprise cycle counts",
            "Check receiving accuracy against POs",
        ],
    },
    "vendor_increase": {
        "severity": "high",
        "category": "Cost Pressure",
        "description": "Vendor raised costs without corresponding price adjustment",
        "multi_hop_depth": 1,  # Direct evidence at SKU level (cost change)
        "recommendations": [
            "Review recent vendor invoices for cost changes",
            "Compare current vs historical cost trends",
            "Negotiate volume discounts or alternatives",
            "Update retail prices to maintain margin",
        ],
    },
    "rebate_timing": {
        "severity": "high",
        "category": "Vendor Terms",
        "description": "Rebate or dating terms causing temporary margin squeeze",
        "multi_hop_depth": 2,  # Requires SKU -> Vendor hop
        "recommendations": [
            "Check vendor payment terms (net-60, etc.)",
            "Review rebate accrual schedule",
            "Verify rebate credits are being applied",
            "Consider early payment discounts vs dating",
        ],
    },
    "margin_leak": {
        "severity": "critical",
        "category": "Pricing",
        "description": "Systematic margin erosion from pricing strategy issues",
        "multi_hop_depth": 1,
        "recommendations": [
            "Audit promotional pricing end dates",
            "Check for stuck sale prices",
            "Review competitive price matching rules",
            "Analyze margin by category for patterns",
        ],
    },
    "demand_shift": {
        "severity": "medium",
        "category": "Market",
        "description": "Market demand changed due to seasonality or trends",
        "multi_hop_depth": 1,
        "recommendations": [
            "Compare YoY sales trends",
            "Review category performance vs market",
            "Consider markdown strategy for declining items",
            "Adjust reorder points based on new demand",
        ],
    },
    "quality_issue": {
        "severity": "high",
        "category": "Product",
        "description": "Product quality problems causing returns or complaints",
        "multi_hop_depth": 3,  # Requires SKU -> Vendor -> Factory hops
        "recommendations": [
            "Review return rate by SKU",
            "Check customer complaints and reviews",
            "Investigate vendor quality controls",
            "Consider alternative suppliers",
        ],
    },
    "pricing_error": {
        "severity": "medium",
        "category": "Operations",
        "description": "Incorrect pricing configuration in POS or catalog",
        "multi_hop_depth": 1,
        "recommendations": [
            "Audit price vs MSRP discrepancies",
            "Check for promotional pricing errors",
            "Verify cost entry accuracy",
            "Review recent price change logs",
        ],
    },
    "inventory_drift": {
        "severity": "medium",
        "category": "Operations",
        "description": "Gradual inventory accuracy degradation",
        "multi_hop_depth": 1,
        "recommendations": [
            "Schedule full physical inventory",
            "Implement cycle counting program",
            "Review receiving procedures",
            "Check for phantom inventory issues",
        ],
    },
}


def get_cause_metadata(cause_key: str) -> dict:
    """Get metadata for a specific cause."""
    return CAUSE_METADATA.get(cause_key, {
        "severity": "info",
        "category": "Unknown",
        "description": "Unknown cause type",
        "multi_hop_depth": 1,
        "recommendations": ["Manual investigation required"],
    })


def create_cause_vectors(ctx: "AnalysisContext") -> CauseVectors:
    """
    Factory function to create CauseVectors for a context.

    Args:
        ctx: Analysis context (provides seed_hash and device)

    Returns:
        CauseVectors instance bound to the context
    """
    return CauseVectors(ctx=ctx)
