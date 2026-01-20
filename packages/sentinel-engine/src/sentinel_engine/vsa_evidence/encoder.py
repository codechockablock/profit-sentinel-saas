"""
Evidence-Based Encoder - Encodes facts by what causes they support.

This is the key insight from the validated research (grounded_v2.py approach):

    Instead of: fact -> [entity x attribute x value] (generic encoding)
    We encode:  fact -> [weighted bundle of cause vectors it supports]

This creates a direct mapping from evidence to hypotheses, making
similarity meaningful for cause identification.

Validated performance:
- 0% quantitative hallucination (vs 39.6% ungrounded)
- 100% multi-hop reasoning accuracy
- +586% improvement over random baseline
- 95% reduction in retrieval operations vs chunk RAG

Reference: RESEARCH_SUMMARY.md - Algorithm (v2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..context import AnalysisContext

from .causes import CauseVectors, create_cause_vectors
from .rules import RuleEngine, create_rule_engine, extract_evidence_facts

logger = logging.getLogger(__name__)


# =============================================================================
# EVIDENCE ENCODER
# =============================================================================

@dataclass
class EvidenceEncoder:
    """
    Encodes facts as weighted bundles of cause vectors.

    The encoding process:
    1. Extract evidence facts from raw data
    2. Apply rules to get cause weights
    3. Create weighted bundle of cause vectors
    4. Normalize the bundle

    This ensures that:
    - Facts with strong evidence for a cause have high similarity to that cause vector
    - Facts with conflicting evidence have distributed similarity (ambiguity)
    - Facts with no matching rules encode to near-zero (routes to cold path)

    Thread-safe: Uses context-isolated state.
    """

    ctx: "AnalysisContext"
    cause_vectors: CauseVectors = field(default=None, repr=False)
    rule_engine: RuleEngine = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize cause vectors and rule engine."""
        if self.cause_vectors is None:
            self.cause_vectors = create_cause_vectors(self.ctx)
        if self.rule_engine is None:
            self.rule_engine = create_rule_engine()

    def encode_fact(
        self,
        fact: dict[str, Any],
        include_sku_binding: bool = True,
    ) -> torch.Tensor:
        """
        Encode a single fact as a weighted bundle of cause vectors.

        This is the core of evidence-based encoding:
            fact -> weighted_bundle([w * cause_vec for cause, w in weights])

        Args:
            fact: Dictionary of attribute -> value pairs (from extract_evidence_facts)
            include_sku_binding: Whether to bind with SKU vector for identification

        Returns:
            Normalized hypervector representing the evidence
        """
        # Apply rules to get cause weights
        cause_weights = self.rule_engine.apply(fact)

        if not cause_weights:
            # No rules matched - return zero vector (will route to cold path)
            logger.debug("No rules matched - returning zero vector")
            return self.ctx.zeros()

        # Create weighted bundle of cause vectors
        bundle = self.ctx.zeros()

        for cause, weight in cause_weights.items():
            cause_vec = self.cause_vectors.get(cause)
            if cause_vec is not None:
                bundle = bundle + weight * cause_vec

        # Normalize the bundle
        bundle = self.ctx.normalize(bundle)

        # Optionally bind with SKU for identification
        if include_sku_binding and "sku" in fact:
            sku_vec = self.ctx.get_or_create(str(fact["sku"]))
            bundle = bundle * sku_vec
            bundle = self.ctx.normalize(bundle)

        return bundle

    def encode_facts(
        self,
        facts: list[dict[str, Any]],
        include_sku_binding: bool = True,
    ) -> list[torch.Tensor]:
        """
        Encode multiple facts.

        Args:
            facts: List of fact dictionaries
            include_sku_binding: Whether to bind with SKU vectors

        Returns:
            List of encoded hypervectors
        """
        return [
            self.encode_fact(fact, include_sku_binding)
            for fact in facts
        ]

    def encode_row(
        self,
        row: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Convenience method: extract facts from POS row and encode.

        Args:
            row: Raw POS data row
            context: Optional context with dataset-level stats

        Returns:
            Encoded evidence hypervector
        """
        facts = extract_evidence_facts(row, context)
        return self.encode_fact(facts)

    def encode_rows(
        self,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> list[torch.Tensor]:
        """
        Encode multiple POS rows.

        Args:
            rows: List of raw POS data rows
            context: Optional context with dataset-level stats

        Returns:
            List of encoded hypervectors
        """
        return [self.encode_row(row, context) for row in rows]

    def bundle_evidence(
        self,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """
        Bundle all evidence from multiple rows into single vector.

        Useful for aggregate queries like "what's causing margin issues?"

        Known limitation: Bundle capacity cliff at ~200-290 facts.
        For larger datasets, use hierarchical encoding.

        Args:
            rows: List of raw POS data rows
            context: Optional context with dataset-level stats

        Returns:
            Bundled evidence hypervector
        """
        if len(rows) > 200:
            logger.warning(
                f"Bundling {len(rows)} facts may hit capacity cliff (~200-290). "
                "Consider hierarchical encoding for better accuracy."
            )

        evidence_vecs = self.encode_rows(rows, context)

        # Filter out zero vectors (no matching rules)
        non_zero_vecs = [
            v for v in evidence_vecs
            if torch.norm(v).item() > 1e-6
        ]

        if not non_zero_vecs:
            return self.ctx.zeros()

        # Sum and normalize
        bundle = sum(non_zero_vecs)
        return self.ctx.normalize(bundle)

    def get_rule_explanations(self, fact: dict[str, Any]) -> list[dict]:
        """
        Get human-readable explanations for why rules matched.

        Useful for transparency and debugging.

        Args:
            fact: Fact dictionary

        Returns:
            List of matched rule explanations
        """
        return self.rule_engine.explain_match(fact)

    def add_rule(
        self,
        attribute: str,
        pattern: str,
        cause: str,
        weight: float,
        description: str = "",
    ):
        """
        Add a new evidence rule (for cold path feedback loop).

        When LLM discovers new patterns, they can be added as hot path rules.

        Args:
            attribute: Fact attribute to match
            pattern: Pattern for matching
            cause: Cause this supports
            weight: Strength of evidence
            description: Human-readable explanation
        """
        from .rules import EvidenceRule

        rule = EvidenceRule(
            attribute=attribute,
            pattern=pattern,
            cause=cause,
            weight=weight,
            description=description,
        )
        self.rule_engine.add_rule(rule)


# =============================================================================
# HIERARCHICAL ENCODER (for large datasets)
# =============================================================================

@dataclass
class HierarchicalEvidenceEncoder:
    """
    Hierarchical encoder for datasets larger than bundle capacity (~200 facts).

    Strategy:
    1. Group facts by category/vendor
    2. Encode each group into a sub-bundle
    3. Combine sub-bundles with weighting

    This extends capacity beyond the ~200-290 cliff.

    Validated: H9 showed >500 items with hierarchical encoding.
    """

    ctx: "AnalysisContext"
    encoder: EvidenceEncoder = field(default=None, repr=False)
    group_size: int = 150  # Stay below capacity cliff

    def __post_init__(self):
        """Initialize base encoder."""
        if self.encoder is None:
            self.encoder = create_evidence_encoder(self.ctx)

    def encode_with_hierarchy(
        self,
        rows: list[dict[str, Any]],
        group_key: str = "category",
        context: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Encode rows with hierarchical grouping.

        Args:
            rows: List of raw POS data rows
            group_key: Key to group by (default: "category")
            context: Optional context with dataset-level stats

        Returns:
            Tuple of (overall_bundle, group_bundles)
        """
        # Group rows
        groups: dict[str, list[dict]] = {}
        for row in rows:
            key = str(row.get(group_key, "unknown")).lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(row)

        logger.info(f"Hierarchical encoding: {len(rows)} rows in {len(groups)} groups")

        # Encode each group
        group_bundles: dict[str, torch.Tensor] = {}
        for group_name, group_rows in groups.items():
            # Chunk large groups
            if len(group_rows) > self.group_size:
                chunks = [
                    group_rows[i:i + self.group_size]
                    for i in range(0, len(group_rows), self.group_size)
                ]
                chunk_bundles = [
                    self.encoder.bundle_evidence(chunk, context)
                    for chunk in chunks
                ]
                group_bundle = self.ctx.normalize(sum(chunk_bundles))
            else:
                group_bundle = self.encoder.bundle_evidence(group_rows, context)

            group_bundles[group_name] = group_bundle

        # Combine group bundles with weighting by size
        total_rows = len(rows)
        overall_bundle = self.ctx.zeros()
        for group_name, group_rows in groups.items():
            weight = len(group_rows) / total_rows
            overall_bundle = overall_bundle + weight * group_bundles[group_name]

        overall_bundle = self.ctx.normalize(overall_bundle)

        return overall_bundle, group_bundles


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_evidence_encoder(ctx: "AnalysisContext") -> EvidenceEncoder:
    """
    Factory function to create an EvidenceEncoder.

    Args:
        ctx: Analysis context (provides vector operations and isolation)

    Returns:
        Configured EvidenceEncoder instance
    """
    return EvidenceEncoder(ctx=ctx)


def create_hierarchical_encoder(
    ctx: "AnalysisContext",
    group_size: int = 150,
) -> HierarchicalEvidenceEncoder:
    """
    Factory function to create a HierarchicalEvidenceEncoder.

    Args:
        ctx: Analysis context
        group_size: Maximum facts per group (default 150, below capacity cliff)

    Returns:
        Configured HierarchicalEvidenceEncoder instance
    """
    return HierarchicalEvidenceEncoder(ctx=ctx, group_size=group_size)
