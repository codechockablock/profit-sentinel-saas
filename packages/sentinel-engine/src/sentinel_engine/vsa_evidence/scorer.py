"""
Cause Scorer - Scores causes by summing positive similarities.

The scoring algorithm from validated research (grounded_v2.py):

    def score_causes(facts):
        evidence_vecs = [encode_fact(f) for f in facts]
        scores = {}
        for cause in CAUSES:
            scores[cause] = sum(
                max(0, similarity(ev, cause_vec))
                for ev in evidence_vecs
            )
        return scores

Key insight: Use positive similarity summing, not averaging.
- Averaging dilutes strong signals
- Summing accumulates evidence
- max(0, sim) prevents negative evidence from canceling

Validated performance:
- +159% actionability improvement over rule-based only
- 100% multi-hop accuracy
- Ambiguity detection: 100% (flags cases needing human review)

Reference: RESEARCH_SUMMARY.md - H3, H8 results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..context import AnalysisContext

from .causes import CauseVectors, create_cause_vectors, get_cause_metadata
from .encoder import EvidenceEncoder, create_evidence_encoder

logger = logging.getLogger(__name__)


# =============================================================================
# SCORING RESULT TYPES
# =============================================================================


@dataclass
class CauseScore:
    """Score for a single cause with metadata."""

    cause: str
    score: float
    confidence: float  # 0.0-1.0, derived from score magnitude
    evidence_count: int  # Number of facts supporting this cause
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "cause": self.cause,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "evidence_count": self.evidence_count,
            "severity": self.metadata.get("severity", "info"),
            "category": self.metadata.get("category", "Unknown"),
            "description": self.metadata.get("description", ""),
            "recommendations": self.metadata.get("recommendations", []),
        }


@dataclass
class ScoringResult:
    """Complete scoring result with all causes and diagnostics."""

    scores: list[CauseScore]
    top_cause: str | None
    ambiguity_score: float  # 0.0 = clear winner, 1.0 = highly ambiguous
    confidence: float  # Overall confidence in diagnosis
    needs_cold_path: bool  # Whether to route to LLM for deeper analysis
    cold_path_reason: str | None
    evidence_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "scores": [s.to_dict() for s in self.scores],
            "top_cause": self.top_cause,
            "ambiguity_score": round(self.ambiguity_score, 4),
            "confidence": round(self.confidence, 4),
            "needs_cold_path": self.needs_cold_path,
            "cold_path_reason": self.cold_path_reason,
            "evidence_summary": self.evidence_summary,
        }

    @property
    def is_confident(self) -> bool:
        """Check if result is confident enough for hot path resolution."""
        return self.confidence >= 0.6 and self.ambiguity_score < 0.5


# =============================================================================
# CAUSE SCORER
# =============================================================================


@dataclass
class CauseScorer:
    """
    Scores causes by summing positive similarities from evidence vectors.

    Architecture:
    1. Encode facts as evidence vectors (weighted cause bundles)
    2. For each cause, sum positive similarities from all evidence
    3. Rank causes by accumulated evidence
    4. Detect ambiguity (multiple high scores)
    5. Route to cold path if needed

    Thread-safe: Uses context-isolated state.
    """

    ctx: AnalysisContext
    encoder: EvidenceEncoder = field(default=None, repr=False)
    cause_vectors: CauseVectors = field(default=None, repr=False)

    # Routing thresholds (from validated research)
    confidence_threshold: float = 0.6  # Below this -> cold path
    ambiguity_threshold: float = 0.5  # Above this -> cold path
    min_evidence_count: int = 2  # Below this -> cold path
    severity_cold_path: bool = True  # High severity always routes to cold path

    def __post_init__(self):
        """Initialize encoder and cause vectors."""
        if self.encoder is None:
            self.encoder = create_evidence_encoder(self.ctx)
        if self.cause_vectors is None:
            self.cause_vectors = create_cause_vectors(self.ctx)

    def score_facts(
        self,
        facts: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> ScoringResult:
        """
        Score all causes based on evidence facts.

        This is the main scoring algorithm:
            scores[cause] = sum(max(0, similarity(ev, cause_vec)) for ev in evidence_vecs)

        Args:
            facts: List of fact dictionaries (from extract_evidence_facts)
            context: Optional dataset-level context

        Returns:
            ScoringResult with ranked causes and diagnostics
        """
        if not facts:
            return self._empty_result("No facts provided")

        # Encode all facts as evidence vectors
        evidence_vecs = self.encoder.encode_facts(facts, include_sku_binding=False)

        # Filter out zero vectors (no matching rules)
        non_zero = [
            (f, v) for f, v in zip(facts, evidence_vecs) if torch.norm(v).item() > 1e-6
        ]

        if not non_zero:
            return self._empty_result("No rules matched any facts")

        facts_matched, evidence_vecs = zip(*non_zero)
        evidence_count = len(evidence_vecs)

        # Score each cause by summing positive similarities
        cause_scores: dict[str, float] = {}
        cause_evidence_counts: dict[str, int] = {}

        for cause_key in self.cause_vectors.keys():
            cause_vec = self.cause_vectors.get(cause_key)
            if cause_vec is None:
                continue

            total_score = 0.0
            supporting_count = 0

            for ev_vec in evidence_vecs:
                # Phasor similarity: real part of complex dot product
                sim = torch.real(torch.dot(torch.conj(cause_vec), ev_vec)).item()

                # Sum only positive similarities (key insight!)
                if sim > 0:
                    total_score += sim
                    supporting_count += 1

            cause_scores[cause_key] = total_score
            cause_evidence_counts[cause_key] = supporting_count

        # Calculate confidence and ambiguity
        scores_sorted = sorted(cause_scores.items(), key=lambda x: x[1], reverse=True)
        max_score = scores_sorted[0][1] if scores_sorted else 0

        # Normalize scores to 0-1 range for confidence
        if max_score > 0:
            confidence = min(1.0, max_score / (evidence_count * 0.5))  # Scaled
        else:
            confidence = 0.0

        # Calculate ambiguity: how close is second place to first?
        if len(scores_sorted) >= 2 and max_score > 0:
            second_score = scores_sorted[1][1]
            ambiguity = second_score / max_score if max_score > 0 else 0.0
        else:
            ambiguity = 0.0

        # Build CauseScore objects
        result_scores = []
        for cause_key, score in scores_sorted:
            metadata = get_cause_metadata(cause_key)
            cause_confidence = score / max_score if max_score > 0 else 0.0

            result_scores.append(
                CauseScore(
                    cause=cause_key,
                    score=score,
                    confidence=cause_confidence,
                    evidence_count=cause_evidence_counts[cause_key],
                    metadata=metadata,
                )
            )

        # Determine top cause
        top_cause = scores_sorted[0][0] if scores_sorted and max_score > 0 else None

        # Determine if cold path is needed
        needs_cold_path, cold_path_reason = self._check_cold_path_routing(
            confidence=confidence,
            ambiguity=ambiguity,
            evidence_count=evidence_count,
            top_cause=top_cause,
            top_score=max_score,
        )

        return ScoringResult(
            scores=result_scores,
            top_cause=top_cause,
            ambiguity_score=ambiguity,
            confidence=confidence,
            needs_cold_path=needs_cold_path,
            cold_path_reason=cold_path_reason,
            evidence_summary={
                "total_facts": len(facts),
                "facts_with_rules": evidence_count,
                "facts_no_rules": len(facts) - evidence_count,
                "max_score": max_score,
            },
        )

    def score_rows(
        self,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> ScoringResult:
        """
        Convenience method: extract facts from POS rows and score.

        Args:
            rows: Raw POS data rows
            context: Optional dataset-level context (avg_margin, etc.)

        Returns:
            ScoringResult with ranked causes
        """
        from .rules import extract_evidence_facts

        facts = [extract_evidence_facts(row, context) for row in rows]
        return self.score_facts(facts, context)

    def score_single_row(
        self,
        row: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ScoringResult:
        """
        Score a single row for quick diagnosis.

        Args:
            row: Single POS data row
            context: Optional dataset-level context

        Returns:
            ScoringResult for this row
        """
        return self.score_rows([row], context)

    def _check_cold_path_routing(
        self,
        confidence: float,
        ambiguity: float,
        evidence_count: int,
        top_cause: str | None,
        top_score: float,
    ) -> tuple[bool, str | None]:
        """
        Determine if this query should route to cold path (LLM).

        Routing criteria (from validated research):
        - Low confidence (<0.6): Not enough signal
        - High ambiguity (>0.5): Multiple plausible causes
        - Insufficient evidence (<2 facts): Can't make reliable diagnosis
        - High severity cause: Always verify critical findings
        - Novel pattern (no rules matched): Unknown territory

        Args:
            confidence: Overall confidence score
            ambiguity: Ambiguity score
            evidence_count: Number of evidence facts
            top_cause: Identified top cause
            top_score: Score of top cause

        Returns:
            Tuple of (needs_cold_path, reason)
        """
        if top_score == 0 or top_cause is None:
            return True, "no_matching_rules"

        if confidence < self.confidence_threshold:
            return (
                True,
                f"low_confidence ({confidence:.2f} < {self.confidence_threshold})",
            )

        if ambiguity > self.ambiguity_threshold:
            return (
                True,
                f"ambiguous_evidence ({ambiguity:.2f} > {self.ambiguity_threshold})",
            )

        if evidence_count < self.min_evidence_count:
            return (
                True,
                f"insufficient_evidence ({evidence_count} < {self.min_evidence_count})",
            )

        # High severity causes should be verified
        if self.severity_cold_path and top_cause:
            metadata = get_cause_metadata(top_cause)
            if metadata.get("severity") == "critical":
                return True, f"critical_severity_verification ({top_cause})"

        return False, None

    def _empty_result(self, reason: str) -> ScoringResult:
        """Create empty result when scoring can't proceed."""
        return ScoringResult(
            scores=[],
            top_cause=None,
            ambiguity_score=1.0,
            confidence=0.0,
            needs_cold_path=True,
            cold_path_reason=reason,
            evidence_summary={"error": reason},
        )


# =============================================================================
# BATCH SCORER (for analyzing multiple items at once)
# =============================================================================


@dataclass
class BatchScorer:
    """
    Scores multiple items efficiently in batch.

    Use for analyzing entire datasets or categories.
    """

    ctx: AnalysisContext
    scorer: CauseScorer = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize scorer."""
        if self.scorer is None:
            self.scorer = create_cause_scorer(self.ctx)

    def score_by_item(
        self,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, ScoringResult]:
        """
        Score each item (SKU) individually.

        Args:
            rows: POS data rows (may have multiple rows per SKU)
            context: Optional dataset-level context

        Returns:
            Dict mapping SKU to ScoringResult
        """
        from ..core import SKU_ALIASES, _get_field

        # Group rows by SKU
        sku_rows: dict[str, list[dict]] = {}
        for row in rows:
            sku = str(_get_field(row, SKU_ALIASES, "unknown"))
            if sku not in sku_rows:
                sku_rows[sku] = []
            sku_rows[sku].append(row)

        # Score each SKU
        results: dict[str, ScoringResult] = {}
        for sku, item_rows in sku_rows.items():
            results[sku] = self.scorer.score_rows(item_rows, context)

        return results

    def score_by_category(
        self,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, ScoringResult]:
        """
        Score each category as a whole.

        Useful for finding category-wide issues (vendor problems, etc.)

        Args:
            rows: POS data rows
            context: Optional dataset-level context

        Returns:
            Dict mapping category to ScoringResult
        """
        from ..core import CATEGORY_ALIASES, _get_field

        # Group rows by category
        cat_rows: dict[str, list[dict]] = {}
        for row in rows:
            cat = str(_get_field(row, CATEGORY_ALIASES, "unknown"))
            if cat not in cat_rows:
                cat_rows[cat] = []
            cat_rows[cat].append(row)

        # Score each category
        results: dict[str, ScoringResult] = {}
        for cat, category_rows in cat_rows.items():
            results[cat] = self.scorer.score_rows(category_rows, context)

        return results

    def get_hot_path_resolvable(
        self,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[str, ScoringResult]], list[tuple[str, ScoringResult]]]:
        """
        Separate items into hot path resolvable vs cold path needed.

        Args:
            rows: POS data rows
            context: Optional dataset-level context

        Returns:
            Tuple of (hot_path_items, cold_path_items)
            Each is list of (sku, ScoringResult) tuples
        """
        item_results = self.score_by_item(rows, context)

        hot_path = []
        cold_path = []

        for sku, result in item_results.items():
            if result.needs_cold_path:
                cold_path.append((sku, result))
            else:
                hot_path.append((sku, result))

        logger.info(
            f"Hot/cold split: {len(hot_path)} hot path, {len(cold_path)} cold path "
            f"({len(hot_path)/(len(hot_path)+len(cold_path))*100:.1f}% hot)"
        )

        return hot_path, cold_path


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_cause_scorer(
    ctx: AnalysisContext,
    confidence_threshold: float = 0.6,
    ambiguity_threshold: float = 0.5,
    min_evidence_count: int = 2,
) -> CauseScorer:
    """
    Factory function to create a CauseScorer.

    Args:
        ctx: Analysis context
        confidence_threshold: Below this routes to cold path
        ambiguity_threshold: Above this routes to cold path
        min_evidence_count: Minimum facts for reliable diagnosis

    Returns:
        Configured CauseScorer instance
    """
    return CauseScorer(
        ctx=ctx,
        confidence_threshold=confidence_threshold,
        ambiguity_threshold=ambiguity_threshold,
        min_evidence_count=min_evidence_count,
    )


def create_batch_scorer(ctx: AnalysisContext) -> BatchScorer:
    """
    Factory function to create a BatchScorer.

    Args:
        ctx: Analysis context

    Returns:
        Configured BatchScorer instance
    """
    return BatchScorer(ctx=ctx)
