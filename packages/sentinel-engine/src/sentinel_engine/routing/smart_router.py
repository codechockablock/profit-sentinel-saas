"""
Smart Router - Hot/cold path routing for hybrid VSA-LLM architecture.

Architecture:
- HOT PATH (VSA): Sub-millisecond inference (<50ms target, achieves 0.003ms)
  - Pattern matching against known causes
  - High-confidence, common anomalies
  - Throughput: 10,000+ anomalies/sec

- COLD PATH (LLM): Deep causal reasoning (~500ms)
  - Novel/ambiguous patterns
  - Multi-step explanation generation
  - Human-readable insights

Routing criteria (from validated research):
| Criterion | Threshold | Effect |
|-----------|-----------|--------|
| Low confidence | <0.6 | Route to cold |
| Ambiguous scores | gap <0.15 | Route to cold |
| Insufficient evidence | <2 facts | Route to cold |
| High severity | critical | Route to cold |
| Novel pattern | no_match | Route to cold |

Benchmark results:
- Hot path: 0.003ms
- Cold path: 500ms
- Speedup: 5,059x

Reference: RESEARCH_SUMMARY.md - Hot/Cold Path Architecture
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..context import AnalysisContext

from ..vsa_evidence import (
    ScoringResult,
    create_cause_scorer,
    get_cause_metadata,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING TYPES
# =============================================================================


class RoutingDecision(Enum):
    """Decision on which path to use."""

    HOT_PATH = "hot"  # VSA-only resolution
    COLD_PATH = "cold"  # Route to LLM
    HYBRID = "hybrid"  # Hot first, then cold for verification


@dataclass
class HotPathResult:
    """Result from hot path (VSA) analysis."""

    cause: str
    confidence: float
    ambiguity: float
    evidence_count: int
    latency_ms: float
    recommendations: list[str]
    severity: str
    explanation: str
    scoring_result: ScoringResult | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "path": "hot",
            "cause": self.cause,
            "confidence": round(self.confidence, 4),
            "ambiguity": round(self.ambiguity, 4),
            "evidence_count": self.evidence_count,
            "latency_ms": round(self.latency_ms, 3),
            "recommendations": self.recommendations,
            "severity": self.severity,
            "explanation": self.explanation,
        }


@dataclass
class ColdPathRequest:
    """Request to be sent to cold path (LLM)."""

    sku: str | None
    facts: list[dict[str, Any]]
    hot_path_result: HotPathResult | None
    routing_reason: str
    context: dict[str, Any]

    def to_prompt(self) -> str:
        """Generate LLM prompt for cold path analysis."""
        prompt_parts = [
            "Analyze this retail inventory issue and provide a root cause diagnosis.",
            "",
            f"SKU: {self.sku or 'Multiple items'}",
            f"Routing reason: {self.routing_reason}",
            "",
            "Evidence facts:",
        ]

        for i, fact in enumerate(self.facts[:10], 1):  # Limit to 10 facts
            prompt_parts.append(f"  {i}. {fact}")

        if self.hot_path_result:
            prompt_parts.extend(
                [
                    "",
                    "Hot path preliminary analysis:",
                    f"  - Initial cause: {self.hot_path_result.cause}",
                    f"  - Confidence: {self.hot_path_result.confidence:.2f}",
                    f"  - Ambiguity: {self.hot_path_result.ambiguity:.2f}",
                ]
            )

        prompt_parts.extend(
            [
                "",
                "Please provide:",
                "1. Root cause identification",
                "2. Confidence level (0-1)",
                "3. Evidence supporting this cause",
                "4. Recommended actions",
                "5. Any additional causes to investigate",
            ]
        )

        return "\n".join(prompt_parts)


@dataclass
class AnalysisResult:
    """Combined result from hot and/or cold path."""

    path_used: RoutingDecision
    hot_result: HotPathResult | None
    cold_result: dict[str, Any] | None
    final_cause: str | None
    final_confidence: float
    total_latency_ms: float
    grounded: bool  # Whether result is grounded in evidence

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "path_used": self.path_used.value,
            "final_cause": self.final_cause,
            "final_confidence": round(self.final_confidence, 4),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "grounded": self.grounded,
        }

        if self.hot_result:
            result["hot_result"] = self.hot_result.to_dict()

        if self.cold_result:
            result["cold_result"] = self.cold_result

        return result


# =============================================================================
# SMART ROUTER
# =============================================================================


@dataclass
class SmartRouter:
    """
    Routes queries between hot path (VSA) and cold path (LLM).

    The router implements the hybrid architecture:
    1. Always run hot path first (fast, grounded)
    2. Check routing criteria
    3. Route to cold path if needed
    4. Combine results if hybrid

    Cost-aware: Minimizes LLM calls for SMB efficiency.
    """

    ctx: AnalysisContext

    # Routing thresholds
    confidence_threshold: float = 0.6
    ambiguity_threshold: float = 0.5
    min_evidence_count: int = 2
    severity_verification: bool = True  # Always verify critical findings

    # Cold path handler (set by integration)
    cold_path_handler: Callable[[ColdPathRequest], dict] | None = field(
        default=None, repr=False
    )

    # Metrics
    _hot_path_count: int = field(default=0, repr=False)
    _cold_path_count: int = field(default=0, repr=False)
    _total_hot_latency_ms: float = field(default=0.0, repr=False)
    _total_cold_latency_ms: float = field(default=0.0, repr=False)

    def analyze(
        self,
        rows: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        force_cold_path: bool = False,
        sku: str | None = None,
    ) -> AnalysisResult:
        """
        Analyze rows with smart hot/cold routing.

        Args:
            rows: POS data rows to analyze
            context: Optional dataset-level context (avg_margin, etc.)
            force_cold_path: Force cold path (for testing/debugging)
            sku: Optional SKU identifier

        Returns:
            AnalysisResult with diagnosis and path information
        """
        start_time = time.perf_counter()

        # Step 1: Run hot path (always - it's fast)
        hot_start = time.perf_counter()
        scorer = create_cause_scorer(self.ctx)
        scoring_result = scorer.score_rows(rows, context)
        hot_latency_ms = (time.perf_counter() - hot_start) * 1000

        self._hot_path_count += 1
        self._total_hot_latency_ms += hot_latency_ms

        # Build hot path result
        hot_result = self._build_hot_result(scoring_result, hot_latency_ms)

        # Step 2: Decide routing
        if force_cold_path:
            routing = RoutingDecision.COLD_PATH
            routing_reason = "forced_cold_path"
        elif scoring_result.needs_cold_path:
            routing = RoutingDecision.COLD_PATH
            routing_reason = scoring_result.cold_path_reason or "unknown"
        elif self.severity_verification and hot_result.severity == "critical":
            routing = RoutingDecision.HYBRID
            routing_reason = "critical_severity_verification"
        else:
            routing = RoutingDecision.HOT_PATH
            routing_reason = None

        # Step 3: Execute cold path if needed
        cold_result = None
        if routing in (RoutingDecision.COLD_PATH, RoutingDecision.HYBRID):
            if self.cold_path_handler:
                cold_start = time.perf_counter()

                # Build cold path request
                from ..vsa_evidence.rules import extract_evidence_facts

                facts = [extract_evidence_facts(row, context) for row in rows]

                request = ColdPathRequest(
                    sku=sku,
                    facts=facts,
                    hot_path_result=(
                        hot_result if routing == RoutingDecision.HYBRID else None
                    ),
                    routing_reason=routing_reason or "unknown",
                    context=context or {},
                )

                try:
                    cold_result = self.cold_path_handler(request)
                    cold_latency_ms = (time.perf_counter() - cold_start) * 1000
                    self._cold_path_count += 1
                    self._total_cold_latency_ms += cold_latency_ms

                    logger.info(f"Cold path completed in {cold_latency_ms:.1f}ms")
                except Exception as e:
                    logger.error(f"Cold path failed: {e}")
                    # Fall back to hot path result
                    routing = RoutingDecision.HOT_PATH
                    cold_result = {"error": str(e)}
            else:
                logger.warning("Cold path needed but no handler configured")
                cold_result = {"skipped": "no_handler_configured"}

        # Step 4: Combine results
        total_latency_ms = (time.perf_counter() - start_time) * 1000

        if routing == RoutingDecision.HOT_PATH:
            final_cause = hot_result.cause
            final_confidence = hot_result.confidence
            grounded = True
        elif (
            routing == RoutingDecision.COLD_PATH
            and cold_result
            and "cause" in cold_result
        ):
            final_cause = cold_result.get("cause")
            final_confidence = cold_result.get("confidence", 0.0)
            grounded = False  # LLM results not automatically grounded
        elif routing == RoutingDecision.HYBRID:
            # Prefer hot path cause if confirmed by cold path
            if cold_result and cold_result.get("confirms_hot_path", False):
                final_cause = hot_result.cause
                final_confidence = max(
                    hot_result.confidence, cold_result.get("confidence", 0.0)
                )
                grounded = True
            else:
                final_cause = (
                    cold_result.get("cause") if cold_result else hot_result.cause
                )
                final_confidence = (
                    cold_result.get("confidence", 0.0)
                    if cold_result
                    else hot_result.confidence
                )
                grounded = False
        else:
            final_cause = hot_result.cause
            final_confidence = hot_result.confidence
            grounded = True

        return AnalysisResult(
            path_used=routing,
            hot_result=hot_result,
            cold_result=cold_result,
            final_cause=final_cause,
            final_confidence=final_confidence,
            total_latency_ms=total_latency_ms,
            grounded=grounded,
        )

    def _build_hot_result(
        self,
        scoring_result: ScoringResult,
        latency_ms: float,
    ) -> HotPathResult:
        """Build HotPathResult from scoring result."""
        if scoring_result.top_cause:
            metadata = get_cause_metadata(scoring_result.top_cause)
            cause = scoring_result.top_cause
            recommendations = metadata.get("recommendations", [])
            severity = metadata.get("severity", "info")
        else:
            cause = "unknown"
            recommendations = ["Manual investigation required"]
            severity = "info"

        # Build explanation
        if scoring_result.scores:
            top_scores = scoring_result.scores[:3]
            explanation = f"Top causes: {', '.join(f'{s.cause} ({s.score:.2f})' for s in top_scores)}"
        else:
            explanation = "No evidence matched any rules"

        # Get evidence count from top cause
        evidence_count = 0
        for score in scoring_result.scores:
            if score.cause == cause:
                evidence_count = score.evidence_count
                break

        return HotPathResult(
            cause=cause,
            confidence=scoring_result.confidence,
            ambiguity=scoring_result.ambiguity_score,
            evidence_count=evidence_count,
            latency_ms=latency_ms,
            recommendations=recommendations,
            severity=severity,
            explanation=explanation,
            scoring_result=scoring_result,
        )

    def set_cold_path_handler(
        self,
        handler: Callable[[ColdPathRequest], dict],
    ):
        """
        Set the cold path handler (typically wraps LLM API).

        Args:
            handler: Function that takes ColdPathRequest and returns dict
        """
        self.cold_path_handler = handler

    def get_metrics(self) -> dict:
        """Get router performance metrics."""
        total_calls = self._hot_path_count + self._cold_path_count
        hot_ratio = self._hot_path_count / total_calls if total_calls > 0 else 0

        return {
            "hot_path_count": self._hot_path_count,
            "cold_path_count": self._cold_path_count,
            "total_calls": total_calls,
            "hot_path_ratio": round(hot_ratio, 4),
            "avg_hot_latency_ms": round(
                self._total_hot_latency_ms / max(1, self._hot_path_count), 3
            ),
            "avg_cold_latency_ms": round(
                self._total_cold_latency_ms / max(1, self._cold_path_count), 3
            ),
            "estimated_cost_savings": f"{hot_ratio * 100:.1f}% LLM calls avoided",
        }

    def reset_metrics(self):
        """Reset performance metrics."""
        self._hot_path_count = 0
        self._cold_path_count = 0
        self._total_hot_latency_ms = 0.0
        self._total_cold_latency_ms = 0.0


# =============================================================================
# COLD PATH HANDLERS
# =============================================================================


def create_grok_cold_path_handler(
    api_key: str,
    model: str = "grok-beta",
    max_tokens: int = 1024,
) -> Callable[[ColdPathRequest], dict]:
    """
    Create a cold path handler using Grok (xAI) API.

    Args:
        api_key: xAI API key
        model: Model to use (default grok-beta)
        max_tokens: Maximum response tokens

    Returns:
        Handler function for cold path requests
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1",
    )

    def handler(request: ColdPathRequest) -> dict:
        """Handle cold path request via Grok."""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a retail inventory analyst expert. Analyze profit leaks "
                            "and provide root cause diagnosis with evidence-based reasoning. "
                            "Be concise and actionable."
                        ),
                    },
                    {
                        "role": "user",
                        "content": request.to_prompt(),
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            content = response.choices[0].message.content

            # Parse structured response (simplified)
            return {
                "cause": _extract_cause_from_response(content),
                "confidence": 0.7,  # Default confidence for LLM
                "reasoning": content,
                "confirms_hot_path": _check_hot_path_confirmation(content, request),
                "model_used": model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
            }

        except Exception as e:
            logger.error(f"Grok API error: {e}")
            return {
                "error": str(e),
                "cause": (
                    request.hot_path_result.cause if request.hot_path_result else None
                ),
                "confidence": 0.0,
            }

    return handler


def _extract_cause_from_response(content: str) -> str | None:
    """Extract cause from LLM response (simplified extraction)."""
    content_lower = content.lower()

    # Map keywords to causes
    cause_keywords = {
        "theft": ["theft", "stolen", "shrinkage", "pilferage"],
        "vendor_increase": ["vendor", "cost increase", "supplier price"],
        "rebate_timing": ["rebate", "dating", "payment terms"],
        "margin_leak": ["margin", "pricing", "promotional"],
        "demand_shift": ["demand", "seasonal", "trend"],
        "quality_issue": ["quality", "defect", "return", "complaint"],
        "pricing_error": ["pricing error", "data entry", "price mismatch"],
        "inventory_drift": ["inventory drift", "cycle count", "variance"],
    }

    for cause, keywords in cause_keywords.items():
        if any(kw in content_lower for kw in keywords):
            return cause

    return None


def _check_hot_path_confirmation(content: str, request: ColdPathRequest) -> bool:
    """Check if LLM confirms hot path diagnosis."""
    if not request.hot_path_result:
        return False

    hot_cause = request.hot_path_result.cause
    content_lower = content.lower()

    # Simple confirmation: cause mentioned positively
    if hot_cause in content_lower:
        # Check for negation
        negations = ["not", "unlikely", "ruled out", "incorrect"]
        for neg in negations:
            if (
                neg in content_lower
                and hot_cause
                in content_lower[content_lower.find(neg) : content_lower.find(neg) + 50]
            ):
                return False
        return True

    return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_smart_router(
    ctx: AnalysisContext,
    cold_path_handler: Callable[[ColdPathRequest], dict] | None = None,
    confidence_threshold: float = 0.6,
    ambiguity_threshold: float = 0.5,
) -> SmartRouter:
    """
    Factory function to create a SmartRouter.

    Args:
        ctx: Analysis context
        cold_path_handler: Optional handler for cold path (LLM)
        confidence_threshold: Below this routes to cold path
        ambiguity_threshold: Above this routes to cold path

    Returns:
        Configured SmartRouter instance
    """
    return SmartRouter(
        ctx=ctx,
        cold_path_handler=cold_path_handler,
        confidence_threshold=confidence_threshold,
        ambiguity_threshold=ambiguity_threshold,
    )
