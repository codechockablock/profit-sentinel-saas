"""Automated Vendor Performance Scoring.

Scores vendors across four dimensions:
    1. Quality — damaged goods rate, return frequency
    2. Delivery — short-ship incidents, on-order fulfillment
    3. Pricing — margin consistency, cost competitiveness
    4. Compliance — rebate tier achievement, co-op participation

Each dimension produces a 0–100 score. The overall vendor score is a
weighted average: Quality 30%, Delivery 25%, Pricing 25%, Compliance 20%.

Every score includes dollar-quantified impact — "Martin's Supply scores 62/100.
Quality issues cost you $8,400 this year."
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from .coop_models import VendorRebateStatus
from .models import Digest, Issue, IssueType, Sku

logger = logging.getLogger("sentinel.vendor_scoring")


# ---------------------------------------------------------------------------
# Score data classes
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    """Score for a single performance dimension (0-100)."""

    dimension: str
    score: float
    weight: float
    findings: list[str] = field(default_factory=list)
    dollar_impact: float = 0.0

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    @property
    def grade(self) -> str:
        if self.score >= 90:
            return "A"
        if self.score >= 80:
            return "B"
        if self.score >= 70:
            return "C"
        if self.score >= 60:
            return "D"
        return "F"

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 1),
            "weight": self.weight,
            "grade": self.grade,
            "findings": self.findings,
            "dollar_impact": round(self.dollar_impact, 2),
        }


@dataclass
class VendorScorecard:
    """Complete vendor performance scorecard."""

    vendor_id: str
    vendor_name: str
    overall_score: float
    overall_grade: str
    dimensions: list[DimensionScore]
    total_skus: int
    total_dollar_exposure: float
    total_quality_cost: float
    recommendations: list[str]
    trend: str = "stable"  # improving, stable, worsening

    @property
    def risk_level(self) -> str:
        if self.overall_score >= 80:
            return "low"
        if self.overall_score >= 60:
            return "medium"
        if self.overall_score >= 40:
            return "high"
        return "critical"

    def to_dict(self) -> dict:
        return {
            "vendor_id": self.vendor_id,
            "vendor_name": self.vendor_name,
            "overall_score": round(self.overall_score, 1),
            "overall_grade": self.overall_grade,
            "risk_level": self.risk_level,
            "trend": self.trend,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "total_skus": self.total_skus,
            "total_dollar_exposure": round(self.total_dollar_exposure, 2),
            "total_quality_cost": round(self.total_quality_cost, 2),
            "recommendations": self.recommendations,
        }


@dataclass
class VendorScoringReport:
    """Complete vendor scoring report for a store."""

    store_id: str
    scorecards: list[VendorScorecard]
    total_vendors_scored: int
    average_score: float
    high_risk_vendors: int
    total_quality_cost: float
    top_recommendation: str

    def to_dict(self) -> dict:
        return {
            "store_id": self.store_id,
            "scorecards": [s.to_dict() for s in self.scorecards],
            "total_vendors_scored": self.total_vendors_scored,
            "average_score": round(self.average_score, 1),
            "high_risk_vendors": self.high_risk_vendors,
            "total_quality_cost": round(self.total_quality_cost, 2),
            "top_recommendation": self.top_recommendation,
        }


# ---------------------------------------------------------------------------
# Vendor catalog (shared with vendor_assist — stub for production)
# ---------------------------------------------------------------------------

_VENDOR_CATALOG: dict[str, dict] = {
    "DMG": {"name": "Martin's Supply Co.", "contact": "Dave Martin"},
    "ELC": {"name": "National Electrical Distributors", "contact": "Sarah Chen"},
    "PNT": {"name": "ColorMax Paint Supply", "contact": "Jim Rodriguez"},
    "PLB": {"name": "ProPlumb Wholesale", "contact": "Linda Park"},
    "SEA": {"name": "SeaCoast Hardware", "contact": "Tom Burke"},
    "FLR": {"name": "FloorCraft Distribution", "contact": "Maria Santos"},
    "HRD": {"name": "HardLine Tools & Fasteners", "contact": "Keith Wong"},
    "TLS": {"name": "ToolSource National", "contact": "Gary Fletcher"},
    "SSN": {"name": "Seasonal Products Inc.", "contact": "Amy Brooks"},
    "NRM": {"name": "Various / Normal Stock", "contact": "N/A"},
}

# Dimension weights
QUALITY_WEIGHT = 0.30
DELIVERY_WEIGHT = 0.25
PRICING_WEIGHT = 0.25
COMPLIANCE_WEIGHT = 0.20

# Thresholds
DAMAGED_RATE_THRESHOLD = 0.05  # >5% damaged = poor quality
MARGIN_BENCHMARK = 0.35  # 35% target margin
SHORT_SHIP_THRESHOLD = 0.10  # >10% on-order with damage = delivery concern


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------


def _vendor_prefix(sku_id: str) -> str:
    """Extract vendor prefix from SKU ID."""
    return sku_id.split("-")[0] if "-" in sku_id else sku_id[:3]


def _vendor_name(prefix: str) -> str:
    """Look up vendor name from prefix."""
    info = _VENDOR_CATALOG.get(prefix)
    return info["name"] if info else f"Unknown ({prefix})"


class VendorPerformanceScorer:
    """Score vendors based on pipeline issue data.

    Usage:
        scorer = VendorPerformanceScorer()
        report = scorer.score_from_digest(digest, store_id="store-7")
        for card in report.scorecards:
            print(f"{card.vendor_name}: {card.overall_score}/100 ({card.overall_grade})")
    """

    def __init__(
        self,
        rebate_statuses: list[VendorRebateStatus] | None = None,
    ):
        self.rebate_statuses = rebate_statuses or []

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------

    def score_from_digest(
        self,
        digest: Digest,
        store_id: str | None = None,
    ) -> VendorScoringReport:
        """Score all vendors from a pipeline digest.

        Groups issues and SKUs by vendor prefix, scores each vendor
        across quality, delivery, pricing, and compliance dimensions.

        Args:
            digest: Pipeline digest containing issues and SKUs.
            store_id: Optional store filter.

        Returns:
            VendorScoringReport with scorecards for each vendor.
        """
        issues = digest.issues
        if store_id:
            issues = [i for i in issues if i.store_id == store_id]

        # Group SKUs by vendor
        vendor_skus: dict[str, list[Sku]] = defaultdict(list)
        vendor_issues: dict[str, list[Issue]] = defaultdict(list)

        for issue in issues:
            for sku in issue.skus:
                prefix = _vendor_prefix(sku.sku_id)
                vendor_skus[prefix].append(sku)
            # Associate issue with vendor via first SKU
            if issue.skus:
                prefix = _vendor_prefix(issue.skus[0].sku_id)
                vendor_issues[prefix].append(issue)

        # Score each vendor
        scorecards: list[VendorScorecard] = []
        for prefix in sorted(vendor_skus.keys()):
            skus = vendor_skus[prefix]
            issues_for_vendor = vendor_issues.get(prefix, [])
            rebate_status = self._find_rebate_status(prefix)

            card = self._score_vendor(prefix, skus, issues_for_vendor, rebate_status)
            scorecards.append(card)

        # Sort by score ascending (worst vendors first for action prioritization)
        scorecards.sort(key=lambda c: c.overall_score)

        # Build report
        avg_score = (
            sum(c.overall_score for c in scorecards) / len(scorecards)
            if scorecards
            else 0
        )
        high_risk = sum(1 for c in scorecards if c.risk_level in ("high", "critical"))
        total_quality = sum(c.total_quality_cost for c in scorecards)

        top_rec = "All vendors performing well." if not scorecards else ""
        if scorecards and scorecards[0].recommendations:
            worst = scorecards[0]
            top_rec = (
                f"Priority: {worst.vendor_name} scores {worst.overall_score:.0f}/100. "
                f"{worst.recommendations[0]}"
            )

        return VendorScoringReport(
            store_id=store_id or "all",
            scorecards=scorecards,
            total_vendors_scored=len(scorecards),
            average_score=avg_score,
            high_risk_vendors=high_risk,
            total_quality_cost=total_quality,
            top_recommendation=top_rec,
        )

    # -----------------------------------------------------------------
    # Per-vendor scoring
    # -----------------------------------------------------------------

    def _score_vendor(
        self,
        prefix: str,
        skus: list[Sku],
        issues: list[Issue],
        rebate_status: VendorRebateStatus | None,
    ) -> VendorScorecard:
        """Score a single vendor across all dimensions."""
        vendor_name = _vendor_name(prefix)

        quality = self._score_quality(skus, issues)
        delivery = self._score_delivery(skus, issues)
        pricing = self._score_pricing(skus, issues)
        compliance = self._score_compliance(prefix, rebate_status)

        dimensions = [quality, delivery, pricing, compliance]
        overall = sum(d.weighted_score for d in dimensions)
        grade = self._overall_grade(overall)

        # Dollar exposure = total cost of all SKUs from this vendor
        total_exposure = sum(abs(s.qty_on_hand) * s.unit_cost for s in skus)
        quality_cost = quality.dollar_impact + delivery.dollar_impact

        # Generate recommendations
        recommendations = self._build_recommendations(vendor_name, dimensions, issues)

        return VendorScorecard(
            vendor_id=prefix,
            vendor_name=vendor_name,
            overall_score=overall,
            overall_grade=grade,
            dimensions=dimensions,
            total_skus=len(skus),
            total_dollar_exposure=total_exposure,
            total_quality_cost=quality_cost,
            recommendations=recommendations,
        )

    # -----------------------------------------------------------------
    # Dimension scorers
    # -----------------------------------------------------------------

    def _score_quality(self, skus: list[Sku], issues: list[Issue]) -> DimensionScore:
        """Score vendor quality: damaged rate, quality-related issues."""
        total = len(skus)
        if total == 0:
            return DimensionScore("Quality", 100.0, QUALITY_WEIGHT)

        damaged = sum(1 for s in skus if s.is_damaged)
        damaged_rate = damaged / total
        damaged_value = sum(
            abs(s.qty_on_hand) * s.unit_cost for s in skus if s.is_damaged
        )

        # Count quality-related issues
        quality_issues = [
            i
            for i in issues
            if i.issue_type
            in (
                IssueType.VENDOR_SHORT_SHIP,
                IssueType.SHRINKAGE_PATTERN,
            )
        ]
        quality_issue_impact = sum(i.dollar_impact for i in quality_issues)

        # Score: start at 100, deduct for problems
        score = 100.0

        # Deduct for damaged rate (0-40 points)
        if damaged_rate > 0:
            deduction = min(40.0, (damaged_rate / DAMAGED_RATE_THRESHOLD) * 20)
            score -= deduction

        # Deduct for quality issues (0-30 points)
        if quality_issues:
            deduction = min(30.0, len(quality_issues) * 10)
            score -= deduction

        # Deduct for high damaged value (0-30 points)
        if damaged_value > 5000:
            deduction = min(30.0, (damaged_value / 10000) * 15)
            score -= deduction

        score = max(0.0, score)

        findings = []
        if damaged > 0:
            findings.append(
                f"{damaged} of {total} SKUs received damaged "
                f"({damaged_rate:.1%} rate, ${damaged_value:,.0f} exposure)"
            )
        if quality_issues:
            findings.append(
                f"{len(quality_issues)} quality-related issue(s) "
                f"totaling ${quality_issue_impact:,.0f}"
            )
        if not findings:
            findings.append("No quality issues detected")

        return DimensionScore(
            "Quality",
            score,
            QUALITY_WEIGHT,
            findings=findings,
            dollar_impact=damaged_value + quality_issue_impact,
        )

    def _score_delivery(self, skus: list[Sku], issues: list[Issue]) -> DimensionScore:
        """Score delivery: short-ship rate, fulfillment reliability."""
        total = len(skus)
        if total == 0:
            return DimensionScore("Delivery", 100.0, DELIVERY_WEIGHT)

        # On-order items with damage suggest delivery problems
        on_order_damaged = sum(1 for s in skus if s.on_order_qty > 0 and s.is_damaged)
        on_order_total = sum(1 for s in skus if s.on_order_qty > 0)
        on_order_value = sum(
            s.on_order_qty * s.unit_cost for s in skus if s.on_order_qty > 0
        )

        # Short-ship issues
        short_ship_issues = [
            i for i in issues if i.issue_type == IssueType.VENDOR_SHORT_SHIP
        ]
        short_ship_impact = sum(i.dollar_impact for i in short_ship_issues)

        score = 100.0

        # Deduct for short ships (0-40 points)
        if short_ship_issues:
            deduction = min(40.0, len(short_ship_issues) * 15)
            score -= deduction

        # Deduct for on-order with damage pattern (0-30 points)
        if on_order_total > 0 and on_order_damaged > 0:
            risk_rate = on_order_damaged / on_order_total
            deduction = min(30.0, risk_rate * 60)
            score -= deduction

        # Deduct for high on-order exposure (0-30 points)
        if on_order_value > 10000:
            deduction = min(30.0, (on_order_value / 50000) * 15)
            score -= deduction

        score = max(0.0, score)

        findings = []
        if short_ship_issues:
            findings.append(
                f"{len(short_ship_issues)} short-ship incident(s) "
                f"totaling ${short_ship_impact:,.0f}"
            )
        if on_order_total > 0:
            findings.append(
                f"{on_order_total} SKU(s) on order " f"(${on_order_value:,.0f} pending)"
            )
        if on_order_damaged > 0:
            findings.append(
                f"{on_order_damaged} on-order SKU(s) have prior damage — "
                f"fulfillment risk elevated"
            )
        if not findings:
            findings.append("No delivery issues detected")

        return DimensionScore(
            "Delivery",
            score,
            DELIVERY_WEIGHT,
            findings=findings,
            dollar_impact=short_ship_impact,
        )

    def _score_pricing(self, skus: list[Sku], issues: list[Issue]) -> DimensionScore:
        """Score pricing: margin consistency, cost competitiveness."""
        total = len(skus)
        if total == 0:
            return DimensionScore("Pricing", 100.0, PRICING_WEIGHT)

        avg_margin = sum(s.margin_pct for s in skus) / total
        below_benchmark = sum(1 for s in skus if s.margin_pct < MARGIN_BENCHMARK)
        below_rate = below_benchmark / total

        # Pricing-related issues
        pricing_issues = [
            i
            for i in issues
            if i.issue_type
            in (
                IssueType.MARGIN_EROSION,
                IssueType.PURCHASING_LEAKAGE,
                IssueType.PRICE_DISCREPANCY,
            )
        ]
        pricing_impact = sum(i.dollar_impact for i in pricing_issues)

        score = 100.0

        # Deduct for below-benchmark margins (0-35 points)
        if below_rate > 0:
            deduction = min(35.0, below_rate * 50)
            score -= deduction

        # Deduct for low average margin (0-25 points)
        margin_gap = max(0, MARGIN_BENCHMARK - avg_margin)
        if margin_gap > 0:
            deduction = min(25.0, (margin_gap / MARGIN_BENCHMARK) * 40)
            score -= deduction

        # Deduct for pricing issues (0-40 points)
        if pricing_issues:
            deduction = min(40.0, len(pricing_issues) * 12)
            score -= deduction

        score = max(0.0, score)

        findings = []
        if below_benchmark > 0:
            findings.append(
                f"{below_benchmark} of {total} SKUs below {MARGIN_BENCHMARK:.0%} "
                f"margin benchmark ({below_rate:.0%})"
            )
        findings.append(
            f"Average margin: {avg_margin:.1%} " f"(benchmark: {MARGIN_BENCHMARK:.0%})"
        )
        if pricing_issues:
            findings.append(
                f"{len(pricing_issues)} pricing issue(s) "
                f"totaling ${pricing_impact:,.0f}"
            )

        return DimensionScore(
            "Pricing",
            score,
            PRICING_WEIGHT,
            findings=findings,
            dollar_impact=pricing_impact,
        )

    def _score_compliance(
        self,
        prefix: str,
        rebate_status: VendorRebateStatus | None,
    ) -> DimensionScore:
        """Score compliance: rebate achievement, co-op participation."""
        if rebate_status is None:
            # No rebate program — neutral score
            return DimensionScore(
                "Compliance",
                75.0,
                COMPLIANCE_WEIGHT,
                findings=["No rebate program tracked for this vendor"],
            )

        score = 100.0
        findings = []

        # On track for next tier?
        if rebate_status.on_track:
            findings.append(
                f"On track for {rebate_status.next_tier.tier_name} tier"
                if rebate_status.next_tier
                else "At top tier — maximizing rebates"
            )
        else:
            # Deduct for at-risk status (0-40 points)
            if rebate_status.next_tier:
                shortfall_pct = (
                    rebate_status.shortfall / rebate_status.next_tier.threshold
                    if rebate_status.next_tier.threshold > 0
                    else 0
                )
                deduction = min(40.0, shortfall_pct * 100)
                score -= deduction
                findings.append(
                    f"At risk: ${rebate_status.shortfall:,.0f} short of "
                    f"{rebate_status.next_tier.tier_name} tier "
                    f"({rebate_status.days_remaining} days remaining)"
                )

        # Current rebate value
        if rebate_status.current_rebate_value > 0:
            findings.append(
                f"YTD rebate earned: ${rebate_status.current_rebate_value:,.0f}"
            )

        # Incremental opportunity
        dollar_impact = 0.0
        if rebate_status.incremental_value > 0 and not rebate_status.on_track:
            dollar_impact = rebate_status.incremental_value
            findings.append(
                f"Potential missed rebate: ${rebate_status.incremental_value:,.0f}"
            )

        return DimensionScore(
            "Compliance",
            max(0.0, score),
            COMPLIANCE_WEIGHT,
            findings=findings,
            dollar_impact=dollar_impact,
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _find_rebate_status(self, prefix: str) -> VendorRebateStatus | None:
        """Find rebate status for a vendor prefix."""
        for status in self.rebate_statuses:
            if status.program.vendor_id == prefix:
                return status
        return None

    @staticmethod
    def _overall_grade(score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    @staticmethod
    def _build_recommendations(
        vendor_name: str,
        dimensions: list[DimensionScore],
        issues: list[Issue],
    ) -> list[str]:
        """Generate actionable recommendations based on scores."""
        recs: list[str] = []

        # Find worst dimension
        worst = min(dimensions, key=lambda d: d.score)

        if worst.dimension == "Quality" and worst.score < 70:
            recs.append(
                f"Schedule quality review with {vendor_name}. "
                f"Request credit for ${worst.dollar_impact:,.0f} in damaged goods."
            )

        if worst.dimension == "Delivery" and worst.score < 70:
            recs.append(
                f"Request delivery improvement plan from {vendor_name}. "
                f"Consider adding backup vendor for critical SKUs."
            )

        if worst.dimension == "Pricing" and worst.score < 70:
            recs.append(
                f"Renegotiate pricing terms with {vendor_name}. "
                f"Volume discount or cost reduction needed on "
                f"${worst.dollar_impact:,.0f} in below-benchmark items."
            )

        if worst.dimension == "Compliance" and worst.score < 70:
            recs.append(
                f"Accelerate purchases from {vendor_name} to hit next "
                f"rebate tier. Potential missed value: "
                f"${worst.dollar_impact:,.0f}."
            )

        # General recommendations for multi-dimension weakness
        low_dims = [d for d in dimensions if d.score < 60]
        if len(low_dims) >= 2:
            recs.append(
                f"Evaluate alternative vendors for {vendor_name}'s categories. "
                f"Multiple performance dimensions scoring below 60."
            )

        if not recs:
            recs.append(f"{vendor_name} is performing well across all dimensions.")

        return recs


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def score_vendors(
    digest: Digest,
    store_id: str | None = None,
    rebate_statuses: list[VendorRebateStatus] | None = None,
) -> VendorScoringReport:
    """Score all vendors from a digest. Convenience wrapper.

    Args:
        digest: Pipeline digest.
        store_id: Optional store filter.
        rebate_statuses: Optional rebate status list for compliance scoring.

    Returns:
        VendorScoringReport with scorecards for each vendor.
    """
    scorer = VendorPerformanceScorer(rebate_statuses=rebate_statuses)
    return scorer.score_from_digest(digest, store_id=store_id)
