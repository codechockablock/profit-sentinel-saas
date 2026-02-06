"""Cross-Report Pattern Detection engine.

Compares two analysis results and identifies:
    - New issues that appeared since last analysis
    - Resolved issues that were present before but are gone now
    - Worsening issues (higher count or impact)
    - Improving issues (lower count or impact)
    - Impact trajectory (total dollar impact trend)
    - Leak type trends (which primitives are getting better/worse)

Usage:
    from .cross_report import compare_analyses, CrossReportResult

    result = compare_analyses(current_result, previous_result)
    print(result.summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sentinel.cross_report")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LeakTrend:
    """Trend for a single leak type between two analyses."""

    leak_key: str
    current_count: int
    previous_count: int
    count_delta: int  # positive = worsening
    current_impact: float
    previous_impact: float
    impact_delta: float  # positive = worsening
    status: str  # "new", "resolved", "worsening", "improving", "stable"

    @property
    def severity(self) -> str:
        """Map status to severity level."""
        if self.status == "new":
            return "critical"
        if self.status == "worsening":
            return "warning"
        if self.status == "resolved":
            return "success"
        if self.status == "improving":
            return "info"
        return "neutral"


@dataclass
class CrossReportResult:
    """Full comparison between two analysis results."""

    # Overall metrics
    current_total_issues: int = 0
    previous_total_issues: int = 0
    issues_delta: int = 0

    current_total_impact_low: float = 0
    current_total_impact_high: float = 0
    previous_total_impact_low: float = 0
    previous_total_impact_high: float = 0
    impact_delta_low: float = 0
    impact_delta_high: float = 0

    # Dataset comparison
    current_rows: int = 0
    previous_rows: int = 0

    # Per-leak trends
    leak_trends: list[LeakTrend] = field(default_factory=list)

    # Categorized trends
    new_leaks: list[LeakTrend] = field(default_factory=list)
    resolved_leaks: list[LeakTrend] = field(default_factory=list)
    worsening_leaks: list[LeakTrend] = field(default_factory=list)
    improving_leaks: list[LeakTrend] = field(default_factory=list)
    stable_leaks: list[LeakTrend] = field(default_factory=list)

    # Metadata
    current_analysis_id: str | None = None
    previous_analysis_id: str | None = None
    current_label: str | None = None
    previous_label: str | None = None
    current_created_at: str | None = None
    previous_created_at: str | None = None

    @property
    def overall_trend(self) -> str:
        """Overall trajectory: 'improving', 'worsening', or 'stable'."""
        if self.issues_delta < -2 or self.impact_delta_high < -500:
            return "improving"
        if self.issues_delta > 2 or self.impact_delta_high > 500:
            return "worsening"
        return "stable"

    @property
    def summary(self) -> dict:
        """Generate a summary dict suitable for API response."""
        return {
            "overall_trend": self.overall_trend,
            "issues_delta": self.issues_delta,
            "current_total_issues": self.current_total_issues,
            "previous_total_issues": self.previous_total_issues,
            "impact_delta_low": round(self.impact_delta_low, 2),
            "impact_delta_high": round(self.impact_delta_high, 2),
            "new_leak_count": len(self.new_leaks),
            "resolved_leak_count": len(self.resolved_leaks),
            "worsening_leak_count": len(self.worsening_leaks),
            "improving_leak_count": len(self.improving_leaks),
            "current_rows": self.current_rows,
            "previous_rows": self.previous_rows,
        }

    def to_dict(self) -> dict:
        """Full serializable representation."""
        return {
            "summary": self.summary,
            "leak_trends": [
                {
                    "leak_key": t.leak_key,
                    "current_count": t.current_count,
                    "previous_count": t.previous_count,
                    "count_delta": t.count_delta,
                    "current_impact": round(t.current_impact, 2),
                    "previous_impact": round(t.previous_impact, 2),
                    "impact_delta": round(t.impact_delta, 2),
                    "status": t.status,
                    "severity": t.severity,
                }
                for t in self.leak_trends
            ],
            "new_leaks": [t.leak_key for t in self.new_leaks],
            "resolved_leaks": [t.leak_key for t in self.resolved_leaks],
            "worsening_leaks": [t.leak_key for t in self.worsening_leaks],
            "improving_leaks": [t.leak_key for t in self.improving_leaks],
            "metadata": {
                "current_analysis_id": self.current_analysis_id,
                "previous_analysis_id": self.previous_analysis_id,
                "current_label": self.current_label,
                "previous_label": self.previous_label,
                "current_created_at": self.current_created_at,
                "previous_created_at": self.previous_created_at,
            },
        }


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def _extract_detection_counts(result: dict) -> dict[str, int]:
    """Extract leak type → count from a full analysis result."""
    # Try the detection_counts field first (stored in analysis_synopses)
    if "detection_counts" in result and isinstance(result["detection_counts"], dict):
        return {k: int(v) for k, v in result["detection_counts"].items() if v}

    # Fall back to the leaks structure in full_result
    counts = {}
    leaks = result.get("full_result", result).get("leaks", {})
    for key, data in leaks.items():
        if isinstance(data, dict):
            counts[key] = data.get("count", 0)
        elif isinstance(data, (int, float)):
            counts[key] = int(data)
    return counts


def _extract_impact_breakdown(result: dict) -> dict[str, float]:
    """Extract leak type → dollar impact from full analysis result."""
    breakdown = {}
    full = result.get("full_result", result)
    summary = full.get("summary", {})
    estimated_impact = summary.get("estimated_impact", {})

    # Try breakdown field
    if "breakdown" in estimated_impact:
        return {k: float(v) for k, v in estimated_impact["breakdown"].items() if v}

    # Estimate from leak counts * average
    total_impact = estimated_impact.get("high", 0) or estimated_impact.get("annual_estimate", 0)
    counts = _extract_detection_counts(result)
    total_count = sum(counts.values()) or 1

    for key, count in counts.items():
        breakdown[key] = (count / total_count) * total_impact

    return breakdown


def compare_analyses(
    current: dict,
    previous: dict,
) -> CrossReportResult:
    """Compare two analysis records and produce a CrossReportResult.

    Args:
        current: The more recent analysis record (from analysis store).
        previous: The older analysis record.

    Returns:
        CrossReportResult with categorized trends and overall trajectory.
    """
    result = CrossReportResult()

    # Metadata
    result.current_analysis_id = current.get("id")
    result.previous_analysis_id = previous.get("id")
    result.current_label = current.get("analysis_label")
    result.previous_label = previous.get("analysis_label")
    result.current_created_at = current.get("created_at")
    result.previous_created_at = previous.get("created_at")

    # Row counts
    result.current_rows = current.get("file_row_count", 0)
    result.previous_rows = previous.get("file_row_count", 0)

    # Impact
    result.current_total_impact_low = float(current.get("total_impact_estimate_low", 0) or 0)
    result.current_total_impact_high = float(current.get("total_impact_estimate_high", 0) or 0)
    result.previous_total_impact_low = float(previous.get("total_impact_estimate_low", 0) or 0)
    result.previous_total_impact_high = float(previous.get("total_impact_estimate_high", 0) or 0)
    result.impact_delta_low = result.current_total_impact_low - result.previous_total_impact_low
    result.impact_delta_high = result.current_total_impact_high - result.previous_total_impact_high

    # Detection counts
    current_counts = _extract_detection_counts(current)
    previous_counts = _extract_detection_counts(previous)

    result.current_total_issues = sum(current_counts.values())
    result.previous_total_issues = sum(previous_counts.values())
    result.issues_delta = result.current_total_issues - result.previous_total_issues

    # Impact breakdown
    current_impacts = _extract_impact_breakdown(current)
    previous_impacts = _extract_impact_breakdown(previous)

    # All leak keys across both analyses
    all_keys = sorted(set(current_counts.keys()) | set(previous_counts.keys()))

    for key in all_keys:
        cur_count = current_counts.get(key, 0)
        prev_count = previous_counts.get(key, 0)
        cur_impact = current_impacts.get(key, 0)
        prev_impact = previous_impacts.get(key, 0)

        count_delta = cur_count - prev_count
        impact_delta = cur_impact - prev_impact

        # Determine status
        if prev_count == 0 and cur_count > 0:
            status = "new"
        elif cur_count == 0 and prev_count > 0:
            status = "resolved"
        elif count_delta > 0 or impact_delta > 100:
            status = "worsening"
        elif count_delta < 0 or impact_delta < -100:
            status = "improving"
        else:
            status = "stable"

        trend = LeakTrend(
            leak_key=key,
            current_count=cur_count,
            previous_count=prev_count,
            count_delta=count_delta,
            current_impact=cur_impact,
            previous_impact=prev_impact,
            impact_delta=impact_delta,
            status=status,
        )

        result.leak_trends.append(trend)

        if status == "new":
            result.new_leaks.append(trend)
        elif status == "resolved":
            result.resolved_leaks.append(trend)
        elif status == "worsening":
            result.worsening_leaks.append(trend)
        elif status == "improving":
            result.improving_leaks.append(trend)
        else:
            result.stable_leaks.append(trend)

    # Sort trends: critical first, then by impact delta magnitude
    result.leak_trends.sort(key=lambda t: abs(t.impact_delta), reverse=True)
    result.worsening_leaks.sort(key=lambda t: t.impact_delta, reverse=True)
    result.improving_leaks.sort(key=lambda t: t.impact_delta)

    logger.info(
        "Cross-report comparison: %d issues → %d issues (%+d), "
        "new=%d, resolved=%d, worsening=%d, improving=%d",
        result.previous_total_issues,
        result.current_total_issues,
        result.issues_delta,
        len(result.new_leaks),
        len(result.resolved_leaks),
        len(result.worsening_leaks),
        len(result.improving_leaks),
    )

    return result
