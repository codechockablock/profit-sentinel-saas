"""Template-based natural language rendering.

No LLM API calls — just structured templates that produce readable,
actionable text. The LLM integration can come later for interpretation
and conversation. Right now, this transforms structured pipeline output
into text a real operations executive would find useful at 6 AM.
"""

from __future__ import annotations

from .coop_models import (
    CategoryMixAnalysis,
    CoopAlert,
    CoopAlertType,
    CoopIntelligenceReport,
    InventoryHealthReport,
    VendorRebateStatus,
)
from .models import (
    CallPrep,
    Digest,
    Issue,
    IssueType,
    Sku,
    Task,
    TrendDirection,
)


def format_dollars(amount: float) -> str:
    """Format a dollar amount with commas and no cents."""
    if amount < 0:
        return f"-${abs(amount):,.0f}"
    return f"${amount:,.0f}"


def format_qty(qty: float) -> str:
    """Format quantity, showing negatives clearly."""
    if qty < 0:
        return f"{int(abs(qty))} units short"
    return f"{int(qty)} units"


# ---------------------------------------------------------------------------
# Issue → Natural Language
# ---------------------------------------------------------------------------


def render_issue_headline(issue: Issue, index: int) -> str:
    """One-line headline for an issue in the digest.

    Example: "1. Store 7 — Negative Inventory ($1,104)"
    """
    return (
        f"{index}. {_store_display(issue.store_id)} "
        f"\u2014 {issue.issue_type.display_name} "
        f"({format_dollars(issue.dollar_impact)})"
    )


def render_issue_detail(issue: Issue) -> str:
    """Multi-line detail block for an issue.

    Returns 3-4 lines of actionable context including SKU info,
    root cause attribution, trend direction, and recommended next step.
    """
    lines: list[str] = []

    # SKU detail line
    if issue.sku_count == 1:
        sku = issue.skus[0]
        lines.append(_render_single_sku(issue.issue_type, sku))
    else:
        lines.append(_render_multi_sku(issue))

    # Root cause attribution line (if available)
    if issue.root_cause is not None:
        conf = issue.root_cause_confidence or 0.0
        lines.append(
            f"Root cause: {issue.root_cause.display_name} ({conf:.0%} confidence)"
        )

    # Context / trend line
    lines.append(_render_trend_context(issue))

    # Action line
    lines.append(_render_action(issue))

    return "\n".join(f"   {line}" for line in lines)


def _store_display(store_id: str) -> str:
    """Convert store-7 → Store 7."""
    if store_id.startswith("store-"):
        return f"Store {store_id[6:]}"
    return store_id


def _render_single_sku(issue_type: IssueType, sku: Sku) -> str:
    """Detail line for a single-SKU issue."""
    match issue_type:
        case IssueType.NEGATIVE_INVENTORY:
            return (
                f"SKU {sku.sku_id}: {int(abs(sku.qty_on_hand))} units short "
                f"@ {format_dollars(sku.unit_cost)}"
            )
        case IssueType.DEAD_STOCK:
            return (
                f"SKU {sku.sku_id}: {int(sku.qty_on_hand)} units, "
                f"zero sales for {int(sku.days_since_receipt)}+ days"
            )
        case IssueType.MARGIN_EROSION:
            return (
                f"SKU {sku.sku_id}: {sku.margin_display} margin "
                f"vs 35% benchmark ({int(sku.qty_on_hand)} units @ "
                f"{format_dollars(sku.unit_cost)})"
            )
        case IssueType.VENDOR_SHORT_SHIP:
            parts = f"SKU {sku.sku_id}: damaged goods"
            if sku.on_order_qty > 0:
                parts += f", {int(sku.on_order_qty)} on order"
            return parts
        case IssueType.PATRONAGE_MISS:
            months = sku.days_since_receipt / 30
            return (
                f"SKU {sku.sku_id}: {int(sku.qty_on_hand)} seasonal units, "
                f"held {months:.0f} months"
            )
        case IssueType.PURCHASING_LEAKAGE:
            return (
                f"SKU {sku.sku_id}: {format_dollars(sku.unit_cost)}/unit at "
                f"{sku.margin_display} margin"
            )
        case IssueType.SHRINKAGE_PATTERN:
            value = abs(sku.qty_on_hand) * sku.unit_cost
            return (
                f"SKU {sku.sku_id}: {format_dollars(value)} inventory, "
                f"{sku.margin_display} margin, "
                f"{int(sku.sales_last_30d)} sales/month"
            )
        case IssueType.ZERO_COST_ANOMALY:
            return (
                f"SKU {sku.sku_id}: $0 cost, "
                f"{format_dollars(sku.retail_price)} retail, "
                f"{int(sku.qty_on_hand)} on hand"
            )
        case IssueType.PRICE_DISCREPANCY:
            loss = sku.unit_cost - sku.retail_price
            return (
                f"SKU {sku.sku_id}: cost {format_dollars(sku.unit_cost)} > "
                f"retail {format_dollars(sku.retail_price)} "
                f"({format_dollars(loss)}/unit loss)"
            )
        case IssueType.OVERSTOCK:
            months = (
                sku.qty_on_hand / sku.sales_last_30d if sku.sales_last_30d > 0 else 24
            )
            return (
                f"SKU {sku.sku_id}: {int(sku.qty_on_hand)} units, "
                f"{months:.0f} months supply"
            )
        case _:
            return f"SKU {sku.sku_id}: {format_qty(sku.qty_on_hand)}"


def _render_multi_sku(issue: Issue) -> str:
    """Detail line for a multi-SKU issue."""
    match issue.issue_type:
        case IssueType.DEAD_STOCK:
            avg_days = sum(s.days_since_receipt for s in issue.skus) / len(issue.skus)
            return f"{issue.sku_count} SKUs with zero sales for {int(avg_days)}+ days"
        case IssueType.MARGIN_EROSION:
            avg_margin = sum(s.margin_pct for s in issue.skus) / len(issue.skus)
            return (
                f"{issue.sku_count} SKUs averaging "
                f"{avg_margin * 100:.0f}% margin vs 35% benchmark"
            )
        case IssueType.NEGATIVE_INVENTORY:
            total_short = sum(
                abs(s.qty_on_hand) for s in issue.skus if s.qty_on_hand < 0
            )
            return f"{issue.sku_count} SKUs, {int(total_short)} total units short"
        case IssueType.SHRINKAGE_PATTERN:
            total_value = sum(abs(s.qty_on_hand) * s.unit_cost for s in issue.skus)
            return (
                f"{issue.sku_count} SKUs with {format_dollars(total_value)} "
                f"inventory at risk of shrinkage"
            )
        case IssueType.ZERO_COST_ANOMALY:
            selling = sum(1 for s in issue.skus if s.sales_last_30d > 0)
            return (
                f"{issue.sku_count} SKUs with $0 cost data ({selling} actively selling)"
            )
        case IssueType.PRICE_DISCREPANCY:
            return f"{issue.sku_count} SKUs priced below cost"
        case IssueType.OVERSTOCK:
            total_excess = sum(max(0, s.qty_on_hand - 30) for s in issue.skus)
            return (
                f"{issue.sku_count} SKUs, "
                f"{int(total_excess)} excess units above optimal stock"
            )
        case _:
            sku_ids = ", ".join(s.sku_id for s in issue.skus[:3])
            if issue.sku_count > 3:
                sku_ids += f", +{issue.sku_count - 3} more"
            return f"SKUs: {sku_ids}"


def _render_trend_context(issue: Issue) -> str:
    """Contextual line about trend and timing."""
    trend = issue.trend_direction

    if trend == TrendDirection.WORSENING:
        return f"This issue is {trend.description} and needs attention soon."
    elif trend == TrendDirection.STABLE:
        return f"Trend is {trend.description}. Monitor or act proactively."
    else:
        return f"Trend is {trend.description}. Continue current approach."


def _render_action(issue: Issue) -> str:
    """Suggested action line."""
    match issue.issue_type:
        case IssueType.NEGATIVE_INVENTORY:
            return "[Investigate] [Delegate to Store Manager]"
        case IssueType.DEAD_STOCK:
            return "[Review SKU List] [Create Markdown Plan]"
        case IssueType.MARGIN_EROSION:
            return "[Review Pricing] [Contact Vendor]"
        case IssueType.VENDOR_SHORT_SHIP:
            return "[Prepare Vendor Call] [File Claim]"
        case IssueType.PATRONAGE_MISS:
            return "[Markdown] [Return to Vendor]"
        case IssueType.PURCHASING_LEAKAGE:
            return "[Review Purchase Orders] [Renegotiate]"
        case IssueType.RECEIVING_GAP:
            return "[Correct Data] [Audit Receiving Process]"
        case IssueType.SHRINKAGE_PATTERN:
            return "[Investigate] [Physical Count] [Review Security]"
        case IssueType.ZERO_COST_ANOMALY:
            return "[Update Cost Data] [Review Purchase Orders]"
        case IssueType.PRICE_DISCREPANCY:
            return "[Correct Retail Price] [Review Vendor Cost]"
        case IssueType.OVERSTOCK:
            return "[Reduce Orders] [Arrange Transfer] [Markdown]"
        case _:
            return "[Review]"


# ---------------------------------------------------------------------------
# Digest → Morning Briefing
# ---------------------------------------------------------------------------


def render_digest(digest: Digest) -> str:
    """Render a full morning digest as natural language text.

    This is what an executive sees at 6 AM on their phone.
    """
    lines: list[str] = []

    # Greeting
    count = digest.summary.total_issues
    if count == 0:
        lines.append("Good morning. No issues need your attention today. All clear.")
        return "\n".join(lines)

    lines.append(
        f"Good morning. {count} item{'s' if count != 1 else ''} "
        f"need{'s' if count == 1 else ''} your attention today."
    )
    lines.append(
        f"Total exposure: {digest.summary.total_dollar_display} "
        f"across {digest.summary.stores_affected} store"
        f"{'s' if digest.summary.stores_affected != 1 else ''}."
    )
    lines.append("")

    # Issues
    for i, issue in enumerate(digest.issues, 1):
        lines.append(render_issue_headline(issue, i))
        lines.append(render_issue_detail(issue))
        lines.append("")

    # Engine 3: cost of inaction summary
    # (only if counterfactual data was enriched into the digest)
    # This will be wired when the digest pipeline passes through Engine 3

    # Footer
    lines.append(
        f"Pipeline analyzed {digest.summary.records_processed} records "
        f"in {digest.pipeline_ms}ms."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task → Store Manager View
# ---------------------------------------------------------------------------


def render_task_for_manager(task: Task) -> str:
    """Render a delegated task as a store manager would see it.

    Different framing than executive view: specific action items,
    bin locations, count lists.
    """
    lines: list[str] = []

    lines.append(f"TASK: {task.title}")
    lines.append(f"Priority: {task.priority.value.upper()}")
    lines.append(f"Due: {task.deadline.strftime('%B %d, %Y')}")
    lines.append(f"Store: {_store_display(task.store_id)}")
    lines.append(f"Impact: {format_dollars(task.dollar_impact)}")
    lines.append("")

    lines.append(task.description)
    lines.append("")

    if task.action_items:
        lines.append("Action Items:")
        for item in task.action_items:
            lines.append(f"  [ ] {item}")
        lines.append("")

    if task.skus:
        lines.append("Affected SKUs:")
        for sku in task.skus:
            status = "DAMAGED" if sku.is_damaged else ""
            if sku.qty_on_hand < 0:
                status = f"{int(abs(sku.qty_on_hand))} SHORT"
            lines.append(
                f"  {sku.sku_id}: {format_qty(sku.qty_on_hand)} "
                f"@ {format_dollars(sku.unit_cost)}  {status}".rstrip()
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CallPrep → Vendor Call Brief
# ---------------------------------------------------------------------------


def render_call_prep(prep: CallPrep) -> str:
    """Render a vendor call preparation brief."""
    lines: list[str] = []

    lines.append(f"VENDOR CALL BRIEF: {prep.vendor_name}")
    lines.append(f"Store: {_store_display(prep.store_id)}")
    lines.append(f"Total at stake: {format_dollars(prep.total_dollar_impact)}")
    lines.append("")

    lines.append("Summary:")
    lines.append(f"  {prep.issue_summary}")
    lines.append("")

    if prep.talking_points:
        lines.append("Talking Points:")
        for point in prep.talking_points:
            lines.append(f"  \u2022 {point}")
        lines.append("")

    if prep.questions_to_ask:
        lines.append("Questions to Ask:")
        for q in prep.questions_to_ask:
            lines.append(f"  ? {q}")
        lines.append("")

    if prep.affected_skus:
        lines.append("Affected Items:")
        for sku in prep.affected_skus:
            lines.append(
                f"  {sku.sku_id}: {format_qty(sku.qty_on_hand)} "
                f"@ {format_dollars(sku.unit_cost)}"
            )
        lines.append("")

    if prep.historical_context:
        lines.append("History:")
        lines.append(f"  {prep.historical_context}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Co-op Intelligence → Natural Language
# ---------------------------------------------------------------------------


def render_coop_alert(alert: CoopAlert) -> str:
    """Render a single co-op intelligence alert.

    Example:
        !! Patronage Leakage: $18,400 in Paint to ColorMax ($2,044/yr)
           Shift to co-op warehouse (11.1% patronage). Annual gain: $2,044.
    """
    lines: list[str] = []

    icon = alert.alert_type.icon
    lines.append(f"{icon} {alert.title} ({format_dollars(alert.dollar_impact)}/yr)")
    lines.append(f"   {alert.detail}")
    lines.append(f"   >> {alert.recommendation}")

    return "\n".join(lines)


def render_coop_report(report: CoopIntelligenceReport) -> str:
    """Render a full co-op intelligence report.

    This is the co-op section of the morning briefing — complements
    the pipeline digest with financial optimization insights.
    """
    lines: list[str] = []

    lines.append("CO-OP INTELLIGENCE REPORT")
    lines.append(f"Store: {_store_display(report.store_id)}")
    lines.append(f"Total opportunity: {format_dollars(report.total_opportunity)}/year")
    lines.append("")

    if not report.alerts:
        lines.append("No co-op optimization alerts at this time.")
        return "\n".join(lines)

    # Group alerts by type
    for alert in report.alerts:
        lines.append(render_coop_alert(alert))
        lines.append("")

    return "\n".join(lines)


def render_inventory_health_summary(report: InventoryHealthReport) -> str:
    """Render a concise inventory health summary.

    One paragraph suitable for inclusion in the morning digest.
    """
    lines: list[str] = []

    lines.append("INVENTORY HEALTH")
    lines.append(f"Store: {_store_display(report.store_id)}")
    lines.append(
        f"Total inventory: {format_dollars(report.total_inventory_value)} | "
        f"GMROI: {report.overall_gmroi:.2f} | "
        f"Turn rate: {report.overall_turn_rate:.1f}x"
    )
    lines.append(
        f"Dead stock: {report.dead_stock_display} "
        f"({report.dead_stock_pct * 100:.0f}% of inventory) | "
        f"Carrying cost: {report.carrying_cost_display}/year"
    )
    lines.append("")

    # Category breakdown (top 3 by issue count)
    problem_categories = [
        a for a in report.category_analyses if a.dead_count + a.weak_count > 0
    ]
    problem_categories.sort(
        key=lambda a: a.dead_count + a.weak_count,
        reverse=True,
    )

    if problem_categories:
        lines.append("Categories needing attention:")
        for cat in problem_categories[:3]:
            lines.append(
                f"  {cat.category}: GMROI {cat.gmroi:.2f}, "
                f"{cat.dead_count} dead + {cat.weak_count} weak "
                f"of {cat.sku_count} SKUs"
            )
        lines.append("")

    return "\n".join(lines)


def render_rebate_status(status: VendorRebateStatus) -> str:
    """Render a single vendor rebate status line.

    Example:
        Martin's Supply - Silver tier (3.5%)
        YTD: $18,500 / $25,000 (74%) | 183 days left
        >> On track. $6,500 remaining. Projected value: $875 additional.
    """
    lines: list[str] = []

    current_name = status.current_tier.tier_name if status.current_tier else "None"
    current_rate = status.current_tier.rebate_pct * 100 if status.current_tier else 0

    lines.append(
        f"{status.program.vendor_name} — {current_name} tier ({current_rate:.1f}%)"
    )

    if status.next_tier:
        pct_complete = (
            status.ytd_purchases / status.next_tier.threshold * 100
            if status.next_tier.threshold > 0
            else 100
        )
        lines.append(
            f"  YTD: {format_dollars(status.ytd_purchases)} / "
            f"{format_dollars(status.next_tier.threshold)} "
            f"({pct_complete:.0f}%) | "
            f"{status.days_remaining} days left"
        )
    else:
        lines.append(
            f"  YTD: {format_dollars(status.ytd_purchases)} (top tier reached)"
        )

    lines.append(f"  >> {status.recommendation}")

    return "\n".join(lines)


def render_category_mix_summary(analysis: CategoryMixAnalysis) -> str:
    """Render a category mix analysis summary.

    Shows the most impactful expansion and contraction opportunities.
    """
    lines: list[str] = []

    lines.append("CATEGORY MIX ANALYSIS")
    lines.append(f"Store: {_store_display(analysis.store_id)}")
    lines.append(
        f"Total revenue: {format_dollars(analysis.total_revenue)} | "
        f"Blended margin: {analysis.total_margin_pct * 100:.0f}%"
    )
    lines.append(f"Total optimization opportunity: {analysis.opportunity_display}/year")
    lines.append("")

    if analysis.top_expansion_categories:
        lines.append("Expand (under-indexed vs NHPA high-profit):")
        for comp in analysis.comparisons:
            if comp.category in analysis.top_expansion_categories:
                lines.append(
                    f"  {comp.category}: "
                    f"{comp.store_pct * 100:.0f}% → "
                    f"{comp.benchmark_pct * 100:.0f}% target "
                    f"(+{comp.opportunity_display} profit)"
                )
        lines.append("")

    if analysis.top_contraction_categories:
        lines.append("Monitor (over-indexed):")
        for comp in analysis.comparisons:
            if comp.category in analysis.top_contraction_categories:
                lines.append(
                    f"  {comp.category}: "
                    f"{comp.store_pct * 100:.0f}% vs "
                    f"{comp.benchmark_pct * 100:.0f}% target"
                )
        lines.append("")

    return "\n".join(lines)
