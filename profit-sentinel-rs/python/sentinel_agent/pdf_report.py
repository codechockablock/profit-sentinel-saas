"""PDF Report Generator for Profit Sentinel.

Generates a professional shrinkage diagnostic report matching the
reference format: What (findings) → Why (COGS/margin/tax impact) → How (actions).

Report sections:
  1. Header (store context, date, items analyzed, report ID)
  2. Executive Summary (apparent shrinkage → to investigate → reduction)
  3. Potential Financial Impact table (COGS, Gross Margin, Inventory, Tax)
  4. Understanding the COGS Impact (process issues breakdown)
  5. Profit Margin Considerations
  6. Tax & Compliance Considerations
  7. Industry Context (Your Results vs NRF benchmarks)
  8. Summary of Findings (numbered action items)
  9. Pattern Analysis (leak type breakdown)
  10. Recommended Actions (prioritized, specific)
  11. Complete Inventory Analysis (all flagged SKUs by category)
  12. Inventory Statistics Summary
"""

from __future__ import annotations

import io
import logging
import uuid
from datetime import UTC, datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("sentinel.pdf")

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
EMERALD = colors.HexColor("#10b981")
SLATE_900 = colors.HexColor("#0f172a")
SLATE_700 = colors.HexColor("#334155")
SLATE_500 = colors.HexColor("#64748b")
SLATE_400 = colors.HexColor("#94a3b8")
SLATE_200 = colors.HexColor("#e2e8f0")
RED = colors.HexColor("#dc2626")
AMBER = colors.HexColor("#f59e0b")
LIGHT_BG = colors.HexColor("#f8fafc")
WHITE = colors.white

SEVERITY_COLORS = {
    "critical": colors.HexColor("#dc2626"),
    "high": colors.HexColor("#f97316"),
    "medium": colors.HexColor("#eab308"),
    "low": colors.HexColor("#3b82f6"),
    "info": colors.HexColor("#6b7280"),
}

# NRF 2024 National Retail Security Survey benchmarks
NRF_SHRINKAGE_RATE = 1.40  # percent of retail sales
NRF_RANGE_LOW = 1.2
NRF_RANGE_HIGH = 1.6
NRF_SOURCE = "National Retail Federation Retail Security Survey"


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "brand": ParagraphStyle(
            "Brand",
            parent=base["Title"],
            fontSize=28,
            textColor=EMERALD,
            alignment=TA_CENTER,
            spaceAfter=2,
            fontName="Helvetica-Bold",
        ),
        "report_title": ParagraphStyle(
            "ReportTitle",
            parent=base["Normal"],
            fontSize=12,
            textColor=SLATE_700,
            alignment=TA_LEFT,
            spaceBefore=4,
            spaceAfter=2,
            fontName="Helvetica-Bold",
        ),
        "meta": ParagraphStyle(
            "Meta",
            parent=base["Normal"],
            fontSize=9,
            textColor=SLATE_500,
            leading=14,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=16,
            textColor=SLATE_900,
            spaceBefore=20,
            spaceAfter=10,
            fontName="Helvetica-Bold",
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontSize=12,
            textColor=SLATE_900,
            spaceBefore=14,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontSize=9,
            textColor=SLATE_700,
            leading=14,
        ),
        "body_bold": ParagraphStyle(
            "BodyBold",
            parent=base["Normal"],
            fontSize=9,
            textColor=SLATE_700,
            leading=14,
            fontName="Helvetica-Bold",
        ),
        "note": ParagraphStyle(
            "Note",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_500,
            leading=11,
            fontStyle="italic",
        ),
        "big_number": ParagraphStyle(
            "BigNumber",
            parent=base["Normal"],
            fontSize=28,
            textColor=EMERALD,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "big_number_red": ParagraphStyle(
            "BigNumberRed",
            parent=base["Normal"],
            fontSize=28,
            textColor=RED,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontSize=9,
            textColor=SLATE_400,
            alignment=TA_CENTER,
        ),
        "arrow": ParagraphStyle(
            "Arrow",
            parent=base["Normal"],
            fontSize=20,
            textColor=SLATE_400,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TH",
            parent=base["Normal"],
            fontSize=8,
            textColor=WHITE,
            leading=11,
            fontName="Helvetica-Bold",
        ),
        "table_cell": ParagraphStyle(
            "TD",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_700,
            leading=11,
        ),
        "table_cell_right": ParagraphStyle(
            "TDR",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_700,
            leading=11,
            alignment=TA_RIGHT,
        ),
        "table_cell_bold": ParagraphStyle(
            "TDB",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_900,
            leading=11,
            fontName="Helvetica-Bold",
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=7,
            textColor=SLATE_400,
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_dollar(amount: float) -> str:
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:,.1f}M"
    if abs(amount) >= 1_000:
        return f"${amount:,.0f}"
    return f"${amount:,.2f}"


def _fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def _make_table(
    data: list[list],
    col_widths: list[float] | None = None,
    header: bool = True,
) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds: list[tuple] = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, SLATE_200),
    ]
    if header:
        style_cmds.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SLATE_900),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    # Zebra striping
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), LIGHT_BG))
    t.setStyle(TableStyle(style_cmds))
    return t


def _extract_metrics(analysis_result: dict) -> dict:
    """Pull all computed values from analysis_result into a flat dict."""
    leaks = analysis_result.get("leaks", {})
    summary = analysis_result.get("summary", {})
    impact = summary.get("estimated_impact", {})
    cause = analysis_result.get("cause_diagnosis", {})
    neg_alert = impact.get("negative_inventory_alert") or {}
    breakdown = impact.get("breakdown", {})

    total_rows = summary.get("total_rows_analyzed", 0)
    total_flagged = summary.get("total_items_flagged", 0)
    low_est = impact.get("low_estimate", 0)
    high_est = impact.get("high_estimate", 0)

    # Compute active leaks
    active_leaks = {k: v for k, v in leaks.items() if v.get("count", 0) > 0}

    # Compute apparent shrinkage (sum of all impact breakdown values)
    apparent_shrinkage = sum(breakdown.values())
    neg_inv_value = neg_alert.get("potential_untracked_cogs", 0)
    apparent_shrinkage += neg_inv_value

    # Process issues: things we can classify (not just "investigate")
    process_issues = 0
    for key in (
        "negative_inventory",
        "dead_item",
        "overstock",
        "zero_cost_anomaly",
        "price_discrepancy",
    ):
        process_issues += breakdown.get(key, 0)
    process_issues += neg_inv_value

    to_investigate = max(0, apparent_shrinkage - process_issues)
    reduction_pct = (
        (process_issues / apparent_shrinkage * 100) if apparent_shrinkage > 0 else 0
    )

    # Compute shrinkage rates as % of estimated annual retail sales.
    #
    # We don't have total store revenue in the analysis data, so we
    # back-calculate from the low/high impact estimates.  The Rust pipeline
    # computes these from actual inventory data with reasonable caps, so
    # they're the most reliable dollar amounts we have.
    #
    # Strategy: use the midpoint of low_est/high_est as the "total annual
    # shrinkage at retail" and derive the rate from a conservative revenue
    # estimate.  For the revenue estimate, we use the relationship:
    #   shrinkage ≈ 1-2% of revenue (NRF benchmark)
    # but anchor on the flagged-item data when available.
    (low_est + high_est) / 2 if (low_est + high_est) > 0 else 0

    # Use flagged items' retail value (qty × revenue) to get a sense of
    # per-item inventory value, then extrapolate across all items.
    flagged_retail_total = 0
    flagged_count_for_val = 0
    for leak_data in leaks.values():
        for item in leak_data.get("item_details", []):
            revenue = item.get("revenue", 0)
            qty = abs(item.get("quantity", 0))
            if revenue > 0 and qty > 0:
                flagged_retail_total += qty * revenue
                flagged_count_for_val += 1

    if flagged_count_for_val > 0 and total_rows > 0:
        avg_retail_value = flagged_retail_total / flagged_count_for_val
        # Flagged items skew high — use median-like discount.
        # Estimated total inventory at retail, then × turns for annual rev.
        # Hardware stores: ~4 inventory turns/year.
        estimated_annual_revenue = avg_retail_value * 0.3 * total_rows * 4
    else:
        estimated_annual_revenue = 0

    # Sanity bounds: rate should be in a reasonable range (0.5%-5%).
    # If estimated revenue gives a rate outside this, use NRF anchor.
    if apparent_shrinkage > 0 and estimated_annual_revenue > 0:
        test_rate = apparent_shrinkage / estimated_annual_revenue * 100
        if test_rate < 0.3 or test_rate > 8.0:
            # Outside plausible range — fall back to NRF-anchored estimate
            estimated_annual_revenue = apparent_shrinkage / (NRF_SHRINKAGE_RATE / 100)
    elif apparent_shrinkage > 0:
        estimated_annual_revenue = apparent_shrinkage / (NRF_SHRINKAGE_RATE / 100)

    shrinkage_rate = (
        (apparent_shrinkage / estimated_annual_revenue * 100)
        if estimated_annual_revenue > 0
        else 0
    )
    adjusted_rate = (
        (to_investigate / estimated_annual_revenue * 100)
        if estimated_annual_revenue > 0
        else 0
    )
    process_rate = (
        (process_issues / estimated_annual_revenue * 100)
        if estimated_annual_revenue > 0
        else 0
    )

    # Margin variance: how much the reported gross margin may be off.
    # Express as percentage-points (e.g. 1.2% means margin is ~1.2pp lower
    # than actual).  Same magnitude as the shrinkage rate.
    margin_variance = shrinkage_rate

    return {
        "total_rows": total_rows,
        "total_flagged": total_flagged,
        "low_est": low_est,
        "high_est": high_est,
        "apparent_shrinkage": apparent_shrinkage,
        "process_issues": process_issues,
        "to_investigate": to_investigate,
        "reduction_pct": reduction_pct,
        "shrinkage_rate": shrinkage_rate,
        "adjusted_rate": adjusted_rate,
        "process_rate": process_rate,
        "margin_variance": margin_variance,
        "neg_alert": neg_alert,
        "neg_inv_value": neg_inv_value,
        "active_leaks": active_leaks,
        "leaks": leaks,
        "cause": cause,
        "impact": impact,
        "breakdown": breakdown,
        "analysis_time": summary.get("analysis_time_seconds", 0),
    }


# ---------------------------------------------------------------------------
# Footer callback
# ---------------------------------------------------------------------------
_FOOTER_TEXT = "Profit Sentinel - AI-Powered Shrinkage Diagnostic"


def _add_page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(SLATE_400)
    canvas.drawString(
        doc.leftMargin,
        0.35 * inch,
        _FOOTER_TEXT,
    )
    canvas.drawRightString(
        letter[0] - doc.rightMargin,
        0.35 * inch,
        f"Page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main PDF generation
# ---------------------------------------------------------------------------
def generate_report_pdf(analysis_result: dict[str, Any]) -> bytes:
    """Generate a complete shrinkage diagnostic report as PDF bytes.

    Matches the reference format: executive summary with big numbers,
    COGS impact analysis, margin considerations, tax implications,
    NRF industry benchmarks, findings, pattern analysis, actions,
    and complete inventory analysis.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )

    S = _build_styles()
    story: list = []
    report_id = f"PS-{datetime.now(UTC).strftime('%Y%m%d%H%M')}"
    now = datetime.now(UTC)
    m = _extract_metrics(analysis_result)

    # =====================================================================
    # 1. HEADER
    # =====================================================================
    story.append(Paragraph("PROFIT SENTINEL", S["brand"]))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Shrinkage Diagnostic Report", S["report_title"]))
    story.append(Spacer(1, 8))

    # Meta grid
    meta_data = [
        [
            Paragraph("Store:", S["meta"]),
            Paragraph("<b>Guest Analysis</b>", S["meta"]),
            Paragraph("Date:", S["meta"]),
            Paragraph(f"<b>{now.strftime('%B %d, %Y')}</b>", S["meta"]),
        ],
        [
            Paragraph("Items Analyzed:", S["meta"]),
            Paragraph(f"<b>{m['total_rows']:,}</b>", S["meta"]),
            Paragraph("Report ID:", S["meta"]),
            Paragraph(f"<b>{report_id}</b>", S["meta"]),
        ],
    ]
    meta_table = Table(
        meta_data,
        colWidths=[1.0 * inch, 2.0 * inch, 1.0 * inch, 2.5 * inch],
    )
    meta_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 6))
    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=SLATE_200,
            spaceAfter=12,
            spaceBefore=4,
            dash=[2, 2],
        )
    )

    # =====================================================================
    # 2. EXECUTIVE SUMMARY — Big Numbers
    # =====================================================================
    story.append(Paragraph("Executive Summary", S["h2"]))

    num_style = S["big_number"] if m["apparent_shrinkage"] == 0 else S["big_number_red"]
    big_data = [
        [
            Paragraph("Apparent Shrinkage", S["metric_label"]),
            Paragraph("", S["arrow"]),
            Paragraph("To Investigate", S["metric_label"]),
            Paragraph("", S["arrow"]),
            Paragraph("Reduction", S["metric_label"]),
        ],
        [
            Paragraph(_fmt_dollar(m["apparent_shrinkage"]), num_style),
            Paragraph("\u2192", S["arrow"]),
            Paragraph(_fmt_dollar(m["to_investigate"]), num_style),
            Paragraph("", S["arrow"]),
            Paragraph(_fmt_pct(m["reduction_pct"]), S["big_number"]),
        ],
    ]
    big_table = Table(
        big_data,
        colWidths=[
            1.6 * inch,
            0.4 * inch,
            1.6 * inch,
            0.4 * inch,
            1.6 * inch,
        ],
    )
    big_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(big_table)
    story.append(Spacer(1, 16))

    # Potential Financial Impact Table
    story.append(Paragraph("<b>Potential Financial Impact</b>", S["h3"]))

    fi_rows = [["Area", "Potential Impact", "Estimated Amount", "Notes"]]
    fi_rows.append(
        [
            "Cost of Goods Sold",
            "May be overstated",
            _fmt_dollar(m["apparent_shrinkage"]),
            "Depends on accounting method",
        ]
    )
    fi_rows.append(
        [
            "Gross Margin",
            "May be understated",
            f"~{_fmt_pct(m['margin_variance'])} variance",
            "Review with accountant",
        ]
    )
    fi_rows.append(
        [
            "Inventory Valuation",
            "May be understated",
            _fmt_dollar(m["apparent_shrinkage"]),
            "Balance sheet consideration",
        ]
    )
    fi_rows.append(
        [
            "Tax Implications",
            "Varies by situation",
            f"Up to {_fmt_dollar(m['apparent_shrinkage'])}",
            "Consult your CPA",
        ]
    )
    story.append(
        _make_table(
            fi_rows,
            col_widths=[
                1.3 * inch,
                1.3 * inch,
                1.4 * inch,
                2.2 * inch,
            ],
        )
    )
    story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "<i>Note: These figures represent potential impacts based on the "
            "diagnostic findings. Actual financial impact depends on your specific "
            "accounting methods, how inventory adjustments are currently recorded, "
            "and your tax situation. We recommend reviewing these findings with "
            "your accountant or CPA.</i>",
            S["note"],
        )
    )
    story.append(Spacer(1, 12))

    # =====================================================================
    # 3. UNDERSTANDING THE COGS IMPACT
    # =====================================================================
    story.append(Paragraph("Understanding the COGS Impact", S["h2"]))

    story.append(
        Paragraph(
            f"<b>What This Diagnostic Found:</b> This analysis identified "
            f"{_fmt_dollar(m['apparent_shrinkage'])} in inventory discrepancies "
            f"that appear related to process issues rather than actual theft or "
            f"loss. Depending on how your inventory system feeds into your "
            f"accounting, this could affect your Cost of Goods Sold in several "
            f"ways:",
            S["body"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "<b>Process Issues Identified:</b>",
            S["body_bold"],
        )
    )

    # Build process issue bullets from actual leak data
    neg_inv = m["leaks"].get("negative_inventory", {})
    neg_count = neg_inv.get("count", 0)
    neg_val = m["neg_inv_value"]

    shrinkage = m["leaks"].get("shrinkage_pattern", {})
    shrink_count = shrinkage.get("count", 0)
    shrink_val = m["breakdown"].get("shrinkage_pattern", 0)

    margin_val = m["breakdown"].get("high_margin_leak", 0)
    dead_val = m["breakdown"].get("dead_item", 0)
    overstock_val = m["breakdown"].get("overstock", 0)
    zero_val = m["breakdown"].get("zero_cost_anomaly", 0)

    process_bullets = []
    if neg_count > 0:
        process_bullets.append(
            f"\u2022 <b>Receiving Gaps ({_fmt_dollar(neg_val)}):</b> "
            f"{neg_count} items showing negative inventory \u2014 these may "
            f"have been sold at the register but not received into the "
            f"inventory system. Common with lumber, sheet goods, and items "
            f"sold from the yard."
        )
    if shrink_count > 0:
        process_bullets.append(
            f"\u2022 <b>Shrinkage Patterns ({_fmt_dollar(shrink_val)}):</b> "
            f"{shrink_count} items with high value and low margin suggesting "
            f"potential unrecorded losses, vendor short-ships, or theft."
        )
    if dead_val > 0:
        process_bullets.append(
            f"\u2022 <b>Dead &amp; Obsolete Stock ({_fmt_dollar(dead_val)}):"
            f"</b> Items with zero or minimal sales tying up capital. These "
            f"may need write-off or clearance to accurately reflect inventory "
            f"value."
        )
    if overstock_val > 0:
        process_bullets.append(
            f"\u2022 <b>Overstock ({_fmt_dollar(overstock_val)}):</b> "
            f"Excess inventory beyond reasonable demand. May have been "
            f"over-ordered or reflects declining demand patterns."
        )
    if zero_val > 0:
        process_bullets.append(
            f"\u2022 <b>Missing Cost Data ({_fmt_dollar(zero_val)}):</b> "
            f"Items with zero cost recorded \u2014 cannot calculate true "
            f"margin. Update from recent vendor invoices."
        )
    if margin_val > 0:
        process_bullets.append(
            f"\u2022 <b>Margin Leaks ({_fmt_dollar(margin_val)}):</b> "
            f"Items selling below expected margin thresholds. May indicate "
            f"unauthorized discounting or vendor cost increases not reflected "
            f"in pricing."
        )

    if not process_bullets:
        process_bullets.append(
            "\u2022 No significant process issues identified in this analysis."
        )

    for bullet in process_bullets:
        story.append(Paragraph(bullet, S["body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 8))

    # =====================================================================
    # 4. PROFIT MARGIN CONSIDERATIONS
    # =====================================================================
    story.append(Paragraph("Profit Margin Considerations", S["h2"]))

    story.append(
        Paragraph(
            "<b>How Process Issues May Affect Reported Margins:</b>",
            S["body_bold"],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        Paragraph(
            "When inventory shows discrepancies due to process issues (rather "
            "than actual loss), it can create a mismatch between what your "
            "system reports and actual business performance. Here's what to "
            "consider:",
            S["body"],
        )
    )
    story.append(Spacer(1, 6))

    margin_bullets = [
        (
            "If COGS is being inflated:",
            "Your gross margin may appear lower than your actual performance. "
            "This could lead to overly conservative pricing decisions or "
            "unnecessary concern about profitability.",
        ),
        (
            "If adjustments are made elsewhere:",
            "Some businesses make periodic inventory adjustments or use "
            "different methods to account for shrinkage. If you're already "
            "accounting for these items through other means, the impact may "
            "already be reflected correctly.",
        ),
        (
            "What this means for you:",
            f"Based on this analysis, there may be up to a "
            f"<b>{_fmt_pct(m['margin_variance'])}</b> variance in your "
            f"reported margin. We recommend discussing these findings with "
            f"your accountant to understand how they apply to your specific "
            f"situation.",
        ),
    ]
    for title, desc in margin_bullets:
        story.append(
            Paragraph(
                f"\u2022 <b>{title}</b> {desc}",
                S["body"],
            )
        )
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 8))

    # =====================================================================
    # 5. TAX & COMPLIANCE CONSIDERATIONS
    # =====================================================================
    story.append(Paragraph("Tax &amp; Compliance Considerations", S["h2"]))

    story.append(
        Paragraph(
            "<b>Important: Please consult your CPA or tax advisor regarding "
            "these findings.</b>",
            S["body_bold"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Inventory discrepancies can have various tax implications depending "
            "on your situation:",
            S["body"],
        )
    )
    story.append(Spacer(1, 6))

    tax_scenarios = [
        (
            "Scenario A - COGS Impact",
            "If negative inventory balances are inflating your Cost of Goods "
            "Sold, this would reduce your taxable income. While this means "
            "lower taxes in the short term, it may not accurately reflect "
            "your business performance and could create issues if audited.",
        ),
        (
            "Scenario B - Separate Shrinkage Deductions",
            "If you're also claiming shrinkage as a separate deduction on "
            "your tax returns (which is allowable), AND your COGS is inflated "
            "from process issues, there may be a risk of inadvertent "
            "double-counting. Your CPA can review whether this applies to "
            "your situation.",
        ),
        (
            "Scenario C - Inventory Corrections",
            "If you make adjustments to correct these process issues, it "
            "could affect your taxable income in the period of adjustment. "
            "Your accountant can help you understand the timing and approach "
            "that works best for your business.",
        ),
    ]
    for title, desc in tax_scenarios:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", S["body"]))
        story.append(Spacer(1, 6))

    story.append(
        Paragraph(
            "<b>Balance Sheet Consideration:</b> Persistent negative inventory "
            "balances represent an understatement of assets. If inventory is "
            "used as collateral for financing, or if you're preparing financial "
            "statements for investors or lenders, accurate inventory valuation "
            "is important.",
            S["body"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "<b>Documentation:</b> This report provides documentation of the "
            "analysis performed. Maintaining records of how shrinkage was "
            "identified and categorized can be valuable for audit purposes.",
            S["body"],
        )
    )
    story.append(Spacer(1, 8))

    # =====================================================================
    # 6. INDUSTRY CONTEXT — Your Results vs NRF
    # =====================================================================
    story.append(Paragraph("Industry Context", S["h2"]))

    bench_rows = [["Metric", "Your Results", "Industry Average*", "Context"]]
    bench_rows.append(
        [
            "Apparent Shrinkage Rate",
            _fmt_pct(m["shrinkage_rate"], 2),
            _fmt_pct(NRF_SHRINKAGE_RATE, 2),
            "Before diagnostic analysis",
        ]
    )
    bench_rows.append(
        [
            "Adjusted Shrinkage Rate",
            _fmt_pct(m["adjusted_rate"], 2),
            _fmt_pct(NRF_SHRINKAGE_RATE, 2),
            "After identifying process issues",
        ]
    )
    bench_rows.append(
        [
            "Process Issues Identified",
            _fmt_pct(m["process_rate"], 2),
            "Varies",
            "Opportunity for improvement",
        ]
    )
    story.append(
        _make_table(
            bench_rows,
            col_widths=[
                1.5 * inch,
                1.1 * inch,
                1.2 * inch,
                2.4 * inch,
            ],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"<i>*Industry benchmark based on {NRF_SOURCE}. Hardware and "
            f"building materials retailers typically experience shrinkage rates "
            f"between {_fmt_pct(NRF_RANGE_LOW)}-{_fmt_pct(NRF_RANGE_HIGH)}. "
            f"These benchmarks include all sources of shrinkage (theft, "
            f"administrative error, vendor fraud, etc.).</i>",
            S["note"],
        )
    )
    story.append(Spacer(1, 12))

    # =====================================================================
    # 7. SUMMARY OF FINDINGS
    # =====================================================================
    story.append(Paragraph("Summary of Findings", S["h2"]))

    findings_rows = [["#", "Finding", "Amount", "Recommended Action"]]
    findings_rows.append(
        [
            "1",
            "Process issues identified in inventory",
            _fmt_dollar(m["process_issues"]),
            "Review with operations team",
        ]
    )
    findings_rows.append(
        [
            "2",
            "Potential margin variance identified",
            f"~{_fmt_pct(m['margin_variance'])}",
            "Discuss with accountant",
        ]
    )
    findings_rows.append(
        [
            "3",
            "Inventory valuation may need review",
            _fmt_dollar(m["apparent_shrinkage"]),
            "Consider adjustment",
        ]
    )
    findings_rows.append(
        [
            "4",
            "Tax implications should be evaluated",
            "Varies",
            "Consult your CPA",
        ]
    )
    findings_rows.append(
        [
            "5",
            "Remaining shrinkage to investigate",
            _fmt_dollar(m["to_investigate"]),
            "Investigate root causes",
        ]
    )
    story.append(
        _make_table(
            findings_rows,
            col_widths=[
                0.3 * inch,
                2.5 * inch,
                1.1 * inch,
                2.3 * inch,
            ],
        )
    )
    story.append(Spacer(1, 12))

    # =====================================================================
    # 8. PATTERN ANALYSIS
    # =====================================================================
    story.append(Paragraph("Pattern Analysis", S["h2"]))

    active = m["active_leaks"]
    story.append(
        Paragraph(
            f"During the diagnostic, <b>{len(active)}</b> leak patterns were "
            f"identified and reviewed:",
            S["body"],
        )
    )
    story.append(Spacer(1, 6))

    if active:
        pattern_rows = [
            ["Pattern", "Items", "Value", "Severity", "Category"],
        ]
        for key, leak_data in sorted(
            active.items(),
            key=lambda x: x[1].get("priority", 99),
        ):
            count = leak_data.get("count", 0)
            val = m["breakdown"].get(key, 0)
            if key == "negative_inventory":
                val = m["neg_inv_value"]
            pattern_rows.append(
                [
                    leak_data.get("title", key),
                    str(count),
                    _fmt_dollar(val),
                    leak_data.get("severity", "").upper(),
                    leak_data.get("category", ""),
                ]
            )
        story.append(
            _make_table(
                pattern_rows,
                col_widths=[
                    1.6 * inch,
                    0.6 * inch,
                    0.9 * inch,
                    0.9 * inch,
                    1.3 * inch,
                ],
            )
        )
    story.append(Spacer(1, 8))

    # Summary by classification
    story.append(Paragraph("Summary by Classification", S["h3"]))

    explained_items = 0
    investigate_items = 0
    for key, leak_data in active.items():
        count = leak_data.get("count", 0)
        sev = leak_data.get("severity", "")
        if sev in ("low", "medium", "info"):
            explained_items += count
        else:
            investigate_items += count

    class_rows = [
        ["Classification", "Description", "Items", "Value", "Status"],
    ]
    if m["process_issues"] > 0 or explained_items > 0:
        class_rows.append(
            [
                "Total Explained",
                "",
                str(explained_items),
                _fmt_dollar(m["process_issues"]),
                "EXPLAINED",
            ]
        )
    if m["to_investigate"] > 0 or investigate_items > 0:
        class_rows.append(
            [
                "Total to Investigate",
                "",
                str(investigate_items),
                _fmt_dollar(m["to_investigate"]),
                "INVESTIGATE",
            ]
        )
    if len(class_rows) > 1:
        story.append(
            _make_table(
                class_rows,
                col_widths=[
                    1.3 * inch,
                    1.3 * inch,
                    0.6 * inch,
                    0.8 * inch,
                    1.2 * inch,
                ],
            )
        )
    story.append(Spacer(1, 12))

    # =====================================================================
    # 9. RECOMMENDED ACTIONS
    # =====================================================================
    story.append(Paragraph("Recommended Actions", S["h2"]))

    actions = []

    if neg_count > 0:
        actions.append(
            f"<b>Review receiving processes</b> - {_fmt_dollar(neg_val)} in "
            f"shrinkage is due to items sold but not received. Consider "
            f"updating POS/inventory integration."
        )

    actions.append(
        "<b>Implement proper write-off procedures</b> - Ensure expired or "
        "damaged items are written off before disposal to maintain accurate "
        "inventory counts."
    )

    if m["to_investigate"] > 0:
        actions.append(
            f"<b>Investigate high-value unexplained items</b> - Focus "
            f"investigation efforts on the "
            f"{_fmt_dollar(m['to_investigate'])} that remains unexplained "
            f"after this diagnostic."
        )

    actions.append(
        "<b>Document vendor-managed inventory</b> - Ensure vendor-managed "
        "categories are clearly documented so future inventory counts "
        "account for these items correctly."
    )
    actions.append(
        "<b>Schedule follow-up diagnostic</b> - Run this diagnostic "
        "quarterly to track progress and identify new patterns."
    )

    for i, action in enumerate(actions, 1):
        story.append(Paragraph(f"<b>{i}.</b> {action}", S["body"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 8))

    # =====================================================================
    # 10. COMPLETE INVENTORY ANALYSIS
    # =====================================================================
    story.append(PageBreak())
    story.append(Paragraph("Complete Inventory Analysis", S["h2"]))

    story.append(
        Paragraph(
            f"The following pages contain the complete analysis of all "
            f"{m['total_rows']:,} inventory items. This includes "
            f"{m['total_flagged']:,} items that were flagged in this diagnostic.",
            S["body"],
        )
    )
    story.append(Spacer(1, 12))

    for key, leak_data in sorted(
        m["leaks"].items(),
        key=lambda x: x[1].get("priority", 99),
    ):
        items = leak_data.get("item_details", [])
        if not items:
            continue

        story.append(
            Paragraph(
                f'<b>{leak_data.get("title", key)}</b> '
                f'({len(items)} item{"s" if len(items) != 1 else ""})',
                S["h3"],
            )
        )

        sku_rows = [
            ["SKU", "Description", "Stock", "Cost", "Value", "Classification"],
        ]
        for item in items:
            qty = item.get("quantity", 0)
            cost = item.get("cost", 0)
            value = abs(qty) * cost
            sev = leak_data.get("severity", "low")
            sku_rows.append(
                [
                    str(item.get("sku", ""))[:20],
                    str(item.get("description", ""))[:35],
                    str(int(qty)),
                    _fmt_dollar(cost),
                    _fmt_dollar(value),
                    sev.upper(),
                ]
            )

        t = _make_table(
            sku_rows,
            col_widths=[
                0.9 * inch,
                1.8 * inch,
                0.6 * inch,
                0.7 * inch,
                0.8 * inch,
                1.0 * inch,
            ],
        )
        story.append(t)
        story.append(Spacer(1, 10))

    # =====================================================================
    # 11. INVENTORY STATISTICS
    # =====================================================================
    story.append(Paragraph("Inventory Statistics", S["h3"]))
    story.append(Spacer(1, 4))

    stats_lines = [
        f"<b>Total Items in Inventory:</b> {m['total_rows']:,}",
        f"<b>Items Flagged:</b> {m['total_flagged']:,}",
        f"<b>Total Apparent Shrinkage:</b> " f"{_fmt_dollar(m['apparent_shrinkage'])}",
        f"<b>Explained as Process Issues:</b> "
        f"{_fmt_dollar(m['process_issues'])} "
        f"({_fmt_pct(m['reduction_pct'])})",
        f"<b>Remaining to Investigate:</b> " f"{_fmt_dollar(m['to_investigate'])}",
    ]
    for line in stats_lines:
        story.append(Paragraph(line, S["body"]))

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            f"<i>This report was generated by Profit Sentinel on "
            f"{now.strftime('%B %d, %Y')} at {now.strftime('%I:%M %p')} UTC.</i>",
            S["note"],
        )
    )

    # Build PDF with footer callback
    doc.build(
        story,
        onFirstPage=_add_page_footer,
        onLaterPages=_add_page_footer,
    )
    pdf_bytes = buf.getvalue()
    buf.close()

    logger.info(
        "Generated PDF report %s: %d bytes",
        report_id,
        len(pdf_bytes),
    )
    return pdf_bytes
