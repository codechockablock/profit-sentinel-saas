"""PDF Report Generator for Profit Sentinel.

Generates a professional shrinkage diagnostic report as a PDF using reportlab.
This is the full, unanonymized report delivered via email to guest users.

Report sections:
  1. Header (store context, date, items analyzed, report ID)
  2. Executive Summary (shrinkage totals, reduction %)
  3. Financial Impact Table (COGS, gross margin, inventory valuation, tax)
  4. Industry Context Benchmarks (NRF data)
  5. Summary of Findings (all leak types)
  6. Pattern Analysis with Classifications
  7. Recommended Actions
  8. Complete Inventory Analysis (all flagged SKUs)
  9. Inventory Statistics Summary
"""

from __future__ import annotations

import io
import uuid
import logging
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

logger = logging.getLogger("sentinel.pdf")

# ---------------------------------------------------------------------------
# Color palette (matches frontend emerald theme)
# ---------------------------------------------------------------------------
EMERALD = colors.HexColor("#10b981")
SLATE_900 = colors.HexColor("#0f172a")
SLATE_700 = colors.HexColor("#334155")
SLATE_400 = colors.HexColor("#94a3b8")
RED = colors.HexColor("#dc2626")
AMBER = colors.HexColor("#f59e0b")
LIGHT_BG = colors.HexColor("#f8fafc")

# Severity → color mapping
SEVERITY_COLORS = {
    "critical": colors.HexColor("#dc2626"),
    "high": colors.HexColor("#f97316"),
    "medium": colors.HexColor("#eab308"),
    "low": colors.HexColor("#3b82f6"),
    "info": colors.HexColor("#6b7280"),
}

# NRF Industry benchmarks (hardcoded reference values)
NRF_BENCHMARKS = {
    "average_shrinkage_rate": 1.6,  # % of retail sales
    "average_shrinkage_dollars": 112.1,  # billions industry-wide
    "employee_theft_pct": 28.5,
    "shoplifting_pct": 36.5,
    "admin_error_pct": 15.3,
    "vendor_fraud_pct": 5.4,
    "unknown_pct": 14.3,
    "source": "NRF 2024 National Retail Security Survey",
}


# ---------------------------------------------------------------------------
# Custom styles
# ---------------------------------------------------------------------------
def _build_styles() -> dict[str, ParagraphStyle]:
    """Build custom paragraph styles for the report."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=22,
            textColor=SLATE_900,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=SLATE_400,
            spaceAfter=16,
        ),
        "h2": ParagraphStyle(
            "SectionH2",
            parent=base["Heading2"],
            fontSize=14,
            textColor=SLATE_900,
            spaceBefore=16,
            spaceAfter=8,
            borderPadding=(0, 0, 4, 0),
        ),
        "h3": ParagraphStyle(
            "SectionH3",
            parent=base["Heading3"],
            fontSize=11,
            textColor=SLATE_700,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=9,
            textColor=SLATE_700,
            leading=13,
        ),
        "body_small": ParagraphStyle(
            "BodySmall",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_400,
            leading=11,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=base["Normal"],
            fontSize=20,
            textColor=EMERALD,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_400,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.white,
            leading=10,
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_700,
            leading=10,
        ),
        "table_cell_right": ParagraphStyle(
            "TableCellRight",
            parent=base["Normal"],
            fontSize=8,
            textColor=SLATE_700,
            leading=10,
            alignment=TA_RIGHT,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=7,
            textColor=SLATE_400,
            alignment=TA_CENTER,
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_dollar(amount: float) -> str:
    """Format dollar amount for display."""
    if abs(amount) >= 1_000_000:
        return f"${amount / 1_000_000:,.1f}M"
    if abs(amount) >= 1_000:
        return f"${amount:,.0f}"
    return f"${amount:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


def _severity_label(sev: str) -> str:
    return sev.upper()


def _make_table(
    data: list[list],
    col_widths: list[float] | None = None,
    header: bool = True,
) -> Table:
    """Build a styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds: list[tuple] = [
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]
    if header:
        style_cmds.extend([
            ("BACKGROUND", (0, 0), (-1, 0), SLATE_900),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    # Zebra striping
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(
                ("BACKGROUND", (0, i), (-1, i), LIGHT_BG)
            )
    t.setStyle(TableStyle(style_cmds))
    return t


# ---------------------------------------------------------------------------
# Main PDF generation
# ---------------------------------------------------------------------------
def generate_report_pdf(analysis_result: dict[str, Any]) -> bytes:
    """Generate a complete shrinkage diagnostic report as PDF bytes.

    Parameters
    ----------
    analysis_result : dict
        The full analysis result dict from /analysis/analyze endpoint.

    Returns
    -------
    bytes
        PDF file content ready for email attachment.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
    )

    styles = _build_styles()
    story: list = []
    report_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now(timezone.utc)

    leaks: dict = analysis_result.get("leaks", {})
    summary: dict = analysis_result.get("summary", {})
    impact: dict = summary.get("estimated_impact", {})
    cause: dict = analysis_result.get("cause_diagnosis", {})

    total_rows = summary.get("total_rows_analyzed", 0)
    total_flagged = summary.get("total_items_flagged", 0)
    analysis_time = summary.get("analysis_time_seconds", 0)

    # ── 1. HEADER ────────────────────────────────────────────────
    story.append(Paragraph("Profit Sentinel", styles["title"]))
    story.append(Paragraph("Shrinkage Diagnostic Report", styles["h2"]))
    story.append(Paragraph(
        f"Generated: {now.strftime('%B %d, %Y at %H:%M UTC')} &nbsp;|&nbsp; "
        f"Report ID: PS-{report_id} &nbsp;|&nbsp; "
        f"Items Analyzed: {total_rows:,}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(
        width="100%", thickness=2, color=EMERALD,
        spaceAfter=12, spaceBefore=4,
    ))

    # ── 2. EXECUTIVE SUMMARY ─────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles["h2"]))

    low_est = impact.get("low_estimate", 0)
    high_est = impact.get("high_estimate", 0)

    active_leaks = [(k, v) for k, v in leaks.items() if v.get("count", 0) > 0]
    active_leaks.sort(key=lambda x: x[1].get("priority", 99))

    exec_text = (
        f"Analysis of <b>{total_rows:,}</b> inventory items identified "
        f"<b>{total_flagged:,}</b> items with potential profit leaks across "
        f"<b>{len(active_leaks)}</b> of 11 detection categories. "
        f"Estimated annual impact: <b>{_fmt_dollar(low_est)}</b> to "
        f"<b>{_fmt_dollar(high_est)}</b>. "
        f"Analysis completed in {analysis_time:.1f} seconds."
    )
    story.append(Paragraph(exec_text, styles["body"]))
    story.append(Spacer(1, 8))

    # Negative inventory alert
    neg_alert = impact.get("negative_inventory_alert")
    if neg_alert and neg_alert.get("requires_audit"):
        alert_text = (
            f'<font color="#dc2626"><b>AUDIT REQUIRED:</b></font> '
            f'{neg_alert["items_found"]} item(s) with negative on-hand quantities detected. '
            f'Potential untracked COGS: {_fmt_dollar(neg_alert.get("potential_untracked_cogs", 0))}. '
            f'This is excluded from the annual estimate and requires a physical audit.'
        )
        story.append(Paragraph(alert_text, styles["body"]))
        story.append(Spacer(1, 8))

    # ── 3. FINANCIAL IMPACT TABLE ────────────────────────────────
    story.append(Paragraph("Financial Impact Breakdown", styles["h2"]))

    breakdown = impact.get("breakdown", {})
    impact_rows = [["Leak Type", "Category", "Severity", "Items", "Est. Impact"]]
    for key, leak_data in sorted(
        leaks.items(), key=lambda x: x[1].get("priority", 99)
    ):
        count = leak_data.get("count", 0)
        if count == 0:
            continue
        impact_amt = breakdown.get(key, 0)
        impact_rows.append([
            leak_data.get("title", key),
            leak_data.get("category", ""),
            _severity_label(leak_data.get("severity", "low")),
            str(count),
            _fmt_dollar(impact_amt) if impact_amt > 0 else "Audit Required",
        ])

    if len(impact_rows) > 1:
        # Totals row
        total_impact_items = sum(
            v.get("count", 0) for v in leaks.values() if v.get("count", 0) > 0
        )
        impact_rows.append([
            "TOTAL", "", "", str(total_impact_items),
            f"{_fmt_dollar(low_est)} – {_fmt_dollar(high_est)}",
        ])
        t = _make_table(
            impact_rows,
            col_widths=[2.0 * inch, 1.2 * inch, 0.8 * inch, 0.6 * inch, 1.4 * inch],
        )
        # Bold the totals row
        t.setStyle(TableStyle([
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ("LINEABOVE", (0, -1), (-1, -1), 1, SLATE_900),
        ]))
        story.append(t)
    story.append(Spacer(1, 8))

    # ── Tax implications note ──
    if low_est > 0:
        tax_text = (
            f"<b>Tax Implications:</b> Unresolved inventory shrinkage of "
            f"{_fmt_dollar(low_est)}–{_fmt_dollar(high_est)} may affect COGS calculations, "
            f"gross margin reporting, and inventory valuation for tax purposes. "
            f"Consult your CPA for specific guidance on write-down timing."
        )
        story.append(Paragraph(tax_text, styles["body_small"]))
        story.append(Spacer(1, 12))

    # ── 4. INDUSTRY CONTEXT BENCHMARKS ───────────────────────────
    story.append(Paragraph("Industry Context", styles["h2"]))

    bench_data = [
        ["Metric", "Industry Average", "Source"],
        ["Average Shrinkage Rate", f"{NRF_BENCHMARKS['average_shrinkage_rate']}% of retail sales", NRF_BENCHMARKS["source"]],
        ["Employee Theft", f"{NRF_BENCHMARKS['employee_theft_pct']}% of shrinkage", NRF_BENCHMARKS["source"]],
        ["Shoplifting", f"{NRF_BENCHMARKS['shoplifting_pct']}% of shrinkage", NRF_BENCHMARKS["source"]],
        ["Administrative Error", f"{NRF_BENCHMARKS['admin_error_pct']}% of shrinkage", NRF_BENCHMARKS["source"]],
        ["Vendor Fraud/Error", f"{NRF_BENCHMARKS['vendor_fraud_pct']}% of shrinkage", NRF_BENCHMARKS["source"]],
    ]
    story.append(_make_table(
        bench_data,
        col_widths=[1.8 * inch, 1.8 * inch, 2.4 * inch],
    ))
    story.append(Spacer(1, 12))

    # ── 5. SUMMARY OF FINDINGS ───────────────────────────────────
    story.append(Paragraph("Summary of Findings", styles["h2"]))

    for key, leak_data in sorted(
        leaks.items(), key=lambda x: x[1].get("priority", 99)
    ):
        count = leak_data.get("count", 0)
        if count == 0:
            continue

        sev = leak_data.get("severity", "low")
        sev_color = SEVERITY_COLORS.get(sev, SLATE_400)

        heading = (
            f'<font color="{sev_color.hexval()}">\u25cf</font> '
            f'<b>{leak_data.get("title", key)}</b> '
            f'({leak_data.get("category", "")}) — '
            f'{count} item{"s" if count != 1 else ""}'
        )
        story.append(Paragraph(heading, styles["h3"]))

        # Context for this leak type
        items = leak_data.get("item_details", [])
        if items:
            top_item = items[0]
            context = top_item.get("context", "")
            if context:
                story.append(Paragraph(
                    f"<i>Example: {context[:200]}</i>",
                    styles["body_small"],
                ))

        # Recommendations
        recs = leak_data.get("recommendations", [])
        if recs:
            rec_text = " &bull; ".join(recs[:3])
            story.append(Paragraph(
                f"<b>Actions:</b> {rec_text}",
                styles["body_small"],
            ))
        story.append(Spacer(1, 4))

    # ── 6. ROOT CAUSE ANALYSIS ───────────────────────────────────
    if cause and cause.get("top_cause"):
        story.append(Paragraph("Pattern Analysis", styles["h2"]))
        top = cause["top_cause"].replace("_", " ").title()
        conf = cause.get("confidence", 0)
        story.append(Paragraph(
            f"<b>Primary Root Cause:</b> {top} (confidence: {_fmt_pct(conf * 100)})",
            styles["body"],
        ))

        hypotheses = cause.get("hypotheses", [])
        if hypotheses:
            for h in hypotheses[:5]:
                cause_name = h.get("cause", "").replace("_", " ").title()
                prob = h.get("probability", 0)
                evidence = h.get("evidence", [])
                ev_text = "; ".join(evidence[:2]) if evidence else ""
                story.append(Paragraph(
                    f"&bull; {cause_name} ({_fmt_pct(prob * 100)})"
                    + (f" — {ev_text}" if ev_text else ""),
                    styles["body_small"],
                ))
        story.append(Spacer(1, 12))

    # ── 7. COMPLETE INVENTORY ANALYSIS ───────────────────────────
    story.append(Paragraph("Complete Inventory Analysis — Flagged Items", styles["h2"]))

    for key, leak_data in sorted(
        leaks.items(), key=lambda x: x[1].get("priority", 99)
    ):
        items = leak_data.get("item_details", [])
        if not items:
            continue

        story.append(Paragraph(
            f'<b>{leak_data.get("title", key)}</b> ({len(items)} items)',
            styles["h3"],
        ))

        sku_rows = [["SKU", "Description", "QOH", "Cost", "Retail", "Sold", "Score"]]
        for item in items:  # Show ALL items, not truncated
            sku_rows.append([
                str(item.get("sku", ""))[:20],
                str(item.get("description", ""))[:30],
                str(item.get("quantity", 0)),
                _fmt_dollar(item.get("cost", 0)),
                _fmt_dollar(item.get("revenue", 0)),
                str(item.get("sold", 0)),
                f"{item.get('score', 0):.0%}",
            ])

        # Only paginate if very large
        table_chunk = sku_rows
        t = _make_table(
            table_chunk,
            col_widths=[0.9 * inch, 1.5 * inch, 0.5 * inch, 0.7 * inch, 0.7 * inch, 0.5 * inch, 0.5 * inch],
        )
        story.append(t)
        story.append(Spacer(1, 8))

    # ── 8. INVENTORY STATISTICS SUMMARY ──────────────────────────
    story.append(Paragraph("Inventory Statistics Summary", styles["h2"]))

    stats_data = [
        ["Metric", "Value"],
        ["Total Items Analyzed", f"{total_rows:,}"],
        ["Total Items Flagged", f"{total_flagged:,}"],
        ["Flag Rate", _fmt_pct((total_flagged / total_rows * 100) if total_rows > 0 else 0)],
        ["Leak Types Detected", str(len(active_leaks))],
        ["Analysis Time", f"{analysis_time:.1f} seconds"],
        ["Estimated Annual Impact (Low)", _fmt_dollar(low_est)],
        ["Estimated Annual Impact (High)", _fmt_dollar(high_est)],
    ]

    # Per-severity counts
    for sev in ["critical", "high", "medium", "low"]:
        count = sum(
            v.get("count", 0) for v in leaks.values()
            if v.get("severity") == sev and v.get("count", 0) > 0
        )
        if count > 0:
            stats_data.append([f"{sev.title()} Severity Items", str(count)])

    story.append(_make_table(stats_data, col_widths=[3.0 * inch, 3.0 * inch]))
    story.append(Spacer(1, 16))

    # ── FOOTER ───────────────────────────────────────────────────
    story.append(HRFlowable(
        width="100%", thickness=1, color=SLATE_400,
        spaceAfter=8, spaceBefore=8,
    ))
    story.append(Paragraph(
        f"Report PS-{report_id} &bull; Generated by Profit Sentinel &bull; "
        f"{now.strftime('%Y-%m-%d %H:%M UTC')} &bull; "
        f"This report contains sensitive business data. Do not share publicly.",
        styles["footer"],
    ))
    story.append(Paragraph(
        "Analysis results are algorithmic suggestions. Verify with qualified "
        "personnel before taking business action. &copy; 2026 Profit Sentinel.",
        styles["footer"],
    ))

    # Build PDF
    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()

    logger.info(
        "Generated PDF report PS-%s: %d pages, %d bytes",
        report_id, doc.page, len(pdf_bytes),
    )
    return pdf_bytes
