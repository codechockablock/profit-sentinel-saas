"""Email service for Profit Sentinel.

Sends morning digest reports and analysis summaries via Resend.
Supports both HTML and plain text emails.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger("sentinel.email")

# ---------------------------------------------------------------------------
# Resend API client (lightweight — no SDK dependency)
# ---------------------------------------------------------------------------

RESEND_API_URL = "https://api.resend.com/emails"


async def send_email(
    *,
    api_key: str,
    to: str | list[str],
    subject: str,
    html: str,
    text: str | None = None,
    from_addr: str = "Profit Sentinel <reports@profitsentinel.com>",
    reply_to: str = "support@profitsentinel.com",
    attachments: list[dict] | None = None,
) -> dict:
    """Send an email via Resend API.

    Parameters
    ----------
    attachments : list[dict], optional
        List of attachment dicts with keys:
        - filename: str (e.g., "report.pdf")
        - content: str (base64-encoded file content)

    Returns the Resend response dict (contains 'id' on success).
    Raises httpx.HTTPStatusError on failure.
    """
    if isinstance(to, str):
        to = [to]

    payload = {
        "from": from_addr,
        "to": to,
        "subject": subject,
        "html": html,
        "headers": {
            "List-Unsubscribe": "<mailto:unsubscribe@profitsentinel.com>",
        },
    }
    if text:
        payload["text"] = text
    if reply_to:
        payload["reply_to"] = reply_to
    if attachments:
        payload["attachments"] = attachments

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            RESEND_API_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info("Email sent to %s, id=%s", to, result.get("id"))
        return result


# ---------------------------------------------------------------------------
# Digest HTML template
# ---------------------------------------------------------------------------


def _format_dollar(amount: float) -> str:
    """Format dollar amount for display."""
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.1f}K"
    return f"${amount:.0f}"


def _priority_color(score: float) -> tuple[str, str]:
    """Return (bg_color, text_color) for email HTML."""
    if score >= 10:
        return "#fef2f2", "#dc2626"  # red
    if score >= 8:
        return "#fff7ed", "#ea580c"  # orange
    if score >= 5:
        return "#fefce8", "#ca8a04"  # yellow
    return "#eff6ff", "#2563eb"  # blue


def _trend_arrow(direction: str) -> str:
    if direction == "Worsening":
        return '<span style="color:#dc2626">&#9650;</span>'
    if direction == "Improving":
        return '<span style="color:#16a34a">&#9660;</span>'
    return '<span style="color:#9ca3af">&#8212;</span>'


def render_digest_html(
    *,
    plain_text: str,
    issue_count: int,
    total_impact: float,
    store_count: int,
    pipeline_ms: int,
    issues: list[dict] | None = None,
    generated_at: str | None = None,
) -> str:
    """Render morning digest as branded HTML email.

    Parameters
    ----------
    plain_text : str
        The rendered text from render_digest() — used as readable fallback
    issue_count : int
        Total issues found
    total_impact : float
        Dollar exposure
    store_count : int
        Number of stores affected
    pipeline_ms : int
        Pipeline execution time
    issues : list[dict], optional
        Issue dicts with keys: issue_type, store_id, dollar_impact,
        priority_score, trend_direction, root_cause, skus
    generated_at : str, optional
        ISO timestamp for when digest was generated
    """
    now = generated_at or datetime.now(timezone.utc).strftime("%B %d, %Y")

    # Build issue rows
    issue_rows = ""
    if issues:
        for issue in issues[:10]:
            bg, fg = _priority_color(issue.get("priority_score", 0))
            issue_type = (
                issue.get("issue_type", "Unknown")
                .replace("_", " ")
                # CamelCase to spaces
            )
            # Insert space before capitals for CamelCase
            spaced_type = ""
            for ch in issue_type:
                if ch.isupper() and spaced_type and spaced_type[-1] != " ":
                    spaced_type += " "
                spaced_type += ch

            store = issue.get("store_id", "")
            impact = _format_dollar(issue.get("dollar_impact", 0))
            trend = _trend_arrow(issue.get("trend_direction", "Stable"))
            root_cause = issue.get("root_cause", "")
            sku_count = len(issue.get("skus", []))

            issue_rows += f"""
            <tr>
              <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;">
                <div style="font-weight:600;color:#0f172a;">{spaced_type} {trend}</div>
                <div style="font-size:12px;color:#64748b;margin-top:2px;">
                  {store} &middot; {sku_count} SKU{"s" if sku_count != 1 else ""}
                  {f" &middot; {root_cause}" if root_cause else ""}
                </div>
              </td>
              <td style="padding:12px 16px;border-bottom:1px solid #f1f5f9;text-align:right;">
                <span style="display:inline-block;padding:4px 10px;border-radius:6px;font-size:13px;font-weight:700;background:{bg};color:{fg};">
                  {impact}
                </span>
              </td>
            </tr>"""

    # No issues — all clear
    if issue_count == 0:
        issue_section = """
        <div style="text-align:center;padding:40px 20px;">
          <div style="font-size:48px;margin-bottom:12px;">&#10003;</div>
          <h2 style="color:#16a34a;margin:0 0 8px;">All Clear</h2>
          <p style="color:#64748b;margin:0;">No priority issues detected today.</p>
        </div>
        """
    else:
        issue_section = f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
          {issue_rows}
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background-color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:600px;margin:0 auto;padding:20px;">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);border-radius:16px 16px 0 0;padding:32px 24px;text-align:center;">
      <h1 style="color:#10b981;margin:0 0 4px;font-size:20px;font-weight:700;">Profit Sentinel</h1>
      <p style="color:#94a3b8;margin:0;font-size:13px;">Morning Digest &middot; {now}</p>
    </div>

    <!-- Summary bar -->
    <div style="background:#0f172a;padding:16px 24px;display:flex;">
      <table width="100%" cellpadding="0" cellspacing="0"><tr>
        <td style="text-align:center;padding:8px;">
          <div style="font-size:24px;font-weight:700;color:#ffffff;">{issue_count}</div>
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Issues</div>
        </td>
        <td style="text-align:center;padding:8px;">
          <div style="font-size:24px;font-weight:700;color:#10b981;">{_format_dollar(total_impact)}</div>
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Exposure</div>
        </td>
        <td style="text-align:center;padding:8px;">
          <div style="font-size:24px;font-weight:700;color:#ffffff;">{store_count}</div>
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Stores</div>
        </td>
        <td style="text-align:center;padding:8px;">
          <div style="font-size:24px;font-weight:700;color:#ffffff;">{pipeline_ms}ms</div>
          <div style="font-size:11px;color:#64748b;text-transform:uppercase;">Pipeline</div>
        </td>
      </tr></table>
    </div>

    <!-- Issues -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;padding:0;">
      {issue_section}
    </div>

    <!-- CTA -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-top:none;padding:24px;text-align:center;border-radius:0 0 16px 16px;">
      <a href="https://profitsentinel.com/dashboard"
         style="display:inline-block;padding:12px 32px;background:#10b981;color:#ffffff;text-decoration:none;border-radius:8px;font-weight:600;font-size:14px;">
        View Full Dashboard
      </a>
    </div>

    <!-- Footer -->
    <div style="text-align:center;padding:24px 16px;color:#94a3b8;font-size:11px;">
      <p style="margin:0 0 8px;">Profit Sentinel &middot; Inventory Intelligence for Hardware Retailers</p>
      <p style="margin:0;">
        <a href="https://profitsentinel.com/dashboard" style="color:#10b981;text-decoration:none;">Dashboard</a>
        &nbsp;&middot;&nbsp;
        <a href="https://profitsentinel.com/privacy" style="color:#94a3b8;text-decoration:none;">Privacy</a>
        &nbsp;&middot;&nbsp;
        <a href="mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe" style="color:#94a3b8;text-decoration:none;">Unsubscribe</a>
      </p>
    </div>

  </div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# High-level digest email sender
# ---------------------------------------------------------------------------


async def send_digest_email(
    *,
    api_key: str,
    to: str | list[str],
    plain_text: str,
    issue_count: int,
    total_impact: float,
    store_count: int,
    pipeline_ms: int,
    issues: list[dict] | None = None,
    generated_at: str | None = None,
) -> dict:
    """Render and send a morning digest email.

    Combines plain text digest with HTML template and sends via Resend.
    """
    html = render_digest_html(
        plain_text=plain_text,
        issue_count=issue_count,
        total_impact=total_impact,
        store_count=store_count,
        pipeline_ms=pipeline_ms,
        issues=issues,
        generated_at=generated_at,
    )

    subject = (
        f"Morning Digest: {issue_count} issue{'s' if issue_count != 1 else ''}"
        f" ({_format_dollar(total_impact)} exposure)"
        if issue_count > 0
        else "Morning Digest: All Clear"
    )

    return await send_email(
        api_key=api_key,
        to=to,
        subject=subject,
        html=html,
        text=plain_text,
    )


# ---------------------------------------------------------------------------
# Guest report email with PDF attachment
# ---------------------------------------------------------------------------


def _render_report_email_html(total_items: int, total_flagged: int, leak_count: int) -> str:
    """Render the HTML body for the guest report delivery email."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background-color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:600px;margin:0 auto;padding:20px;">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);border-radius:16px 16px 0 0;padding:32px 24px;text-align:center;">
      <h1 style="color:#10b981;margin:0 0 4px;font-size:20px;font-weight:700;">Profit Sentinel</h1>
      <p style="color:#94a3b8;margin:0;font-size:13px;">Your Shrinkage Diagnostic Report</p>
    </div>

    <!-- Body -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;padding:32px 24px;">
      <h2 style="color:#0f172a;margin:0 0 16px;font-size:18px;">Your report is attached.</h2>
      <p style="color:#334155;font-size:14px;line-height:1.6;margin:0 0 16px;">
        We analyzed <strong>{total_items:,}</strong> inventory items and identified
        <strong>{total_flagged:,}</strong> items with potential profit leaks across
        <strong>{leak_count}</strong> detection categories.
      </p>
      <p style="color:#334155;font-size:14px;line-height:1.6;margin:0 0 16px;">
        The attached PDF contains your complete, unanonymized analysis with specific
        SKU numbers, product names, and actionable recommendations.
      </p>
      <p style="color:#64748b;font-size:12px;line-height:1.5;margin:0;padding:16px;background:#f8fafc;border-radius:8px;">
        <strong>Privacy note:</strong> Your uploaded files are automatically deleted
        within 24 hours. We only retain anonymized aggregate statistics. This report
        is the only copy of your detailed analysis — save it for your records.
      </p>
    </div>

    <!-- CTA -->
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-top:none;padding:24px;text-align:center;border-radius:0 0 16px 16px;">
      <a href="https://profitsentinel.com/analyze"
         style="display:inline-block;padding:12px 32px;background:#10b981;color:#ffffff;text-decoration:none;border-radius:8px;font-weight:600;font-size:14px;">
        Run Another Analysis
      </a>
    </div>

    <!-- Footer -->
    <div style="text-align:center;padding:24px 16px;color:#94a3b8;font-size:11px;">
      <p style="margin:0 0 8px;">Profit Sentinel &middot; Inventory Intelligence for Retailers</p>
      <p style="margin:0;">
        <a href="https://profitsentinel.com/privacy" style="color:#94a3b8;text-decoration:none;">Privacy</a>
        &nbsp;&middot;&nbsp;
        <a href="mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe" style="color:#94a3b8;text-decoration:none;">Unsubscribe</a>
      </p>
    </div>

  </div>
</body>
</html>"""


async def send_report_email(
    *,
    api_key: str,
    to: str,
    pdf_bytes: bytes,
    total_items: int,
    total_flagged: int,
    leak_count: int,
) -> dict:
    """Send the guest analysis report with PDF attachment via Resend.

    Parameters
    ----------
    api_key : str
        Resend API key
    to : str
        Recipient email address
    pdf_bytes : bytes
        Generated PDF report content
    total_items : int
        Total items analyzed (for email body)
    total_flagged : int
        Total items flagged (for email body)
    leak_count : int
        Number of active leak types detected (for email body)

    Returns
    -------
    dict
        Resend API response
    """
    import base64

    html = _render_report_email_html(total_items, total_flagged, leak_count)

    return await send_email(
        api_key=api_key,
        to=to,
        subject="Your Profit Sentinel Shrinkage Diagnostic Report",
        html=html,
        text=(
            f"Your Profit Sentinel report is attached.\n\n"
            f"We analyzed {total_items:,} items and found {total_flagged:,} "
            f"with potential profit leaks across {leak_count} categories.\n\n"
            f"Save this report — your uploaded files are automatically deleted "
            f"within 24 hours."
        ),
        attachments=[{
            "filename": "Profit_Sentinel_Shrinkage_Report.pdf",
            "content": base64.b64encode(pdf_bytes).decode("ascii"),
        }],
    )
