"""
Email Delivery via Resend

Send diagnostic reports to users via Resend email API.

Setup:
    1. Get API key from https://resend.com
    2. Set environment variable: RESEND_API_KEY=re_xxxxx
    3. Or pass api_key to functions directly

Example:
    from diagnostic.email import email_report

    result = email_report(
        to_email="user@example.com",
        pdf_path="/path/to/report.pdf",
        store_name="Demo Hardware Store"
    )
    print(f"Email sent: {result['id']}")
"""

import base64
import os
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import resend

    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False


def init_resend(api_key: str | None = None):
    """Initialize Resend with API key."""
    if not RESEND_AVAILABLE:
        raise ImportError("resend package not installed. Run: pip install resend")

    key = api_key or os.environ.get("RESEND_API_KEY")
    if not key:
        raise ValueError(
            "Resend API key required. Set RESEND_API_KEY env var or pass api_key."
        )

    resend.api_key = key


def email_report(
    to_email: str,
    pdf_path: str,
    store_name: str,
    api_key: str | None = None,
    from_email: str = "Profit Sentinel <reports@profitsentinel.com>",
    reply_to: str | None = None,
) -> dict[str, Any]:
    """
    Email a diagnostic report via Resend.

    Args:
        to_email: Recipient email address
        pdf_path: Path to the PDF report file
        store_name: Name of the store (for subject line)
        api_key: Resend API key (optional if env var set)
        from_email: Sender email address
        reply_to: Reply-to address (optional)

    Returns:
        Resend API response with email ID

    Raises:
        ImportError: If resend package not installed
        ValueError: If API key not provided
        FileNotFoundError: If PDF file not found
    """
    init_resend(api_key)

    # Read PDF file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_content = base64.standard_b64encode(f.read()).decode("utf-8")

    # Generate filename
    date_str = datetime.now().strftime("%Y%m%d")
    safe_store_name = store_name.lower().replace(" ", "_").replace("'", "")
    filename = f"profit_sentinel_report_{safe_store_name}_{date_str}.pdf"

    # Build email
    params = {
        "from": from_email,
        "to": [to_email],
        "subject": f"Shrinkage Diagnostic Report - {store_name}",
        "html": _build_email_html(store_name),
        "attachments": [
            {
                "filename": filename,
                "content": pdf_content,
            }
        ],
    }

    if reply_to:
        params["reply_to"] = reply_to

    # Send
    response = resend.Emails.send(params)
    return response


def email_report_summary(
    to_email: str,
    store_name: str,
    total_shrinkage: float,
    explained_value: float,
    unexplained_value: float,
    reduction_percent: float,
    report_url: str | None = None,
    api_key: str | None = None,
    from_email: str = "Profit Sentinel <reports@profitsentinel.com>",
) -> dict[str, Any]:
    """
    Email a summary of the diagnostic (without PDF attachment).

    Useful for quick notifications or when PDF is hosted elsewhere.
    """
    init_resend(api_key)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 30px; border-radius: 12px 12px 0 0; }}
            .content {{ background: #f9fafb; padding: 30px; border-radius: 0 0 12px 12px; }}
            .stat {{ display: inline-block; text-align: center; padding: 15px 25px; }}
            .stat-value {{ font-size: 28px; font-weight: bold; }}
            .stat-label {{ font-size: 12px; color: #6b7280; }}
            .red {{ color: #ef4444; }}
            .green {{ color: #10b981; }}
            .button {{ display: inline-block; background: #10b981; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin: 0;">⚡ Profit Sentinel</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">Shrinkage Diagnostic Complete</p>
            </div>
            <div class="content">
                <h2>Results for {store_name}</h2>

                <div style="text-align: center; margin: 30px 0;">
                    <div class="stat">
                        <div class="stat-value red">${total_shrinkage:,.0f}</div>
                        <div class="stat-label">Apparent Shrinkage</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">→</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value green">${unexplained_value:,.0f}</div>
                        <div class="stat-label">To Investigate</div>
                    </div>
                </div>

                <p style="text-align: center; font-size: 18px;">
                    <strong class="green">{reduction_percent:.1f}%</strong> identified as process issues
                </p>

                <p>
                    Through the diagnostic conversation, we identified <strong>${explained_value:,.0f}</strong>
                    in apparent shrinkage that may be related to process issues rather than actual loss.
                </p>

                {"<p><a href='" + report_url + "' class='button'>View Full Report</a></p>" if report_url else ""}

                <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 30px 0;">

                <p style="color: #6b7280; font-size: 14px;">
                    Questions about this report? Reply to this email.<br>
                    — The Profit Sentinel Team
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    params = {
        "from": from_email,
        "to": [to_email],
        "subject": f"Diagnostic Complete: {reduction_percent:.0f}% Shrinkage Explained - {store_name}",
        "html": html,
    }

    response = resend.Emails.send(params)
    return response


def _build_email_html(store_name: str) -> str:
    """Build the HTML email body."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #1f2937;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
                padding: 30px;
                border-radius: 12px 12px 0 0;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .content {{
                background: #f9fafb;
                padding: 30px;
                border-radius: 0 0 12px 12px;
            }}
            .highlight {{
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 20px;
                margin: 20px 0;
            }}
            .footer {{
                color: #6b7280;
                font-size: 14px;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #e5e7eb;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚡ Profit Sentinel</h1>
                <p style="margin: 10px 0 0 0; opacity: 0.9;">Shrinkage Diagnostic Report</p>
            </div>

            <div class="content">
                <h2>Your Report is Ready</h2>

                <p>Hi,</p>

                <p>
                    Attached is your comprehensive shrinkage diagnostic report for
                    <strong>{store_name}</strong>.
                </p>

                <div class="highlight">
                    <strong>What's in this report:</strong>
                    <ul style="margin: 10px 0 0 0; padding-left: 20px;">
                        <li>Executive Summary with key findings</li>
                        <li>Financial impact analysis (COGS, margins, tax considerations)</li>
                        <li>Pattern-by-pattern breakdown</li>
                        <li>Industry benchmarks</li>
                        <li>Complete SKU listing for verification</li>
                        <li>Recommended action items</li>
                    </ul>
                </div>

                <p>
                    We recommend reviewing the Executive Summary first, then sharing relevant
                    sections with your operations team and accountant.
                </p>

                <div class="footer">
                    <p>
                        Questions about this report? Reply to this email and we'll help you
                        interpret the findings.
                    </p>
                    <p>— The Profit Sentinel Team</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


# Batch sending for multiple stores
def email_reports_batch(
    reports: list[dict[str, Any]],
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """
    Send multiple reports in batch.

    Args:
        reports: List of dicts with keys: to_email, pdf_path, store_name
        api_key: Resend API key

    Returns:
        List of Resend API responses
    """
    results = []
    for report in reports:
        try:
            result = email_report(
                to_email=report["to_email"],
                pdf_path=report["pdf_path"],
                store_name=report["store_name"],
                api_key=api_key,
            )
            results.append({"success": True, "response": result, **report})
        except Exception as e:
            results.append({"success": False, "error": str(e), **report})

    return results
