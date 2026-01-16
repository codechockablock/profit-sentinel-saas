"""
Email Service - Send analysis reports via Resend or SendGrid.

Supports GDPR/CCPA compliant email delivery with:
- Consent tracking
- Unsubscribe links
- Privacy-compliant content
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending email reports via Resend or SendGrid."""

    def __init__(self):
        """Initialize email service with provider detection."""
        self.resend_api_key = os.getenv("RESEND_API_KEY")
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("EMAIL_FROM", "reports@profitsentinel.com")
        self.from_name = os.getenv("EMAIL_FROM_NAME", "Profit Sentinel")

        # Determine which provider to use
        if self.resend_api_key:
            self.provider = "resend"
            logger.info("Email service initialized with Resend")
        elif self.sendgrid_api_key:
            self.provider = "sendgrid"
            logger.info("Email service initialized with SendGrid")
        else:
            self.provider = None
            logger.warning("No email provider configured (RESEND_API_KEY or SENDGRID_API_KEY required)")

    @property
    def is_configured(self) -> bool:
        """Check if email service is configured."""
        return self.provider is not None

    async def send_analysis_report(
        self,
        to_email: str,
        results: List[Dict],
        consent_given: bool = True,
        consent_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send analysis report email.

        Args:
            to_email: Recipient email address
            results: Analysis results to include in report
            consent_given: Whether user gave consent
            consent_timestamp: When consent was given

        Returns:
            Dict with success status and message ID
        """
        if not self.is_configured:
            logger.warning("Email service not configured, skipping send")
            return {"success": False, "error": "Email service not configured"}

        if not consent_given:
            logger.warning("Consent not given, refusing to send email")
            return {"success": False, "error": "Consent required"}

        # Generate HTML content
        html_content = self._generate_report_html(results, to_email)

        # Generate plain text fallback
        text_content = self._generate_report_text(results)

        subject = f"Your Profit Leak Analysis Report - {datetime.now().strftime('%B %d, %Y')}"

        try:
            if self.provider == "resend":
                return await self._send_via_resend(to_email, subject, html_content, text_content)
            elif self.provider == "sendgrid":
                return await self._send_via_sendgrid(to_email, subject, html_content, text_content)
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {"success": False, "error": str(e)}

    async def _send_via_resend(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str
    ) -> Dict[str, Any]:
        """Send email via Resend API."""
        try:
            import resend
            resend.api_key = self.resend_api_key

            response = resend.Emails.send({
                "from": f"{self.from_name} <{self.from_email}>",
                "to": [to_email],
                "subject": subject,
                "html": html_content,
                "text": text_content,
                "headers": {
                    "List-Unsubscribe": f"<mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe {to_email}>"
                }
            })

            logger.info(f"Email sent via Resend to {to_email}, ID: {response.get('id')}")
            return {"success": True, "message_id": response.get("id")}

        except ImportError:
            logger.error("Resend package not installed: pip install resend")
            return {"success": False, "error": "Resend package not installed"}
        except Exception as e:
            logger.error(f"Resend API error: {e}")
            return {"success": False, "error": str(e)}

    async def _send_via_sendgrid(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str
    ) -> Dict[str, Any]:
        """Send email via SendGrid API."""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import (
                Mail, Email, To, Content, Header
            )

            sg = SendGridAPIClient(self.sendgrid_api_key)

            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=To(to_email),
                subject=subject,
            )
            message.add_content(Content("text/plain", text_content))
            message.add_content(Content("text/html", html_content))

            # Add unsubscribe header
            message.add_header(Header(
                "List-Unsubscribe",
                f"<mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe {to_email}>"
            ))

            response = sg.send(message)

            logger.info(f"Email sent via SendGrid to {to_email}, status: {response.status_code}")
            return {
                "success": response.status_code in [200, 201, 202],
                "status_code": response.status_code
            }

        except ImportError:
            logger.error("SendGrid package not installed: pip install sendgrid")
            return {"success": False, "error": "SendGrid package not installed"}
        except Exception as e:
            logger.error(f"SendGrid API error: {e}")
            return {"success": False, "error": str(e)}

    def _generate_report_html(self, results: List[Dict], to_email: str) -> str:
        """Generate HTML email content for analysis report."""
        # Calculate totals
        total_flagged = 0
        total_impact_low = 0
        total_impact_high = 0

        for result in results:
            summary = result.get("summary", {})
            total_flagged += summary.get("total_items_flagged", 0)
            impact = summary.get("estimated_impact", {})
            total_impact_low += impact.get("low_estimate", 0)
            total_impact_high += impact.get("high_estimate", 0)

        leak_sections = ""
        for result in results:
            filename = result.get("filename", "Unknown File")
            leaks = result.get("leaks", {})

            leak_sections += f"""
            <div style="margin-bottom: 30px;">
                <h3 style="color: #10b981; margin-bottom: 15px; font-size: 18px;">
                    {filename}
                </h3>
            """

            for leak_type, data in leaks.items():
                items = data.get("top_items", [])[:5]
                scores = data.get("scores", [])
                count = data.get("count", len(items))

                if count == 0:
                    continue

                title = leak_type.replace("_", " ").title()
                color = self._get_leak_color(leak_type)

                leak_sections += f"""
                <div style="background: #1e293b; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 4px solid {color};">
                    <h4 style="color: {color}; margin: 0 0 10px 0; font-size: 16px;">{title}</h4>
                    <p style="color: #94a3b8; margin: 0 0 10px 0; font-size: 14px;">{count} items flagged</p>
                    <ul style="margin: 0; padding-left: 20px; color: #cbd5e1;">
                """

                for i, item in enumerate(items):
                    score = scores[i] if i < len(scores) else 0
                    pct = int(score * 100)
                    leak_sections += f'<li style="margin-bottom: 5px;">{item} - <span style="color: {color};">{pct}% risk</span></li>'

                leak_sections += """
                    </ul>
                </div>
                """

            leak_sections += "</div>"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profit Sentinel Analysis Report</title>
</head>
<body style="margin: 0; padding: 0; background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <!-- Header -->
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="color: #10b981; margin: 0; font-size: 28px;">Profit Sentinel</h1>
            <p style="color: #64748b; margin: 10px 0 0 0; font-size: 14px;">AI-Powered Profit Forensics</p>
        </div>

        <!-- Summary Card -->
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 25px; margin-bottom: 30px;">
            <h2 style="color: #10b981; margin: 0 0 20px 0; font-size: 20px;">Analysis Summary</h2>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div style="text-align: center; min-width: 100px; margin-bottom: 15px;">
                    <p style="color: #f97316; font-size: 28px; font-weight: bold; margin: 0;">{total_flagged}</p>
                    <p style="color: #94a3b8; font-size: 12px; margin: 5px 0 0 0;">Items Flagged</p>
                </div>
                <div style="text-align: center; min-width: 140px; margin-bottom: 15px;">
                    <p style="color: #10b981; font-size: 28px; font-weight: bold; margin: 0;">
                        ${total_impact_low:,.0f} - ${total_impact_high:,.0f}
                    </p>
                    <p style="color: #94a3b8; font-size: 12px; margin: 5px 0 0 0;">Est. Annual Impact</p>
                </div>
            </div>
            <p style="color: #10b981; font-size: 14px; margin: 15px 0 0 0; padding: 10px; background: rgba(16, 185, 129, 0.1); border-radius: 8px;">
                Addressing these leaks could recover ${total_impact_low:,.0f} to ${total_impact_high:,.0f} annually.
            </p>
        </div>

        <!-- Leak Details -->
        <div style="background: #1e293b; border-radius: 16px; padding: 25px; margin-bottom: 30px;">
            <h2 style="color: #f1f5f9; margin: 0 0 20px 0; font-size: 20px;">Detailed Findings</h2>
            {leak_sections}
        </div>

        <!-- Action Items -->
        <div style="background: #1e293b; border-radius: 16px; padding: 25px; margin-bottom: 30px;">
            <h2 style="color: #f1f5f9; margin: 0 0 15px 0; font-size: 20px;">Recommended Next Steps</h2>
            <ol style="color: #cbd5e1; padding-left: 20px; margin: 0;">
                <li style="margin-bottom: 10px;">Review critical and high-severity items first</li>
                <li style="margin-bottom: 10px;">Verify negative inventory items with physical counts</li>
                <li style="margin-bottom: 10px;">Update pricing on margin leak items</li>
                <li style="margin-bottom: 10px;">Investigate shrinkage patterns for theft or vendor issues</li>
                <li style="margin-bottom: 10px;">Plan clearance for dead inventory to free up capital</li>
            </ol>
        </div>

        <!-- Footer -->
        <div style="text-align: center; padding-top: 30px; border-top: 1px solid #334155;">
            <p style="color: #64748b; font-size: 12px; margin: 0 0 10px 0;">
                This report was generated by Profit Sentinel based on your uploaded POS data.
            </p>
            <p style="color: #64748b; font-size: 12px; margin: 0 0 10px 0;">
                Your data has been processed and deleted per our privacy policy.
            </p>
            <p style="color: #475569; font-size: 11px; margin: 20px 0 0 0;">
                You received this email because you requested an analysis report at profitsentinel.com.<br>
                <a href="mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe {to_email}" style="color: #10b981;">Unsubscribe</a> |
                <a href="https://profitsentinel.com/privacy" style="color: #10b981;">Privacy Policy</a>
            </p>
            <p style="color: #334155; font-size: 10px; margin: 15px 0 0 0;">
                Profit Sentinel - Protecting retail margins with AI-powered forensic analysis.
            </p>
        </div>
    </div>
</body>
</html>
        """
        return html

    def _generate_report_text(self, results: List[Dict]) -> str:
        """Generate plain text email content for analysis report."""
        lines = [
            "PROFIT SENTINEL - Analysis Report",
            "=" * 40,
            "",
        ]

        total_flagged = 0
        for result in results:
            summary = result.get("summary", {})
            total_flagged += summary.get("total_items_flagged", 0)

        lines.append(f"Total Items Flagged: {total_flagged}")
        lines.append("")

        for result in results:
            filename = result.get("filename", "Unknown File")
            lines.append(f"\n--- {filename} ---")

            leaks = result.get("leaks", {})
            for leak_type, data in leaks.items():
                items = data.get("top_items", [])[:5]
                count = data.get("count", len(items))

                if count == 0:
                    continue

                title = leak_type.replace("_", " ").title()
                lines.append(f"\n{title} ({count} items):")

                for item in items:
                    lines.append(f"  - {item}")

        lines.extend([
            "",
            "=" * 40,
            "RECOMMENDED NEXT STEPS:",
            "1. Review critical and high-severity items first",
            "2. Verify negative inventory with physical counts",
            "3. Update pricing on margin leak items",
            "4. Investigate shrinkage patterns",
            "5. Plan clearance for dead inventory",
            "",
            "---",
            "This report was generated by Profit Sentinel.",
            "Your uploaded data has been processed and deleted.",
            "Visit profitsentinel.com for more information.",
            "",
            "To unsubscribe, reply with 'Unsubscribe' in the subject.",
        ])

        return "\n".join(lines)

    def _get_leak_color(self, leak_type: str) -> str:
        """Get color for leak type."""
        colors = {
            "high_margin_leak": "#ef4444",
            "negative_inventory": "#dc2626",
            "low_stock": "#f59e0b",
            "shrinkage_pattern": "#f97316",
            "margin_erosion": "#ec4899",
            "dead_item": "#6b7280",
            "overstock": "#3b82f6",
            "price_discrepancy": "#8b5cf6",
        }
        return colors.get(leak_type, "#6b7280")


# Singleton instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get or create email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
