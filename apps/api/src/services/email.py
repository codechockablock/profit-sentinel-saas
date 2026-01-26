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
from typing import Any

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
            logger.warning(
                "No email provider configured (RESEND_API_KEY or SENDGRID_API_KEY required)"
            )

    @property
    def is_configured(self) -> bool:
        """Check if email service is configured."""
        return self.provider is not None

    async def send_analysis_report(
        self,
        to_email: str,
        results: list[dict],
        consent_given: bool = True,
        consent_timestamp: str | None = None,
    ) -> dict[str, Any]:
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

        # Check if this is mock data (sentinel engine not available)
        is_mock = any(r.get("mock", False) for r in results)
        if is_mock:
            logger.warning(
                "Sending email with MOCK DATA - sentinel engine was not available. "
                "SKUs will be placeholders like 'SKU-001'."
            )

        # Generate HTML content
        html_content = self._generate_report_html(results, to_email, is_mock=is_mock)

        # Generate plain text fallback
        text_content = self._generate_report_text(results)

        subject = (
            f"Your Profit Leak Analysis Report - {datetime.now().strftime('%B %d, %Y')}"
        )

        try:
            if self.provider == "resend":
                return await self._send_via_resend(
                    to_email, subject, html_content, text_content
                )
            elif self.provider == "sendgrid":
                return await self._send_via_sendgrid(
                    to_email, subject, html_content, text_content
                )
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {"success": False, "error": str(e)}

    async def _send_via_resend(
        self, to_email: str, subject: str, html_content: str, text_content: str
    ) -> dict[str, Any]:
        """Send email via Resend API."""
        try:
            import resend

            resend.api_key = self.resend_api_key

            response = resend.Emails.send(
                {
                    "from": f"{self.from_name} <{self.from_email}>",
                    "to": [to_email],
                    "subject": subject,
                    "html": html_content,
                    "text": text_content,
                    "headers": {
                        "List-Unsubscribe": f"<mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe {to_email}>"
                    },
                }
            )

            # Log success without PII (no email address)
            logger.info(f"Email sent via Resend, ID: {response.get('id')}")
            return {"success": True, "message_id": response.get("id")}

        except ImportError:
            logger.error("Resend package not installed: pip install resend")
            return {"success": False, "error": "Resend package not installed"}
        except Exception as e:
            logger.error(f"Resend API error: {e}")
            return {"success": False, "error": str(e)}

    async def _send_via_sendgrid(
        self, to_email: str, subject: str, html_content: str, text_content: str
    ) -> dict[str, Any]:
        """Send email via SendGrid API."""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Content, Email, Header, Mail, To

            sg = SendGridAPIClient(self.sendgrid_api_key)

            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=To(to_email),
                subject=subject,
            )
            message.add_content(Content("text/plain", text_content))
            message.add_content(Content("text/html", html_content))

            # Add unsubscribe header
            message.add_header(
                Header(
                    "List-Unsubscribe",
                    f"<mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe {to_email}>",
                )
            )

            response = sg.send(message)

            # Log success without PII (no email address)
            logger.info(f"Email sent via SendGrid, status: {response.status_code}")
            return {
                "success": response.status_code in [200, 201, 202],
                "status_code": response.status_code,
            }

        except ImportError:
            logger.error("SendGrid package not installed: pip install sendgrid")
            return {"success": False, "error": "SendGrid package not installed"}
        except Exception as e:
            logger.error(f"SendGrid API error: {e}")
            return {"success": False, "error": str(e)}

    def _generate_report_html(
        self, results: list[dict], to_email: str, is_mock: bool = False
    ) -> str:
        """Generate HTML email content for analysis report."""
        # Mock data warning banner
        mock_warning = ""
        if is_mock:
            mock_warning = """
            <div style="background: #fef3c7; border: 2px solid #f59e0b; border-radius: 12px; padding: 20px; margin-bottom: 30px; text-align: center;">
                <p style="color: #92400e; font-size: 16px; font-weight: bold; margin: 0 0 10px 0;">
                    ⚠️ DEMO MODE - Sample Data Only
                </p>
                <p style="color: #a16207; font-size: 14px; margin: 0;">
                    This report contains placeholder data (SKU-001, SKU-002, etc.) because the analysis engine
                    could not process your actual file. Please contact support@profitsentinel.com for assistance.
                </p>
            </div>
            """

        # Calculate totals
        total_flagged = 0
        total_impact_low = 0
        total_impact_high = 0
        negative_inventory_alert = None

        for result in results:
            summary = result.get("summary", {})
            total_flagged += summary.get("total_items_flagged", 0)
            impact = summary.get("estimated_impact", {})
            total_impact_low += impact.get("low_estimate", 0)
            total_impact_high += impact.get("high_estimate", 0)
            # Extract negative inventory alert (data integrity issue)
            if impact.get("negative_inventory_alert"):
                negative_inventory_alert = impact["negative_inventory_alert"]

        # Build negative inventory data integrity alert section
        data_integrity_section = ""
        if negative_inventory_alert:
            items_found = negative_inventory_alert.get("items_found", 0)
            untracked_cogs = negative_inventory_alert.get("potential_untracked_cogs", 0)
            is_anomalous = negative_inventory_alert.get("is_anomalous", False)

            anomaly_warning = ""
            if is_anomalous:
                anomaly_warning = """
                <p style="color: #fbbf24; font-size: 13px; margin: 12px 0 0 0; padding: 10px; background: #fbbf2415; border-radius: 6px; border-left: 3px solid #fbbf24;">
                    ⚠️ This figure exceeds normal thresholds and requires physical audit.<br>
                    Impact excluded from annual estimate until verified.
                </p>
                """

            data_integrity_section = f"""
            <div style="background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(220, 38, 38, 0.05)); border: 1px solid rgba(220, 38, 38, 0.4); border-radius: 16px; padding: 25px; margin-bottom: 30px;">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 24px; margin-right: 10px;">🚨</span>
                    <h2 style="color: #f87171; margin: 0; font-size: 20px;">Data Integrity Alert</h2>
                    <span style="margin-left: auto; background: rgba(220, 38, 38, 0.2); color: #dc2626; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">CRITICAL</span>
                </div>

                <div style="background: #0f172a; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                    <h3 style="color: #f87171; margin: 0 0 12px 0; font-size: 16px;">Negative Inventory</h3>
                    <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                        <div>
                            <span style="color: #64748b; font-size: 11px; text-transform: uppercase;">Items Found</span>
                            <p style="color: #f87171; font-size: 24px; font-weight: bold; margin: 4px 0 0 0;">{items_found:,}</p>
                        </div>
                        <div>
                            <span style="color: #64748b; font-size: 11px; text-transform: uppercase;">Potential Untracked COGS</span>
                            <p style="color: #fbbf24; font-size: 24px; font-weight: bold; margin: 4px 0 0 0;">${untracked_cogs:,.0f}</p>
                            <span style="color: #94a3b8; font-size: 10px;">(estimated, requires verification)</span>
                        </div>
                    </div>
                    {anomaly_warning}
                </div>

                <div style="background: #1e293b; border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                    <h4 style="color: #fbbf24; margin: 0 0 10px 0; font-size: 14px;">📋 What This Means</h4>
                    <ul style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 13px;">
                        <li style="margin-bottom: 6px;">{items_found:,} items show negative quantities (sold without being received)</li>
                        <li style="margin-bottom: 6px;">Financial records likely do not reflect true COGS</li>
                        <li style="margin-bottom: 6px;">Tax and margin calculations may be materially misstated</li>
                    </ul>
                </div>

                <div style="background: #1e293b; border-radius: 8px; padding: 15px;">
                    <h4 style="color: #10b981; margin: 0 0 10px 0; font-size: 14px;">✅ Recommended Actions</h4>
                    <ol style="margin: 0; padding-left: 20px; color: #cbd5e1; font-size: 13px;">
                        <li style="margin-bottom: 6px;">Conduct physical inventory count</li>
                        <li style="margin-bottom: 6px;">Reconcile receiving records against vendor invoices</li>
                        <li style="margin-bottom: 6px;">Consult accountant regarding COGS and tax implications</li>
                    </ol>
                </div>
            </div>
            """

        # Extract cause diagnosis if available
        cause_diagnosis_section = ""
        for result in results:
            cause_diagnosis = result.get("cause_diagnosis")
            if cause_diagnosis and cause_diagnosis.get("top_cause"):
                top_cause = cause_diagnosis.get("top_cause", "Unknown")
                confidence = cause_diagnosis.get("confidence", 0) * 100
                cause_details = cause_diagnosis.get("cause_details", {})
                severity = cause_details.get("severity", "info")
                category = cause_details.get("category", "Unknown")
                description = cause_details.get("description", "")
                recommendations = cause_details.get("recommendations", [])
                alternative_causes = cause_diagnosis.get("alternative_causes", [])

                severity_colors = {
                    "critical": "#dc2626",
                    "high": "#f97316",
                    "medium": "#f59e0b",
                    "info": "#3b82f6",
                }
                severity_color = severity_colors.get(severity, "#6b7280")

                cause_diagnosis_section = f"""
                <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(139, 92, 246, 0.05)); border: 1px solid rgba(139, 92, 246, 0.4); border-radius: 16px; padding: 25px; margin-bottom: 30px;">
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <span style="font-size: 24px; margin-right: 10px;">🔍</span>
                        <h2 style="color: #a78bfa; margin: 0; font-size: 20px;">Root Cause Analysis</h2>
                        <span style="margin-left: auto; background: rgba(16, 185, 129, 0.2); color: #10b981; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold;">VSA-Grounded</span>
                    </div>

                    <div style="background: #0f172a; border-radius: 12px; padding: 20px; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="color: #f1f5f9; font-size: 18px; font-weight: bold;">{top_cause.replace("_", " ").title()}</span>
                            <span style="background: {severity_color}22; color: {severity_color}; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: bold; text-transform: uppercase;">{severity}</span>
                        </div>
                        <p style="color: #94a3b8; font-size: 14px; margin: 0 0 12px 0;">{description}</p>
                        <div style="display: flex; gap: 20px; margin-top: 10px;">
                            <div>
                                <span style="color: #64748b; font-size: 11px; text-transform: uppercase;">Confidence</span>
                                <p style="color: #10b981; font-size: 20px; font-weight: bold; margin: 4px 0 0 0;">{confidence:.0f}%</p>
                            </div>
                            <div>
                                <span style="color: #64748b; font-size: 11px; text-transform: uppercase;">Category</span>
                                <p style="color: #f1f5f9; font-size: 14px; margin: 4px 0 0 0;">{category}</p>
                            </div>
                        </div>
                    </div>
                """

                # Add recommendations
                if recommendations:
                    cause_diagnosis_section += """
                    <div style="margin-bottom: 15px;">
                        <h4 style="color: #fbbf24; margin: 0 0 10px 0; font-size: 14px;">📋 Targeted Actions</h4>
                        <ul style="margin: 0; padding-left: 20px; color: #cbd5e1;">
                    """
                    for rec in recommendations[:4]:
                        cause_diagnosis_section += f'<li style="margin-bottom: 6px; font-size: 13px;">{rec}</li>'
                    cause_diagnosis_section += "</ul></div>"

                # Add alternative causes
                if alternative_causes:
                    cause_diagnosis_section += """
                    <div style="background: #1e293b; border-radius: 8px; padding: 12px; margin-top: 10px;">
                        <p style="color: #64748b; font-size: 11px; text-transform: uppercase; margin: 0 0 8px 0;">Also Consider</p>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    """
                    for alt in alternative_causes[:3]:
                        alt_cause = alt.get("cause", "").replace("_", " ").title()
                        alt_conf = alt.get("confidence", 0) * 100
                        cause_diagnosis_section += f"""
                            <span style="background: #0f172a; color: #94a3b8; padding: 4px 10px; border-radius: 4px; font-size: 12px;">
                                {alt_cause} ({alt_conf:.0f}%)
                            </span>
                        """
                    cause_diagnosis_section += "</div></div>"

                cause_diagnosis_section += "</div>"
                break  # Only show first cause diagnosis

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
                item_details = data.get("item_details", [])
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
                """

                # Use item_details if available (includes inventory context)
                if item_details:
                    for i, detail in enumerate(item_details[:5]):
                        sku = detail.get("sku", "Unknown")
                        score = detail.get("score", 0)
                        pct = int(score * 100)
                        context = detail.get("context", "")
                        qty = detail.get("quantity", 0)
                        cost = detail.get("cost", 0)
                        sold = detail.get("sold", 0)
                        desc = detail.get("description", "")[:40]

                        leak_sections += f"""
                        <div style="background: #0f172a; border-radius: 6px; padding: 12px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <span style="color: #f1f5f9; font-family: monospace; font-weight: bold;">{sku}</span>
                                <span style="background: {color}22; color: {color}; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">{pct}% risk</span>
                            </div>
                            {f'<p style="color: #64748b; font-size: 12px; margin: 0 0 8px 0;">{desc}</p>' if desc else ""}
                            <div style="display: flex; gap: 15px; font-size: 12px; margin-bottom: 8px;">
                                <span style="color: #94a3b8;">QOH: <strong style="color: #f1f5f9;">{qty:.0f}</strong></span>
                                <span style="color: #94a3b8;">Cost: <strong style="color: #f1f5f9;">${cost:.2f}</strong></span>
                                <span style="color: #94a3b8;">Sold: <strong style="color: #f1f5f9;">{sold:.0f}</strong></span>
                            </div>
                            <p style="color: #fbbf24; font-size: 13px; margin: 0; padding: 8px; background: #fbbf2410; border-radius: 4px;">
                                💡 {context}
                            </p>
                        </div>
                        """
                else:
                    # Fallback to simple list if no details
                    leak_sections += (
                        '<ul style="margin: 0; padding-left: 20px; color: #cbd5e1;">'
                    )
                    for i, item in enumerate(items):
                        score = scores[i] if i < len(scores) else 0
                        pct = int(score * 100)
                        leak_sections += f'<li style="margin-bottom: 5px;">{item} - <span style="color: {color};">{pct}% risk</span></li>'
                    leak_sections += "</ul>"

                leak_sections += """
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

        <!-- Mock Data Warning (if applicable) -->
        {mock_warning}

        <!-- Data Integrity Alert (Negative Inventory) -->
        {data_integrity_section}

        <!-- Root Cause Analysis (VSA-Grounded) -->
        {cause_diagnosis_section}

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

    def _generate_report_text(self, results: list[dict]) -> str:
        """Generate plain text email content for analysis report."""
        lines = [
            "PROFIT SENTINEL - Analysis Report",
            "=" * 40,
            "",
        ]

        # Check for negative inventory data integrity alert
        for result in results:
            impact = result.get("summary", {}).get("estimated_impact", {})
            neg_alert = impact.get("negative_inventory_alert")
            if neg_alert:
                items_found = neg_alert.get("items_found", 0)
                untracked_cogs = neg_alert.get("potential_untracked_cogs", 0)
                is_anomalous = neg_alert.get("is_anomalous", False)

                lines.extend(
                    [
                        "!!! DATA INTEGRITY ALERT !!!",
                        "-" * 35,
                        "NEGATIVE INVENTORY - CRITICAL",
                        f"Items Found: {items_found:,}",
                        f"Potential Untracked COGS: ${untracked_cogs:,.0f} (estimated)",
                        "",
                    ]
                )

                if is_anomalous:
                    lines.extend(
                        [
                            "WARNING: This figure exceeds normal thresholds.",
                            "Impact excluded from annual estimate until verified.",
                            "",
                        ]
                    )

                lines.extend(
                    [
                        "What this means:",
                        f"- {items_found:,} items show negative quantities (sold without being received)",
                        "- Financial records likely do not reflect true COGS",
                        "- Tax and margin calculations may be materially misstated",
                        "",
                        "Recommended Actions:",
                        "1. Conduct physical inventory count",
                        "2. Reconcile receiving records against vendor invoices",
                        "3. Consult accountant regarding COGS and tax implications",
                        "",
                        "=" * 40,
                        "",
                    ]
                )
                break

        # Add root cause analysis if available
        for result in results:
            cause_diagnosis = result.get("cause_diagnosis")
            if cause_diagnosis and cause_diagnosis.get("top_cause"):
                top_cause = cause_diagnosis.get("top_cause", "Unknown")
                confidence = cause_diagnosis.get("confidence", 0) * 100
                cause_details = cause_diagnosis.get("cause_details", {})
                severity = cause_details.get("severity", "info")
                description = cause_details.get("description", "")
                recommendations = cause_details.get("recommendations", [])

                lines.extend(
                    [
                        "ROOT CAUSE ANALYSIS (VSA-Grounded)",
                        "-" * 35,
                        f"Primary Cause: {top_cause.replace('_', ' ').title()}",
                        f"Confidence: {confidence:.0f}%",
                        f"Severity: {severity.upper()}",
                        "",
                        f"Description: {description}",
                        "",
                    ]
                )

                if recommendations:
                    lines.append("Targeted Actions:")
                    for i, rec in enumerate(recommendations[:4], 1):
                        lines.append(f"  {i}. {rec}")
                    lines.append("")

                lines.append("=" * 40)
                lines.append("")
                break

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
                item_details = data.get("item_details", [])
                items = data.get("top_items", [])[:5]
                count = data.get("count", len(items))

                if count == 0:
                    continue

                title = leak_type.replace("_", " ").title()
                lines.append(f"\n{title} ({count} items):")

                # Use item_details if available for richer context
                if item_details:
                    for detail in item_details[:5]:
                        sku = detail.get("sku", "Unknown")
                        qty = detail.get("quantity", 0)
                        cost = detail.get("cost", 0)
                        sold = detail.get("sold", 0)
                        context = detail.get("context", "")
                        lines.append(f"  - {sku}")
                        lines.append(
                            f"    QOH: {qty:.0f} | Cost: ${cost:.2f} | Sold: {sold:.0f}"
                        )
                        if context:
                            lines.append(f"    -> {context}")
                else:
                    for item in items:
                        lines.append(f"  - {item}")

        lines.extend(
            [
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
            ]
        )

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

    async def send_diagnostic_report(
        self,
        to_email: str,
        pdf_path: str,
        store_name: str,
        summary: dict | None = None,
    ) -> dict[str, Any]:
        """
        Send diagnostic report email with PDF attachment.

        Args:
            to_email: Recipient email address
            pdf_path: Path to the generated PDF report
            store_name: Name of the store for the report
            summary: Optional summary data to include in email body

        Returns:
            Dict with success status and message ID
        """
        if not self.is_configured:
            logger.warning("Email service not configured, skipping send")
            return {"success": False, "error": "Email service not configured"}

        # Read PDF file
        try:
            import base64

            with open(pdf_path, "rb") as f:
                pdf_content = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}")
            return {"success": False, "error": f"Failed to read PDF: {str(e)}"}

        # Generate email content
        html_content = self._generate_diagnostic_email_html(store_name, summary)
        text_content = self._generate_diagnostic_email_text(store_name, summary)

        subject = f"Profit Sentinel Diagnostic Report - {store_name}"

        try:
            if self.provider == "resend":
                return await self._send_diagnostic_via_resend(
                    to_email,
                    subject,
                    html_content,
                    text_content,
                    pdf_content,
                    store_name,
                )
            elif self.provider == "sendgrid":
                return await self._send_diagnostic_via_sendgrid(
                    to_email,
                    subject,
                    html_content,
                    text_content,
                    pdf_content,
                    store_name,
                )
        except Exception as e:
            logger.error(f"Failed to send diagnostic email: {e}")
            return {"success": False, "error": str(e)}

        return {"success": False, "error": "No email provider configured"}

    async def _send_diagnostic_via_resend(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
        pdf_content: str,
        store_name: str,
    ) -> dict[str, Any]:
        """Send diagnostic email via Resend API with PDF attachment."""
        try:
            import resend

            resend.api_key = self.resend_api_key

            safe_store_name = store_name.lower().replace(" ", "_")
            filename = f"profit_sentinel_report_{safe_store_name}.pdf"

            response = resend.Emails.send(
                {
                    "from": f"{self.from_name} <{self.from_email}>",
                    "to": [to_email],
                    "subject": subject,
                    "html": html_content,
                    "text": text_content,
                    "attachments": [
                        {
                            "filename": filename,
                            "content": pdf_content,
                        }
                    ],
                    "headers": {
                        "List-Unsubscribe": f"<mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe {to_email}>"
                    },
                }
            )

            # Log success without PII (no email address)
            logger.info(f"Diagnostic email sent via Resend, ID: {response.get('id')}")
            return {"success": True, "message_id": response.get("id")}

        except ImportError:
            logger.error("Resend package not installed: pip install resend")
            return {"success": False, "error": "Resend package not installed"}
        except Exception as e:
            logger.error(f"Resend API error: {e}")
            return {"success": False, "error": str(e)}

    async def _send_diagnostic_via_sendgrid(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
        pdf_content: str,
        store_name: str,
    ) -> dict[str, Any]:
        """Send diagnostic email via SendGrid API with PDF attachment."""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import (
                Attachment,
                Content,
                Disposition,
                Email,
                FileContent,
                FileName,
                FileType,
                Header,
                Mail,
                To,
            )

            sg = SendGridAPIClient(self.sendgrid_api_key)

            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=To(to_email),
                subject=subject,
            )
            message.add_content(Content("text/plain", text_content))
            message.add_content(Content("text/html", html_content))

            # Add PDF attachment
            safe_store_name = store_name.lower().replace(" ", "_")
            filename = f"profit_sentinel_report_{safe_store_name}.pdf"

            attachment = Attachment(
                FileContent(pdf_content),
                FileName(filename),
                FileType("application/pdf"),
                Disposition("attachment"),
            )
            message.add_attachment(attachment)

            # Add unsubscribe header
            message.add_header(
                Header(
                    "List-Unsubscribe",
                    f"<mailto:unsubscribe@profitsentinel.com?subject=Unsubscribe {to_email}>",
                )
            )

            response = sg.send(message)

            # Log success without PII (no email address)
            logger.info(
                f"Diagnostic email sent via SendGrid, status: {response.status_code}"
            )
            return {
                "success": response.status_code in [200, 201, 202],
                "status_code": response.status_code,
            }

        except ImportError:
            logger.error("SendGrid package not installed: pip install sendgrid")
            return {"success": False, "error": "SendGrid package not installed"}
        except Exception as e:
            logger.error(f"SendGrid API error: {e}")
            return {"success": False, "error": str(e)}

    def _generate_diagnostic_email_html(
        self, store_name: str, summary: dict | None = None
    ) -> str:
        """Generate HTML email content for diagnostic report."""
        summary_section = ""
        if summary:
            total_shrinkage = summary.get("total_shrinkage", 0)
            explained = summary.get("explained_value", 0)
            unexplained = summary.get("unexplained_value", 0)
            reduction = summary.get("reduction_percent", 0)

            summary_section = f"""
            <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05)); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 25px; margin-bottom: 30px;">
                <h2 style="color: #10b981; margin: 0 0 20px 0; font-size: 20px;">Diagnostic Summary</h2>
                <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
                    <div style="text-align: center; min-width: 100px;">
                        <p style="color: #ef4444; font-size: 24px; font-weight: bold; margin: 0;">${total_shrinkage:,.0f}</p>
                        <p style="color: #94a3b8; font-size: 12px; margin: 5px 0 0 0;">Apparent Shrinkage</p>
                    </div>
                    <div style="text-align: center; min-width: 100px;">
                        <p style="color: #10b981; font-size: 24px; font-weight: bold; margin: 0;">${explained:,.0f}</p>
                        <p style="color: #94a3b8; font-size: 12px; margin: 5px 0 0 0;">Explained</p>
                    </div>
                    <div style="text-align: center; min-width: 100px;">
                        <p style="color: #f97316; font-size: 24px; font-weight: bold; margin: 0;">${unexplained:,.0f}</p>
                        <p style="color: #94a3b8; font-size: 12px; margin: 5px 0 0 0;">To Investigate</p>
                    </div>
                    <div style="text-align: center; min-width: 100px;">
                        <p style="color: #10b981; font-size: 24px; font-weight: bold; margin: 0;">{reduction:.1f}%</p>
                        <p style="color: #94a3b8; font-size: 12px; margin: 5px 0 0 0;">Reduction</p>
                    </div>
                </div>
            </div>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Profit Sentinel Diagnostic Report</title>
</head>
<body style="margin: 0; padding: 0; background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
        <div style="text-align: center; margin-bottom: 40px;">
            <h1 style="color: #10b981; margin: 0; font-size: 28px;">Profit Sentinel</h1>
            <p style="color: #64748b; margin: 10px 0 0 0; font-size: 14px;">Diagnostic Report for {store_name}</p>
        </div>

        {summary_section}

        <div style="background: #1e293b; border-radius: 16px; padding: 25px; margin-bottom: 30px;">
            <h2 style="color: #f1f5f9; margin: 0 0 15px 0; font-size: 18px;">Your Report is Attached</h2>
            <p style="color: #94a3b8; font-size: 14px; line-height: 1.6; margin: 0;">
                Your comprehensive diagnostic report is attached to this email as a PDF file.
                The report includes:
            </p>
            <ul style="color: #cbd5e1; padding-left: 20px; margin: 15px 0 0 0;">
                <li style="margin-bottom: 8px;">Executive summary with financial impact</li>
                <li style="margin-bottom: 8px;">Pattern-by-pattern breakdown of findings</li>
                <li style="margin-bottom: 8px;">Complete SKU listing for verification</li>
                <li style="margin-bottom: 8px;">Recommended next steps</li>
            </ul>
        </div>

        <div style="text-align: center; padding-top: 30px; border-top: 1px solid #334155;">
            <p style="color: #64748b; font-size: 12px; margin: 0 0 10px 0;">
                This report was generated by Profit Sentinel.
            </p>
            <p style="color: #334155; font-size: 10px; margin: 15px 0 0 0;">
                Profit Sentinel - Protecting retail margins with AI-powered forensic analysis.
            </p>
        </div>
    </div>
</body>
</html>
        """

    def _generate_diagnostic_email_text(
        self, store_name: str, summary: dict | None = None
    ) -> str:
        """Generate plain text email content for diagnostic report."""
        lines = [
            "PROFIT SENTINEL - Diagnostic Report",
            f"Store: {store_name}",
            "=" * 40,
            "",
        ]

        if summary:
            total_shrinkage = summary.get("total_shrinkage", 0)
            explained = summary.get("explained_value", 0)
            unexplained = summary.get("unexplained_value", 0)
            reduction = summary.get("reduction_percent", 0)

            lines.extend(
                [
                    "SUMMARY",
                    "-" * 20,
                    f"Apparent Shrinkage: ${total_shrinkage:,.0f}",
                    f"Explained: ${explained:,.0f}",
                    f"To Investigate: ${unexplained:,.0f}",
                    f"Reduction: {reduction:.1f}%",
                    "",
                ]
            )

        lines.extend(
            [
                "Your comprehensive diagnostic report is attached to this email as a PDF.",
                "",
                "The report includes:",
                "- Executive summary with financial impact",
                "- Pattern-by-pattern breakdown of findings",
                "- Complete SKU listing for verification",
                "- Recommended next steps",
                "",
                "---",
                "This report was generated by Profit Sentinel.",
            ]
        )

        return "\n".join(lines)


# Singleton instance
_email_service: EmailService | None = None


def get_email_service() -> EmailService:
    """Get or create email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
