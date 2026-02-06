"""Vendor Call Assistant.

Prepares call briefs for vendor-related issues (VendorShortShip,
PurchasingLeakage). Generates talking points, questions, and
historical context from the structured pipeline data.
"""

from __future__ import annotations

from .llm_layer import format_dollars, render_call_prep
from .models import CallPrep, Issue, IssueType

# ---------------------------------------------------------------------------
# Vendor catalog (stub — would come from a database in production)
# ---------------------------------------------------------------------------

_VENDOR_CATALOG: dict[str, dict] = {
    "DMG": {
        "name": "Martin's Supply Co.",
        "contact": "Regional Rep: Dave Martin",
        "history": "3 quality issues in the past 6 months. Last claim filed 2025-11-15.",
    },
    "ELC": {
        "name": "National Electrical Distributors",
        "contact": "Account Manager: Sarah Chen",
        "history": "Generally reliable. One short-ship last quarter.",
    },
    "PNT": {
        "name": "ColorMax Paint Supply",
        "contact": "Sales: Jim Rodriguez",
        "history": "Pricing has drifted above market over the past 2 quarters.",
    },
    "PLB": {
        "name": "ProPlumb Wholesale",
        "contact": "Account Manager: Linda Park",
        "history": "Good relationship. Volume discount expires next month.",
    },
    "SEA": {
        "name": "SeaCoast Hardware",
        "contact": "Regional Rep: Tom Burke",
        "history": "No recent issues.",
    },
    "FLR": {
        "name": "FloorCraft Distribution",
        "contact": "Account Manager: Maria Santos",
        "history": "Long lead times. Consider alternate supplier for fast-moving items.",
    },
    "HRD": {
        "name": "HardLine Tools & Fasteners",
        "contact": "Sales: Keith Wong",
        "history": "Solid vendor. No recent issues.",
    },
    "TLS": {
        "name": "ToolSource National",
        "contact": "Account Manager: Gary Fletcher",
        "history": "Good pricing but occasional quality concerns on imports.",
    },
    "SSN": {
        "name": "Seasonal Products Inc.",
        "contact": "Sales: Amy Brooks",
        "history": "Standard return policy: 30 days post-season for unsold seasonal goods.",
    },
    "NRM": {
        "name": "Various / Normal Stock",
        "contact": "N/A",
        "history": "No vendor-specific issues.",
    },
}


def _lookup_vendor(sku_id: str) -> dict:
    """Look up vendor info from SKU prefix."""
    prefix = sku_id.split("-")[0] if "-" in sku_id else sku_id[:3]
    return _VENDOR_CATALOG.get(
        prefix,
        {
            "name": f"Unknown Vendor ({prefix})",
            "contact": "N/A",
            "history": "No historical data available.",
        },
    )


class VendorCallAssistant:
    """Prepare vendor call briefs from pipeline issues.

    Usage:
        assistant = VendorCallAssistant()
        prep = assistant.prepare_call(issue)
        print(assistant.render(prep))
    """

    def prepare_call(self, issue: Issue) -> CallPrep:
        """Generate a call preparation package for a vendor-related issue.

        Works best with VendorShortShip and PurchasingLeakage issues,
        but can prepare a brief for any issue type where vendor contact
        would be valuable.

        Args:
            issue: An Issue from the pipeline digest.

        Returns:
            CallPrep with vendor info, talking points, and questions.
        """
        # Look up vendor from the first SKU
        vendor_info = _lookup_vendor(issue.skus[0].sku_id if issue.skus else "")

        talking_points = self._build_talking_points(issue, vendor_info)
        questions = self._build_questions(issue)

        return CallPrep(
            issue_id=issue.id,
            store_id=issue.store_id,
            vendor_name=vendor_info["name"],
            issue_summary=self._build_summary(issue),
            affected_skus=issue.skus,
            total_dollar_impact=issue.dollar_impact,
            talking_points=talking_points,
            questions_to_ask=questions,
            historical_context=vendor_info.get("history", ""),
        )

    def render(self, prep: CallPrep) -> str:
        """Render a CallPrep into formatted text."""
        return render_call_prep(prep)

    def _build_summary(self, issue: Issue) -> str:
        """Build a one-line issue summary for the vendor call."""
        match issue.issue_type:
            case IssueType.VENDOR_SHORT_SHIP:
                damaged_count = sum(1 for s in issue.skus if s.is_damaged)
                on_order = sum(s.on_order_qty for s in issue.skus)
                parts = []
                if damaged_count:
                    parts.append(
                        f"{damaged_count} SKU{'s' if damaged_count > 1 else ''} "
                        f"received damaged"
                    )
                if on_order > 0:
                    parts.append(f"{int(on_order)} units currently on order")
                return (
                    ". ".join(parts)
                    + f". Total exposure: {format_dollars(issue.dollar_impact)}."
                )

            case IssueType.PURCHASING_LEAKAGE:
                return (
                    f"{issue.sku_count} high-cost SKU{'s' if issue.sku_count > 1 else ''} "
                    f"at below-benchmark margins. "
                    f"Estimated overpayment: {format_dollars(issue.dollar_impact)}."
                )

            case IssueType.MARGIN_EROSION:
                avg_margin = (
                    sum(s.margin_pct for s in issue.skus) / len(issue.skus)
                    if issue.skus
                    else 0
                )
                return (
                    f"{issue.sku_count} SKU{'s' if issue.sku_count > 1 else ''} "
                    f"averaging {avg_margin * 100:.0f}% margin vs 35% benchmark. "
                    f"Profit gap: {format_dollars(issue.dollar_impact)}."
                )

            case _:
                return (
                    f"{issue.issue_type.display_name} issue affecting "
                    f"{issue.sku_count} SKU{'s' if issue.sku_count > 1 else ''} "
                    f"with {format_dollars(issue.dollar_impact)} exposure."
                )

    def _build_talking_points(self, issue: Issue, vendor: dict) -> list[str]:
        """Generate talking points for the vendor call."""
        points: list[str] = []

        match issue.issue_type:
            case IssueType.VENDOR_SHORT_SHIP:
                points.append(
                    f"We received damaged goods on recent shipment "
                    f"({issue.sku_count} SKU{'s' if issue.sku_count > 1 else ''})."
                )
                total_damaged_value = sum(
                    s.qty_on_hand * s.unit_cost for s in issue.skus if s.is_damaged
                )
                if total_damaged_value > 0:
                    points.append(
                        f"Value of damaged inventory: "
                        f"{format_dollars(total_damaged_value)}."
                    )
                on_order_skus = [s for s in issue.skus if s.on_order_qty > 0]
                if on_order_skus:
                    points.append(
                        f"We have {int(sum(s.on_order_qty for s in on_order_skus))} "
                        f"units currently on order — need assurance on quality."
                    )
                points.append("Request credit or replacement for damaged goods.")

            case IssueType.PURCHASING_LEAKAGE:
                points.append(
                    f"Current pricing puts us at below-benchmark margins "
                    f"on {issue.sku_count} item{'s' if issue.sku_count > 1 else ''}."
                )
                points.append(
                    f"Our volume should qualify for better terms. "
                    f"Estimated gap: {format_dollars(issue.dollar_impact)}."
                )
                points.append("Request updated pricing schedule or volume rebate.")

            case IssueType.MARGIN_EROSION:
                points.append(
                    f"Margins on {issue.sku_count} item{'s' if issue.sku_count > 1 else ''} "
                    f"have fallen below our 35% benchmark."
                )
                points.append(
                    f"Combined profit gap: {format_dollars(issue.dollar_impact)}."
                )
                points.append(
                    "Need either a cost reduction or authorization to adjust retail."
                )

            case _:
                points.append(
                    f"Issue type: {issue.issue_type.display_name}. "
                    f"Total exposure: {format_dollars(issue.dollar_impact)}."
                )
                points.append(
                    "Request information on resolution timeline and next steps."
                )

        return points

    def _build_questions(self, issue: Issue) -> list[str]:
        """Generate questions to ask during the vendor call."""
        questions: list[str] = []

        match issue.issue_type:
            case IssueType.VENDOR_SHORT_SHIP:
                questions.append(
                    "What is your process for handling damaged goods claims?"
                )
                questions.append("Can you expedite replacements for the damaged items?")
                questions.append("What quality controls are in place for shipping?")
                if any(s.on_order_qty > 0 for s in issue.skus):
                    questions.append("Can you confirm quality on our pending order?")

            case IssueType.PURCHASING_LEAKAGE:
                questions.append("What volume tiers are available for better pricing?")
                questions.append(
                    "Are there quarterly rebate programs we should explore?"
                )
                questions.append("What are competitors paying for similar items?")

            case IssueType.MARGIN_EROSION:
                questions.append("Has your cost basis changed recently?")
                questions.append(
                    "Are there alternative product lines with better margins?"
                )
                questions.append("Can we negotiate promotional pricing support?")

            case _:
                questions.append("What is the expected resolution timeline?")
                questions.append("What can we do to prevent this issue going forward?")

        return questions
