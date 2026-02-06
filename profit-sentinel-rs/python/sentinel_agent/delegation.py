"""Delegation Manager.

Creates task packages from pipeline issues and formats them for
different audiences (executive summary vs store manager action items).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from .llm_layer import format_dollars, render_task_for_manager
from .models import Issue, IssueType, Task, TaskPriority, TrendDirection


class DelegationManager:
    """Create and manage delegated tasks from pipeline issues.

    Usage:
        dm = DelegationManager()
        task = dm.create_task(issue, assignee="Store 7 Manager")
        print(dm.format_for_store_manager(task))
    """

    def create_task(
        self,
        issue: Issue,
        assignee: str,
        deadline: datetime | None = None,
    ) -> Task:
        """Create a task from an issue with full context for the assignee.

        Args:
            issue: Pipeline issue to delegate.
            assignee: Name or role of the person receiving the task.
            deadline: Optional explicit deadline. If None, auto-calculated
                     based on issue urgency.

        Returns:
            Task with title, description, action items, and SKU details.
        """
        priority = self._determine_priority(issue)

        if deadline is None:
            deadline = self._auto_deadline(issue, priority)

        title = self._build_title(issue)
        description = self._build_description(issue)
        action_items = self._build_action_items(issue)

        # Enrich with root cause recommendations if available
        if issue.root_cause is not None:
            rc_recs = issue.root_cause.recommendations
            # Add first 2 root-cause-specific recommendations that aren't
            # already covered by the issue-type action items
            for rec in rc_recs[:2]:
                if not any(rec.lower() in item.lower() for item in action_items):
                    action_items.append(f"[{issue.root_cause.display_name}] {rec}")

        return Task(
            task_id=f"task-{uuid4().hex[:8]}",
            issue_id=issue.id,
            issue_type=issue.issue_type,
            store_id=issue.store_id,
            assignee=assignee,
            deadline=deadline,
            priority=priority,
            title=title,
            description=description,
            action_items=action_items,
            dollar_impact=issue.dollar_impact,
            skus=issue.skus,
        )

    def format_for_store_manager(self, task: Task) -> str:
        """Format a task for store manager consumption.

        Different framing than executive view: specific action items,
        bin locations, count lists.
        """
        return render_task_for_manager(task)

    def format_for_executive(self, task: Task) -> str:
        """Format a task summary for executive tracking.

        Brief status-oriented view suitable for a dashboard.
        """
        status = "ASSIGNED"
        return (
            f"[{status}] {task.title}\n"
            f"  Assigned to: {task.assignee}\n"
            f"  Due: {task.deadline.strftime('%b %d')}\n"
            f"  Impact: {format_dollars(task.dollar_impact)} | "
            f"Priority: {task.priority.value.upper()}"
        )

    def _determine_priority(self, issue: Issue) -> TaskPriority:
        """Map issue characteristics to task priority."""
        if issue.priority_score >= 10.0:
            return TaskPriority.CRITICAL
        if issue.priority_score >= 8.0:
            return TaskPriority.HIGH
        if issue.priority_score >= 5.0:
            return TaskPriority.MEDIUM
        return TaskPriority.LOW

    def _auto_deadline(self, issue: Issue, priority: TaskPriority) -> datetime:
        """Calculate deadline based on priority and trend."""
        base = datetime.now()

        days = {
            TaskPriority.CRITICAL: 1,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 7,
            TaskPriority.LOW: 14,
        }[priority]

        # Worsening trends get tighter deadlines
        if issue.trend_direction == TrendDirection.WORSENING:
            days = max(1, days - 1)

        return base + timedelta(days=days)

    def _build_title(self, issue: Issue) -> str:
        """Build a clear, actionable task title."""
        store_label = issue.store_id
        if store_label.startswith("store-"):
            store_label = f"Store {store_label[6:]}"

        return (
            f"{issue.issue_type.action_verb} — "
            f"{store_label} ({format_dollars(issue.dollar_impact)})"
        )

    def _build_description(self, issue: Issue) -> str:
        """Build task description with full context."""
        lines: list[str] = []

        match issue.issue_type:
            case IssueType.NEGATIVE_INVENTORY:
                short_count = sum(
                    abs(s.qty_on_hand) for s in issue.skus if s.qty_on_hand < 0
                )
                lines.append(
                    f"System shows {int(short_count)} units with negative on-hand "
                    f"across {issue.sku_count} SKU{'s' if issue.sku_count > 1 else ''}."
                )
                lines.append(
                    "This typically indicates shrinkage, receiving errors, or "
                    "system sync issues."
                )
                lines.append(
                    "Please conduct a physical count of the affected items "
                    "and reconcile with system records."
                )

            case IssueType.DEAD_STOCK:
                total_value = sum(s.qty_on_hand * s.unit_cost for s in issue.skus)
                lines.append(
                    f"{format_dollars(total_value)} in inventory with "
                    f"zero sales in the past 30 days."
                )
                lines.append(
                    "Review each item for markdown, return-to-vendor, "
                    "or transfer to a higher-velocity location."
                )

            case IssueType.MARGIN_EROSION:
                avg_margin = (
                    sum(s.margin_pct for s in issue.skus) / len(issue.skus)
                    if issue.skus
                    else 0
                )
                lines.append(
                    f"Margins averaging {avg_margin * 100:.0f}% on these items, "
                    f"well below our 35% benchmark."
                )
                lines.append(
                    "Verify retail prices are current and check for "
                    "unauthorized discounting."
                )
                lines.append(
                    "If pricing is correct, escalate to purchasing for "
                    "vendor cost review."
                )

            case IssueType.VENDOR_SHORT_SHIP:
                lines.append(
                    "Damaged goods received from vendor. "
                    "Document damage with photos and file claim."
                )
                on_order = sum(s.on_order_qty for s in issue.skus)
                if on_order > 0:
                    lines.append(
                        f"Note: {int(on_order)} additional units on order. "
                        f"Inspect carefully on receipt."
                    )

            case IssueType.PATRONAGE_MISS:
                lines.append(
                    "Seasonal inventory past its sales window. "
                    "Every day held increases carrying cost."
                )
                lines.append(
                    "Markdown to move quickly, or arrange return to vendor "
                    "if return policy permits."
                )

            case IssueType.PURCHASING_LEAKAGE:
                lines.append(
                    "High-cost items at below-benchmark margins suggest "
                    "we're paying too much."
                )
                lines.append(
                    "Review purchase orders and compare to Do It Best "
                    "recommended pricing."
                )

            case IssueType.RECEIVING_GAP:
                lines.append(
                    "Pricing data anomaly detected. "
                    "Items show negative retail price in system."
                )
                lines.append(
                    "Correct retail pricing and audit the receiving process "
                    "for data entry errors."
                )

            case IssueType.SHRINKAGE_PATTERN:
                total_value = sum(abs(s.qty_on_hand) * s.unit_cost for s in issue.skus)
                lines.append(
                    f"{format_dollars(total_value)} in inventory with "
                    f"suspiciously low margins and minimal sales."
                )
                lines.append(
                    "This pattern may indicate theft, unrecorded damage, "
                    "or vendor billing errors."
                )
                lines.append("Conduct physical counts and compare to system records.")

            case IssueType.ZERO_COST_ANOMALY:
                selling = sum(1 for s in issue.skus if s.sales_last_30d > 0)
                lines.append(
                    f"{issue.sku_count} item{'s' if issue.sku_count > 1 else ''} "
                    f"show $0 cost in the system."
                )
                if selling > 0:
                    lines.append(
                        f"{selling} of these are actively selling. "
                        f"Margin and profitability data is invalid until corrected."
                    )
                lines.append(
                    "Update cost data from purchase orders or vendor invoices."
                )

            case IssueType.PRICE_DISCREPANCY:
                lines.append("Items are priced below cost — every sale is a loss.")
                lines.append(
                    "Verify retail pricing is correct in the POS system. "
                    "If pricing is intentional (clearance), document the reason."
                )

            case IssueType.OVERSTOCK:
                total_excess = sum(max(0, int(s.qty_on_hand) - 30) for s in issue.skus)
                lines.append(
                    f"{total_excess} excess units above optimal stock levels "
                    f"across {issue.sku_count} SKU{'s' if issue.sku_count > 1 else ''}."
                )
                lines.append(
                    "Reduce upcoming orders and evaluate transfers to "
                    "higher-velocity locations."
                )

        # Add root cause context if available
        if issue.root_cause is not None:
            conf = issue.root_cause_confidence or 0.0
            lines.append("")
            lines.append(
                f"Likely root cause: {issue.root_cause.display_name} "
                f"({conf:.0%} confidence)."
            )

        return "\n".join(lines)

    def _build_action_items(self, issue: Issue) -> list[str]:
        """Build specific action items for the assignee."""
        items: list[str] = []

        match issue.issue_type:
            case IssueType.NEGATIVE_INVENTORY:
                items.append("Physical count of affected SKUs")
                items.append("Check receiving logs for the past 7 days")
                items.append("Review POS transactions for potential shrinkage")
                items.append("Reconcile system counts and submit corrections")
                items.append("Report findings to operations")

            case IssueType.DEAD_STOCK:
                items.append("Pull affected SKUs for visual inspection")
                items.append("Check if any items are damaged or expired")
                items.append("Identify candidates for markdown (suggest pricing)")
                items.append("Check return-to-vendor eligibility")
                items.append("Create markdown or disposal plan")

            case IssueType.MARGIN_EROSION:
                items.append("Verify current retail prices match planogram")
                items.append("Check for unauthorized local discounts")
                items.append("Compare to Do It Best suggested retail")
                items.append("Escalate to purchasing if vendor cost is the issue")

            case IssueType.VENDOR_SHORT_SHIP:
                items.append("Photograph all damaged goods")
                items.append("Separate damaged inventory from sellable stock")
                items.append("File vendor damage claim with documentation")
                items.append("Note damage on any incoming PO receipts")

            case IssueType.PATRONAGE_MISS:
                items.append("Identify remaining seasonal inventory")
                items.append("Calculate markdown pricing to clear by end of season")
                items.append("Check vendor return policy and deadlines")
                items.append("Move to clearance endcap or promotional display")

            case IssueType.PURCHASING_LEAKAGE:
                items.append("Pull purchase orders for affected SKUs")
                items.append("Compare unit costs to Do It Best catalog pricing")
                items.append("Document price variance for purchasing review")
                items.append("Request updated vendor quotes")

            case IssueType.RECEIVING_GAP:
                items.append("Correct retail pricing in POS system")
                items.append("Audit recent receiving entries for data errors")
                items.append("Verify shelf labels match system pricing")

            case IssueType.SHRINKAGE_PATTERN:
                items.append("Physical count of affected SKUs")
                items.append("Compare counts to system records")
                items.append("Review security footage for high-value items")
                items.append("Check for unrecorded damages or returns")
                items.append("Report variance findings to loss prevention")

            case IssueType.ZERO_COST_ANOMALY:
                items.append("Look up vendor invoices for affected SKUs")
                items.append("Update unit cost in POS system")
                items.append("Verify margin calculations after correction")
                items.append("Flag for purchasing review if cost unknown")

            case IssueType.PRICE_DISCREPANCY:
                items.append("Verify retail price in POS matches shelf tags")
                items.append("Compare to Do It Best suggested retail")
                items.append("Correct pricing or document clearance reason")
                items.append("Review vendor cost to confirm it's current")

            case IssueType.OVERSTOCK:
                items.append("Review upcoming purchase orders and reduce quantities")
                items.append("Identify items for transfer to other locations")
                items.append("Evaluate markdown opportunities for slow movers")
                items.append("Check if vendor return is an option")

        return items
