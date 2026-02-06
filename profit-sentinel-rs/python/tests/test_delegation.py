"""Tests for the DelegationManager."""

import json
from datetime import datetime, timedelta

import pytest
from sentinel_agent.delegation import DelegationManager
from sentinel_agent.models import Digest, IssueType, TaskPriority, TrendDirection

SAMPLE_JSON = json.dumps(
    {
        "generated_at": "2026-02-05T06:00:00+00:00",
        "store_filter": ["store-7"],
        "pipeline_ms": 10,
        "issues": [
            {
                "id": "store-7-DeadStock-001",
                "issue_type": "DeadStock",
                "store_id": "store-7",
                "dollar_impact": 5000.0,
                "confidence": 0.89,
                "trend_direction": "Worsening",
                "priority_score": 11.37,
                "urgency_score": 0.9,
                "detection_timestamp": "2025-01-02T00:00:00Z",
                "skus": [
                    {
                        "sku_id": "SEA-1201",
                        "qty_on_hand": 100.0,
                        "unit_cost": 50.0,
                        "retail_price": 67.5,
                        "margin_pct": 0.35,
                        "sales_last_30d": 0.0,
                        "days_since_receipt": 180.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    }
                ],
                "context": "Dead stock in store 7.",
            },
            {
                "id": "store-7-NegativeInventory-002",
                "issue_type": "NegativeInventory",
                "store_id": "store-7",
                "dollar_impact": 1104.5,
                "confidence": 0.87,
                "trend_direction": "Stable",
                "priority_score": 4.45,
                "urgency_score": 0.5,
                "detection_timestamp": "2025-01-02T00:00:00Z",
                "skus": [
                    {
                        "sku_id": "ELC-4401",
                        "qty_on_hand": -47.0,
                        "unit_cost": 23.5,
                        "retail_price": 31.73,
                        "margin_pct": 0.35,
                        "sales_last_30d": 10.0,
                        "days_since_receipt": 30.0,
                        "is_damaged": False,
                        "on_order_qty": 0.0,
                        "is_seasonal": False,
                    }
                ],
                "context": "Negative inventory.",
            },
        ],
        "summary": {
            "total_issues": 2,
            "total_dollar_impact": 6104.5,
            "stores_affected": 1,
            "records_processed": 10,
            "issues_detected": 2,
            "issues_filtered_out": 0,
        },
    }
)


class TestDelegationManager:
    def setup_method(self):
        self.dm = DelegationManager()
        self.digest = Digest.model_validate_json(SAMPLE_JSON)

    def test_create_task_basic(self):
        issue = self.digest.issues[0]
        task = self.dm.create_task(issue, assignee="Store 7 Manager")

        assert task.issue_id == "store-7-DeadStock-001"
        assert task.store_id == "store-7"
        assert task.assignee == "Store 7 Manager"
        assert task.dollar_impact == 5000.0
        assert len(task.skus) == 1

    def test_task_id_generated(self):
        issue = self.digest.issues[0]
        task = self.dm.create_task(issue, assignee="Manager")
        assert task.task_id.startswith("task-")
        assert len(task.task_id) > 5

    def test_critical_priority_for_high_score(self):
        """Score 11.37 >= 10.0 should be CRITICAL."""
        issue = self.digest.issues[0]
        task = self.dm.create_task(issue, assignee="Manager")
        assert task.priority == TaskPriority.CRITICAL

    def test_low_priority_for_low_score(self):
        """Score 4.45 < 5.0 should be LOW."""
        issue = self.digest.issues[1]
        task = self.dm.create_task(issue, assignee="Manager")
        assert task.priority == TaskPriority.LOW

    def test_auto_deadline_critical(self):
        """Critical + worsening should be ~1 day (tightened)."""
        issue = self.digest.issues[0]  # CRITICAL + Worsening
        task = self.dm.create_task(issue, assignee="Manager")
        # Deadline should be within 1 day (worsening tightens from 1 to... still 1)
        delta = task.deadline - datetime.now()
        assert delta.days <= 1

    def test_auto_deadline_low(self):
        """Low priority + stable should be ~14 days."""
        issue = self.digest.issues[1]  # LOW + Stable
        task = self.dm.create_task(issue, assignee="Manager")
        delta = task.deadline - datetime.now()
        assert 12 <= delta.days <= 15

    def test_explicit_deadline(self):
        issue = self.digest.issues[0]
        explicit = datetime(2026, 3, 1)
        task = self.dm.create_task(issue, assignee="Manager", deadline=explicit)
        assert task.deadline == explicit

    def test_title_contains_action_and_store(self):
        issue = self.digest.issues[0]
        task = self.dm.create_task(issue, assignee="Manager")
        assert "Store 7" in task.title
        assert "$5,000" in task.title
        assert "dead" in task.title.lower() or "markdown" in task.title.lower()

    def test_description_not_empty(self):
        for issue in self.digest.issues:
            task = self.dm.create_task(issue, assignee="Manager")
            assert task.description, f"Empty description for {issue.issue_type}"

    def test_action_items_not_empty(self):
        for issue in self.digest.issues:
            task = self.dm.create_task(issue, assignee="Manager")
            assert task.action_items, f"No action items for {issue.issue_type}"
            assert len(task.action_items) >= 3

    def test_negative_inventory_action_items(self):
        issue = self.digest.issues[1]  # NegativeInventory
        task = self.dm.create_task(issue, assignee="Manager")
        all_items = " ".join(task.action_items).lower()
        assert "count" in all_items  # physical count
        assert "receiving" in all_items or "reconcile" in all_items

    def test_dead_stock_action_items(self):
        issue = self.digest.issues[0]  # DeadStock
        task = self.dm.create_task(issue, assignee="Manager")
        all_items = " ".join(task.action_items).lower()
        assert "markdown" in all_items or "return" in all_items

    def test_format_for_store_manager(self):
        issue = self.digest.issues[0]
        task = self.dm.create_task(issue, assignee="Store 7 Manager")
        text = self.dm.format_for_store_manager(task)

        assert "TASK:" in text
        assert "Priority:" in text
        assert "Due:" in text
        assert "Store 7" in text
        assert "$5,000" in text
        assert "SEA-1201" in text
        assert "Action Items:" in text

    def test_format_for_executive(self):
        issue = self.digest.issues[0]
        task = self.dm.create_task(issue, assignee="Store 7 Manager")
        text = self.dm.format_for_executive(task)

        assert "ASSIGNED" in text
        assert "Store 7 Manager" in text
        assert "CRITICAL" in text
        assert "$5,000" in text

    def test_manager_format_shows_sku_details(self):
        issue = self.digest.issues[1]  # Negative inventory
        task = self.dm.create_task(issue, assignee="Manager")
        text = self.dm.format_for_store_manager(task)

        assert "ELC-4401" in text
        assert "47" in text  # units short
        assert "$23" in text or "$24" in text  # unit cost
