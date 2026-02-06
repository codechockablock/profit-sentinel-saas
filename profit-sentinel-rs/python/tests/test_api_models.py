"""Tests for API request/response models."""

from datetime import datetime

import pytest
from sentinel_agent.api_models import (
    CoopReportResponse,
    DelegateRequest,
    DelegateResponse,
    DigestResponse,
    ErrorResponse,
    HealthResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
    VendorCallResponse,
)
from sentinel_agent.coop_models import CoopIntelligenceReport
from sentinel_agent.models import (
    CallPrep,
    Digest,
    Issue,
    IssueType,
    Sku,
    Summary,
    Task,
    TaskPriority,
    TrendDirection,
)


def _make_sku() -> Sku:
    return Sku(
        sku_id="PNT-1001",
        qty_on_hand=50,
        unit_cost=100,
        retail_price=150,
        margin_pct=0.33,
        sales_last_30d=10,
        days_since_receipt=30,
        is_damaged=False,
        on_order_qty=0,
        is_seasonal=False,
    )


def _make_issue() -> Issue:
    return Issue(
        id="store-7-DeadStock-001",
        issue_type=IssueType.DEAD_STOCK,
        store_id="store-7",
        dollar_impact=5000,
        confidence=0.89,
        trend_direction=TrendDirection.WORSENING,
        priority_score=11.37,
        urgency_score=0.9,
        detection_timestamp="2026-01-02T00:00:00Z",
        skus=[_make_sku()],
        context="Dead stock in paint category.",
    )


def _make_digest() -> Digest:
    return Digest(
        generated_at="2026-02-05T06:00:00+00:00",
        store_filter=["store-7"],
        pipeline_ms=10,
        issues=[_make_issue()],
        summary=Summary(
            total_issues=1,
            total_dollar_impact=5000,
            stores_affected=1,
            records_processed=10,
            issues_detected=1,
            issues_filtered_out=0,
        ),
    )


def _make_task() -> Task:
    return Task(
        task_id="task-abc12345",
        issue_id="store-7-DeadStock-001",
        issue_type=IssueType.DEAD_STOCK,
        store_id="store-7",
        assignee="Store 7 Manager",
        deadline=datetime(2026, 2, 10),
        priority=TaskPriority.HIGH,
        title="Review dead inventory - Store 7 ($5,000)",
        description="Dead stock needs attention.",
        action_items=["Pull SKUs", "Create markdown plan"],
        dollar_impact=5000,
        skus=[_make_sku()],
    )


class TestErrorResponse:
    def test_basic_error(self):
        err = ErrorResponse(code="NOT_FOUND", message="Resource not found")
        assert err.code == "NOT_FOUND"
        assert err.detail is None

    def test_error_with_detail(self):
        err = ErrorResponse(
            code="PIPELINE_ERROR",
            message="Pipeline failed",
            detail="Binary not found",
        )
        assert err.detail == "Binary not found"

    def test_serialization(self):
        err = ErrorResponse(code="ERR", message="msg")
        data = err.model_dump()
        assert data["code"] == "ERR"
        roundtrip = ErrorResponse.model_validate(data)
        assert roundtrip.code == "ERR"


class TestTaskStatus:
    def test_all_statuses(self):
        assert TaskStatus.OPEN.value == "open"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.ESCALATED.value == "escalated"


class TestDelegateRequest:
    def test_minimal_request(self):
        req = DelegateRequest(
            issue_id="store-7-DeadStock-001",
            assignee="Store Manager",
        )
        assert req.deadline is None
        assert req.notes is None

    def test_full_request(self):
        req = DelegateRequest(
            issue_id="store-7-DeadStock-001",
            assignee="Store Manager",
            deadline=datetime(2026, 2, 15),
            notes="Urgent - handle today",
        )
        assert req.deadline.year == 2026
        assert req.notes == "Urgent - handle today"


class TestTaskStatusUpdate:
    def test_complete(self):
        update = TaskStatusUpdate(status=TaskStatus.COMPLETED)
        assert update.status == TaskStatus.COMPLETED
        assert update.notes is None

    def test_escalate_with_notes(self):
        update = TaskStatusUpdate(
            status=TaskStatus.ESCALATED,
            notes="Need regional manager approval",
        )
        assert update.status == TaskStatus.ESCALATED
        assert "regional" in update.notes


class TestDigestResponse:
    def test_full_response(self):
        digest = _make_digest()
        resp = DigestResponse(
            digest=digest,
            rendered_text="Good morning. 1 item needs attention.",
            generated_at=digest.generated_at,
            store_ids=["store-7"],
            issue_count=1,
            total_dollar_impact=5000,
        )
        assert resp.issue_count == 1
        assert resp.total_dollar_impact == 5000
        assert "Good morning" in resp.rendered_text

    def test_serialization_roundtrip(self):
        digest = _make_digest()
        resp = DigestResponse(
            digest=digest,
            rendered_text="test",
            generated_at=digest.generated_at,
            store_ids=["store-7"],
            issue_count=1,
            total_dollar_impact=5000,
        )
        data = resp.model_dump()
        roundtrip = DigestResponse.model_validate(data)
        assert roundtrip.issue_count == 1


class TestDelegateResponse:
    def test_response(self):
        task = _make_task()
        resp = DelegateResponse(
            task=task,
            rendered_text="TASK: Review dead inventory",
            task_id=task.task_id,
        )
        assert resp.task_id == "task-abc12345"


class TestTaskResponse:
    def test_defaults(self):
        task = _make_task()
        resp = TaskResponse(
            task=task,
            rendered_text="Rendered text",
        )
        assert resp.status == TaskStatus.OPEN
        assert resp.notes == []

    def test_with_status_and_notes(self):
        task = _make_task()
        resp = TaskResponse(
            task=task,
            status=TaskStatus.COMPLETED,
            rendered_text="Done",
            notes=["Completed by manager"],
        )
        assert resp.status == TaskStatus.COMPLETED
        assert len(resp.notes) == 1


class TestTaskListResponse:
    def test_empty_list(self):
        resp = TaskListResponse(tasks=[], total=0)
        assert resp.total == 0

    def test_with_tasks(self):
        task = _make_task()
        task_resp = TaskResponse(task=task, rendered_text="text")
        resp = TaskListResponse(tasks=[task_resp], total=1)
        assert resp.total == 1
        assert resp.tasks[0].task.task_id == "task-abc12345"


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(binary_found=True, binary_path="/usr/bin/sentinel")
        assert resp.status == "ok"
        assert resp.version == "0.13.0"
        assert resp.dev_mode is False

    def test_degraded(self):
        resp = HealthResponse(
            status="degraded",
            binary_found=False,
            dev_mode=True,
        )
        assert resp.status == "degraded"
        assert resp.binary_path is None


class TestCoopReportResponse:
    def test_basic(self):
        report = CoopIntelligenceReport(
            store_id="store-7",
            total_opportunity=5000,
        )
        resp = CoopReportResponse(
            report=report,
            rendered_text="CO-OP INTELLIGENCE REPORT",
            total_opportunity=5000,
        )
        assert resp.total_opportunity == 5000
        assert resp.health_summary is None
        assert resp.rebate_statuses == []
