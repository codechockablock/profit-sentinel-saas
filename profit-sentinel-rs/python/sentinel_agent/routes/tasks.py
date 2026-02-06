"""Task delegation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ..api_models import (
    DelegateRequest,
    DelegateResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    TaskStatusUpdate,
)
from ..llm_layer import render_task_for_manager
from .state import AppState


def create_tasks_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["tasks"])

    @router.post(
        "/delegate",
        response_model=DelegateResponse,
        dependencies=[Depends(require_auth)],
    )
    async def delegate_issue(body: DelegateRequest) -> DelegateResponse:
        """Create a delegated task from a pipeline issue."""
        issue = state.find_issue(body.issue_id)

        task = state.delegation_mgr.create_task(
            issue,
            assignee=body.assignee,
            deadline=body.deadline,
        )

        rendered = render_task_for_manager(task)

        notes = [body.notes] if body.notes else []
        task_resp = TaskResponse(
            task=task,
            status=TaskStatus.OPEN,
            rendered_text=rendered,
            notes=notes,
        )
        state.task_store[task.task_id] = task_resp

        return DelegateResponse(
            task=task,
            rendered_text=rendered,
            task_id=task.task_id,
        )

    @router.get(
        "/tasks",
        response_model=TaskListResponse,
        dependencies=[Depends(require_auth)],
    )
    async def list_tasks(
        store_id: str | None = Query(default=None),
        priority: str | None = Query(default=None),
        status: str | None = Query(default=None),
    ) -> TaskListResponse:
        """List delegated tasks with optional filtering."""
        tasks = list(state.task_store.values())

        if store_id:
            tasks = [t for t in tasks if t.task.store_id == store_id]
        if priority:
            tasks = [t for t in tasks if t.task.priority.value == priority]
        if status:
            tasks = [t for t in tasks if t.status.value == status]

        return TaskListResponse(tasks=tasks, total=len(tasks))

    @router.get(
        "/tasks/{task_id}",
        response_model=TaskResponse,
        dependencies=[Depends(require_auth)],
    )
    async def get_task(task_id: str) -> TaskResponse:
        """Get a single task by ID."""
        task_resp = state.task_store.get(task_id)
        if not task_resp:
            raise HTTPException(
                status_code=404, detail=f"Task '{task_id}' not found"
            )
        return task_resp

    @router.patch(
        "/tasks/{task_id}",
        response_model=TaskResponse,
        dependencies=[Depends(require_auth)],
    )
    async def update_task(
        task_id: str,
        body: TaskStatusUpdate,
    ) -> TaskResponse:
        """Update a task's status (complete, escalate, etc.)."""
        task_resp = state.task_store.get(task_id)
        if not task_resp:
            raise HTTPException(
                status_code=404, detail=f"Task '{task_id}' not found"
            )

        task_resp.status = body.status
        if body.notes:
            task_resp.notes.append(body.notes)

        state.task_store[task_id] = task_resp
        return task_resp

    return router
