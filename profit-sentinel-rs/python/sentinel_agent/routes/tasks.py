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
from ..dual_auth import UserContext
from ..llm_layer import render_task_for_manager
from .state import AppState


def create_tasks_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["tasks"])

    def _get_user_tasks(user_id: str) -> dict[str, TaskResponse]:
        """Get or create the per-user task store."""
        if user_id not in state.task_store:
            state.task_store[user_id] = {}
        return state.task_store[user_id]

    @router.post(
        "/delegate",
        response_model=DelegateResponse,
    )
    async def delegate_issue(
        body: DelegateRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> DelegateResponse:
        """Create a delegated task from a pipeline issue."""
        issue = state.find_issue(ctx.user_id, body.issue_id)

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
        user_tasks = _get_user_tasks(ctx.user_id)
        user_tasks[task.task_id] = task_resp

        return DelegateResponse(
            task=task,
            rendered_text=rendered,
            task_id=task.task_id,
        )

    @router.get(
        "/tasks",
        response_model=TaskListResponse,
    )
    async def list_tasks(
        store_id: str | None = Query(default=None),
        priority: str | None = Query(default=None),
        status: str | None = Query(default=None),
        ctx: UserContext = Depends(require_auth),
    ) -> TaskListResponse:
        """List delegated tasks with optional filtering."""
        user_tasks = _get_user_tasks(ctx.user_id)
        tasks = list(user_tasks.values())

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
    )
    async def get_task(
        task_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> TaskResponse:
        """Get a single task by ID (owner check: returns 404 if not yours)."""
        user_tasks = _get_user_tasks(ctx.user_id)
        task_resp = user_tasks.get(task_id)
        if not task_resp:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return task_resp

    @router.patch(
        "/tasks/{task_id}",
        response_model=TaskResponse,
    )
    async def update_task(
        task_id: str,
        body: TaskStatusUpdate,
        ctx: UserContext = Depends(require_auth),
    ) -> TaskResponse:
        """Update a task's status (owner check: returns 404 if not yours)."""
        user_tasks = _get_user_tasks(ctx.user_id)
        task_resp = user_tasks.get(task_id)
        if not task_resp:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

        task_resp.status = body.status
        if body.notes:
            task_resp.notes.append(body.notes)

        user_tasks[task_id] = task_resp
        return task_resp

    @router.delete(
        "/tasks/{task_id}",
        status_code=204,
    )
    async def delete_task(
        task_id: str,
        ctx: UserContext = Depends(require_auth),
    ) -> None:
        """Delete a task (owner check: returns 404 if not yours)."""
        user_tasks = _get_user_tasks(ctx.user_id)
        if task_id not in user_tasks:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        del user_tasks[task_id]

    return router
