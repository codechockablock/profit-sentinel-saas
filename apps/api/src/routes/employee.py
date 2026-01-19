"""
Employee API Endpoints

Provides endpoints for employee management and stats:
- GET /employee/{employee_id} - Get employee profile
- GET /employee/{employee_id}/stats - Get employee activity stats
- GET /employee/{employee_id}/corrections - Get correction history
- POST /employee/register - Register new employee
"""

import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_current_user, get_supabase_client

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class EmployeeRegisterRequest(BaseModel):
    """Request to register a new employee."""
    employee_id: str = Field(..., min_length=1)
    store_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = None
    role: str = Field(default="associate", pattern="^(associate|manager|admin)$")


class EmployeeResponse(BaseModel):
    """Employee profile response."""
    employee_id: str
    store_id: str
    name: str
    email: Optional[str] = None
    role: str
    created_at: str
    is_active: bool = True


class EmployeeStatsResponse(BaseModel):
    """Employee activity statistics."""
    employee_id: str
    total_assists: int
    total_corrections: int
    corrections_accepted: int
    accuracy_rate: float  # % of corrections that were accurate
    categories_helped: List[str]
    last_activity: Optional[str] = None


class CorrectionHistoryItem(BaseModel):
    """Single correction in history."""
    correction_id: str
    problem_id: str
    original_category: str
    corrected_category: str
    correction_notes: Optional[str] = None
    created_at: str
    was_accepted: bool = True


class CorrectionHistoryResponse(BaseModel):
    """Response for correction history."""
    employee_id: str
    corrections: List[CorrectionHistoryItem]
    total_count: int


# =============================================================================
# IN-MEMORY STORAGE (would be database in production)
# =============================================================================

_employees: dict[str, dict] = {}
_employee_stats: dict[str, dict] = {}
_corrections: dict[str, List[dict]] = {}


# =============================================================================
# ENDPOINTS
# =============================================================================

# Health endpoint MUST come before /{employee_id} routes to avoid being matched as an ID
@router.get("/health")
async def employee_health():
    """Health check for employee service."""
    return {
        "status": "healthy",
        "employees_registered": len(_employees),
    }


@router.post("/register", response_model=EmployeeResponse)
async def register_employee(
    request: EmployeeRegisterRequest,
    user_id: Optional[str] = Depends(get_current_user),
):
    """
    Register a new employee in the system.
    """
    if request.employee_id in _employees:
        raise HTTPException(status_code=400, detail="Employee already exists")

    now = datetime.utcnow()

    employee = {
        "employee_id": request.employee_id,
        "store_id": request.store_id,
        "name": request.name,
        "email": request.email,
        "role": request.role,
        "created_at": now.isoformat(),
        "is_active": True,
    }

    _employees[request.employee_id] = employee

    # Initialize stats
    _employee_stats[request.employee_id] = {
        "total_assists": 0,
        "total_corrections": 0,
        "corrections_accepted": 0,
        "categories_helped": [],
        "last_activity": None,
    }

    _corrections[request.employee_id] = []

    return EmployeeResponse(**employee)


@router.get("/{employee_id}", response_model=EmployeeResponse)
async def get_employee(
    employee_id: str,
    user_id: Optional[str] = Depends(get_current_user),
):
    """
    Get employee profile by ID.
    """
    if employee_id not in _employees:
        raise HTTPException(status_code=404, detail="Employee not found")

    return EmployeeResponse(**_employees[employee_id])


@router.get("/{employee_id}/stats", response_model=EmployeeStatsResponse)
async def get_employee_stats(
    employee_id: str,
    user_id: Optional[str] = Depends(get_current_user),
):
    """
    Get employee activity statistics.
    """
    if employee_id not in _employees:
        raise HTTPException(status_code=404, detail="Employee not found")

    stats = _employee_stats.get(employee_id, {
        "total_assists": 0,
        "total_corrections": 0,
        "corrections_accepted": 0,
        "categories_helped": [],
        "last_activity": None,
    })

    # Calculate accuracy rate
    total = stats["total_corrections"]
    accepted = stats["corrections_accepted"]
    accuracy = (accepted / total * 100) if total > 0 else 0.0

    return EmployeeStatsResponse(
        employee_id=employee_id,
        total_assists=stats["total_assists"],
        total_corrections=stats["total_corrections"],
        corrections_accepted=stats["corrections_accepted"],
        accuracy_rate=accuracy,
        categories_helped=stats["categories_helped"],
        last_activity=stats["last_activity"],
    )


@router.get("/{employee_id}/corrections", response_model=CorrectionHistoryResponse)
async def get_correction_history(
    employee_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: Optional[str] = Depends(get_current_user),
):
    """
    Get employee's correction history.
    """
    if employee_id not in _employees:
        raise HTTPException(status_code=404, detail="Employee not found")

    all_corrections = _corrections.get(employee_id, [])
    total = len(all_corrections)

    # Paginate
    paginated = all_corrections[offset:offset + limit]

    items = [CorrectionHistoryItem(**c) for c in paginated]

    return CorrectionHistoryResponse(
        employee_id=employee_id,
        corrections=items,
        total_count=total,
    )


@router.put("/{employee_id}/deactivate")
async def deactivate_employee(
    employee_id: str,
    user_id: Optional[str] = Depends(get_current_user),
):
    """
    Deactivate an employee account.
    """
    if employee_id not in _employees:
        raise HTTPException(status_code=404, detail="Employee not found")

    _employees[employee_id]["is_active"] = False

    return {"status": "deactivated", "employee_id": employee_id}


# =============================================================================
# INTERNAL HELPERS (called by other routes)
# =============================================================================

def record_assist(employee_id: str, category_slug: str):
    """Record an assist for stats tracking."""
    if employee_id in _employee_stats:
        stats = _employee_stats[employee_id]
        stats["total_assists"] += 1
        if category_slug not in stats["categories_helped"]:
            stats["categories_helped"].append(category_slug)
        stats["last_activity"] = datetime.utcnow().isoformat()


def record_correction(
    employee_id: str,
    correction_id: str,
    problem_id: str,
    original_category: str,
    corrected_category: str,
    notes: Optional[str] = None,
):
    """Record a correction for stats and history."""
    if employee_id in _employee_stats:
        stats = _employee_stats[employee_id]
        stats["total_corrections"] += 1
        stats["corrections_accepted"] += 1
        if corrected_category not in stats["categories_helped"]:
            stats["categories_helped"].append(corrected_category)
        stats["last_activity"] = datetime.utcnow().isoformat()

    if employee_id in _corrections:
        _corrections[employee_id].append({
            "correction_id": correction_id,
            "problem_id": problem_id,
            "original_category": original_category,
            "corrected_category": corrected_category,
            "correction_notes": notes,
            "created_at": datetime.utcnow().isoformat(),
            "was_accepted": True,
        })
