"""
tests/api/test_employee_routes.py - Tests for Employee API Routes

Tests the employee API endpoints:
- POST /employee/register
- GET /employee/{employee_id}
- GET /employee/{employee_id}/stats
- GET /employee/{employee_id}/corrections
- PUT /employee/{employee_id}/deactivate
- GET /employee/health
"""
import pytest
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# =============================================================================
# MINIMAL MODELS (copied for isolation)
# =============================================================================

class EmployeeRegisterRequest(BaseModel):
    employee_id: str = Field(..., min_length=1)
    store_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = None
    role: str = Field(default="associate", pattern="^(associate|manager|admin)$")


class EmployeeResponse(BaseModel):
    employee_id: str
    store_id: str
    name: str
    email: Optional[str] = None
    role: str
    created_at: str
    is_active: bool = True


class EmployeeStatsResponse(BaseModel):
    employee_id: str
    total_assists: int
    total_corrections: int
    corrections_accepted: int
    accuracy_rate: float
    categories_helped: List[str]
    last_activity: Optional[str] = None


class CorrectionHistoryItem(BaseModel):
    correction_id: str
    problem_id: str
    original_category: str
    corrected_category: str
    correction_notes: Optional[str] = None
    created_at: str
    was_accepted: bool = True


class CorrectionHistoryResponse(BaseModel):
    employee_id: str
    corrections: List[CorrectionHistoryItem]
    total_count: int


# =============================================================================
# IN-MEMORY STORAGE
# =============================================================================

_employees: dict[str, dict] = {}
_employee_stats: dict[str, dict] = {}
_corrections: dict[str, List[dict]] = {}


def reset_storage():
    """Reset storage for test isolation."""
    global _employees, _employee_stats, _corrections
    _employees = {}
    _employee_stats = {}
    _corrections = {}


# =============================================================================
# CREATE TEST APP
# =============================================================================

def create_test_app():
    """Create minimal FastAPI app with employee routes for testing."""
    app = FastAPI()

    # Health endpoint MUST come before /{employee_id} to avoid being matched as an ID
    @app.get("/employee/health")
    async def employee_health():
        return {
            "status": "healthy",
            "employees_registered": len(_employees),
        }

    @app.post("/employee/register", response_model=EmployeeResponse)
    async def register_employee(request: EmployeeRegisterRequest):
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
        _employee_stats[request.employee_id] = {
            "total_assists": 0,
            "total_corrections": 0,
            "corrections_accepted": 0,
            "categories_helped": [],
            "last_activity": None,
        }
        _corrections[request.employee_id] = []

        return EmployeeResponse(**employee)

    @app.get("/employee/{employee_id}", response_model=EmployeeResponse)
    async def get_employee(employee_id: str):
        if employee_id not in _employees:
            raise HTTPException(status_code=404, detail="Employee not found")
        return EmployeeResponse(**_employees[employee_id])

    @app.get("/employee/{employee_id}/stats", response_model=EmployeeStatsResponse)
    async def get_employee_stats(employee_id: str):
        if employee_id not in _employees:
            raise HTTPException(status_code=404, detail="Employee not found")

        stats = _employee_stats.get(employee_id, {
            "total_assists": 0,
            "total_corrections": 0,
            "corrections_accepted": 0,
            "categories_helped": [],
            "last_activity": None,
        })

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

    @app.get("/employee/{employee_id}/corrections", response_model=CorrectionHistoryResponse)
    async def get_correction_history(
        employee_id: str,
        limit: int = Query(50, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ):
        if employee_id not in _employees:
            raise HTTPException(status_code=404, detail="Employee not found")

        all_corrections = _corrections.get(employee_id, [])
        total = len(all_corrections)
        paginated = all_corrections[offset:offset + limit]
        items = [CorrectionHistoryItem(**c) for c in paginated]

        return CorrectionHistoryResponse(
            employee_id=employee_id,
            corrections=items,
            total_count=total,
        )

    @app.put("/employee/{employee_id}/deactivate")
    async def deactivate_employee(employee_id: str):
        if employee_id not in _employees:
            raise HTTPException(status_code=404, detail="Employee not found")
        _employees[employee_id]["is_active"] = False
        return {"status": "deactivated", "employee_id": employee_id}

    return app


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Test client with fresh storage."""
    reset_storage()
    app = create_test_app()
    return TestClient(app)


@pytest.fixture
def registered_employee(client):
    """Register an employee and return the response."""
    response = client.post("/employee/register", json={
        "employee_id": "emp-001",
        "store_id": "store-001",
        "name": "John Doe",
        "email": "john@example.com",
        "role": "associate",
    })
    return response.json()


# =============================================================================
# REGISTER ENDPOINT TESTS
# =============================================================================

class TestRegisterEndpoint:
    """Tests for POST /employee/register."""

    def test_register_employee(self, client):
        """Registration should create new employee."""
        response = client.post("/employee/register", json={
            "employee_id": "emp-001",
            "store_id": "store-001",
            "name": "John Doe",
            "role": "associate",
        })

        assert response.status_code == 200
        data = response.json()

        assert data["employee_id"] == "emp-001"
        assert data["store_id"] == "store-001"
        assert data["name"] == "John Doe"
        assert data["role"] == "associate"
        assert data["is_active"] is True
        assert "created_at" in data

    def test_register_with_email(self, client):
        """Registration with email should work."""
        response = client.post("/employee/register", json={
            "employee_id": "emp-002",
            "store_id": "store-001",
            "name": "Jane Smith",
            "email": "jane@example.com",
            "role": "manager",
        })

        assert response.status_code == 200
        assert response.json()["email"] == "jane@example.com"

    def test_register_duplicate_fails(self, client, registered_employee):
        """Duplicate registration should fail."""
        response = client.post("/employee/register", json={
            "employee_id": "emp-001",
            "store_id": "store-001",
            "name": "Another John",
            "role": "associate",
        })

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_register_missing_fields_fails(self, client):
        """Missing required fields should fail."""
        response = client.post("/employee/register", json={
            "employee_id": "emp-001",
        })

        assert response.status_code == 422

    def test_register_invalid_role_fails(self, client):
        """Invalid role should fail validation."""
        response = client.post("/employee/register", json={
            "employee_id": "emp-001",
            "store_id": "store-001",
            "name": "Test",
            "role": "invalid_role",
        })

        assert response.status_code == 422


# =============================================================================
# GET EMPLOYEE ENDPOINT TESTS
# =============================================================================

class TestGetEmployeeEndpoint:
    """Tests for GET /employee/{employee_id}."""

    def test_get_employee(self, client, registered_employee):
        """Should return employee profile."""
        response = client.get("/employee/emp-001")

        assert response.status_code == 200
        data = response.json()

        assert data["employee_id"] == "emp-001"
        assert data["name"] == "John Doe"

    def test_get_nonexistent_employee(self, client):
        """Should return 404 for nonexistent employee."""
        response = client.get("/employee/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# =============================================================================
# STATS ENDPOINT TESTS
# =============================================================================

class TestStatsEndpoint:
    """Tests for GET /employee/{employee_id}/stats."""

    def test_get_stats_new_employee(self, client, registered_employee):
        """New employee should have zero stats."""
        response = client.get("/employee/emp-001/stats")

        assert response.status_code == 200
        data = response.json()

        assert data["employee_id"] == "emp-001"
        assert data["total_assists"] == 0
        assert data["total_corrections"] == 0
        assert data["accuracy_rate"] == 0.0
        assert data["categories_helped"] == []
        assert data["last_activity"] is None

    def test_get_stats_nonexistent_employee(self, client):
        """Should return 404 for nonexistent employee."""
        response = client.get("/employee/nonexistent/stats")

        assert response.status_code == 404


# =============================================================================
# CORRECTIONS ENDPOINT TESTS
# =============================================================================

class TestCorrectionsEndpoint:
    """Tests for GET /employee/{employee_id}/corrections."""

    def test_get_corrections_empty(self, client, registered_employee):
        """New employee should have empty corrections."""
        response = client.get("/employee/emp-001/corrections")

        assert response.status_code == 200
        data = response.json()

        assert data["employee_id"] == "emp-001"
        assert data["corrections"] == []
        assert data["total_count"] == 0

    def test_get_corrections_nonexistent(self, client):
        """Should return 404 for nonexistent employee."""
        response = client.get("/employee/nonexistent/corrections")

        assert response.status_code == 404

    def test_corrections_pagination(self, client, registered_employee):
        """Pagination parameters should be accepted."""
        response = client.get("/employee/emp-001/corrections?limit=5&offset=0")

        assert response.status_code == 200


# =============================================================================
# DEACTIVATE ENDPOINT TESTS
# =============================================================================

class TestDeactivateEndpoint:
    """Tests for PUT /employee/{employee_id}/deactivate."""

    def test_deactivate_employee(self, client, registered_employee):
        """Should deactivate employee."""
        response = client.put("/employee/emp-001/deactivate")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "deactivated"
        assert data["employee_id"] == "emp-001"

        # Verify employee is deactivated
        get_response = client.get("/employee/emp-001")
        assert get_response.json()["is_active"] is False

    def test_deactivate_nonexistent(self, client):
        """Should return 404 for nonexistent employee."""
        response = client.put("/employee/nonexistent/deactivate")

        assert response.status_code == 404


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

class TestHealthEndpoint:
    """Tests for GET /employee/health."""

    def test_health_check(self, client):
        """Health check should return status."""
        response = client.get("/employee/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "employees_registered" in data

    def test_health_shows_count(self, client, registered_employee):
        """Health should show employee count."""
        response = client.get("/employee/health")

        assert response.status_code == 200
        assert response.json()["employees_registered"] == 1


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestEmployeeIntegration:
    """End-to-end integration tests."""

    def test_full_employee_lifecycle(self, client):
        """Test complete employee lifecycle."""
        # Step 1: Register
        register_response = client.post("/employee/register", json={
            "employee_id": "emp-lifecycle",
            "store_id": "store-001",
            "name": "Lifecycle Test",
            "role": "associate",
        })
        assert register_response.status_code == 200

        # Step 2: Get profile
        get_response = client.get("/employee/emp-lifecycle")
        assert get_response.status_code == 200
        assert get_response.json()["is_active"] is True

        # Step 3: Check stats
        stats_response = client.get("/employee/emp-lifecycle/stats")
        assert stats_response.status_code == 200
        assert stats_response.json()["total_assists"] == 0

        # Step 4: Check corrections
        corrections_response = client.get("/employee/emp-lifecycle/corrections")
        assert corrections_response.status_code == 200
        assert corrections_response.json()["total_count"] == 0

        # Step 5: Deactivate
        deactivate_response = client.put("/employee/emp-lifecycle/deactivate")
        assert deactivate_response.status_code == 200

        # Step 6: Verify deactivated
        final_response = client.get("/employee/emp-lifecycle")
        assert final_response.json()["is_active"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
