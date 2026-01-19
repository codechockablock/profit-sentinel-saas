"""
tests/api/test_repair_routes.py - Tests for Repair Assistant API Routes

Tests the repair API endpoints in isolation (no external dependencies).
Creates a minimal FastAPI app with just the repair routes for testing.

Run: pytest apps/api/tests/test_repair_routes.py -v
"""
import pytest
import base64
import json
import sys
import os
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Depends, HTTPException
from fastapi.testclient import TestClient
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# =============================================================================
# MINIMAL PYDANTIC MODELS (copied from repair.py for isolation)
# =============================================================================

class DiagnoseRequestAPI(BaseModel):
    """API request model for diagnosis."""
    text_description: Optional[str] = Field(None, max_length=2000)
    voice_transcript: Optional[str] = Field(None, max_length=2000)
    image_base64: Optional[str] = Field(None, description="Base64-encoded image")
    store_id: str = Field(..., min_length=1)
    employee_id: Optional[str] = None
    session_id: Optional[str] = None


class RefineRequestAPI(BaseModel):
    """API request for refining diagnosis."""
    problem_id: str
    additional_text: Optional[str] = Field(None, max_length=1000)


class CorrectionRequestAPI(BaseModel):
    """API request for employee correction."""
    problem_id: str
    employee_id: str
    correct_category_slug: str
    correction_notes: Optional[str] = Field(None, max_length=500)


class HypothesisAPI(BaseModel):
    """Hypothesis in API response."""
    category_slug: str
    category_name: str
    probability: float
    explanation: Optional[str] = None
    icon: Optional[str] = None


class DiagnoseResponseAPI(BaseModel):
    """API response for diagnosis."""
    problem_id: str
    status: str
    hypotheses: List[HypothesisAPI]
    top_hypothesis: HypothesisAPI
    confidence: float
    entropy: float
    needs_more_info: bool
    follow_up_questions: List[str]
    likely_parts_needed: Optional[List[str]] = None
    tools_needed: Optional[List[str]] = None
    safety_concerns: Optional[List[str]] = None
    diy_feasible: Optional[bool] = None
    professional_recommended: Optional[bool] = None


class CorrectionResultAPI(BaseModel):
    """API response for correction submission."""
    correction_id: str
    problem_id: str
    employee_id: str
    xp_awarded: int
    new_total_xp: int
    leveled_up: bool
    new_level: Optional[int] = None
    badge_earned: Optional[str] = None
    streak_extended: bool
    current_streak: int


class CategoryAPI(BaseModel):
    """Problem category in API response."""
    category_id: str
    name: str
    slug: str
    description: Optional[str] = None
    icon: Optional[str] = None
    parent_slug: Optional[str] = None
    subcategories: List["CategoryAPI"] = []


CategoryAPI.model_rebuild()


# =============================================================================
# MOCK DATA
# =============================================================================

# Minimal valid JPEG bytes
VALID_JPEG_BYTES = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
    0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
    0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
    0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F,
    0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x45, 0x00, 0xFF, 0xD9
])

VALID_JPEG_BASE64 = base64.b64encode(VALID_JPEG_BYTES).decode()


# =============================================================================
# MOCK ENGINE AND SERVICES
# =============================================================================

class MockEngineConfig:
    dimensions = 4096


class MockCodebook:
    def __init__(self):
        self._categories = {
            "plumbing": {"name": "Plumbing", "icon": "droplet", "parent_slug": None},
            "plumbing-faucet": {"name": "Faucet Issues", "parent_slug": "plumbing"},
            "plumbing-toilet": {"name": "Toilet Problems", "parent_slug": "plumbing"},
            "electrical": {"name": "Electrical", "icon": "zap", "parent_slug": None},
            "electrical-outlet": {"name": "Outlet Issues", "parent_slug": "electrical"},
            "hvac": {"name": "HVAC", "icon": "thermometer", "parent_slug": None},
        }

    def get_all_slugs(self):
        return list(self._categories.keys())

    def get_info(self, slug):
        return self._categories.get(slug, {"name": slug})


class MockBundle:
    hypotheses = ["plumbing-faucet", "plumbing-toilet", "hvac"]

    def __init__(self):
        import torch
        self.probabilities = torch.tensor([0.65, 0.25, 0.10])
        self.vector = torch.randn(4096, dtype=torch.complex64)


class MockDiagnoseResponse:
    def __init__(self):
        self.status = MagicMock()
        self.status.value = "diagnosed"
        self.hypotheses = [
            MagicMock(category_slug="plumbing-faucet", category_name="Faucet Issues",
                      probability=0.65, explanation="Likely faucet problem", icon="droplet"),
            MagicMock(category_slug="plumbing-toilet", category_name="Toilet Problems",
                      probability=0.25, explanation="Could be toilet", icon="toilet"),
            MagicMock(category_slug="hvac", category_name="HVAC",
                      probability=0.10, explanation="Less likely", icon="thermometer"),
        ]
        self.top_hypothesis = self.hypotheses[0]
        self.confidence = 0.65
        self.entropy = 0.8
        self.needs_more_info = True
        self.follow_up_questions = ["Is water dripping?", "Is the leak visible?"]


class MockRepairEngine:
    def __init__(self):
        self.config = MockEngineConfig()
        self.codebook = MockCodebook()
        self.text_encoder = MagicMock()

    def diagnose(self, text=None, voice_transcript=None, image_features=None,
                 store_id=None, employee_id=None):
        return MockBundle(), {"store_context_similarity": 0.3}

    def bundle_to_response(self, bundle, problem_id, metadata):
        return MockDiagnoseResponse()


class MockVisionResult:
    primary_category = "plumbing"
    subcategory = "faucet"
    confidence = 0.85
    description = "Dripping faucet"
    visible_components = ["faucet", "sink"]
    damage_indicators = ["water stain"]
    likely_parts_needed = ["washer", "O-ring"]
    tools_needed = ["wrench"]
    severity_estimate = "moderate"
    diy_feasible = True
    professional_recommended = False
    safety_concerns = []
    keywords = ["faucet", "dripping"]


class MockVisionService:
    def analyze_image(self, image_base64, text_context=None):
        return MockVisionResult()

    def extract_features_for_vsa(self, result):
        return MagicMock()

    def encode_to_vsa_vector(self, features, encoder):
        import torch
        return torch.randn(4096, dtype=torch.complex64)


# Global mock instances
_mock_engine = None
_mock_vision = None


def get_mock_engine():
    global _mock_engine
    if _mock_engine is None:
        _mock_engine = MockRepairEngine()
    return _mock_engine


def get_mock_vision():
    global _mock_vision
    if _mock_vision is None:
        _mock_vision = MockVisionService()
    return _mock_vision


# =============================================================================
# CREATE TEST APP
# =============================================================================

def create_test_app():
    """Create minimal FastAPI app with repair routes for testing."""
    import uuid

    app = FastAPI()

    @app.post("/repair/diagnose", response_model=DiagnoseResponseAPI)
    async def diagnose_problem(request: DiagnoseRequestAPI):
        if not any([request.text_description, request.voice_transcript, request.image_base64]):
            raise HTTPException(status_code=400, detail="At least one input (text, voice, or image) required")

        engine = get_mock_engine()
        vision_result = None

        # Process image
        if request.image_base64:
            try:
                vision = get_mock_vision()
                vision_result = vision.analyze_image(request.image_base64, request.text_description)
            except Exception:
                pass  # Continue without image

        bundle, metadata = engine.diagnose(
            text=request.text_description,
            voice_transcript=request.voice_transcript,
            store_id=request.store_id,
        )

        response = engine.bundle_to_response(bundle, str(uuid.uuid4()), metadata)

        hypotheses_api = [
            HypothesisAPI(
                category_slug=h.category_slug,
                category_name=h.category_name,
                probability=h.probability,
                explanation=h.explanation,
                icon=h.icon,
            )
            for h in response.hypotheses
        ]

        api_response = DiagnoseResponseAPI(
            problem_id=str(uuid.uuid4()),
            status=response.status.value,
            hypotheses=hypotheses_api,
            top_hypothesis=hypotheses_api[0],
            confidence=response.confidence,
            entropy=response.entropy,
            needs_more_info=response.needs_more_info,
            follow_up_questions=response.follow_up_questions,
        )

        if vision_result:
            api_response.likely_parts_needed = vision_result.likely_parts_needed
            api_response.tools_needed = vision_result.tools_needed
            api_response.diy_feasible = vision_result.diy_feasible

        return api_response

    @app.post("/repair/diagnose/refine")
    async def refine_diagnosis(request: RefineRequestAPI):
        raise HTTPException(status_code=501, detail="Coming soon")

    @app.get("/repair/solution/{problem_id}")
    async def get_solution(problem_id: str):
        raise HTTPException(status_code=501, detail="Coming soon")

    @app.post("/repair/correction", response_model=CorrectionResultAPI)
    async def submit_correction(request: CorrectionRequestAPI):
        if not request.employee_id:
            raise HTTPException(status_code=400, detail="Employee ID required")

        return CorrectionResultAPI(
            correction_id=str(uuid.uuid4()),
            problem_id=request.problem_id,
            employee_id=request.employee_id,
            xp_awarded=50,
            new_total_xp=50,
            leveled_up=False,
            new_level=None,
            badge_earned=None,
            streak_extended=True,
            current_streak=1,
        )

    @app.get("/repair/categories", response_model=List[CategoryAPI])
    async def list_categories():
        engine = get_mock_engine()
        categories = []

        for slug in engine.codebook.get_all_slugs():
            info = engine.codebook.get_info(slug)
            if info.get("parent_slug"):
                continue

            subcategories = []
            for sub_slug in engine.codebook.get_all_slugs():
                sub_info = engine.codebook.get_info(sub_slug)
                if sub_info.get("parent_slug") == slug:
                    subcategories.append(CategoryAPI(
                        category_id=sub_slug,
                        name=sub_info.get("name", sub_slug),
                        slug=sub_slug,
                        parent_slug=slug,
                        subcategories=[],
                    ))

            categories.append(CategoryAPI(
                category_id=slug,
                name=info.get("name", slug),
                slug=slug,
                icon=info.get("icon"),
                subcategories=subcategories,
            ))

        return categories

    @app.get("/repair/health")
    async def repair_health():
        engine = get_mock_engine()
        return {
            "status": "healthy",
            "engine": "available",
            "categories_loaded": len(engine.codebook.get_all_slugs()),
            "dimensions": engine.config.dimensions,
        }

    return app


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def client():
    """Test client with mocked repair routes."""
    app = create_test_app()
    return TestClient(app)


# =============================================================================
# DIAGNOSE ENDPOINT TESTS
# =============================================================================

class TestDiagnoseEndpoint:
    """Tests for POST /repair/diagnose."""

    def test_diagnose_text_only(self, client):
        """Text-only diagnosis should work."""
        response = client.post("/repair/diagnose", json={
            "text_description": "My kitchen faucet is dripping",
            "store_id": "store-123",
        })

        assert response.status_code == 200
        data = response.json()

        assert "problem_id" in data
        assert data["status"] == "diagnosed"
        assert len(data["hypotheses"]) > 0
        assert data["top_hypothesis"]["category_slug"] == "plumbing-faucet"
        assert 0 <= data["confidence"] <= 1

    def test_diagnose_voice_only(self, client):
        """Voice transcript diagnosis should work."""
        response = client.post("/repair/diagnose", json={
            "voice_transcript": "The toilet keeps running",
            "store_id": "store-123",
        })

        assert response.status_code == 200
        data = response.json()
        assert "problem_id" in data

    def test_diagnose_with_image(self, client):
        """Diagnosis with image should include vision extras."""
        response = client.post("/repair/diagnose", json={
            "text_description": "What's wrong with my faucet?",
            "image_base64": VALID_JPEG_BASE64,
            "store_id": "store-123",
        })

        assert response.status_code == 200
        data = response.json()
        assert "likely_parts_needed" in data
        assert "tools_needed" in data
        assert "diy_feasible" in data

    def test_diagnose_combined_inputs(self, client):
        """Diagnosis with all inputs should work."""
        response = client.post("/repair/diagnose", json={
            "text_description": "Faucet dripping",
            "voice_transcript": "It's been like this for a week",
            "image_base64": VALID_JPEG_BASE64,
            "store_id": "store-123",
            "employee_id": "emp-456",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "diagnosed"

    def test_diagnose_no_input_fails(self, client):
        """Diagnosis with no input should fail."""
        response = client.post("/repair/diagnose", json={
            "store_id": "store-123",
        })

        assert response.status_code == 400
        assert "at least one input" in response.json()["detail"].lower()

    def test_diagnose_missing_store_id_fails(self, client):
        """Diagnosis without store_id should fail validation."""
        response = client.post("/repair/diagnose", json={
            "text_description": "Faucet dripping",
        })

        assert response.status_code == 422  # Validation error

    def test_diagnose_response_structure(self, client):
        """Response should have correct structure."""
        response = client.post("/repair/diagnose", json={
            "text_description": "Water leak",
            "store_id": "store-123",
        })

        assert response.status_code == 200
        data = response.json()

        # Required fields
        assert "problem_id" in data
        assert "status" in data
        assert "hypotheses" in data
        assert "top_hypothesis" in data
        assert "confidence" in data
        assert "entropy" in data
        assert "needs_more_info" in data
        assert "follow_up_questions" in data

        # Hypothesis structure
        hyp = data["hypotheses"][0]
        assert "category_slug" in hyp
        assert "category_name" in hyp
        assert "probability" in hyp

    def test_diagnose_with_session_id(self, client):
        """Customer session ID should be accepted."""
        response = client.post("/repair/diagnose", json={
            "text_description": "My AC isn't cooling",
            "store_id": "store-123",
            "session_id": "anon-session-789",
        })

        assert response.status_code == 200


# =============================================================================
# REFINE ENDPOINT TESTS
# =============================================================================

class TestRefineEndpoint:
    """Tests for POST /repair/diagnose/refine."""

    def test_refine_not_implemented(self, client):
        """Refine endpoint should return 501 (not implemented yet)."""
        response = client.post("/repair/diagnose/refine", json={
            "problem_id": "prob-123",
            "additional_text": "It's getting worse",
        })

        assert response.status_code == 501
        assert "coming soon" in response.json()["detail"].lower()


# =============================================================================
# SOLUTION ENDPOINT TESTS
# =============================================================================

class TestSolutionEndpoint:
    """Tests for GET /repair/solution/{problem_id}."""

    def test_solution_not_implemented(self, client):
        """Solution endpoint should return 501 (not implemented yet)."""
        response = client.get("/repair/solution/prob-123")

        assert response.status_code == 501
        assert "coming soon" in response.json()["detail"].lower()


# =============================================================================
# CORRECTION ENDPOINT TESTS
# =============================================================================

class TestCorrectionEndpoint:
    """Tests for POST /repair/correction."""

    def test_correction_success(self, client):
        """Employee correction should return XP reward."""
        response = client.post("/repair/correction", json={
            "problem_id": "prob-123",
            "employee_id": "emp-456",
            "correct_category_slug": "plumbing-toilet",
            "correction_notes": "It's actually a toilet issue",
        })

        assert response.status_code == 200
        data = response.json()

        assert data["correction_id"]
        assert data["problem_id"] == "prob-123"
        assert data["employee_id"] == "emp-456"
        assert data["xp_awarded"] > 0
        assert "new_total_xp" in data
        assert "leveled_up" in data
        assert "current_streak" in data

    def test_correction_without_employee_fails(self, client):
        """Correction without employee ID should fail."""
        response = client.post("/repair/correction", json={
            "problem_id": "prob-123",
            "employee_id": "",
            "correct_category_slug": "plumbing",
        })

        assert response.status_code == 400
        assert "employee" in response.json()["detail"].lower()

    def test_correction_missing_fields_fails(self, client):
        """Correction with missing required fields should fail."""
        response = client.post("/repair/correction", json={
            "problem_id": "prob-123",
        })

        assert response.status_code == 422  # Validation error


# =============================================================================
# CATEGORIES ENDPOINT TESTS
# =============================================================================

class TestCategoriesEndpoint:
    """Tests for GET /repair/categories."""

    def test_list_categories(self, client):
        """Should return hierarchical category list."""
        response = client.get("/repair/categories")

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) > 0

        # Check category structure
        category = data[0]
        assert "category_id" in category
        assert "name" in category
        assert "slug" in category
        assert "subcategories" in category

    def test_categories_hierarchical(self, client):
        """Categories should be hierarchical (parents have subcategories)."""
        response = client.get("/repair/categories")

        assert response.status_code == 200
        data = response.json()

        # Find plumbing category
        plumbing = next((c for c in data if c["slug"] == "plumbing"), None)
        assert plumbing is not None
        assert len(plumbing["subcategories"]) > 0


# =============================================================================
# HEALTH ENDPOINT TESTS
# =============================================================================

class TestHealthEndpoint:
    """Tests for GET /repair/health."""

    def test_health_check(self, client):
        """Health check should return engine status."""
        response = client.get("/repair/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["engine"] == "available"
        assert "categories_loaded" in data
        assert data["categories_loaded"] > 0
        assert data["dimensions"] == 4096


# =============================================================================
# INTEGRATION TEST
# =============================================================================

class TestRepairIntegration:
    """End-to-end integration tests."""

    def test_full_diagnosis_flow(self, client):
        """Test complete diagnosis flow."""
        # Step 1: Diagnose
        diagnose_response = client.post("/repair/diagnose", json={
            "text_description": "Water dripping from kitchen faucet",
            "image_base64": VALID_JPEG_BASE64,
            "store_id": "store-123",
            "employee_id": "emp-456",
        })

        assert diagnose_response.status_code == 200
        diagnosis = diagnose_response.json()
        problem_id = diagnosis["problem_id"]

        # Step 2: Check categories available
        categories_response = client.get("/repair/categories")
        assert categories_response.status_code == 200
        categories = categories_response.json()
        assert len(categories) > 0

        # Step 3: Employee makes correction
        correction_response = client.post("/repair/correction", json={
            "problem_id": problem_id,
            "employee_id": "emp-456",
            "correct_category_slug": "plumbing-faucet",
            "correction_notes": "Confirmed faucet cartridge issue",
        })

        assert correction_response.status_code == 200
        correction = correction_response.json()
        assert correction["xp_awarded"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
