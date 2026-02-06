"""
Repair Assistant API Endpoints

Provides endpoints for the Visual AI Repair Assistant:
- POST /repair/diagnose - Analyze problem from text/image
- POST /repair/diagnose/refine - Refine diagnosis with more info
- GET /repair/solution/{problem_id} - Get repair solution
- POST /repair/correction - Employee corrects diagnosis
- GET /repair/categories - List problem categories

Security:
- Input validation via Pydantic models
- Image guardrails in grok_vision service
- Rate limiting awareness
- Employee authentication for corrections
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_current_user
from ..services.grok_vision import (
    GrokVisionService,
    VisionAnalysisResult,
)

# Import from sentinel-engine (path may need adjustment based on package setup)
try:
    from sentinel_engine.repair_engine import (  # noqa: F401
        RepairDiagnosisEngine,
        create_engine,
    )
    from sentinel_engine.repair_models import (  # noqa: F401
        XP_REWARDS,
        CategoryList,
        CorrectionRequest,
        CorrectionResult,
        DiagnoseRequest,
        DiagnoseResponse,
        Hypothesis,
        ProblemCategory,
        ProblemStatus,
        RefineRequest,
        SolutionPart,
        SolutionResponse,
        SolutionStep,
        calculate_level,
        xp_for_next_level,
    )

    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    # Define placeholder type for annotation when engine not available
    RepairDiagnosisEngine = None  # type: ignore

router = APIRouter()
logger = logging.getLogger(__name__)

# Global engine instance (singleton)
# Use Any type to avoid issues when sentinel_engine not installed
_engine: "RepairDiagnosisEngine | None" = None  # type: ignore


def get_engine() -> "RepairDiagnosisEngine":  # type: ignore
    """Get or create the repair diagnosis engine."""
    global _engine
    if _engine is None:
        if not ENGINE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Repair engine not available")
        _engine = create_engine(dimensions=4096, device="cpu")
    return _engine


# =============================================================================
# REQUEST/RESPONSE MODELS (API-specific)
# =============================================================================


class DiagnoseRequestAPI(BaseModel):
    """API request model for diagnosis."""

    text_description: str | None = Field(None, max_length=2000)
    voice_transcript: str | None = Field(None, max_length=2000)
    image_base64: str | None = Field(None, description="Base64-encoded image")
    store_id: str = Field(..., min_length=1)
    employee_id: str | None = None
    session_id: str | None = None


class RefineRequestAPI(BaseModel):
    """API request for refining diagnosis."""

    problem_id: str
    additional_text: str | None = Field(None, max_length=1000)
    additional_image_base64: str | None = None
    answer_to_question: str | None = Field(None, max_length=500)
    question_index: int | None = None


class CorrectionRequestAPI(BaseModel):
    """API request for employee correction."""

    problem_id: str
    employee_id: str
    correct_category_slug: str
    correction_notes: str | None = Field(None, max_length=500)


class HypothesisAPI(BaseModel):
    """Hypothesis in API response."""

    category_slug: str
    category_name: str
    probability: float
    explanation: str | None = None
    icon: str | None = None


class DiagnoseResponseAPI(BaseModel):
    """API response for diagnosis."""

    problem_id: str
    status: str

    hypotheses: list[HypothesisAPI]
    top_hypothesis: HypothesisAPI

    confidence: float
    entropy: float

    needs_more_info: bool
    follow_up_questions: list[str]

    # Vision analysis extras (if image provided)
    likely_parts_needed: list[str] | None = None
    tools_needed: list[str] | None = None
    safety_concerns: list[str] | None = None
    diy_feasible: bool | None = None
    professional_recommended: bool | None = None


class SolutionPartAPI(BaseModel):
    """Part in solution response."""

    part_name: str
    part_description: str | None = None
    quantity: int = 1
    is_required: bool = True
    sku: str | None = None
    in_stock: bool | None = None
    stock_quantity: int | None = None
    unit_price: float | None = None
    has_substitute: bool = False
    substitute_sku: str | None = None
    substitute_name: str | None = None


class SolutionStepAPI(BaseModel):
    """Step in solution response."""

    order: int
    instruction: str
    tip: str | None = None
    caution: str | None = None


class SolutionResponseAPI(BaseModel):
    """API response for repair solution."""

    solution_id: str
    problem_id: str
    category_slug: str
    category_name: str

    title: str
    summary: str
    steps: list[SolutionStepAPI]
    parts: list[SolutionPartAPI]

    tools_required: list[str]
    estimated_time_minutes: int | None = None
    difficulty_level: int

    video_urls: list[str]

    all_parts_available: bool
    parts_total_cost: float | None = None


class CorrectionResultAPI(BaseModel):
    """API response for correction submission."""

    correction_id: str
    problem_id: str
    employee_id: str

    xp_awarded: int
    new_total_xp: int
    leveled_up: bool
    new_level: int | None = None

    badge_earned: str | None = None
    streak_extended: bool
    current_streak: int


class CategoryAPI(BaseModel):
    """Problem category in API response."""

    category_id: str
    name: str
    slug: str
    description: str | None = None
    icon: str | None = None
    parent_slug: str | None = None
    subcategories: list["CategoryAPI"] = []


CategoryAPI.model_rebuild()


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/diagnose", response_model=DiagnoseResponseAPI)
async def diagnose_problem(
    request: DiagnoseRequestAPI,
    user_id: str | None = Depends(get_current_user),
):
    """
    Diagnose a repair problem from text and/or image.

    Accepts:
    - Text description
    - Voice transcript
    - Photo (base64-encoded)

    Returns hypothesis probabilities and follow-up questions if needed.
    """
    # Validate at least one input
    if not any(
        [request.text_description, request.voice_transcript, request.image_base64]
    ):
        raise HTTPException(
            status_code=400,
            detail="At least one input (text, voice, or image) required",
        )

    engine = get_engine()
    vision_result: VisionAnalysisResult | None = None

    # Process image if provided
    image_features = None
    if request.image_base64:
        try:
            vision_service = GrokVisionService()
            vision_result = vision_service.analyze_image(
                request.image_base64,
                text_context=request.text_description or request.voice_transcript,
            )
            # Get VSA vector from vision
            features = vision_service.extract_features_for_vsa(vision_result)
            image_features = vision_service.encode_to_vsa_vector(
                features, engine.text_encoder
            )
        except ValueError as e:
            logger.warning(f"Image analysis failed: {e}")
            # Continue without image features

    # Run VSA diagnosis
    try:
        bundle, metadata = engine.diagnose(
            text=request.text_description,
            voice_transcript=request.voice_transcript,
            image_features=image_features,
            store_id=request.store_id,
            employee_id=request.employee_id,
        )
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}")
        raise HTTPException(status_code=500, detail="Diagnosis failed")

    # Generate problem ID
    problem_id = str(uuid.uuid4())

    # Convert to API response
    response = engine.bundle_to_response(bundle, problem_id, metadata)

    # Build API response with vision extras
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
        problem_id=problem_id,
        status=response.status.value,
        hypotheses=hypotheses_api,
        top_hypothesis=HypothesisAPI(
            category_slug=response.top_hypothesis.category_slug,
            category_name=response.top_hypothesis.category_name,
            probability=response.top_hypothesis.probability,
            explanation=response.top_hypothesis.explanation,
            icon=response.top_hypothesis.icon,
        ),
        confidence=response.confidence,
        entropy=response.entropy,
        needs_more_info=response.needs_more_info,
        follow_up_questions=response.follow_up_questions,
    )

    # Add vision extras if available
    if vision_result:
        api_response.likely_parts_needed = vision_result.likely_parts_needed
        api_response.tools_needed = vision_result.tools_needed
        api_response.safety_concerns = vision_result.safety_concerns
        api_response.diy_feasible = vision_result.diy_feasible
        api_response.professional_recommended = vision_result.professional_recommended

    # TODO: Store problem in database for later retrieval

    return api_response


@router.post("/diagnose/refine", response_model=DiagnoseResponseAPI)
async def refine_diagnosis(
    request: RefineRequestAPI,
    user_id: str | None = Depends(get_current_user),
):
    """
    Refine an existing diagnosis with additional information.

    Use this when:
    - needs_more_info was True
    - User wants to add more details
    - Answering follow-up questions
    """
    # TODO: Retrieve existing hypothesis bundle from database
    # For now, return error since we need to implement persistence

    raise HTTPException(
        status_code=501,
        detail="Diagnosis refinement requires database persistence (coming soon)",
    )


@router.get("/solution/{problem_id}", response_model=SolutionResponseAPI)
async def get_solution(
    problem_id: str,
    user_id: str | None = Depends(get_current_user),
):
    """
    Get repair solution for a diagnosed problem.

    Returns:
    - Step-by-step repair instructions
    - Parts needed with inventory status
    - Tools required
    - Video references
    """
    # TODO: Retrieve problem from database and generate solution
    # For now, return a mock solution

    raise HTTPException(
        status_code=501,
        detail="Solution generation requires database persistence (coming soon)",
    )


@router.post("/correction", response_model=CorrectionResultAPI)
async def submit_correction(
    request: CorrectionRequestAPI,
    background_tasks: BackgroundTasks,
    user_id: str | None = Depends(get_current_user),
):
    """
    Employee submits correction to AI diagnosis.

    Awards XP based on:
    - Base correction XP
    - Difficulty multiplier
    - First-in-category bonus

    Updates knowledge base via CW-Bundle.
    """
    # Validate employee
    if not request.employee_id:
        raise HTTPException(
            status_code=400, detail="Employee ID required for corrections"
        )

    # TODO: Verify employee exists and retrieve their profile
    # TODO: Retrieve problem and original diagnosis
    # TODO: Apply correction to knowledge base
    # TODO: Award XP and check for level up/badges

    # For now, return mock result
    correction_id = str(uuid.uuid4())
    base_xp = XP_REWARDS["correction_base"] if ENGINE_AVAILABLE else 50

    return CorrectionResultAPI(
        correction_id=correction_id,
        problem_id=request.problem_id,
        employee_id=request.employee_id,
        xp_awarded=base_xp,
        new_total_xp=base_xp,  # Would be cumulative
        leveled_up=False,
        new_level=None,
        badge_earned=None,
        streak_extended=True,
        current_streak=1,
    )


@router.get("/categories", response_model=list[CategoryAPI])
async def list_categories():
    """
    List all problem categories.

    Returns hierarchical category tree.
    """
    engine = get_engine()

    # Get categories from engine codebook
    categories = []
    for slug in engine.codebook.get_all_slugs():
        info = engine.codebook.get_info(slug)

        # Skip subcategories at top level
        if info.get("parent_slug"):
            continue

        # Build subcategories
        subcategories = []
        for sub_slug in engine.codebook.get_all_slugs():
            sub_info = engine.codebook.get_info(sub_slug)
            if sub_info.get("parent_slug") == slug:
                subcategories.append(
                    CategoryAPI(
                        category_id=sub_slug,
                        name=sub_info.get("name", sub_slug),
                        slug=sub_slug,
                        description=sub_info.get("description"),
                        icon=sub_info.get("icon"),
                        parent_slug=slug,
                        subcategories=[],
                    )
                )

        categories.append(
            CategoryAPI(
                category_id=slug,
                name=info.get("name", slug),
                slug=slug,
                description=info.get("description"),
                icon=info.get("icon"),
                parent_slug=None,
                subcategories=subcategories,
            )
        )

    return categories


@router.get("/health")
async def repair_health():
    """Health check for repair service."""
    try:
        engine = get_engine()
        return {
            "status": "healthy",
            "engine": "available",
            "categories_loaded": len(engine.codebook.get_all_slugs()),
            "dimensions": engine.config.dimensions,
        }
    except Exception as e:
        return {
            "status": "degraded",
            "engine": "unavailable",
            "error": str(e)[:100],
        }
