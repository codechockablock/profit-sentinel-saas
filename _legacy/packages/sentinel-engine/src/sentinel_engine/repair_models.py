"""
Repair Assistant Data Models

Pydantic models for the Visual AI Repair Assistant system.
Covers diagnosis requests/responses, solutions, employees, and gamification.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# ENUMS
# =============================================================================


class ProblemStatus(StrEnum):
    """Status of a repair problem through the diagnosis flow."""

    PENDING = "pending"
    DIAGNOSED = "diagnosed"
    REFINED = "refined"
    SOLVED = "solved"
    CORRECTED = "corrected"


class QueryType(StrEnum):
    """Type of employee/customer query."""

    DIAGNOSE = "diagnose"
    REFINE = "refine"
    GET_SOLUTION = "get_solution"
    FIND_PARTS = "find_parts"


class EventType(StrEnum):
    """Types of learning events for gamification."""

    ASSIST = "assist"
    CORRECTION = "correction"
    BADGE_EARNED = "badge_earned"
    LEVEL_UP = "level_up"
    STREAK_EXTENDED = "streak_extended"


class BadgeType(StrEnum):
    """Types of badges employees can earn."""

    FIRST_HELPER = "first_helper"
    KNOWLEDGE_BUILDER = "knowledge_builder"
    WEEK_WARRIOR = "week_warrior"
    CATEGORY_MASTER = "category_master"
    AI_TRAINER = "ai_trainer"
    STREAK_7 = "streak_7"
    STREAK_30 = "streak_30"
    ASSISTS_10 = "assists_10"
    ASSISTS_50 = "assists_50"
    ASSISTS_100 = "assists_100"


# =============================================================================
# GAMIFICATION CONSTANTS
# =============================================================================

# XP thresholds for levels (cumulative)
LEVEL_THRESHOLDS = {
    1: 0,
    2: 100,
    3: 300,
    4: 600,
    5: 1000,
    6: 1500,
    7: 2200,
    8: 3000,
    9: 4000,
    10: 5500,
}

# XP rewards
XP_REWARDS = {
    "assist_base": 10,
    "assist_difficulty_multiplier": 1.5,  # Per difficulty level
    "correction_base": 50,
    "correction_difficulty_multiplier": 2.0,
    "correction_first_in_category": 100,
    "quiz_complete": 20,
    "quiz_perfect": 30,
    "video_watch": 5,
    "video_complete": 10,
}


def calculate_level(xp: int) -> int:
    """Calculate level from total XP."""
    level = 1
    for lvl, threshold in sorted(LEVEL_THRESHOLDS.items()):
        if xp >= threshold:
            level = lvl
    return level


def xp_for_next_level(current_xp: int) -> int:
    """Calculate XP needed for next level."""
    current_level = calculate_level(current_xp)
    if current_level >= 10:
        return 0
    next_threshold = LEVEL_THRESHOLDS.get(current_level + 1, 0)
    return max(0, next_threshold - current_xp)


def expertise_multiplier(level: int) -> float:
    """
    Calculate expertise multiplier for knowledge weighting.
    Higher level employees have more influence on corrections.
    """
    # Level 1: 1.0x, Level 5: 1.5x, Level 10: 2.5x
    return 1.0 + (level - 1) * 0.166


# =============================================================================
# REQUEST MODELS
# =============================================================================


class DiagnoseRequest(BaseModel):
    """Request to diagnose a repair problem."""

    text_description: str | None = Field(
        None, description="Text description of the problem", max_length=2000
    )
    voice_transcript: str | None = Field(
        None, description="Transcript from voice input", max_length=2000
    )
    image_base64: str | None = Field(
        None, description="Base64-encoded image (max 1024px, EXIF stripped)"
    )
    store_id: str = Field(..., description="Store identifier")
    employee_id: str | None = Field(
        None, description="Employee ID (None for anonymous customer)"
    )
    session_id: str | None = Field(
        None, description="Anonymous session ID for customers"
    )

    @field_validator("text_description", "voice_transcript")
    @classmethod
    def not_empty_if_provided(cls, v: str | None) -> str | None:
        if v is not None and len(v.strip()) == 0:
            return None
        return v

    def has_input(self) -> bool:
        """Check if at least one input type is provided."""
        return bool(self.text_description or self.voice_transcript or self.image_base64)


class RefineRequest(BaseModel):
    """Request to refine diagnosis with additional information."""

    problem_id: str = Field(..., description="Problem ID to refine")
    additional_text: str | None = Field(None, max_length=1000)
    additional_image_base64: str | None = None
    answer_to_question: str | None = Field(
        None, description="Answer to a follow-up question"
    )
    question_index: int | None = Field(
        None, description="Index of question being answered"
    )


class CorrectionRequest(BaseModel):
    """Employee correction of AI diagnosis."""

    problem_id: str = Field(..., description="Problem ID being corrected")
    employee_id: str = Field(..., description="Employee making correction")
    correct_category_slug: str = Field(..., description="Correct category slug")
    correction_notes: str | None = Field(
        None, description="Optional notes explaining the correction", max_length=500
    )


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class Hypothesis(BaseModel):
    """Single hypothesis in diagnosis."""

    category_slug: str
    category_name: str
    probability: float = Field(..., ge=0.0, le=1.0)
    explanation: str | None = None
    icon: str | None = None


class DiagnoseResponse(BaseModel):
    """Response from diagnosis endpoint."""

    problem_id: str
    status: ProblemStatus = ProblemStatus.DIAGNOSED

    # Hypotheses (P-Sup results)
    hypotheses: list[Hypothesis]
    top_hypothesis: Hypothesis

    # Confidence metrics
    confidence: float = Field(..., ge=0.0, le=1.0)
    entropy: float = Field(
        ..., description="Uncertainty measure (lower = more certain)"
    )

    # Next steps
    needs_more_info: bool = False
    follow_up_questions: list[str] = Field(default_factory=list)

    # Context (if employee)
    similar_recent_problems: int | None = Field(
        None, description="Count of similar problems at this store recently"
    )


class SolutionStep(BaseModel):
    """Single step in repair solution."""

    order: int
    instruction: str
    tip: str | None = None
    caution: str | None = None
    image_url: str | None = None


class SolutionPart(BaseModel):
    """Part needed for repair solution."""

    part_name: str
    part_description: str | None = None
    quantity: int = 1
    is_required: bool = True

    # Inventory status
    sku: str | None = None
    in_stock: bool | None = None
    stock_quantity: int | None = None
    unit_price: float | None = None

    # Substitute if out of stock
    has_substitute: bool = False
    substitute_sku: str | None = None
    substitute_name: str | None = None


class SolutionResponse(BaseModel):
    """Full repair solution response."""

    solution_id: str
    problem_id: str
    category_slug: str
    category_name: str

    # Solution content
    title: str
    summary: str
    steps: list[SolutionStep]
    parts: list[SolutionPart]

    # Requirements
    tools_required: list[str] = Field(default_factory=list)
    estimated_time_minutes: int | None = None
    difficulty_level: int = Field(..., ge=1, le=5)

    # Video references
    video_urls: list[str] = Field(default_factory=list)

    # Inventory summary
    all_parts_available: bool
    parts_total_cost: float | None = None


# =============================================================================
# EMPLOYEE & GAMIFICATION MODELS
# =============================================================================


class SkillMastery(BaseModel):
    """Employee skill mastery for a category."""

    category_slug: str
    category_name: str
    mastery_score: float = Field(..., ge=0.0, le=100.0)
    assists_count: int = 0
    corrections_count: int = 0


class Badge(BaseModel):
    """Employee badge/achievement."""

    badge_type: BadgeType
    badge_name: str
    badge_description: str
    category_slug: str | None = None
    earned_at: datetime


class EmployeeProfile(BaseModel):
    """Employee profile with gamification stats."""

    employee_id: str
    name: str
    store_id: str

    # Gamification
    xp: int = 0
    level: int = 1
    xp_to_next_level: int = 100
    current_streak: int = 0
    longest_streak: int = 0

    # Stats
    total_assists: int = 0
    total_corrections: int = 0
    corrections_accepted: int = 0
    acceptance_rate: float = 0.0

    # Skills
    skills: list[SkillMastery] = Field(default_factory=list)
    top_categories: list[str] = Field(default_factory=list)

    # Badges
    badges: list[Badge] = Field(default_factory=list)
    badge_count: int = 0

    # Expertise multiplier for knowledge weighting
    expertise_multiplier: float = 1.0


class LearningEvent(BaseModel):
    """Learning event for gamification log."""

    event_id: str
    employee_id: str
    event_type: EventType

    xp_delta: int
    xp_after: int

    problem_id: str | None = None
    category_slug: str | None = None
    badge_type: BadgeType | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class CorrectionResult(BaseModel):
    """Result of submitting a correction."""

    correction_id: str
    problem_id: str
    employee_id: str

    # XP awarded
    xp_awarded: int
    new_total_xp: int
    leveled_up: bool = False
    new_level: int | None = None

    # Badge earned
    badge_earned: Badge | None = None

    # Streak
    streak_extended: bool = False
    current_streak: int = 0


# =============================================================================
# CATEGORY MODELS
# =============================================================================


class ProblemCategory(BaseModel):
    """Problem category for diagnosis taxonomy."""

    category_id: str
    name: str
    slug: str
    description: str | None = None
    icon: str | None = None
    parent_slug: str | None = None
    is_active: bool = True

    # Subcategories (for hierarchical display)
    subcategories: list[ProblemCategory] = Field(default_factory=list)


class CategoryList(BaseModel):
    """List of problem categories."""

    categories: list[ProblemCategory]
    total_count: int


# =============================================================================
# INTERNAL VSA MODELS
# =============================================================================


class VSAHypothesisState(BaseModel):
    """
    Internal state for VSA hypothesis tracking.
    Serialized to hypothesis_blob in database.
    """

    # Hypothesis labels and probabilities
    hypotheses: list[str]
    probabilities: list[float]

    # Vector data (base64 encoded torch tensors)
    superposition_vector: str  # Base64 encoded
    basis_vectors: str  # Base64 encoded (n, d)

    # Metadata
    dimensions: int
    update_count: int = 0
    last_evidence_similarity: float | None = None


class StoreMemoryState(BaseModel):
    """
    Internal state for store T-Bind memory.
    Serialized to memory_vector_blob in database.
    """

    # Memory vector (base64 encoded)
    memory_vector: str

    # Reference time for temporal encoding
    reference_time: float  # Unix timestamp

    # Stats
    total_problems: int
    category_distribution: dict[str, int]

    # Config
    dimensions: int
    decay_rate: float = 0.1


class KnowledgeBaseState(BaseModel):
    """
    Internal state for category knowledge (CW-Bundle).
    Serialized to knowledge_vector_blob in database.
    """

    category_slug: str

    # Knowledge vector (base64 encoded)
    knowledge_vector: str

    # Aggregate confidence
    aggregate_confidence: float

    # Stats
    total_corrections: int
    correction_weights: list[float]  # Expertise weights of contributors

    # Config
    dimensions: int
