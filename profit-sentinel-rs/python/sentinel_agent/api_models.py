"""API request/response models for the Sentinel sidecar.

These models define the HTTP contract. They wrap the existing domain models
(Digest, Task, CallPrep, CoopIntelligenceReport) with API-specific metadata
like rendered text, pagination, and error structure.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from .coop_models import (
    CoopIntelligenceReport,
    InventoryHealthReport,
    VendorRebateStatus,
)
from .models import CallPrep, Digest, Issue, Task, TaskPriority

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    code: str
    message: str
    detail: str | None = None


class TaskStatus(str, Enum):
    """Task lifecycle status."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ESCALATED = "escalated"


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------


class DigestResponse(BaseModel):
    """Response for GET /api/v1/digest."""

    digest: Digest
    rendered_text: str
    generated_at: str
    store_filter: list[str]
    issue_count: int
    total_dollar_impact: float


# ---------------------------------------------------------------------------
# Delegation
# ---------------------------------------------------------------------------


class DelegateRequest(BaseModel):
    """Request body for POST /api/v1/delegate."""

    issue_id: str
    assignee: str
    deadline: datetime | None = None
    notes: str | None = None


class DelegateResponse(BaseModel):
    """Response for POST /api/v1/delegate."""

    task: Task
    rendered_text: str
    task_id: str


class TaskStatusUpdate(BaseModel):
    """Request body for PATCH /api/v1/tasks/{task_id}."""

    status: TaskStatus
    notes: str | None = None


class TaskResponse(BaseModel):
    """Response for GET /api/v1/tasks/{task_id}."""

    task: Task
    status: TaskStatus = TaskStatus.OPEN
    rendered_text: str
    notes: list[str] = Field(default_factory=list)


class TaskListResponse(BaseModel):
    """Response for GET /api/v1/tasks."""

    tasks: list[TaskResponse]
    total: int


# ---------------------------------------------------------------------------
# Vendor Call
# ---------------------------------------------------------------------------


class VendorCallResponse(BaseModel):
    """Response for GET /api/v1/vendor-call/{issue_id}."""

    call_prep: CallPrep
    rendered_text: str


# ---------------------------------------------------------------------------
# Co-op Intelligence
# ---------------------------------------------------------------------------


class CoopReportResponse(BaseModel):
    """Response for GET /api/v1/coop/{store_id}."""

    report: CoopIntelligenceReport
    rendered_text: str
    health_summary: str | None = None
    rebate_statuses: list[VendorRebateStatus] = Field(default_factory=list)
    total_opportunity: float


# ---------------------------------------------------------------------------
# Diagnostics (Phase 11)
# ---------------------------------------------------------------------------


class DiagnosticStartRequest(BaseModel):
    """Request body for POST /api/v1/diagnostic/start."""

    items: list[dict]
    store_name: str = "My Store"


class DiagnosticStartResponse(BaseModel):
    """Response for POST /api/v1/diagnostic/start."""

    session_id: str
    store_name: str
    total_items: int
    negative_items: int
    total_shrinkage: float
    patterns_detected: int


class DiagnosticQuestionResponse(BaseModel):
    """Response for GET /api/v1/diagnostic/{session_id}/question."""

    pattern_id: str
    pattern_name: str
    question: str
    suggested_answers: list[list[str]]
    item_count: int
    total_value: float
    sample_items: list[dict]
    progress: dict
    running_totals: dict


class DiagnosticAnswerRequest(BaseModel):
    """Request body for POST /api/v1/diagnostic/{session_id}/answer."""

    classification: str
    note: str = ""


class DiagnosticAnswerResponse(BaseModel):
    """Response for POST /api/v1/diagnostic/{session_id}/answer."""

    answered: dict
    progress: dict
    running_totals: dict
    is_complete: bool
    next_question: DiagnosticQuestionResponse | None = None


class DiagnosticSummaryResponse(BaseModel):
    """Response for GET /api/v1/diagnostic/{session_id}/summary."""

    session_id: str
    store_name: str
    status: str
    total_items: int
    negative_items: int
    total_shrinkage: float
    explained_value: float
    unexplained_value: float
    reduction_percent: float
    patterns_total: int
    patterns_answered: int


class DiagnosticReportResponse(BaseModel):
    """Response for GET /api/v1/diagnostic/{session_id}/report."""

    session_id: str
    summary: dict
    by_classification: dict
    items_to_investigate: list[dict]
    journey: list[dict]
    rendered_text: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Explain / Proof Tree (Phase 13)
# ---------------------------------------------------------------------------


class ProofNodeResponse(BaseModel):
    """A single node in the proof tree showing one reasoning step."""

    statement: str
    confidence: float
    explanation: str
    source: str  # "signal", "vsa_scoring", "inference", "attribute"
    children: list[ProofNodeResponse] = Field(default_factory=list)


class SignalContributionResponse(BaseModel):
    """How a single signal contributed to the root cause attribution."""

    signal: str
    description: str
    rules_fired: list[dict] = Field(default_factory=list)


class CompetingHypothesisResponse(BaseModel):
    """An alternative cause that was considered but scored lower."""

    cause: str
    cause_display: str
    score: float
    rank: int
    why_lower: str


class ProofTreeResponse(BaseModel):
    """Full proof tree for an issue's root cause attribution."""

    issue_id: str
    issue_type: str
    store_id: str
    dollar_impact: float
    root_cause: str | None
    root_cause_display: str
    root_cause_confidence: float
    root_cause_ambiguity: float
    active_signals: list[str]
    signal_contributions: list[SignalContributionResponse]
    cause_scores: list[dict]
    proof_tree: ProofNodeResponse
    inferred_facts: list[dict]
    competing_hypotheses: list[CompetingHypothesisResponse]
    recommendations: list[str]
    suggested_actions: list[dict]


class ExplainResponse(BaseModel):
    """Response for GET /api/v1/explain/{issue_id}.

    Contains the full proof tree showing why the system attributed
    a particular root cause to the issue. Includes signal contributions,
    VSA cause scores, forward-chained inferences, competing hypotheses,
    and actionable recommendations.
    """

    issue_id: str
    proof_tree: ProofTreeResponse
    rendered_text: str


class BackwardChainRequest(BaseModel):
    """Request body for POST /api/v1/explain/{issue_id}/why."""

    goal: str = Field(description="The predicate to explain, e.g. 'root_cause(Theft)'")


class BackwardChainResponse(BaseModel):
    """Response for POST /api/v1/explain/{issue_id}/why."""

    issue_id: str
    goal: str
    reasoning_steps: list[dict]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    version: str = "0.13.0"
    binary_found: bool
    binary_path: str | None = None
    dev_mode: bool = False


# ---------------------------------------------------------------------------
# Digest Email / Subscriptions
# ---------------------------------------------------------------------------


class SubscribeRequest(BaseModel):
    """Request body for POST /api/v1/digest/subscribe."""

    email: str = Field(description="Subscriber email address")
    stores: list[str] = Field(
        default_factory=list, description="Store IDs to include (empty = all)"
    )
    send_hour: int = Field(
        default=6, ge=0, le=23, description="Hour of day to send (0-23)"
    )
    timezone: str = Field(default="America/New_York", description="IANA timezone")


class SubscribeResponse(BaseModel):
    """Response for POST /api/v1/digest/subscribe."""

    subscription: dict
    message: str = "Subscription created"


class SubscriptionListResponse(BaseModel):
    """Response for GET /api/v1/digest/subscriptions."""

    subscriptions: list[dict]
    total: int


class DigestSendRequest(BaseModel):
    """Request body for POST /api/v1/digest/send."""

    email: str = Field(description="Email to send digest to")


class DigestSendResponse(BaseModel):
    """Response for POST /api/v1/digest/send."""

    email_id: str | None = None
    message: str = "Digest sent"


class SchedulerStatusResponse(BaseModel):
    """Response for GET /api/v1/digest/scheduler-status."""

    enabled: bool
    running: bool
    subscribers: int
    send_hour: int


# ---------------------------------------------------------------------------
# Analysis History
# ---------------------------------------------------------------------------


class _AnalysisBase(BaseModel):
    """Shared fields for analysis list/detail responses."""

    id: str
    analysis_label: str | None = None
    original_filename: str | None = None
    file_row_count: int = 0
    file_column_count: int | None = None
    detection_counts: dict = Field(default_factory=dict)
    total_impact_estimate_low: float = 0
    total_impact_estimate_high: float = 0
    processing_time_seconds: float | None = None
    created_at: str | None = None


class AnalysisListItem(_AnalysisBase):
    """Summary of a saved analysis (without full_result)."""

    has_full_result: bool = False


class AnalysisListResponse(BaseModel):
    """Response for GET /api/v1/analyses."""

    analyses: list[AnalysisListItem]
    total: int


class AnalysisDetailResponse(_AnalysisBase):
    """Response for GET /api/v1/analyses/{id} — includes full_result."""

    full_result: dict | None = None


class AnalysisRenameRequest(BaseModel):
    """Request body for PATCH /api/v1/analyses/{id}."""

    label: str = Field(description="New analysis label")


class AnalysisCompareRequest(BaseModel):
    """Request body for POST /api/v1/analyses/compare."""

    current_id: str = Field(description="ID of the current (newer) analysis")
    previous_id: str = Field(description="ID of the previous (older) analysis")


class AnalysisCompareResponse(BaseModel):
    """Response for POST /api/v1/analyses/compare."""

    summary: dict
    leak_trends: list[dict]
    new_leaks: list[str]
    resolved_leaks: list[str]
    worsening_leaks: list[str]
    improving_leaks: list[str]
    metadata: dict


# ---------------------------------------------------------------------------
# Vendor Performance Scoring
# ---------------------------------------------------------------------------


class VendorScoresResponse(BaseModel):
    """Response for GET /api/v1/vendor-scores."""

    store_id: str
    scorecards: list[dict]
    total_vendors_scored: int
    average_score: float
    high_risk_vendors: int
    total_quality_cost: float
    top_recommendation: str


# ---------------------------------------------------------------------------
# Predictive Inventory Alerts
# ---------------------------------------------------------------------------


class PredictiveReportResponse(BaseModel):
    """Response for GET /api/v1/predictions."""

    store_id: str
    total_predictions: int
    critical_alerts: int
    warning_alerts: int
    total_revenue_at_risk: float
    total_carrying_cost_at_risk: float
    stockout_predictions: list[dict]
    overstock_predictions: list[dict]
    velocity_alerts: list[dict]
    top_recommendation: str


# ---------------------------------------------------------------------------
# Enterprise API Keys
# ---------------------------------------------------------------------------


class CreateApiKeyRequest(BaseModel):
    """Request body for POST /api/v1/api-keys."""

    name: str = Field(default="Default", description="Friendly name for the key")
    tier: str = Field(default="free", description="API tier: free, pro, enterprise")
    test: bool = Field(default=False, description="Create a test key (ps_test_ prefix)")


class CreateApiKeyResponse(BaseModel):
    """Response for POST /api/v1/api-keys.

    IMPORTANT: The plaintext key is returned only once. Store it securely.
    """

    key: str = Field(description="Plaintext API key (shown only once)")
    record: dict


class ApiKeyListResponse(BaseModel):
    """Response for GET /api/v1/api-keys."""

    keys: list[dict]
    total: int


# ---------------------------------------------------------------------------
# POS Integrations
# ---------------------------------------------------------------------------


class PosConnectionRequest(BaseModel):
    """Request body for POST /api/v1/pos/connections."""

    pos_system: str = Field(
        description="POS system: square, lightspeed, clover, shopify"
    )
    store_name: str = Field(description="Store display name")
    sync_frequency: str = Field(
        default="daily", description="Sync frequency: manual, daily, weekly, monthly"
    )
    location_id: str | None = Field(
        default=None, description="POS-specific location ID"
    )
