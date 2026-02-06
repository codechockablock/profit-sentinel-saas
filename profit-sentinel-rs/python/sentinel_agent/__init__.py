"""Profit Sentinel — Python Agent Orchestration Layer.

Bridges the Rust VSA pipeline to natural-language agent output.
Transforms structured inventory analysis into actionable intelligence
for Do It Best operations executives and store managers.

Usage:
    from sentinel_agent import MorningDigestGenerator

    gen = MorningDigestGenerator()
    digest = gen.generate("inventory.csv", stores=["store-7", "store-12"])
    print(gen.render(digest))

With Co-op Intelligence:
    from sentinel_agent import (
        MorningDigestGenerator, CoopIntelligence, InventoryHealthScorer,
        VendorRebateTracker, CategoryMixOptimizer,
    )

    gen = MorningDigestGenerator()
    digest = gen.generate("inventory.csv", stores=["store-7"])
    coop_report = gen.generate_coop_report(
        digest, store_id="store-7",
        purchases=[...], vendor_ytd={"DMG": 18500},
    )
    print(gen.render_full(digest, coop_report))
"""

from .adapters.sales import (
    SalesAdapterResult,
    SalesAggregation,
    SalesDataAdapter,
    SalesOverlay,
    SalesTransaction,
    aggregate_sales_30d,
)
from .category_mix import CategoryMixOptimizer
from .coop_intelligence import CoopIntelligence
from .coop_models import (
    CATEGORY_BENCHMARKS,
    PATRONAGE_RATES,
    WAREHOUSE_CASH_DISCOUNT,
    BenchmarkComparison,
    CategoryMixAnalysis,
    ConsolidationOpportunity,
    CoopAffiliation,
    CoopAlert,
    CoopAlertType,
    CoopIntelligenceReport,
    CoopType,
    GMROIAnalysis,
    InventoryHealthReport,
    PatronageCategory,
    PatronageLeakage,
    PatronageProgram,
    RebateTier,
    SkuHealth,
    TurnClassification,
    VendorPurchase,
    VendorRebateProgram,
    VendorRebateStatus,
)
from .delegation import DelegationManager
from .diagnostics import (
    Classification,
    DetectedPattern,
    DiagnosticEngine,
    DiagnosticSession,
    InventoryItem,
    render_diagnostic_report,
    render_diagnostic_summary,
)
from .digest import MorningDigestGenerator
from .engine import AnalysisResult, PipelineError, SentinelEngine
from .inventory_health import InventoryHealthScorer
from .models import (
    CallPrep,
    CauseScoreDetail,
    Digest,
    Issue,
    IssueType,
    Sku,
    Summary,
    Task,
    TaskPriority,
    TrendDirection,
)
from .symbolic_reasoning import (
    DOMAIN_RULES,
    EVIDENCE_RULES,
    SIGNAL_DESCRIPTIONS,
    CompetingHypothesis,
    DomainRule,
    EvidenceRuleSpec,
    Fact,
    FactSource,
    ProofNode,
    ProofTree,
    SignalContribution,
    SymbolicReasoner,
)
from .vendor_assist import VendorCallAssistant
from .vendor_rebates import VendorRebateTracker

# Phase 6 - API (lazy import to avoid requiring sidecar deps at import time)
try:
    from .api_models import (
        BackwardChainRequest,
        BackwardChainResponse,
        CoopReportResponse,
        DelegateRequest,
        DelegateResponse,
        DiagnosticAnswerRequest,
        DiagnosticAnswerResponse,
        DiagnosticQuestionResponse,
        DiagnosticReportResponse,
        DiagnosticStartRequest,
        DiagnosticStartResponse,
        DiagnosticSummaryResponse,
        DigestResponse,
        ErrorResponse,
        ExplainResponse,
        HealthResponse,
        TaskListResponse,
        TaskResponse,
        TaskStatus,
        TaskStatusUpdate,
        VendorCallResponse,
    )
    from .sidecar import create_app
    from .sidecar_config import SidecarSettings, get_settings

    _HAS_SIDECAR = True
except ImportError:
    _HAS_SIDECAR = False

__all__ = [
    # Phase 4 - Core
    "AnalysisResult",
    "CallPrep",
    "DelegationManager",
    "Digest",
    "Issue",
    "IssueType",
    "MorningDigestGenerator",
    "PipelineError",
    "SentinelEngine",
    "Sku",
    "Summary",
    "Task",
    "TaskPriority",
    "TrendDirection",
    "VendorCallAssistant",
    # Phase 5 - Co-op Intelligence
    "CategoryMixOptimizer",
    "CoopAffiliation",
    "CoopAlert",
    "CoopAlertType",
    "CoopIntelligence",
    "CoopIntelligenceReport",
    "CoopType",
    "ConsolidationOpportunity",
    "GMROIAnalysis",
    "InventoryHealthReport",
    "InventoryHealthScorer",
    "PatronageCategory",
    "PatronageLeakage",
    "PatronageProgram",
    "RebateTier",
    "SkuHealth",
    "TurnClassification",
    "VendorPurchase",
    "VendorRebateProgram",
    "VendorRebateStatus",
    "VendorRebateTracker",
    "BenchmarkComparison",
    "CategoryMixAnalysis",
    # Constants
    "CATEGORY_BENCHMARKS",
    "PATRONAGE_RATES",
    "WAREHOUSE_CASH_DISCOUNT",
    # Phase 6 - API Server & Mobile Interface
    "BackwardChainRequest",
    "BackwardChainResponse",
    "CoopReportResponse",
    "DelegateRequest",
    "DelegateResponse",
    "DigestResponse",
    "ErrorResponse",
    "ExplainResponse",
    "HealthResponse",
    "TaskListResponse",
    "TaskResponse",
    "TaskStatus",
    "TaskStatusUpdate",
    "VendorCallResponse",
    "SidecarSettings",
    "create_app",
    "get_settings",
    # Phase 11 - Conversational Diagnostics
    "Classification",
    "DetectedPattern",
    "DiagnosticEngine",
    "DiagnosticSession",
    "DiagnosticAnswerRequest",
    "DiagnosticAnswerResponse",
    "DiagnosticQuestionResponse",
    "DiagnosticReportResponse",
    "DiagnosticStartRequest",
    "DiagnosticStartResponse",
    "DiagnosticSummaryResponse",
    "InventoryItem",
    "render_diagnostic_report",
    "render_diagnostic_summary",
    # Phase 13 - VSA→Symbolic Bridge
    "CauseScoreDetail",
    "CompetingHypothesis",
    "DomainRule",
    "EvidenceRuleSpec",
    "Fact",
    "FactSource",
    "ProofNode",
    "ProofTree",
    "SignalContribution",
    "SymbolicReasoner",
    "DOMAIN_RULES",
    "EVIDENCE_RULES",
    "SIGNAL_DESCRIPTIONS",
    # Phase 12 - Sales Data Integration
    "SalesAdapterResult",
    "SalesAggregation",
    "SalesDataAdapter",
    "SalesOverlay",
    "SalesTransaction",
    "aggregate_sales_30d",
]
