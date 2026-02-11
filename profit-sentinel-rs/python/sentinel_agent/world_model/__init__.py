from .battery import WarmupPhase, WorldModelBattery
from .config import (
    ClassificationResult,
    ConfigPresets,
    DeadStockConfig,
    DeadStockThresholds,
    DeadStockTier,
    InventoryLifecycleTracker,
)
from .core import (
    PhasorAlgebra,
    StateVector,
    TransitionModel,
    VSAWorldModel,
    WorldModelConfig,
)
from .pipeline import (
    FeedbackEngine,
    Intervention,
    InterventionType,
    MoatMetrics,
    Outcome,
    OutcomeType,
    PredictiveEngine,
    SentinelPipeline,
    TemporalHierarchy,
    VendorIntelligence,
)
from .transfer_matching import (
    EntityHierarchy,
    StoreAgent,
    TransferMatcher,
    TransferRecommendation,
)

# Optional Rust-accelerated backend
try:
    from .rust_algebra import RUST_AVAILABLE, RustPhasorAlgebra
except ImportError:
    RUST_AVAILABLE = False
