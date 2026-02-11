"""Multi-vendor data adapters for Profit Sentinel.

Each vendor data source gets its own adapter that normalizes incoming
data to the canonical schema expected by the Rust pipeline engine.

Supported sources:
    - Orgill POs (CSV with header block, line items, short-ship markers)
    - Inventory reports (POS export with 50+ columns)
    - Sales transaction data (Phase 12)
    - Do It Best (future)
    - Ace Hardware (future)

Usage:
    from sentinel_agent.adapters import detect_and_ingest

    result = detect_and_ingest("/path/to/data")
    print(result.summary)
"""

from .base import (
    AdapterResult,
    BaseAdapter,
    NormalizedInventory,
    POLineItem,
    PurchaseOrder,
)
from .bridge import PipelineBridge, to_pipeline_csv
from .detection import detect_adapter, detect_and_ingest
from .sales import (
    SalesAdapterResult,
    SalesAggregation,
    SalesDataAdapter,
    SalesOverlay,
    SalesTransaction,
    aggregate_sales_30d,
)

__all__ = [
    "AdapterResult",
    "BaseAdapter",
    "NormalizedInventory",
    "PipelineBridge",
    "POLineItem",
    "PurchaseOrder",
    "detect_adapter",
    "detect_and_ingest",
    "to_pipeline_csv",
    # Phase 12 — Sales Data Integration
    "SalesAdapterResult",
    "SalesAggregation",
    "SalesDataAdapter",
    "SalesOverlay",
    "SalesTransaction",
    "aggregate_sales_30d",
]
