"""Auto-detection of data source format.

Given a file or directory, determines which adapter to use
based on file naming patterns, header signatures, and column names.

Detection order (first match wins):
1. Orgill PO — "Shipto:" signature in first line
2. Sample Store Inventory — column signature match
3. Do It Best — (future) TBD
4. Ace Hardware — (future) TBD
"""

from __future__ import annotations

from pathlib import Path

from .ace import AceAdapter
from .base import AdapterResult, BaseAdapter
from .do_it_best import DoItBestAdapter
from .orgill import OrgillPOAdapter
from .generic_pos import GenericPosAdapter
from .sales import SalesDataAdapter

# Registry of all adapters, ordered by detection priority
_ADAPTER_REGISTRY: list[BaseAdapter] = [
    OrgillPOAdapter(),
    GenericPosAdapter(),
    SalesDataAdapter(),
    DoItBestAdapter(),
    AceAdapter(),
]


def detect_adapter(path: str | Path) -> BaseAdapter | None:
    """Detect the appropriate adapter for a data source.

    Args:
        path: Path to a file or directory.

    Returns:
        The first adapter that can handle the path, or None.
    """
    path = Path(path)

    for adapter in _ADAPTER_REGISTRY:
        if adapter.can_handle(path):
            return adapter

    return None


def detect_and_ingest(
    path: str | Path,
    store_id: str = "default-store",
) -> AdapterResult:
    """Detect adapter and ingest data in one step.

    Args:
        path: Path to a file or directory.
        store_id: Store identifier for normalized records.

    Returns:
        AdapterResult from the detected adapter.

    Raises:
        ValueError: If no adapter can handle the path.
    """
    path = Path(path)
    adapter = detect_adapter(path)

    if adapter is None:
        return AdapterResult(
            source=str(path),
            adapter_name="unknown",
            errors=[
                f"No adapter found for: {path}. "
                "Supported formats: Orgill PO, Sample Store Inventory, Sales Data"
            ],
        )

    return adapter.ingest(path, store_id=store_id)


def list_adapters() -> list[dict[str, str]]:
    """List all registered adapters."""
    return [{"name": a.name, "class": type(a).__name__} for a in _ADAPTER_REGISTRY]
