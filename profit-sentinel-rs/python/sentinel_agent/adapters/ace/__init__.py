"""Ace Hardware data adapter (stub).

This adapter will handle Ace Hardware retail network exports,
including:
    - Ace Retail Systems (ARS) inventory files
    - Ace warehouse purchase orders
    - Ace Rewards loyalty data
    - Helpful Ace training / planogram data

Expected data sources:
    - ARS inventory CSV exports
    - Ace warehouse PO confirmations
    - Monthly performance reports
"""

from __future__ import annotations

from pathlib import Path

from ..base import AdapterResult, BaseAdapter


class AceAdapter(BaseAdapter):
    """Stub adapter for Ace Hardware data."""

    @property
    def name(self) -> str:
        return "Ace Hardware"

    def can_handle(self, path: Path) -> bool:
        """Detection not yet implemented."""
        # TODO: Implement when schema is defined
        # Look for Ace header signatures or filename patterns
        return False

    def ingest(self, path: Path, store_id: str = "default-store") -> AdapterResult:
        """Not yet implemented."""
        return AdapterResult(
            source=str(path),
            adapter_name=self.name,
            errors=["Ace Hardware adapter not yet implemented"],
        )


__all__ = ["AceAdapter"]
