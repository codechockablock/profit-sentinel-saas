"""Do It Best data adapter (stub).

This adapter will handle Do It Best co-op inventory exports,
purchase orders, and patronage reports once the schema is defined.

Expected data sources:
    - Member inventory reports
    - Warehouse purchase orders
    - Promotional program purchases
    - Patronage dividend statements
"""

from __future__ import annotations

from pathlib import Path

from ..base import AdapterResult, BaseAdapter


class DoItBestAdapter(BaseAdapter):
    """Stub adapter for Do It Best co-op data."""

    @property
    def name(self) -> str:
        return "Do It Best"

    def can_handle(self, path: Path) -> bool:
        """Detection not yet implemented."""
        # TODO: Implement when schema is defined
        # Look for Do It Best header signatures or filename patterns
        return False

    def ingest(self, path: Path, store_id: str = "default-store") -> AdapterResult:
        """Not yet implemented."""
        return AdapterResult(
            source=str(path),
            adapter_name=self.name,
            errors=["Do It Best adapter not yet implemented"],
        )


__all__ = ["DoItBestAdapter"]
