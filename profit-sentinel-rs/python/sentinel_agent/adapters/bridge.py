"""Bridge: NormalizedInventory → Rust pipeline InventoryRecord CSV.

This is the critical data bridge that connects Python POS adapters to the
Rust analysis pipeline. The Rust pipeline expects a CSV with exactly 11
columns matching the InventoryRecord struct:

    store_id, sku, qty_on_hand, unit_cost, margin_pct, sales_last_30d,
    days_since_receipt, retail_price, is_damaged, on_order_qty, is_seasonal

This module handles:
- Field renaming (sku_id → sku)
- Type coercion (int → float for Rust's f64 fields)
- Derived field computation:
    - margin_pct from (retail - cost) / retail
    - days_since_receipt from last_receipt_date
    - sales_last_30d estimated from sales_ytd
- Default values for fields not present in POS data
- Bidirectional mapping (preserves original NormalizedInventory for enrichment)

Usage:
    from sentinel_agent.adapters.bridge import PipelineBridge

    bridge = PipelineBridge()
    csv_path = bridge.to_pipeline_csv(records, "/tmp/pipeline_input.csv")
    # → Ready for SentinelEngine.run(csv_path)

    # Bidirectional: enrich Rust output with original adapter data
    enrichment = bridge.build_enrichment_index(records)
    original = enrichment["SKU-001"]  # Full NormalizedInventory
"""

from __future__ import annotations

import csv
import logging
import tempfile
from datetime import date
from pathlib import Path

from ..adapters.base import NormalizedInventory

logger = logging.getLogger("sentinel.bridge")

# Rust InventoryRecord CSV column order (must match exactly)
_PIPELINE_COLUMNS = [
    "store_id",
    "sku",
    "qty_on_hand",
    "unit_cost",
    "margin_pct",
    "sales_last_30d",
    "days_since_receipt",
    "retail_price",
    "is_damaged",
    "on_order_qty",
    "is_seasonal",
]

# Seasonal department/category keywords (case-insensitive)
_SEASONAL_KEYWORDS = frozenset(
    {
        "seasonal",
        "christmas",
        "holiday",
        "halloween",
        "easter",
        "spring",
        "summer",
        "winter",
        "fall",
        "outdoor living",
        "lawn & garden",
        "snow",
        "ice melt",
        "pool",
    }
)


class PipelineBridge:
    """Converts NormalizedInventory records to Rust pipeline CSV format.

    The bridge is bidirectional: it can produce the CSV that the Rust
    pipeline consumes, and also maintains an enrichment index so that
    Rust pipeline output can be annotated with original adapter fields
    (description, vendor, bin_location, etc.).
    """

    def __init__(
        self,
        reference_date: date | None = None,
        default_days_since_receipt: float = 365.0,
        months_elapsed_ytd: float | None = None,
    ):
        """Initialize the bridge.

        Args:
            reference_date: Date to compute days_since_receipt against.
                Defaults to today.
            default_days_since_receipt: Default when last_receipt_date
                is unknown. 365 = assume old stock.
            months_elapsed_ytd: Months elapsed in YTD period for
                sales_last_30d estimation. None = auto-calculate from
                reference_date.
        """
        self.reference_date = reference_date or date.today()

        self.default_days_since_receipt = default_days_since_receipt

        if months_elapsed_ytd is not None:
            self._months_elapsed = max(months_elapsed_ytd, 1.0)
        else:
            # Auto-calculate: Jan=1, Feb=2, etc.
            self._months_elapsed = max(self.reference_date.month, 1)

    def convert_record(self, rec: NormalizedInventory) -> dict[str, str]:
        """Convert a single NormalizedInventory to a pipeline CSV row dict.

        Returns a dict with string values matching _PIPELINE_COLUMNS.
        """
        # margin_pct: computed from retail and cost
        margin_pct = rec.margin_pct  # Uses the @property on NormalizedInventory

        # days_since_receipt: computed from last_receipt_date
        if rec.last_receipt_date is not None:
            delta = (self.reference_date - rec.last_receipt_date).days
            days_since_receipt = max(0.0, float(delta))
        else:
            days_since_receipt = self.default_days_since_receipt

        # sales_last_30d: prefer actual data (Phase 12), fall back to YTD estimate
        if rec.sales_last_30d is not None:
            # Actual 30-day sales from transaction data overlay
            sales_last_30d = round(rec.sales_last_30d, 2)
        elif rec.sales_ytd > 0:
            # Fallback: estimate from sales_ytd / months_elapsed
            monthly_avg = rec.sales_ytd / self._months_elapsed
            sales_last_30d = round(monthly_avg, 2)
        else:
            sales_last_30d = 0.0

        # is_seasonal: heuristic from department/category name
        is_seasonal = self._detect_seasonal(rec)

        # is_damaged: not available in POS data, default false
        is_damaged = False

        return {
            "store_id": rec.store_id,
            "sku": rec.sku_id,
            "qty_on_hand": str(float(rec.qty_on_hand)),
            "unit_cost": f"{rec.unit_cost:.2f}",
            "margin_pct": f"{margin_pct:.4f}",
            "sales_last_30d": f"{sales_last_30d:.2f}",
            "days_since_receipt": f"{days_since_receipt:.0f}",
            "retail_price": f"{rec.retail_price:.2f}",
            "is_damaged": str(is_damaged).lower(),
            "on_order_qty": str(float(rec.on_order_qty)),
            "is_seasonal": str(is_seasonal).lower(),
        }

    def to_pipeline_csv(
        self,
        records: list[NormalizedInventory],
        output_path: str | Path | None = None,
    ) -> Path:
        """Convert records to pipeline CSV file.

        Args:
            records: List of NormalizedInventory records from any adapter.
            output_path: Where to write the CSV. If None, uses a temp file.

        Returns:
            Path to the written CSV file.
        """
        if output_path is None:
            fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="sentinel_pipeline_")
            import os

            os.close(fd)
            output_path = Path(tmp)
        else:
            output_path = Path(output_path)

        converted = 0
        skipped = 0

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_PIPELINE_COLUMNS)
            writer.writeheader()

            for rec in records:
                try:
                    row = self.convert_record(rec)
                    writer.writerow(row)
                    converted += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 10:
                        logger.warning(
                            "Skipped SKU %s during bridge conversion: %s",
                            rec.sku_id,
                            e,
                        )

        logger.info(
            "Bridge: %d records converted, %d skipped → %s",
            converted,
            skipped,
            output_path,
        )

        return output_path

    def build_enrichment_index(
        self,
        records: list[NormalizedInventory],
    ) -> dict[str, NormalizedInventory]:
        """Build a lookup index from SKU → original NormalizedInventory.

        This enables bidirectional mapping: after the Rust pipeline
        identifies issues by SKU, we can look up the original adapter
        data (description, vendor, bin location, etc.) for rich output.

        Key format: "{store_id}::{sku_id}" for multi-store support.
        Falls back to just sku_id for single-store convenience.
        """
        index: dict[str, NormalizedInventory] = {}
        for rec in records:
            # Primary key: store_id::sku_id (handles multi-store)
            key = f"{rec.store_id}::{rec.sku_id}"
            index[key] = rec
            # Also index by bare sku_id (for single-store convenience)
            if rec.sku_id not in index:
                index[rec.sku_id] = rec
        return index

    def _detect_seasonal(self, rec: NormalizedInventory) -> bool:
        """Heuristic: detect seasonal items from department/category names."""
        for field in (rec.department, rec.category):
            if field and field.lower().strip() in _SEASONAL_KEYWORDS:
                return True
        # Also check partial matches for multi-word departments
        for field in (rec.department, rec.category):
            if field:
                lower = field.lower()
                for kw in _SEASONAL_KEYWORDS:
                    if kw in lower:
                        return True
        return False


def to_pipeline_csv(
    records: list[NormalizedInventory],
    output_path: str | Path | None = None,
    reference_date: date | None = None,
) -> Path:
    """Convenience function: convert records to pipeline CSV.

    Args:
        records: Normalized inventory records from any adapter.
        output_path: Where to write the CSV. None = temp file.
        reference_date: Date for computing days_since_receipt.

    Returns:
        Path to the pipeline-ready CSV file.
    """
    bridge = PipelineBridge(reference_date=reference_date)
    return bridge.to_pipeline_csv(records, output_path)
