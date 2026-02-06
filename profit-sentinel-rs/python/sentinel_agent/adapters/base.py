"""Base adapter and canonical data models.

All vendor-specific adapters inherit from BaseAdapter and normalize
their source data into NormalizedInventory and/or PurchaseOrder records.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Canonical Models — the pipeline expects these
# ---------------------------------------------------------------------------


class NormalizedInventory(BaseModel):
    """A single inventory row in canonical form.

    This is what the Rust pipeline expects. Every vendor adapter must
    map its source columns to these fields.
    """

    sku_id: str
    description: str | None = None
    vendor: str | None = None
    vendor_sku: str | None = None
    qty_on_hand: int = 0
    unit_cost: float = 0.0
    retail_price: float = 0.0
    last_receipt_date: date | None = None
    last_sale_date: date | None = None
    bin_location: str | None = None
    store_id: str = "default-store"
    category: str | None = None
    department: str | None = None
    # Extra fields useful for analysis
    barcode: str | None = None
    on_order_qty: int = 0
    min_qty: int = 0
    max_qty: int = 0
    sales_ytd: float = 0.0
    cost_ytd: float = 0.0
    # Phase 12: actual 30-day sales from transaction data (overrides YTD estimate)
    sales_last_30d: float | None = None

    @property
    def margin_pct(self) -> float:
        """Calculate margin percentage."""
        if self.retail_price <= 0:
            return 0.0
        return (self.retail_price - self.unit_cost) / self.retail_price

    @property
    def inventory_value_at_cost(self) -> float:
        return self.qty_on_hand * self.unit_cost


class POStatus(str, Enum):
    """Purchase order status."""

    INVOICED = "INVOICED"
    PENDING = "PENDING"
    OPEN = "OPEN"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class POLineItem(BaseModel):
    """A single line item on a purchase order."""

    line_number: int
    sku_id: str
    description: str = ""
    qty_ordered: int = 0
    qty_filled: int = 0
    unit_cost: float = 0.0
    ext_cost: float = 0.0
    retail_price: float = 0.0
    upc: str | None = None
    department: str | None = None
    department_code: str | None = None
    unit_of_measure: str | None = None
    is_short_ship: bool = False
    is_special_price: bool = False
    country_of_origin: str | None = None
    vendor_item_number: str | None = None

    @property
    def short_ship_qty(self) -> int:
        """Quantity not filled."""
        return max(0, self.qty_ordered - self.qty_filled)

    @property
    def short_ship_value(self) -> float:
        """Dollar value of unfilled items."""
        return self.short_ship_qty * self.unit_cost

    @property
    def fill_rate(self) -> float:
        """Fill rate as a fraction (0.0 to 1.0)."""
        if self.qty_ordered <= 0:
            return 1.0
        return self.qty_filled / self.qty_ordered


class PurchaseOrder(BaseModel):
    """A normalized purchase order."""

    po_number: str
    vendor: str = "Orgill"
    order_date: date | None = None
    status: POStatus = POStatus.UNKNOWN
    terms: str | None = None
    ship_to: str | None = None
    bill_to: str | None = None
    usd_amount: float = 0.0
    line_items: list[POLineItem] = Field(default_factory=list)

    @property
    def total_ordered_value(self) -> float:
        """Total value of all items ordered."""
        return sum(
            item.qty_ordered * item.unit_cost
            for item in self.line_items
            if item.qty_ordered > 0
        )

    @property
    def total_filled_value(self) -> float:
        """Total value of items actually filled."""
        return sum(item.ext_cost for item in self.line_items)

    @property
    def total_short_ship_value(self) -> float:
        """Total dollar value of short-shipped items."""
        return sum(item.short_ship_value for item in self.line_items)

    @property
    def total_line_items(self) -> int:
        return len(self.line_items)

    @property
    def short_ship_count(self) -> int:
        return sum(1 for item in self.line_items if item.is_short_ship)

    @property
    def fill_rate(self) -> float:
        """Overall PO fill rate by quantity."""
        total_ordered = sum(
            item.qty_ordered for item in self.line_items if item.qty_ordered > 0
        )
        total_filled = sum(
            item.qty_filled for item in self.line_items if item.qty_ordered > 0
        )
        if total_ordered == 0:
            return 1.0
        return total_filled / total_ordered

    @property
    def product_line_items(self) -> list[POLineItem]:
        """Line items that are actual products (not service charges)."""
        return [item for item in self.line_items if item.sku_id != "0000000"]


# ---------------------------------------------------------------------------
# Adapter Result
# ---------------------------------------------------------------------------


class AdapterResult(BaseModel):
    """Result of running an adapter on a data source."""

    source: str
    adapter_name: str
    inventory_records: list[NormalizedInventory] = Field(default_factory=list)
    purchase_orders: list[PurchaseOrder] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    files_processed: int = 0
    processing_time_ms: int = 0

    @property
    def total_inventory_records(self) -> int:
        return len(self.inventory_records)

    @property
    def total_purchase_orders(self) -> int:
        return len(self.purchase_orders)

    @property
    def total_po_line_items(self) -> int:
        return sum(po.total_line_items for po in self.purchase_orders)

    @property
    def total_short_ships(self) -> int:
        return sum(po.short_ship_count for po in self.purchase_orders)

    @property
    def total_short_ship_value(self) -> float:
        return sum(po.total_short_ship_value for po in self.purchase_orders)

    @property
    def total_ordered_value(self) -> float:
        return sum(po.total_ordered_value for po in self.purchase_orders)

    @property
    def total_filled_value(self) -> float:
        return sum(po.total_filled_value for po in self.purchase_orders)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def summary(self) -> str:
        """Human-readable summary of the ingestion result."""
        parts: list[str] = []
        parts.append(f"Adapter: {self.adapter_name}")
        parts.append(f"Source: {self.source}")
        parts.append(f"Files processed: {self.files_processed}")

        if self.inventory_records:
            parts.append(f"Inventory records: {self.total_inventory_records:,}")
            total_value = sum(r.inventory_value_at_cost for r in self.inventory_records)
            parts.append(f"Total inventory value: ${total_value:,.2f}")

        if self.purchase_orders:
            parts.append(f"Purchase orders: {self.total_purchase_orders}")
            parts.append(f"Total PO line items: {self.total_po_line_items:,}")
            parts.append(f"Total ordered value: ${self.total_ordered_value:,.2f}")
            parts.append(f"Total filled value: ${self.total_filled_value:,.2f}")
            parts.append(
                f"Short-ships: {self.total_short_ships} items "
                f"(${self.total_short_ship_value:,.2f})"
            )
            overall_fill = (
                self.total_filled_value / self.total_ordered_value * 100
                if self.total_ordered_value > 0
                else 100.0
            )
            parts.append(f"Overall fill rate: {overall_fill:.1f}%")

        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            parts.append(f"Warnings: {len(self.warnings)}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Abstract Base Adapter
# ---------------------------------------------------------------------------


class BaseAdapter(ABC):
    """Abstract base class for all data source adapters.

    To add a new vendor:
    1. Subclass BaseAdapter
    2. Implement can_handle() for auto-detection
    3. Implement ingest() to parse and normalize
    4. Register detection signature in detection.py
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""
        ...

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """Return True if this adapter can parse the given file/directory."""
        ...

    @abstractmethod
    def ingest(self, path: Path, store_id: str = "default-store") -> AdapterResult:
        """Parse and normalize the data source.

        Args:
            path: Path to a file or directory.
            store_id: Store identifier for normalized records.

        Returns:
            AdapterResult with normalized records and/or POs.
        """
        ...
