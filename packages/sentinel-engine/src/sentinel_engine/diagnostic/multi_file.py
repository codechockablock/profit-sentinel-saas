"""
Multi-File Diagnostic Engine with Vendor Correlation

PREMIUM PREVIEW FEATURE

Handles up to 200 files, cross-references inventory with vendor invoices,
and discovers causal patterns between vendor behavior and inventory anomalies.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐
    │ Inventory Files │     │ Vendor Invoices │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────────────────────────────┐
    │         SKU Normalization Layer         │
    └─────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │      Vendor Invoice Aggregator          │
    │  • Aggregate by SKU across all files    │
    │  • Calculate fill rates, variances      │
    └─────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │      Causal Correlation Engine          │
    │  • Short ship → Negative inventory      │
    │  • Price change → Margin erosion        │
    │  • Encode discoveries to Dorian         │
    └─────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────┐
    │   Enhanced Conversational Diagnostic    │
    └─────────────────────────────────────────┘

Usage:
    from sentinel_engine.diagnostic.multi_file import MultiFileDiagnostic

    diagnostic = MultiFileDiagnostic()

    # Add inventory files
    diagnostic.add_inventory_file("inventory.csv")

    # Add vendor invoices (up to 200)
    for invoice in invoice_files:
        diagnostic.add_vendor_invoice(invoice)

    # Configure SKU matching
    diagnostic.configure_matching(
        inventory_sku_column="SKU",
        invoice_item_column="Item Number",
        vendor_column="Vendor Code"
    )

    # Start session with correlation
    session = diagnostic.start_session()
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# =============================================================================
# COLUMN MAPPINGS (Universal POS/Invoice Support)
# =============================================================================

INVENTORY_SKU_ALIASES = [
    "sku",
    "item_number",
    "item_no",
    "item",
    "upc",
    "barcode",
    "plu",
    "product_id",
    "item_id",
    "partnumber",
    "itemlookupcode",
    "system_id",
    "custom_sku",
    "handle",
    "variant_sku",
    "article_code",
    "stock_code",
    "item_code",
    "material_number",
    "internal_id",
]

INVOICE_ITEM_ALIASES = [
    "item_number",
    "item_no",
    "item",
    "sku",
    "vendor_item",
    "vendor_sku",
    "product_code",
    "part_number",
    "catalog_number",
    "mfg_part",
    "upc",
    "line_item",
    "material",
    "article",
]

VENDOR_ALIASES = [
    "vendor",
    "vendor_code",
    "vendor_id",
    "supplier",
    "supplier_code",
    "vendor_name",
    "supplier_name",
    "mfg",
    "manufacturer",
    "brand",
    "distributor",
    "wholesaler",
    "vendor_no",
    "vend_no",
]

QTY_ORDERED_ALIASES = [
    "qty_ordered",
    "ordered",
    "order_qty",
    "quantity_ordered",
    "ord_qty",
    "ordered_qty",
    "po_qty",
    "purchase_qty",
    "order_quantity",
]

QTY_SHIPPED_ALIASES = [
    "qty_shipped",
    "shipped",
    "ship_qty",
    "quantity_shipped",
    "filled",
    "qty_filled",
    "received",
    "qty_received",
    "recv_qty",
    "filled_qty",
    "actual_qty",
    "delivered",
]

INVOICE_COST_ALIASES = [
    "unit_cost",
    "cost",
    "price",
    "unit_price",
    "invoice_cost",
    "po_cost",
    "purchase_price",
    "buy_price",
    "vendor_cost",
    "ext_cost",
    "line_cost",
]

INVOICE_DATE_ALIASES = [
    "date",
    "invoice_date",
    "po_date",
    "order_date",
    "ship_date",
    "received_date",
    "document_date",
    "trans_date",
    "transaction_date",
]

PO_NUMBER_ALIASES = [
    "po_number",
    "po_no",
    "po",
    "purchase_order",
    "order_number",
    "order_no",
    "document_number",
    "doc_no",
    "invoice_number",
    "invoice_no",
]


# =============================================================================
# DATA CLASSES
# =============================================================================


class FileType(Enum):
    INVENTORY = "inventory"
    VENDOR_INVOICE = "vendor_invoice"
    PURCHASE_ORDER = "purchase_order"
    UNKNOWN = "unknown"


@dataclass
class ColumnMapping:
    """Mapping of logical columns to actual column names in file."""

    sku: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[str] = None
    cost: Optional[str] = None
    retail: Optional[str] = None
    vendor: Optional[str] = None
    qty_ordered: Optional[str] = None
    qty_shipped: Optional[str] = None
    po_number: Optional[str] = None
    date: Optional[str] = None


@dataclass
class ParsedFile:
    """A parsed CSV file with metadata."""

    filename: str
    file_type: FileType
    columns: list[str]
    column_mapping: ColumnMapping
    rows: list[dict[str, Any]]
    row_count: int
    parse_errors: list[str] = field(default_factory=list)


@dataclass
class InventoryItem:
    """Normalized inventory item."""

    sku: str
    description: str
    quantity: float
    cost: float
    retail: float
    vendor: Optional[str] = None
    margin_pct: float = 0.0
    source_file: str = ""

    def __post_init__(self):
        if self.retail > 0 and self.cost >= 0:
            self.margin_pct = ((self.retail - self.cost) / self.retail) * 100


@dataclass
class VendorInvoiceLine:
    """Single line from a vendor invoice."""

    item_number: str
    description: str
    vendor_code: str
    qty_ordered: float
    qty_shipped: float
    unit_cost: float
    po_number: str
    invoice_date: Optional[datetime] = None
    source_file: str = ""

    @property
    def fill_rate(self) -> float:
        if self.qty_ordered <= 0:
            return 1.0
        return self.qty_shipped / self.qty_ordered

    @property
    def short_qty(self) -> float:
        return max(0, self.qty_ordered - self.qty_shipped)

    @property
    def is_short_ship(self) -> bool:
        return self.qty_shipped < self.qty_ordered and self.qty_ordered > 0


@dataclass
class AggregatedVendorData:
    """Aggregated vendor invoice data for a single SKU."""

    sku: str = ""
    matched_item_numbers: set[str] = field(default_factory=set)
    vendor_codes: set[str] = field(default_factory=set)
    total_ordered: float = 0.0
    total_shipped: float = 0.0
    total_short: float = 0.0
    invoice_count: int = 0
    short_ship_count: int = 0
    cost_history: list[tuple[datetime, float]] = field(default_factory=list)
    min_cost: float = float("inf")
    max_cost: float = 0.0
    avg_cost: float = 0.0
    first_invoice: Optional[datetime] = None
    last_invoice: Optional[datetime] = None

    @property
    def fill_rate(self) -> float:
        if self.total_ordered <= 0:
            return 1.0
        return self.total_shipped / self.total_ordered

    @property
    def short_ship_rate(self) -> float:
        if self.invoice_count <= 0:
            return 0.0
        return self.short_ship_count / self.invoice_count

    @property
    def cost_variance(self) -> float:
        if self.min_cost == float("inf") or self.max_cost == 0:
            return 0.0
        return self.max_cost - self.min_cost

    @property
    def cost_variance_pct(self) -> float:
        if self.min_cost == float("inf") or self.min_cost == 0:
            return 0.0
        return ((self.max_cost - self.min_cost) / self.min_cost) * 100


@dataclass
class VendorSummary:
    """Summary statistics for a vendor across all invoices."""

    vendor_code: str = ""
    vendor_name: Optional[str] = None
    total_lines: int = 0
    total_ordered: float = 0.0
    total_shipped: float = 0.0
    total_short: float = 0.0
    short_ship_lines: int = 0
    unique_skus: set[str] = field(default_factory=set)
    total_cost: float = 0.0
    invoice_count: int = 0

    @property
    def fill_rate(self) -> float:
        if self.total_ordered <= 0:
            return 1.0
        return self.total_shipped / self.total_ordered

    @property
    def short_ship_rate(self) -> float:
        if self.total_lines <= 0:
            return 0.0
        return self.short_ship_lines / self.total_lines


# =============================================================================
# CORRELATION PATTERNS
# =============================================================================


class CorrelationType(Enum):
    SHORT_SHIP_NEGATIVE_STOCK = "short_ship_negative_stock"
    PRICE_INCREASE_MARGIN_EROSION = "price_increase_margin_erosion"
    CHRONIC_SHORT_SHIP = "chronic_short_ship"
    VENDOR_FILL_RATE_ISSUE = "vendor_fill_rate_issue"
    COST_VARIANCE_ANOMALY = "cost_variance_anomaly"
    CREDIT_NOT_APPLIED = "credit_not_applied"
    RECEIVING_GAP = "receiving_gap"


@dataclass
class CorrelationPattern:
    """A discovered correlation between vendor and inventory data."""

    correlation_type: CorrelationType
    confidence: float  # 0.0 to 1.0
    affected_skus: list[str]
    affected_vendors: list[str]
    total_impact: float  # Dollar impact
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)

    # For conversational diagnostic
    question: str = ""
    suggested_answers: list[tuple[str, str]] = field(default_factory=list)


# =============================================================================
# SKU NORMALIZATION
# =============================================================================


class SKUNormalizer:
    """
    Normalizes SKU formats across different files and systems.

    Handles:
    - Leading zeros (0001234 vs 1234)
    - Prefixes/suffixes (SKU-1234 vs 1234)
    - Case variations
    - UPC/EAN matching
    - Vendor item number → internal SKU mapping
    """

    def __init__(self):
        self.sku_map: dict[str, str] = {}  # normalized → canonical
        self.reverse_map: dict[str, set[str]] = defaultdict(
            set
        )  # canonical → all variants
        self.vendor_item_map: dict[str, str] = {}  # vendor_item → canonical SKU

    def normalize(self, sku: str) -> str:
        """Normalize a single SKU to standard format."""
        if not sku:
            return ""

        # Convert to string and strip
        sku = str(sku).strip().upper()

        # Remove common prefixes
        prefixes = ["SKU-", "SKU", "ITEM-", "ITEM", "PRD-", "PRD"]
        for prefix in prefixes:
            if sku.startswith(prefix):
                sku = sku[len(prefix) :]

        # Remove leading zeros (but keep at least one digit)
        sku_stripped = sku.lstrip("0") or "0"

        # Check if we've seen this before
        if sku in self.sku_map:
            return self.sku_map[sku]
        if sku_stripped in self.sku_map:
            return self.sku_map[sku_stripped]

        return sku_stripped

    def register_sku(self, sku: str, canonical: Optional[str] = None) -> str:
        """Register a SKU and optionally set its canonical form."""
        normalized = self.normalize(sku)

        if canonical:
            canonical_norm = self.normalize(canonical)
            self.sku_map[normalized] = canonical_norm
            self.sku_map[sku.upper()] = canonical_norm
            self.reverse_map[canonical_norm].add(normalized)
            self.reverse_map[canonical_norm].add(sku.upper())
            return canonical_norm

        # Use normalized form as canonical
        self.sku_map[normalized] = normalized
        self.sku_map[sku.upper()] = normalized
        self.reverse_map[normalized].add(sku.upper())
        return normalized

    def register_vendor_item(self, vendor_item: str, internal_sku: str):
        """Map a vendor item number to an internal SKU."""
        vendor_norm = self.normalize(vendor_item)
        sku_norm = self.normalize(internal_sku)
        self.vendor_item_map[vendor_norm] = sku_norm
        self.reverse_map[sku_norm].add(vendor_norm)

    def match_vendor_item(self, vendor_item: str) -> Optional[str]:
        """Try to match a vendor item number to a known SKU."""
        vendor_norm = self.normalize(vendor_item)

        # Direct mapping
        if vendor_norm in self.vendor_item_map:
            return self.vendor_item_map[vendor_norm]

        # Try as a SKU directly
        if vendor_norm in self.sku_map:
            return self.sku_map[vendor_norm]

        return None

    def get_all_variants(self, sku: str) -> set[str]:
        """Get all known variants of a SKU."""
        canonical = self.normalize(sku)
        if canonical in self.sku_map:
            canonical = self.sku_map[canonical]
        return self.reverse_map.get(canonical, {canonical})


# =============================================================================
# FILE PARSER
# =============================================================================


class MultiFileParser:
    """
    Parses multiple CSV files with automatic column detection.

    Supports:
    - Inventory exports from any POS
    - Vendor invoices in various formats
    - Up to 200 files
    """

    MAX_FILES = 200

    def __init__(self):
        self.parsed_files: list[ParsedFile] = []
        self.inventory_files: list[ParsedFile] = []
        self.invoice_files: list[ParsedFile] = []

    def detect_file_type(self, columns: list[str]) -> FileType:
        """Detect whether file is inventory or vendor invoice."""
        cols_lower = [c.lower().strip() for c in columns]

        # Check for invoice indicators
        invoice_indicators = [
            "qty_ordered",
            "qty_shipped",
            "ordered",
            "shipped",
            "filled",
            "po_number",
            "invoice",
        ]
        has_invoice_cols = sum(
            1 for ind in invoice_indicators if any(ind in c for c in cols_lower)
        )

        # Check for inventory indicators
        inventory_indicators = [
            "on_hand",
            "in_stock",
            "quantity",
            "qoh",
            "retail",
            "margin",
            "sold",
        ]
        has_inventory_cols = sum(
            1 for ind in inventory_indicators if any(ind in c for c in cols_lower)
        )

        if has_invoice_cols >= 2:
            return FileType.VENDOR_INVOICE
        elif has_inventory_cols >= 2:
            return FileType.INVENTORY
        else:
            return FileType.UNKNOWN

    def detect_column_mapping(
        self, columns: list[str], file_type: FileType
    ) -> ColumnMapping:
        """Auto-detect column mapping based on column names."""
        mapping = ColumnMapping()
        cols_lower = {c.lower().strip(): c for c in columns}

        def find_column(aliases: list[str]) -> Optional[str]:
            for alias in aliases:
                # Exact match
                if alias in cols_lower:
                    return cols_lower[alias]
                # Partial match
                for col_lower, col_orig in cols_lower.items():
                    if alias in col_lower:
                        return col_orig
            return None

        # Common columns
        mapping.sku = find_column(INVENTORY_SKU_ALIASES)
        mapping.vendor = find_column(VENDOR_ALIASES)
        mapping.description = find_column(
            ["description", "desc", "name", "item_name", "product_name"]
        )

        if file_type == FileType.INVENTORY:
            mapping.quantity = find_column(
                [
                    "in stock qty.",
                    "in stock qty",
                    "quantity",
                    "qty",
                    "qoh",
                    "on_hand",
                    "stock",
                ]
            )
            mapping.cost = find_column(["cost", "unit_cost", "avg_cost"])
            mapping.retail = find_column(
                ["retail", "price", "sell_price", "sug. retail"]
            )

        elif file_type == FileType.VENDOR_INVOICE:
            mapping.sku = find_column(INVOICE_ITEM_ALIASES) or mapping.sku
            mapping.qty_ordered = find_column(QTY_ORDERED_ALIASES)
            mapping.qty_shipped = find_column(QTY_SHIPPED_ALIASES)
            mapping.cost = find_column(INVOICE_COST_ALIASES)
            mapping.po_number = find_column(PO_NUMBER_ALIASES)
            mapping.date = find_column(INVOICE_DATE_ALIASES)

        return mapping

    def parse_file(
        self,
        file_path: Union[str, Path],
        content: Optional[str] = None,
        override_type: Optional[FileType] = None,
    ) -> ParsedFile:
        """Parse a single CSV file."""
        filename = str(file_path) if isinstance(file_path, Path) else file_path

        # Read content
        if content is None:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()

        # Parse CSV
        reader = csv.DictReader(StringIO(content))
        columns = reader.fieldnames or []

        # Detect type and mapping
        file_type = override_type or self.detect_file_type(columns)
        column_mapping = self.detect_column_mapping(columns, file_type)

        # Parse rows
        rows = []
        errors = []
        for i, row in enumerate(reader):
            try:
                rows.append(dict(row))
            except Exception as e:
                errors.append(f"Row {i + 2}: {str(e)}")

        parsed = ParsedFile(
            filename=filename,
            file_type=file_type,
            columns=columns,
            column_mapping=column_mapping,
            rows=rows,
            row_count=len(rows),
            parse_errors=errors,
        )

        # Categorize
        if file_type == FileType.INVENTORY:
            self.inventory_files.append(parsed)
        elif file_type == FileType.VENDOR_INVOICE:
            self.invoice_files.append(parsed)

        self.parsed_files.append(parsed)
        return parsed

    def parse_multiple(
        self, files: list[Union[str, Path, tuple[str, str]]]
    ) -> list[ParsedFile]:
        """
        Parse multiple files.

        Args:
            files: List of file paths or (filename, content) tuples
        """
        if len(files) > self.MAX_FILES:
            raise ValueError(
                f"Maximum {self.MAX_FILES} files allowed, got {len(files)}"
            )

        results = []
        for item in files:
            if isinstance(item, tuple):
                filename, content = item
                results.append(self.parse_file(filename, content=content))
            else:
                results.append(self.parse_file(item))

        return results


# =============================================================================
# VENDOR INVOICE AGGREGATOR
# =============================================================================


class VendorInvoiceAggregator:
    """
    Aggregates vendor invoice data across multiple files.

    Calculates:
    - Fill rates by SKU and vendor
    - Cost variances over time
    - Short ship patterns
    - Vendor performance metrics
    """

    def __init__(self, normalizer: SKUNormalizer):
        self.normalizer = normalizer
        self.invoice_lines: list[VendorInvoiceLine] = []
        self.by_sku: dict[str, AggregatedVendorData] = defaultdict(AggregatedVendorData)
        self.by_vendor: dict[str, VendorSummary] = defaultdict(VendorSummary)

    def add_parsed_file(self, parsed: ParsedFile):
        """Add a parsed invoice file to aggregation."""
        if parsed.file_type != FileType.VENDOR_INVOICE:
            return

        mapping = parsed.column_mapping

        for row in parsed.rows:
            try:
                # Extract fields
                item_number = str(row.get(mapping.sku, "")).strip()
                if not item_number:
                    continue

                description = str(row.get(mapping.description, "")).strip()
                vendor_code = str(row.get(mapping.vendor, "UNKNOWN")).strip().upper()

                qty_ordered = self._parse_float(row.get(mapping.qty_ordered, 0))
                qty_shipped = self._parse_float(row.get(mapping.qty_shipped, 0))
                unit_cost = self._parse_float(row.get(mapping.cost, 0))
                po_number = str(row.get(mapping.po_number, "")).strip()

                # Parse date
                invoice_date = None
                if mapping.date and row.get(mapping.date):
                    invoice_date = self._parse_date(row[mapping.date])

                # Create line item
                line = VendorInvoiceLine(
                    item_number=item_number,
                    description=description,
                    vendor_code=vendor_code,
                    qty_ordered=qty_ordered,
                    qty_shipped=qty_shipped,
                    unit_cost=unit_cost,
                    po_number=po_number,
                    invoice_date=invoice_date,
                    source_file=parsed.filename,
                )

                self.invoice_lines.append(line)

                # Aggregate by SKU
                sku = self.normalizer.normalize(item_number)
                self._aggregate_by_sku(sku, line)

                # Aggregate by vendor
                self._aggregate_by_vendor(vendor_code, sku, line)

            except Exception:
                continue

    def _parse_float(self, value: Any) -> float:
        """Safely parse a float value."""
        if value is None:
            return 0.0
        try:
            # Remove currency symbols and commas
            cleaned = str(value).replace("$", "").replace(",", "").strip()
            return float(cleaned) if cleaned else 0.0
        except Exception:
            return 0.0

    def _parse_date(self, value: str) -> Optional[datetime]:
        """Try to parse a date string."""
        if not value:
            return None

        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%d/%m/%Y",
            "%Y%m%d",
            "%m-%d-%Y",
            "%d-%m-%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(str(value).strip(), fmt)
            except Exception:
                continue
        return None

    def _aggregate_by_sku(self, sku: str, line: VendorInvoiceLine):
        """Aggregate data for a single SKU."""
        agg = self.by_sku[sku]
        agg.sku = sku
        agg.matched_item_numbers.add(line.item_number)
        agg.vendor_codes.add(line.vendor_code)

        agg.total_ordered += line.qty_ordered
        agg.total_shipped += line.qty_shipped
        agg.total_short += line.short_qty
        agg.invoice_count += 1

        if line.is_short_ship:
            agg.short_ship_count += 1

        if line.unit_cost > 0:
            agg.cost_history.append(
                (line.invoice_date or datetime.now(), line.unit_cost)
            )
            agg.min_cost = min(agg.min_cost, line.unit_cost)
            agg.max_cost = max(agg.max_cost, line.unit_cost)

        if line.invoice_date:
            if agg.first_invoice is None or line.invoice_date < agg.first_invoice:
                agg.first_invoice = line.invoice_date
            if agg.last_invoice is None or line.invoice_date > agg.last_invoice:
                agg.last_invoice = line.invoice_date

    def _aggregate_by_vendor(self, vendor_code: str, sku: str, line: VendorInvoiceLine):
        """Aggregate data for a vendor."""
        vendor = self.by_vendor[vendor_code]
        vendor.vendor_code = vendor_code
        vendor.total_lines += 1
        vendor.total_ordered += line.qty_ordered
        vendor.total_shipped += line.qty_shipped
        vendor.total_short += line.short_qty
        vendor.total_cost += line.unit_cost * line.qty_shipped
        vendor.unique_skus.add(sku)

        if line.is_short_ship:
            vendor.short_ship_lines += 1

    def finalize(self):
        """Calculate final aggregated metrics."""
        for sku, agg in self.by_sku.items():
            if agg.cost_history:
                costs = [c[1] for c in agg.cost_history]
                agg.avg_cost = sum(costs) / len(costs)

    def get_chronic_short_ships(
        self, min_rate: float = 0.3, min_count: int = 3
    ) -> list[AggregatedVendorData]:
        """Get SKUs with chronic short ship issues."""
        return [
            agg
            for agg in self.by_sku.values()
            if agg.short_ship_rate >= min_rate and agg.short_ship_count >= min_count
        ]

    def get_cost_variance_anomalies(
        self, min_variance_pct: float = 15.0
    ) -> list[AggregatedVendorData]:
        """Get SKUs with significant cost variance."""
        return [
            agg
            for agg in self.by_sku.values()
            if agg.cost_variance_pct >= min_variance_pct
        ]

    def get_poor_vendors(self, max_fill_rate: float = 0.85) -> list[VendorSummary]:
        """Get vendors with poor fill rates."""
        return [
            vendor
            for vendor in self.by_vendor.values()
            if vendor.fill_rate <= max_fill_rate and vendor.total_lines >= 10
        ]


# =============================================================================
# CORRELATION ENGINE
# =============================================================================


class CorrelationEngine:
    """
    Discovers causal patterns between vendor behavior and inventory anomalies.

    Patterns detected:
    - Short ship → Negative inventory
    - Price increase → Margin erosion
    - Chronic vendor issues
    - Receiving gaps
    """

    def __init__(self, normalizer: SKUNormalizer):
        self.normalizer = normalizer
        self.inventory: dict[str, InventoryItem] = {}
        self.vendor_data: dict[str, AggregatedVendorData] = {}
        self.vendor_summaries: dict[str, VendorSummary] = {}
        self.discovered_patterns: list[CorrelationPattern] = []

    def load_inventory(self, items: list[InventoryItem]):
        """Load inventory items for correlation."""
        for item in items:
            sku = self.normalizer.normalize(item.sku)
            self.inventory[sku] = item
            self.normalizer.register_sku(item.sku)

    def load_vendor_data(self, aggregator: VendorInvoiceAggregator):
        """Load aggregated vendor data."""
        self.vendor_data = dict(aggregator.by_sku)
        self.vendor_summaries = dict(aggregator.by_vendor)

    def discover_all_patterns(self) -> list[CorrelationPattern]:
        """Run all pattern detection algorithms."""
        self.discovered_patterns = []

        self.discovered_patterns.extend(self._find_short_ship_negative_stock())
        self.discovered_patterns.extend(self._find_price_increase_margin_erosion())
        self.discovered_patterns.extend(self._find_chronic_short_ships())
        self.discovered_patterns.extend(self._find_vendor_fill_rate_issues())
        self.discovered_patterns.extend(self._find_cost_variance_anomalies())
        self.discovered_patterns.extend(self._find_receiving_gaps())

        # Sort by impact
        self.discovered_patterns.sort(key=lambda p: p.total_impact, reverse=True)

        return self.discovered_patterns

    def _find_short_ship_negative_stock(self) -> list[CorrelationPattern]:
        """Find SKUs where short ships correlate with negative inventory."""
        patterns = []

        affected_skus = []
        affected_vendors: set = set()
        total_impact = 0.0
        evidence = []

        for sku, inv_item in self.inventory.items():
            if inv_item.quantity >= 0:
                continue

            # Check for vendor short ships on this SKU
            vendor_agg = self.vendor_data.get(sku)
            if not vendor_agg or vendor_agg.total_short <= 0:
                continue

            # Correlation found!
            affected_skus.append(sku)
            affected_vendors.update(vendor_agg.vendor_codes)
            impact = abs(inv_item.quantity) * inv_item.cost
            total_impact += impact

            evidence.append(
                {
                    "sku": sku,
                    "description": inv_item.description,
                    "negative_qty": inv_item.quantity,
                    "short_shipped_qty": vendor_agg.total_short,
                    "fill_rate": vendor_agg.fill_rate,
                    "vendors": list(vendor_agg.vendor_codes),
                    "impact": impact,
                }
            )

        if affected_skus:
            patterns.append(
                CorrelationPattern(
                    correlation_type=CorrelationType.SHORT_SHIP_NEGATIVE_STOCK,
                    confidence=min(0.95, 0.5 + (len(affected_skus) / 100)),
                    affected_skus=affected_skus,
                    affected_vendors=list(affected_vendors),
                    total_impact=total_impact,
                    description=f"Found {len(affected_skus)} items with short ships that now show negative inventory",
                    evidence={"items": evidence[:20]},  # Top 20
                    question=f"I found {len(affected_skus)} items where vendors short-shipped and you now have negative stock. "
                    f"Total impact: ${total_impact:,.0f}. Is this a receiving issue or actual shortage?",
                    suggested_answers=[
                        ("Receiving gap - we sell before receiving", "receiving_gap"),
                        ("Vendor reliability issue - need to switch", "vendor_issue"),
                        ("Mixed - some of each", "mixed"),
                        ("Investigate these individually", "investigate"),
                    ],
                )
            )

        return patterns

    def _find_price_increase_margin_erosion(self) -> list[CorrelationPattern]:
        """Find SKUs where cost increases correlate with margin problems."""
        patterns = []

        affected_skus = []
        affected_vendors: set = set()
        total_impact = 0.0
        evidence = []

        for sku, inv_item in self.inventory.items():
            if inv_item.margin_pct >= 20:  # Margin is okay
                continue

            vendor_agg = self.vendor_data.get(sku)
            if not vendor_agg or vendor_agg.cost_variance_pct < 10:
                continue

            # Check if current cost is near max (price went up)
            if vendor_agg.avg_cost < vendor_agg.max_cost * 0.9:
                continue

            # Correlation found!
            affected_skus.append(sku)
            affected_vendors.update(vendor_agg.vendor_codes)

            # Estimate impact (lost margin)
            margin_loss_pct = (vendor_agg.cost_variance_pct / 100) * inv_item.retail
            impact = margin_loss_pct * max(1, abs(inv_item.quantity))
            total_impact += impact

            evidence.append(
                {
                    "sku": sku,
                    "description": inv_item.description,
                    "current_margin": inv_item.margin_pct,
                    "cost_increase_pct": vendor_agg.cost_variance_pct,
                    "min_cost": vendor_agg.min_cost,
                    "max_cost": vendor_agg.max_cost,
                    "current_retail": inv_item.retail,
                    "vendors": list(vendor_agg.vendor_codes),
                    "impact": impact,
                }
            )

        if affected_skus:
            patterns.append(
                CorrelationPattern(
                    correlation_type=CorrelationType.PRICE_INCREASE_MARGIN_EROSION,
                    confidence=min(0.90, 0.5 + (len(affected_skus) / 50)),
                    affected_skus=affected_skus,
                    affected_vendors=list(affected_vendors),
                    total_impact=total_impact,
                    description=f"Found {len(affected_skus)} items where vendor cost increases eroded margins",
                    evidence={"items": evidence[:20]},
                    question=f"I found {len(affected_skus)} items where vendor costs increased but retail prices weren't updated. "
                    f"Estimated margin loss: ${total_impact:,.0f}. Should retail be updated?",
                    suggested_answers=[
                        ("Yes - update retail prices", "update_retail"),
                        ("No - accept lower margin", "accept_margin"),
                        ("Some - review individually", "review"),
                        ("Find alternative vendors", "vendor_switch"),
                    ],
                )
            )

        return patterns

    def _find_chronic_short_ships(self) -> list[CorrelationPattern]:
        """Find SKUs with chronic short ship patterns."""
        patterns = []

        chronic_items = [
            (sku, agg)
            for sku, agg in self.vendor_data.items()
            if agg.short_ship_rate >= 0.3 and agg.short_ship_count >= 3
        ]

        if not chronic_items:
            return patterns

        # Group by vendor
        by_vendor: dict[str, list] = defaultdict(list)
        for sku, agg in chronic_items:
            for vendor in agg.vendor_codes:
                by_vendor[vendor].append((sku, agg))

        for vendor, items in by_vendor.items():
            if len(items) < 3:
                continue

            affected_skus = [sku for sku, _ in items]
            total_short = sum(agg.total_short for _, agg in items)

            # Estimate impact
            total_impact = sum(
                agg.total_short
                * (self.inventory.get(sku, InventoryItem(sku, "", 0, 10, 0)).cost)
                for sku, agg in items
            )

            patterns.append(
                CorrelationPattern(
                    correlation_type=CorrelationType.CHRONIC_SHORT_SHIP,
                    confidence=0.85,
                    affected_skus=affected_skus,
                    affected_vendors=[vendor],
                    total_impact=total_impact,
                    description=f"Vendor {vendor} chronically short-ships {len(items)} items",
                    evidence={
                        "vendor": vendor,
                        "item_count": len(items),
                        "total_short_units": total_short,
                        "avg_fill_rate": sum(agg.fill_rate for _, agg in items)
                        / len(items),
                        "items": [
                            {"sku": sku, "short_rate": agg.short_ship_rate}
                            for sku, agg in items[:10]
                        ],
                    },
                    question=f"Vendor {vendor} has chronically short-shipped {len(items)} items "
                    f"({total_short:,.0f} units short). Should you address this with the vendor?",
                    suggested_answers=[
                        ("Contact vendor about fill rates", "contact_vendor"),
                        ("Find alternative supplier", "switch_vendor"),
                        ("Increase order quantities to compensate", "over_order"),
                        ("Accept - this is normal for these items", "accept"),
                    ],
                )
            )

        return patterns

    def _find_vendor_fill_rate_issues(self) -> list[CorrelationPattern]:
        """Find vendors with overall poor performance."""
        patterns = []

        for vendor_code, summary in self.vendor_summaries.items():
            if summary.fill_rate > 0.85 or summary.total_lines < 20:
                continue

            total_impact = summary.total_short * (
                summary.total_cost / max(1, summary.total_shipped)
            )

            patterns.append(
                CorrelationPattern(
                    correlation_type=CorrelationType.VENDOR_FILL_RATE_ISSUE,
                    confidence=0.80,
                    affected_skus=list(summary.unique_skus)[:50],
                    affected_vendors=[vendor_code],
                    total_impact=total_impact,
                    description=f"Vendor {vendor_code} has {summary.fill_rate * 100:.0f}% fill rate across {summary.total_lines} lines",
                    evidence={
                        "vendor": vendor_code,
                        "fill_rate": summary.fill_rate,
                        "total_lines": summary.total_lines,
                        "short_ship_lines": summary.short_ship_lines,
                        "unique_skus": len(summary.unique_skus),
                        "total_short_units": summary.total_short,
                    },
                    question=f"Vendor {vendor_code} only fills {summary.fill_rate * 100:.0f}% of orders "
                    f"({summary.short_ship_lines} short ships out of {summary.total_lines} lines). "
                    f"Is this acceptable for this vendor?",
                    suggested_answers=[
                        ("No - escalate with vendor", "escalate"),
                        ("No - find alternative", "switch"),
                        ("Yes - this is expected for this vendor", "accept"),
                        ("Review on a case-by-case basis", "review"),
                    ],
                )
            )

        return patterns

    def _find_cost_variance_anomalies(self) -> list[CorrelationPattern]:
        """Find SKUs with unusual cost variance."""
        patterns = []

        anomalies = [
            (sku, agg)
            for sku, agg in self.vendor_data.items()
            if agg.cost_variance_pct >= 20 and agg.invoice_count >= 3
        ]

        if not anomalies:
            return patterns

        affected_skus = [sku for sku, _ in anomalies]
        affected_vendors: set = set()
        for _, agg in anomalies:
            affected_vendors.update(agg.vendor_codes)

        total_impact = sum(
            agg.cost_variance
            * max(
                1,
                abs(self.inventory.get(sku, InventoryItem(sku, "", 1, 0, 0)).quantity),
            )
            for sku, agg in anomalies
        )

        patterns.append(
            CorrelationPattern(
                correlation_type=CorrelationType.COST_VARIANCE_ANOMALY,
                confidence=0.75,
                affected_skus=affected_skus,
                affected_vendors=list(affected_vendors),
                total_impact=total_impact,
                description=f"Found {len(anomalies)} items with cost swings over 20%",
                evidence={
                    "items": [
                        {
                            "sku": sku,
                            "variance_pct": agg.cost_variance_pct,
                            "min_cost": agg.min_cost,
                            "max_cost": agg.max_cost,
                        }
                        for sku, agg in anomalies[:20]
                    ]
                },
                question=f"Found {len(anomalies)} items with cost swings over 20%. "
                f"This could indicate special pricing, rebates, or data entry errors. Review?",
                suggested_answers=[
                    ("Review for pricing errors", "review_errors"),
                    ("These are promotional prices - expected", "promotional"),
                    ("Flag for accounting review", "accounting"),
                    ("Ignore - normal variance", "ignore"),
                ],
            )
        )

        return patterns

    def _find_receiving_gaps(self) -> list[CorrelationPattern]:
        """Find potential receiving gaps (shipped but not received)."""
        patterns = []

        gaps = []
        for sku, inv_item in self.inventory.items():
            if inv_item.quantity >= 0:
                continue

            vendor_agg = self.vendor_data.get(sku)
            if not vendor_agg:
                continue

            # Negative stock but vendor shipped items
            if vendor_agg.total_shipped > 0:
                gap = min(abs(inv_item.quantity), vendor_agg.total_shipped)
                gaps.append((sku, inv_item, vendor_agg, gap))

        if not gaps:
            return patterns

        affected_skus = [sku for sku, _, _, _ in gaps]
        affected_vendors: set = set()
        for _, _, agg, _ in gaps:
            affected_vendors.update(agg.vendor_codes)

        total_gap = sum(gap for _, _, _, gap in gaps)
        total_impact = sum(gap * inv.cost for _, inv, _, gap in gaps)

        patterns.append(
            CorrelationPattern(
                correlation_type=CorrelationType.RECEIVING_GAP,
                confidence=0.70,
                affected_skus=affected_skus,
                affected_vendors=list(affected_vendors),
                total_impact=total_impact,
                description=f"Found {len(gaps)} items where vendor shipped but stock is negative (receiving gap?)",
                evidence={
                    "total_gap_units": total_gap,
                    "items": [
                        {
                            "sku": sku,
                            "current_stock": inv.quantity,
                            "vendor_shipped": agg.total_shipped,
                            "gap": gap,
                        }
                        for sku, inv, agg, gap in gaps[:20]
                    ],
                },
                question=f"Found {len(gaps)} items where vendors shipped product but your stock is negative. "
                f"Potential receiving gap: {total_gap:,.0f} units (${total_impact:,.0f}). "
                f"Are these items being sold before receiving?",
                suggested_answers=[
                    ("Yes - sold at POS before receiving", "receiving_gap"),
                    ("Receiving department issue", "receiving_issue"),
                    ("Need to investigate individually", "investigate"),
                    ("These are drop-ship items", "drop_ship"),
                ],
            )
        )

        return patterns


# =============================================================================
# MULTI-FILE DIAGNOSTIC
# =============================================================================


class MultiFileDiagnostic:
    """
    Main entry point for multi-file diagnostic with vendor correlation.

    PREMIUM PREVIEW FEATURE

    Usage:
        diagnostic = MultiFileDiagnostic()

        # Add files
        diagnostic.add_inventory_file("inventory.csv")
        for f in invoice_files:
            diagnostic.add_vendor_invoice(f)

        # Run
        session = diagnostic.start_session()

        # Get questions (now includes correlation patterns)
        while not session.is_complete:
            q = diagnostic.get_current_question()
            diagnostic.answer_question(classification)
    """

    def __init__(self):
        self.normalizer = SKUNormalizer()
        self.parser = MultiFileParser()
        self.aggregator = VendorInvoiceAggregator(self.normalizer)
        self.correlation_engine = CorrelationEngine(self.normalizer)

        self.inventory_items: list[InventoryItem] = []
        self.correlation_patterns: list[CorrelationPattern] = []

        # Session state
        self._session_started = False
        self._current_pattern_index = 0
        self._answers: list[dict] = []

    def add_inventory_file(
        self, file_path: Union[str, Path], content: Optional[str] = None
    ):
        """Add an inventory file."""
        parsed = self.parser.parse_file(
            file_path, content=content, override_type=FileType.INVENTORY
        )
        self._process_inventory_file(parsed)
        return parsed

    def add_vendor_invoice(
        self, file_path: Union[str, Path], content: Optional[str] = None
    ):
        """Add a vendor invoice file."""
        parsed = self.parser.parse_file(
            file_path, content=content, override_type=FileType.VENDOR_INVOICE
        )
        self.aggregator.add_parsed_file(parsed)
        return parsed

    def add_files_batch(self, files: list[dict[str, Any]]):
        """
        Add multiple files at once.

        Args:
            files: List of dicts with 'path' or 'content', 'filename', and optional 'type'
        """
        for f in files:
            file_type = f.get("type", "auto")
            content = f.get("content")
            path = f.get("path", f.get("filename", "unknown.csv"))

            if file_type == "inventory":
                self.add_inventory_file(path, content=content)
            elif file_type == "invoice":
                self.add_vendor_invoice(path, content=content)
            else:
                # Auto-detect
                parsed = self.parser.parse_file(path, content=content)
                if parsed.file_type == FileType.INVENTORY:
                    self._process_inventory_file(parsed)
                elif parsed.file_type == FileType.VENDOR_INVOICE:
                    self.aggregator.add_parsed_file(parsed)

    def _process_inventory_file(self, parsed: ParsedFile):
        """Process a parsed inventory file into InventoryItems."""
        mapping = parsed.column_mapping

        for row in parsed.rows:
            try:
                sku = str(row.get(mapping.sku, "")).strip()
                if not sku:
                    continue

                item = InventoryItem(
                    sku=sku,
                    description=str(row.get(mapping.description, "")).strip(),
                    quantity=self._parse_float(row.get(mapping.quantity, 0)),
                    cost=self._parse_float(row.get(mapping.cost, 0)),
                    retail=self._parse_float(row.get(mapping.retail, 0)),
                    vendor=str(row.get(mapping.vendor, "")).strip() or None,
                    source_file=parsed.filename,
                )

                self.inventory_items.append(item)
                self.normalizer.register_sku(sku)

            except Exception:
                continue

    def _parse_float(self, value: Any) -> float:
        """Safely parse a float value."""
        if value is None:
            return 0.0
        try:
            cleaned = str(value).replace("$", "").replace(",", "").strip()
            return float(cleaned) if cleaned else 0.0
        except Exception:
            return 0.0

    def start_session(self) -> MultiFileDiagnostic:
        """Start the diagnostic session and discover patterns."""
        # Finalize aggregation
        self.aggregator.finalize()

        # Load data into correlation engine
        self.correlation_engine.load_inventory(self.inventory_items)
        self.correlation_engine.load_vendor_data(self.aggregator)

        # Discover patterns
        self.correlation_patterns = self.correlation_engine.discover_all_patterns()

        self._session_started = True
        self._current_pattern_index = 0

        return self

    @property
    def is_complete(self) -> bool:
        """Check if all patterns have been reviewed."""
        return self._current_pattern_index >= len(self.correlation_patterns)

    @property
    def current_pattern(self) -> Optional[CorrelationPattern]:
        """Get current pattern being reviewed."""
        if self.is_complete:
            return None
        return self.correlation_patterns[self._current_pattern_index]

    def get_current_question(self) -> Optional[dict]:
        """Get the current question for the user."""
        pattern = self.current_pattern
        if not pattern:
            return None

        return {
            "pattern_index": self._current_pattern_index,
            "pattern_type": pattern.correlation_type.value,
            "question": pattern.question,
            "suggested_answers": pattern.suggested_answers,
            "affected_skus_count": len(pattern.affected_skus),
            "affected_vendors": pattern.affected_vendors,
            "total_impact": pattern.total_impact,
            "confidence": pattern.confidence,
            "description": pattern.description,
            "evidence_summary": self._summarize_evidence(pattern),
            "progress": {
                "current": self._current_pattern_index + 1,
                "total": len(self.correlation_patterns),
            },
        }

    def _summarize_evidence(self, pattern: CorrelationPattern) -> dict:
        """Create a summary of the evidence for display."""
        evidence = pattern.evidence
        summary: dict[str, Any] = {
            "type": pattern.correlation_type.value,
            "sample_skus": pattern.affected_skus[:5],
        }

        if "items" in evidence:
            summary["sample_items"] = evidence["items"][:5]

        return summary

    def answer_question(self, classification: str, note: str = "") -> dict:
        """Record answer to current question and advance."""
        pattern = self.current_pattern
        if not pattern:
            return {"error": "No current pattern", "is_complete": True}

        answer = {
            "pattern_index": self._current_pattern_index,
            "pattern_type": pattern.correlation_type.value,
            "classification": classification,
            "note": note,
            "impact": pattern.total_impact,
            "affected_skus": pattern.affected_skus,
        }

        self._answers.append(answer)
        self._current_pattern_index += 1

        return {
            "recorded": True,
            "is_complete": self.is_complete,
            "next_pattern": self.get_current_question(),
            "progress": {
                "answered": len(self._answers),
                "total": len(self.correlation_patterns),
            },
        }

    def get_summary(self) -> dict:
        """Get summary of the diagnostic session."""
        total_impact = sum(p.total_impact for p in self.correlation_patterns)

        by_type: dict[str, list] = defaultdict(list)
        for p in self.correlation_patterns:
            by_type[p.correlation_type.value].append(p)

        return {
            "files_processed": {
                "inventory": len(self.parser.inventory_files),
                "invoices": len(self.parser.invoice_files),
                "total": len(self.parser.parsed_files),
            },
            "inventory_items": len(self.inventory_items),
            "vendor_invoice_lines": len(self.aggregator.invoice_lines),
            "patterns_discovered": len(self.correlation_patterns),
            "patterns_by_type": {k: len(v) for k, v in by_type.items()},
            "total_potential_impact": total_impact,
            "vendors_analyzed": len(self.aggregator.by_vendor),
            "unique_skus_in_invoices": len(self.aggregator.by_sku),
            "answers": self._answers,
            "is_complete": self.is_complete,
        }

    def get_final_report(self) -> dict:
        """Get final report data for PDF generation."""
        return {
            "summary": self.get_summary(),
            "patterns": [
                {
                    "type": p.correlation_type.value,
                    "description": p.description,
                    "confidence": p.confidence,
                    "impact": p.total_impact,
                    "affected_skus": p.affected_skus,
                    "affected_vendors": p.affected_vendors,
                    "evidence": p.evidence,
                }
                for p in self.correlation_patterns
            ],
            "answers": self._answers,
            "vendor_summaries": [
                {
                    "vendor": v.vendor_code,
                    "fill_rate": v.fill_rate,
                    "total_lines": v.total_lines,
                    "short_ship_lines": v.short_ship_lines,
                    "unique_skus": len(v.unique_skus),
                }
                for v in sorted(
                    self.aggregator.by_vendor.values(), key=lambda x: x.fill_rate
                )[:20]
            ],
            "inventory_stats": {
                "total_items": len(self.inventory_items),
                "negative_stock_items": sum(
                    1 for i in self.inventory_items if i.quantity < 0
                ),
                "negative_margin_items": sum(
                    1 for i in self.inventory_items if i.margin_pct < 0
                ),
                "total_negative_value": sum(
                    abs(i.quantity) * i.cost
                    for i in self.inventory_items
                    if i.quantity < 0
                ),
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_multi_file_diagnostic() -> MultiFileDiagnostic:
    """Create a new multi-file diagnostic instance."""
    return MultiFileDiagnostic()


def quick_correlation_scan(inventory_file: str, invoice_files: list[str]) -> dict:
    """
    Quick scan for correlations without full diagnostic session.

    Returns summary of discovered patterns.
    """
    diagnostic = MultiFileDiagnostic()
    diagnostic.add_inventory_file(inventory_file)

    for f in invoice_files:
        diagnostic.add_vendor_invoice(f)

    diagnostic.start_session()
    return diagnostic.get_summary()
