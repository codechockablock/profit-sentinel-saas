#!/usr/bin/env python3
"""
================================================================================
HARDWARE STORE INVENTORY HEALTH ANALYZER
================================================================================

A comprehensive inventory analysis tool for hardware stores (Paladin, Epicor,
and similar POS systems) that identifies inventory primitives (issue patterns)
and generates detailed reports with full accounting impact context.

This script is designed to work with Profit Sentinel's analysis primitives
and generate reports compatible with the SaaS platform's email reporting.

USAGE:
------
    # Basic usage with default settings:
    python hardware_inventory_analyzer.py inventory_export.csv

    # With custom configuration:
    python hardware_inventory_analyzer.py inventory_export.csv --config config.yaml

    # Multiple files:
    python hardware_inventory_analyzer.py file1.csv file2.xlsx --output report.html

    # Specify output format:
    python hardware_inventory_analyzer.py inventory.csv --format html --output my_report.html

EXPECTED COLUMNS (from Paladin/Hardware Store POS exports):
-----------------------------------------------------------
Required:
    - SKU (or "Item SKU", "Product SKU")
    - Description (or "Description Short", "Item Name")
    - In Stock Qty. (or "Qty", "QOH", "Quantity")
    - Cost (unit cost per item)

Recommended:
    - Retail (selling price)
    - Profit Margin %
    - Sold (units sold in period)
    - Last Sale (date of last sale)
    - Sub Total (Cost × Quantity = inventory value)
    - Vendor
    - Category / Dpt.
    - Pkg. Qty. (package quantity)

OUTPUT:
-------
    - Console summary with key metrics
    - Detailed HTML report with:
        1. Executive Summary
        2. Primitive Definitions and Accounting Impacts
        3. Detailed Findings (grouped by primitive, showing QOH, Last Sale, etc.)
        4. Recommendations
        5. Validation Log

INVENTORY PRIMITIVES DETECTED:
------------------------------
    1. Negative Inventory - QOH below zero
    2. Dead Stock - No sales in 12+ months
    3. Slow Moving - No sales in 6-12 months
    4. Overstock - Excessive inventory vs sales velocity
    5. Low Stock - Below safety threshold
    6. Missing Cost - Cost field blank/zero
    7. Zero Cost with Quantity - Suspicious $0 cost
    8. Negative Cost - Invalid negative cost values
    9. High Margin Leak - Selling below expected margin
    10. Margin Erosion - Cost approaching revenue
    11. Cost Exceeds Price - Selling at a loss
    12. Shrinkage Pattern - High value, low margin, minimal sales
    13. Missing Last Sold - Cannot assess inventory age
    14. High-Value Dead Stock - Dead stock with significant $ value

Author: Profit Sentinel
Version: 2.0.0
License: MIT
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

# Optional: YAML config support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class Severity(Enum):
    """Severity levels for inventory issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class PrimitiveType(Enum):
    """
    Inventory primitive types - specific issue patterns that indicate
    inventory problems affecting financial reporting and operations.

    These align with Profit Sentinel's analysis primitives.
    """

    NEGATIVE_INVENTORY = "negative_inventory"
    DEAD_STOCK = "dead_stock"
    SLOW_MOVING = "slow_moving"
    OVERSTOCK = "overstock"
    LOW_STOCK = "low_stock"
    MISSING_COST = "missing_cost"
    ZERO_COST_WITH_QUANTITY = "zero_cost_with_quantity"
    NEGATIVE_COST = "negative_cost"
    HIGH_MARGIN_LEAK = "high_margin_leak"
    MARGIN_EROSION = "margin_erosion"
    COST_EXCEEDS_PRICE = "cost_exceeds_price"
    SHRINKAGE_PATTERN = "shrinkage_pattern"
    MISSING_LAST_SOLD = "missing_last_sold"
    HIGH_VALUE_DEAD_STOCK = "high_value_dead_stock"
    PRICE_DISCREPANCY = "price_discrepancy"


# Column name aliases for flexible matching (hardware store POS exports)
COLUMN_ALIASES: dict[str, list[str]] = {
    "sku": [
        "sku",
        "SKU",
        "item sku",
        "product sku",
        "item_sku",
        "itemsku",
        "item number",
        "item no",
        "product id",
    ],
    "vendor_sku": [
        "vendor sku",
        "Vendor SKU",
        "supplier sku",
        "mfg sku",
        "manufacturer sku",
    ],
    "description": [
        "description",
        "Description",
        "Description ",
        "item name",
        "product name",
        "name",
        "title",
        "description short",
        "Description Short",
    ],
    "quantity": [
        "in stock qty.",
        "In Stock Qty.",
        "in stock qty",
        "qty",
        "Qty",
        "quantity",
        "qoh",
        "QOH",
        "stock",
        "on hand",
        "qty on hand",
        "inventory quantity",
    ],
    "cost": [
        "cost",
        "Cost",
        "unit cost",
        "item cost",
        "cost per item",
        "cogs",
        "landed cost",
    ],
    "sub_total": [
        "sub total",
        "Sub Total",
        "subtotal",
        "inventory value",
        "total value",
        "extended cost",
        "ext cost",
    ],
    "retail": [
        "retail",
        "Retail",
        "price",
        "selling price",
        "sell price",
        "retail price",
        "unit price",
    ],
    "suggested_retail": [
        "sug. retail",
        "Sug. Retail",
        "suggested retail",
        "msrp",
        "MSRP",
        "list price",
    ],
    "margin": [
        "profit margin %",
        "Profit Margin %",
        "margin",
        "margin %",
        "gross margin",
        "gp %",
        "profit %",
    ],
    "sold": [
        "sold",
        "Sold",
        "units sold",
        "qty sold",
        "sales qty",
        "sold qty",
    ],
    "last_sale": [
        "last sale",
        "Last Sale",
        "last sold",
        "last sold date",
        "date last sold",
        "last sale date",
    ],
    "last_purchase": [
        "last pur.",
        "Last Pur.",
        "last purchase",
        "last po",
        "last received",
    ],
    "vendor": [
        "vendor",
        "Vendor",
        "supplier",
        "manufacturer",
        "mfg",
    ],
    "category": [
        "category",
        "Category",
        "dpt.",
        "Dpt.",
        "department",
        "product type",
        "class",
    ],
    "pkg_qty": [
        "pkg. qty.",
        "Pkg. Qty.",
        "package qty",
        "pack qty",
        "case qty",
        "unit qty",
    ],
    "barcode": [
        "barcode",
        "Barcode",
        "upc",
        "UPC",
        "ean",
        "gtin",
    ],
    "bin": [
        "bin",
        "BIN",
        "location",
        "bin location",
        "shelf",
    ],
    "status": [
        "status",
        "Status",
        "item status",
        "active",
    ],
}

# Required columns for validation
REQUIRED_COLUMNS = ["sku", "description", "quantity"]
RECOMMENDED_COLUMNS = ["cost", "retail", "sold", "last_sale", "sub_total", "margin"]


# =============================================================================
# PRIMITIVE DEFINITIONS - Comprehensive accounting and business impact
# =============================================================================


@dataclass
class PrimitiveDefinition:
    """
    Complete definition of an inventory primitive including
    its business and accounting impacts.
    """

    name: str
    primitive_type: PrimitiveType
    severity: Severity
    short_description: str
    full_definition: str
    how_it_occurs: str
    cogs_impact: str
    gross_profit_impact: str
    gross_margin_impact: str
    additional_risks: list[str]
    recommendations: list[str]
    color: str  # For HTML report styling


PRIMITIVE_DEFINITIONS: dict[PrimitiveType, PrimitiveDefinition] = {
    PrimitiveType.NEGATIVE_INVENTORY: PrimitiveDefinition(
        name="Negative Inventory",
        primitive_type=PrimitiveType.NEGATIVE_INVENTORY,
        severity=Severity.CRITICAL,
        short_description="Inventory quantity on hand is less than zero.",
        full_definition="""
Negative inventory occurs when the recorded quantity on hand for a product falls
below zero. This represents a fundamental data integrity issue where the system
shows you owe inventory that doesn't exist, or where sales have been recorded
without corresponding stock being available. In hardware stores, this is often
caused by POS sales not syncing with inventory systems or manual adjustment errors.
        """.strip(),
        how_it_occurs="""
• Overselling at POS: Sales recorded before inventory receipt is logged
• Sync delays: Sales at register not syncing with back-office system in real-time
• Manual adjustments: Incorrect inventory counts entered during cycle counts
• Returns processing: Returns not properly added back to inventory
• Transfer errors: Stock transfers between locations not completed correctly
• Receiving errors: Products sold before properly received into system
• Theft/shrinkage: Products stolen but not accounted for in system
        """.strip(),
        cogs_impact="""
COGS is UNDERSTATED because when inventory goes negative, no inventory value is
relieved upon sale (or a negative value is recorded). The cost expense that should
be recognized doesn't exist because there's no inventory cost to relieve. This means
your reported cost of goods sold is artificially low, potentially by thousands of
dollars depending on the items affected.
        """.strip(),
        gross_profit_impact="""
Gross profit is OVERSTATED because revenue is recorded from the sale, but the
corresponding COGS expense is not fully recognized. The profit & loss statement
shows profit that doesn't actually exist. When you eventually purchase replacement
inventory, you'll recognize cost without corresponding revenue.
        """.strip(),
        gross_margin_impact="""
Gross margins appear ARTIFICIALLY INFLATED. If you're showing a 40% gross margin
but have significant negative inventory, your true margin could be substantially
lower (potentially 30-35%) once the missing COGS is properly accounted for. This
misleads stakeholders about actual business profitability.
        """.strip(),
        additional_risks=[
            "Tax underreporting: Lower COGS means higher taxable income is reported",
            "Customer fulfillment issues: Orders placed for items that don't exist",
            "Special order chaos: Can't tell if item needs ordering or is in stock",
            "Cash flow surprises: Future purchases to cover negative stock hit cash unexpectedly",
            "Audit red flags: Material discrepancies between physical and book inventory",
            "Valuation issues: Business valuation based on incorrect financials",
            "Perpetual inventory unreliable: Physical counts won't match system",
        ],
        recommendations=[
            "Run physical count on all negative SKUs immediately",
            "Investigate sales history for each negative item - were items actually sold?",
            "Check receiving logs - was inventory received but not entered?",
            "Review POS sync settings and timing",
            "Implement real-time inventory sync between POS and back-office",
            "Consider requiring positive inventory check before POS sale",
            "Create journal entry to correct COGS once quantities are fixed",
        ],
        color="#dc2626",  # Red
    ),
    PrimitiveType.DEAD_STOCK: PrimitiveDefinition(
        name="Dead Stock / Obsolete Inventory",
        primitive_type=PrimitiveType.DEAD_STOCK,
        severity=Severity.HIGH,
        short_description="Products with no sales activity for 12+ months.",
        full_definition="""
Dead stock refers to inventory items that have had no sales activity for an extended
period (typically 12+ months). In hardware stores, this often includes discontinued
product lines, seasonal items that didn't sell, or specialty items ordered for
customers who never picked them up. These items are sitting on shelves, tying up
capital, and likely losing value through obsolescence.
        """.strip(),
        how_it_occurs="""
• Over-ordering: Purchasing too much of slow-selling items or vendor minimums
• Product discontinuation: Manufacturers stopped making item, but you still have stock
• Trend changes: Technology or building code changes made items obsolete
• Seasonality misjudgment: Ordering seasonal items that didn't sell during season
• Special orders not picked up: Customer-ordered items never retrieved
• Planogram changes: Items removed from active selling areas but still in inventory
• Vendor closeouts: Buying deeply discounted items that don't move
        """.strip(),
        cogs_impact="""
COGS is UNDERSTATED in the current period because these items haven't sold. However,
under proper accounting (GAAP/IFRS), dead stock should be written down to Net Realizable
Value (NRV). If NRV is lower than cost (often $0 for truly obsolete items), an inventory
write-down expense should be recognized, increasing COGS or being recorded separately.
        """.strip(),
        gross_profit_impact="""
Gross profit is OVERSTATED if dead stock hasn't been written down. The balance sheet
shows inventory at historical cost when its true recoverable value is much lower.
When you eventually liquidate at clearance prices or throw away the inventory, you'll
recognize a large loss that should have been accrued gradually.
        """.strip(),
        gross_margin_impact="""
Gross margins are MISLEADING because they're calculated against sales that exclude
these problematic items. Your overall inventory investment's return is actually lower
than margins suggest. A hardware store showing 35% margin may have true return on
inventory investment of only 25% when dead stock losses are factored in.
        """.strip(),
        additional_risks=[
            "Working capital locked: Cash tied up in unsellable inventory",
            "Storage costs: Shelf space and warehouse space occupied unproductively",
            "Opportunity cost: Space and capital not available for faster-moving items",
            "Physical deterioration: Hardware items can rust, packaging degrades",
            "Safety hazards: Old chemicals, paints may become hazardous",
            "Year-end write-offs: Large, unexpected write-downs at fiscal year-end",
            "Investor/lender concerns: Overstated inventory affects financial ratios",
        ],
        recommendations=[
            "Implement monthly aging analysis - review items with no sales 6+ months",
            "Create clearance endcap or bargain bin with aggressive pricing",
            "Contact vendor about return authorization for slow sellers",
            "Donate obsolete items for tax deduction (check with accountant)",
            "Write down inventory to NRV per accounting standards",
            "Set up automatic reorder point of zero for items flagged as dead",
            "Add 'last sale date' review to weekly manager report",
        ],
        color="#9333ea",  # Purple
    ),
    PrimitiveType.SLOW_MOVING: PrimitiveDefinition(
        name="Slow-Moving Inventory",
        primitive_type=PrimitiveType.SLOW_MOVING,
        severity=Severity.MEDIUM,
        short_description="Products with no sales in 6-12 months (trending toward dead stock).",
        full_definition="""
Slow-moving inventory includes items that haven't sold in 6-12 months but aren't yet
classified as dead stock. These items are warning signs - they're on the path to
becoming obsolete and represent an opportunity for intervention before value is lost.
In hardware retail, this is the critical window to take action.
        """.strip(),
        how_it_occurs="""
• Demand decline: Product lifecycle entering decline phase
• Competition: Big box stores or online sellers offering better prices
• Pricing issues: Price may not match local market expectations
• Visibility problems: Items buried on bottom shelves or back aisles
• Seasonal off-cycle: Items awaiting their selling season
• Customer preference shift: New products replacing older technology
• Economic factors: Contractors reducing activity in your area
        """.strip(),
        cogs_impact="""
COGS timing is delayed - these items' costs sit on the balance sheet instead of
flowing through to expense. While not immediately distorting COGS, the longer items
sit, the greater the risk of eventual large write-downs that hit COGS all at once.
        """.strip(),
        gross_profit_impact="""
Current gross profit appears normal, but FUTURE gross profit is at risk. When these
items eventually sell (likely at discount), the margin will be lower than normal
selling prices would generate. The full impact is deferred, not avoided.
        """.strip(),
        gross_margin_impact="""
Inventory turn metrics are degraded - money is invested in slow assets vs. fast-turning
profitable ones. Your Open-to-Buy is consumed by slow movers instead of generating
healthy turns. GMROI (Gross Margin Return on Inventory Investment) suffers.
        """.strip(),
        additional_risks=[
            "Trending toward dead stock and potential complete write-off",
            "Tying up Open-to-Buy budget that could fund better items",
            "Risk of manufacturer discontinuation increasing daily",
            "Packaging deterioration making items less sellable",
            "Cash flow constraint: money locked in slow inventory",
        ],
        recommendations=[
            "Flag for 20-30% off promotional activity before they become dead stock",
            "Move to end caps or feature displays to increase visibility",
            "Bundle with faster-moving related items",
            "Contact vendor about exchange or return programs",
            "Review why items slowed - price, placement, or demand issue?",
            "Reduce or eliminate reorder quantities",
        ],
        color="#f59e0b",  # Amber
    ),
    PrimitiveType.OVERSTOCK: PrimitiveDefinition(
        name="Overstock / Excess Inventory",
        primitive_type=PrimitiveType.OVERSTOCK,
        severity=Severity.MEDIUM,
        short_description="Quantity on hand significantly exceeds sales velocity.",
        full_definition="""
Overstock occurs when you're holding significantly more inventory than needed to meet
demand. Measured as months of supply (Quantity on Hand ÷ Monthly Sales Velocity),
having more than 12 months of supply typically indicates over-purchasing. Even
items that sell can be overstocked if you've bought too many years' worth.
        """.strip(),
        how_it_occurs="""
• Bulk discount temptation: Buying more to get better unit pricing from vendors
• Demand overestimation: Expecting higher sales than materialized
• Vendor minimum orders: MOQs forcing larger purchases than needed
• Fear of stockouts: Over-ordering as safety stock
• Closeout purchases: Buying large quantities of deeply discounted items
• Seasonal pre-buy gone wrong: Pre-season orders based on optimistic forecasts
• Automatic reorder errors: Min/max levels set too high
        """.strip(),
        cogs_impact="""
COGS per unit may actually be LOWER due to bulk pricing, but total COGS investment
is much higher than necessary. The carrying cost of excess inventory (typically
20-30% of value annually including space, insurance, damage, obsolescence risk)
erodes any unit cost savings achieved through bulk buying.
        """.strip(),
        gross_profit_impact="""
Gross profit dollars are reduced by carrying costs that often aren't tracked at
the SKU level. A product showing 40% gross margin might only deliver 25% after
accounting for the cost of capital, storage, and risk of overstock positions.
        """.strip(),
        gross_margin_impact="""
Inventory turns are LOW, meaning your gross margin dollars are generated slowly
relative to inventory investment. A 50% margin item that turns once per year
generates less return than a 30% margin item turning 4x per year.
        """.strip(),
        additional_risks=[
            "Cash flow strain: Working capital tied up unnecessarily",
            "Carrying costs: Storage, insurance, handling = 20-30% of value/year",
            "Obsolescence risk: Technology or codes may change before sell-through",
            "Markdown risk: May need to discount heavily to move excess",
            "Opportunity cost: Capital not available for new/better items",
            "Warehouse/shelf capacity consumed unproductively",
        ],
        recommendations=[
            "Calculate actual months of supply for flagged items",
            "Negotiate smaller, more frequent deliveries with vendors",
            "Consider vendor-managed inventory for high-volume items",
            "Use promotional velocity to move excess before it ages",
            "Reset min/max levels based on actual sales velocity",
            "Track GMROI (Gross Margin Return on Inventory Investment) by category",
        ],
        color="#3b82f6",  # Blue
    ),
    PrimitiveType.LOW_STOCK: PrimitiveDefinition(
        name="Low Stock / Reorder Point Alert",
        primitive_type=PrimitiveType.LOW_STOCK,
        severity=Severity.HIGH,
        short_description="Quantity on hand is critically low relative to sales velocity.",
        full_definition="""
Low stock indicates that inventory levels have fallen below the safety threshold
needed to maintain sales while awaiting replenishment. For hardware stores, this
means popular items that contractors and DIYers rely on are at risk of stocking out,
potentially sending customers to competitors.
        """.strip(),
        how_it_occurs="""
• Demand spike: Unexpected project or contractor buying more than usual
• Reorder delays: Purchase order not placed on time
• Supplier issues: Extended lead times or backorders from distributors
• Forecasting errors: Underestimating seasonal or project-based demand
• Cash constraints: Delaying orders due to cash flow
• System failures: Reorder alerts not triggering properly
        """.strip(),
        cogs_impact="""
COGS itself isn't directly affected, but the OPPORTUNITY COST of lost sales is
significant. Every stockout represents revenue that would have generated COGS
expense and gross profit that now goes to competitors instead.
        """.strip(),
        gross_profit_impact="""
Gross profit is LOST for every sale that can't be fulfilled. If you stock out of
a $25 item with 40% margin that normally sells 20/week, that's $200 in lost gross
profit per week - often more than the cost of carrying safety stock.
        """.strip(),
        gross_margin_impact="""
While margin percentage isn't affected, TOTAL margin dollars are reduced. Additionally,
rush shipping to expedite restock orders often comes at premium cost, further eroding
margins on subsequent sales.
        """.strip(),
        additional_risks=[
            "Lost sales: Direct revenue loss when customers can't buy",
            "Customer defection: Contractors may switch to competitor permanently",
            "Project delays: Customer projects stalled waiting for your item",
            "Rush order costs: Expedited shipping is expensive",
            "Reputation damage: Word spreads about unreliable stock",
            "Lost basket: Customer who can't find one item may buy nothing",
        ],
        recommendations=[
            "Calculate safety stock based on lead time and demand variability",
            "Review and adjust reorder points for high-velocity items",
            "Set up automated low-stock alerts in POS system",
            "Establish backup supplier relationships for critical items",
            "Consider seasonal safety stock adjustments",
            "Monitor sales velocity trends weekly for top sellers",
        ],
        color="#eab308",  # Yellow
    ),
    PrimitiveType.MISSING_COST: PrimitiveDefinition(
        name="Missing Cost Data",
        primitive_type=PrimitiveType.MISSING_COST,
        severity=Severity.CRITICAL,
        short_description="Cost per item field is blank - COGS cannot be calculated.",
        full_definition="""
Missing cost data means the 'Cost' field is blank or null for a product. Without
this data, you cannot calculate Cost of Goods Sold, gross profit, gross margin,
inventory valuation, or any meaningful financial metric for this item. This is
a critical data quality issue affecting your entire financial reporting.
        """.strip(),
        how_it_occurs="""
• Incomplete product setup: Cost field skipped during item creation
• Vendor pricing not entered: Received goods without logging cost
• Data migration issues: Cost field not mapped during system change
• Free/promotional items: Assumption that free items don't need cost
• Price changes: Old cost deleted, new cost not entered
• System errors: Cost field accidentally cleared
        """.strip(),
        cogs_impact="""
COGS is COMPLETELY UNKNOWN for these items. Your P&L's cost of goods sold line is
understated by the entire cost value of sales from these items. If they represent
10% of sales, your reported COGS could be understated by 10% or more.
        """.strip(),
        gross_profit_impact="""
Gross profit is OVERSTATED by the missing COGS amount. You're recording revenue
but no cost, making profit appear higher than reality. This is a material
misstatement if the affected items represent significant sales volume.
        """.strip(),
        gross_margin_impact="""
Gross margin percentage is UNRELIABLE and likely overstated. You're dividing
profit by revenue, but the profit number excludes costs. Actual margin could be
significantly lower than reported.
        """.strip(),
        additional_risks=[
            "Financial statements are materially misstated",
            "Tax liability may be incorrect (underreported COGS = overpaid taxes)",
            "Inventory valuation wrong: balance sheet overstated or understated",
            "Pricing decisions made without knowing true costs",
            "Cannot calculate margin by item, category, or vendor",
            "Audit findings possible: material weakness in controls",
        ],
        recommendations=[
            "URGENT: Populate cost data for all items immediately",
            "Review supplier invoices and POs to determine correct costs",
            "Set up process requiring cost entry before item goes active",
            "Run weekly report on items missing cost data",
            "Consider cost default based on similar items as temporary fix",
            "Update receiving process to require cost entry",
        ],
        color="#dc2626",  # Red
    ),
    PrimitiveType.ZERO_COST_WITH_QUANTITY: PrimitiveDefinition(
        name="Zero Cost with Positive Quantity",
        primitive_type=PrimitiveType.ZERO_COST_WITH_QUANTITY,
        severity=Severity.HIGH,
        short_description="Cost is $0.00 but quantity on hand is greater than zero.",
        full_definition="""
This flags items where the cost field is explicitly zero (not blank) but inventory
exists. While some items legitimately have zero cost (free samples, promotional items,
damaged items written off), most inventory has acquisition costs. A $0 cost is
suspicious and usually indicates a data entry error or system issue.
        """.strip(),
        how_it_occurs="""
• Data entry error: Accidentally entering 0 instead of actual cost
• Placeholder value: Using 0 as placeholder that was never updated
• Free samples/promos: Legitimately free items (should be documented)
• Received as vendor credit: Items received without new cost entered
• System default: Some POS systems default to 0 if not specified
• Price file issue: Vendor cost file had errors
        """.strip(),
        cogs_impact="""
COGS is recorded as $0 when these items sell, even though you likely paid for them.
This understates COGS. If you sell $5,000 of zero-cost items that actually cost
$3,000, your COGS is understated by $3,000.
        """.strip(),
        gross_profit_impact="""
Gross profit is overstated by the missing cost amount. Sales of these items appear
100% profitable when they're not. This inflates overall gross profit figures.
        """.strip(),
        gross_margin_impact="""
These items show 100% gross margin (Revenue - $0 = 100% profit), which is almost
never accurate for purchased inventory. This artificially inflates blended margin
calculations and category margin analysis.
        """.strip(),
        additional_risks=[
            "Tax implications: overstating profit, potentially overpaying taxes",
            "Pricing errors: thinking you can price lower because 'no cost'",
            "Inventory valuation: balance sheet shows $0 for real assets",
            "Insurance: if inventory is lost, claim would be for $0",
        ],
        recommendations=[
            "Review each zero-cost item - is $0 accurate or an error?",
            "Update costs from supplier invoices or purchase orders",
            "For truly free items, add note/tag documenting why",
            "Implement validation: warn when $0 cost is entered",
            "Check recent receiving for cost entry issues",
        ],
        color="#f97316",  # Orange
    ),
    PrimitiveType.NEGATIVE_COST: PrimitiveDefinition(
        name="Negative Cost Value",
        primitive_type=PrimitiveType.NEGATIVE_COST,
        severity=Severity.CRITICAL,
        short_description="Cost per item is a negative number.",
        full_definition="""
Negative cost means the cost field contains a value less than zero (e.g., -$5.00).
This is almost always a data entry error or system glitch. No inventory item
should have a negative cost - suppliers don't pay you to take their products.
        """.strip(),
        how_it_occurs="""
• Data entry error: Accidentally typing a minus sign
• Copy/paste error: Pasting from spreadsheet with sign errors
• Return/credit adjustments: Incorrectly adjusting costs for credits
• System bugs: Integration or calculation errors
• Rebate mishandling: Vendor rebates incorrectly applied to cost
        """.strip(),
        cogs_impact="""
COGS is calculated as a NEGATIVE value or credit when these items sell. Instead
of recognizing expense, you're recognizing negative expense (income). This makes
COGS significantly understated and corrupts financial reporting.
        """.strip(),
        gross_profit_impact="""
Gross profit is massively overstated. If an item has -$10 cost and sells for $50,
it appears to generate $60 profit ($50 - (-$10)), when real profit should be less.
        """.strip(),
        gross_margin_impact="""
Margin calculations become nonsensical - you can show margins over 100%, which
is a clear indicator of data problems.
        """.strip(),
        additional_risks=[
            "Complete corruption of financial reporting",
            "Tax reporting errors of significant magnitude",
            "Impossible to make rational pricing decisions",
            "Audit would identify this as material misstatement",
        ],
        recommendations=[
            "IMMEDIATE FIX: Correct all negative costs to accurate values",
            "Investigate how negative costs entered the system",
            "Add validation rules to reject negative cost entries",
            "Review data imports for sign-handling issues",
        ],
        color="#dc2626",  # Red
    ),
    PrimitiveType.HIGH_MARGIN_LEAK: PrimitiveDefinition(
        name="High Margin Leak",
        primitive_type=PrimitiveType.HIGH_MARGIN_LEAK,
        severity=Severity.HIGH,
        short_description="Item selling well below expected margin threshold.",
        full_definition="""
High margin leak identifies items where the actual gross margin is significantly
below the expected or target margin for that category. In hardware retail, if
your target is 35% and items are selling at 15%, that's a margin leak that
compounds with every sale.
        """.strip(),
        how_it_occurs="""
• Pricing not updated after cost increase from vendor
• Competitive pressure: matching low prices without considering margin
• Promotion pricing left in place too long
• Cost entered incorrectly (too high) making margin appear low
• Retail price entered incorrectly (too low)
• Category margin targets not set appropriately
        """.strip(),
        cogs_impact="""
COGS itself is accurate, but the RATIO of COGS to revenue is too high. Each
sale generates less gross profit than it should, compounding over time into
significant profit shortfall.
        """.strip(),
        gross_profit_impact="""
Gross profit is LOWER THAN EXPECTED for each sale. If you sell 100 units at 15%
margin instead of 35% margin on a $20 item, you've lost $4 per unit = $400 in
gross profit that should have been earned.
        """.strip(),
        gross_margin_impact="""
Blended gross margin is dragged down by these items. A few high-volume low-margin
items can significantly impact overall margin even if most items are priced correctly.
        """.strip(),
        additional_risks=[
            "Profit shortfall compounds with volume",
            "Working harder for less profit",
            "May indicate vendor cost increases not passed through",
            "Could signal competitive pricing war affecting category",
        ],
        recommendations=[
            "Review pricing strategy: is low margin intentional?",
            "Check for cost increases not reflected in retail price",
            "Evaluate competitive landscape for this item",
            "Consider raising price or discontinuing if not strategic",
            "Review vendor alternatives for better cost",
        ],
        color="#ef4444",  # Red
    ),
    PrimitiveType.MARGIN_EROSION: PrimitiveDefinition(
        name="Margin Erosion",
        primitive_type=PrimitiveType.MARGIN_EROSION,
        severity=Severity.HIGH,
        short_description="Cost is approaching or exceeding 85% of retail price.",
        full_definition="""
Margin erosion flags items where the cost is dangerously close to the selling price,
leaving minimal room for profit after operating expenses. When cost is 85%+ of
retail, the gross margin is only 15% or less - often not enough to cover overhead.
        """.strip(),
        how_it_occurs="""
• Vendor cost increases not passed to retail
• Retail price reduced for competition without cost review
• Vendor deals or rebates expired
• Freight costs increased
• Currency fluctuations (for imported goods)
• Volume purchasing discounts lost
        """.strip(),
        cogs_impact="""
COGS as a percentage of revenue is approaching 85%+, leaving only 15% or less
as gross profit. Most hardware stores need 30-40% gross margin to cover operating
expenses and generate net profit.
        """.strip(),
        gross_profit_impact="""
Gross profit dollars per unit are minimal. Even with volume, these items contribute
little to covering overhead. High volume at eroded margins can actually lose money
after operating costs.
        """.strip(),
        gross_margin_impact="""
Item-level margin is critically low. These items should either be repriced,
discontinued, or recognized as traffic drivers (loss leaders) with intentional
low margin.
        """.strip(),
        additional_risks=[
            "May be selling below true break-even after overhead allocation",
            "Vendor relationship may need renegotiation",
            "Could indicate broader category margin pressure",
            "Risk of going negative if any cost increase occurs",
        ],
        recommendations=[
            "Evaluate retail price increase feasibility",
            "Negotiate with vendor for better cost or terms",
            "Consider alternative suppliers",
            "If intentional loss leader, document and limit quantity",
            "Review entire category for similar erosion patterns",
        ],
        color="#ec4899",  # Pink
    ),
    PrimitiveType.COST_EXCEEDS_PRICE: PrimitiveDefinition(
        name="Cost Exceeds Price (Negative Margin)",
        primitive_type=PrimitiveType.COST_EXCEEDS_PRICE,
        severity=Severity.CRITICAL,
        short_description="Cost per item is higher than the selling price - losing money on every sale.",
        full_definition="""
When cost exceeds selling price, every sale generates a loss. While this may be
intentional for clearance items, it usually indicates a pricing error, cost data
error, or items that need immediate repricing to stop bleeding money.
        """.strip(),
        how_it_occurs="""
• Pricing error: Price set lower than cost by mistake
• Cost increase: Vendor raised prices but retail not updated
• Clearance pricing: Intentional markdown below cost to clear inventory
• Data error: Cost or price field has wrong value
• Currency/unit confusion: Cost in different unit than price
        """.strip(),
        cogs_impact="""
COGS exceeds revenue for these items. While technically correct (cost is recorded
as incurred), every sale increases your losses rather than generating profit.
        """.strip(),
        gross_profit_impact="""
Gross profit is NEGATIVE for these items. They're pulling down your overall gross
profit with every sale. A high-volume negative-margin item can significantly
impact total store profitability.
        """.strip(),
        gross_margin_impact="""
Gross margin is negative (below 0%). A product costing $15 selling for $10 has
-50% margin. This drags down blended margin significantly.
        """.strip(),
        additional_risks=[
            "Losing money on every sale",
            "High volume compounds losses quickly",
            "May indicate vendor cost change you missed",
            "Customer expectations set at unprofitable price",
        ],
        recommendations=[
            "IMMEDIATE: Verify if pricing or cost data is in error",
            "If intentional clearance: document and set end date",
            "If error: correct price immediately",
            "Consider pulling from active sale until corrected",
            "Set up alerts for any items where cost > price",
        ],
        color="#dc2626",  # Red
    ),
    PrimitiveType.SHRINKAGE_PATTERN: PrimitiveDefinition(
        name="Shrinkage Pattern",
        primitive_type=PrimitiveType.SHRINKAGE_PATTERN,
        severity=Severity.HIGH,
        short_description="High-value inventory with low margin and minimal sales - potential theft indicator.",
        full_definition="""
Shrinkage pattern identifies items that have characteristics commonly associated
with inventory shrinkage (theft, damage, or administrative errors): high inventory
value, low or no sales, and often below-expected margin. These items warrant
investigation for potential loss.
        """.strip(),
        how_it_occurs="""
• Employee theft: High-value items taken without ringing sale
• Customer theft: Shoplifting of desirable items
• Receiving fraud: Items received but diverted before shelving
• Damaged and not recorded: Items damaged but not written off
• Mis-picks: Wrong items shipped, returns not processed
• Administrative errors: Data entry mistakes in receiving/adjustments
        """.strip(),
        cogs_impact="""
COGS is understated because items that should have flowed through as sold (or
written off) are sitting in inventory. The cost of shrinkage isn't recognized
until a physical inventory reveals the loss.
        """.strip(),
        gross_profit_impact="""
Gross profit is overstated by the amount of unrecognized shrinkage. When physical
count reveals shortages, a write-down will reduce profit for that period.
        """.strip(),
        gross_margin_impact="""
Margins appear normal but are artificially inflated because shrinkage costs
haven't been recognized. True margins are lower than reported.
        """.strip(),
        additional_risks=[
            "Ongoing theft if not addressed",
            "Cash flow impact when shrinkage discovered",
            "Insurance may not cover undetected shrinkage",
            "Employee morale issues if theft is internal",
            "Large year-end adjustments at physical inventory",
        ],
        recommendations=[
            "Conduct physical count of flagged items immediately",
            "Review security measures for high-value items",
            "Check receiving documentation and processes",
            "Analyze transaction patterns for anomalies",
            "Consider locked display or behind-counter placement",
            "Review employee access controls",
        ],
        color="#f97316",  # Orange
    ),
    PrimitiveType.MISSING_LAST_SOLD: PrimitiveDefinition(
        name="Missing Last Sold Date",
        primitive_type=PrimitiveType.MISSING_LAST_SOLD,
        severity=Severity.MEDIUM,
        short_description="No last sold date recorded - unable to assess inventory age.",
        full_definition="""
Items without a last sold date cannot be properly aged or analyzed for sales velocity.
This could indicate items that have never sold (new products or problems), or it may
be a data gap where sales occurred but weren't tracked. Either way, inventory aging
analysis is compromised.
        """.strip(),
        how_it_occurs="""
• Never sold: New products awaiting first sale
• Data not tracked: Legacy items added before date tracking
• System migration: Date field not mapped during conversion
• Manual sales: Sales processed outside of POS system
• Data corruption: Date field corrupted or cleared
        """.strip(),
        cogs_impact="""
COGS isn't directly affected, but you can't analyze COGS by product age. Without
knowing when items last sold, you can't identify candidates for write-down.
        """.strip(),
        gross_profit_impact="""
Cannot determine if items are contributing to profit or just sitting. Profit
analysis by product lifecycle stage is impossible.
        """.strip(),
        gross_margin_impact="""
Can't correlate margin with inventory age. Items sold at clearance (lower margin)
can't be identified from full-price sales.
        """.strip(),
        additional_risks=[
            "Can't identify dead stock without sale date history",
            "Inventory aging reports are incomplete",
            "May be unknowingly holding obsolete inventory",
            "Demand forecasting compromised",
        ],
        recommendations=[
            "For new items: no action needed, date will populate on first sale",
            "For older items: research if sales occurred elsewhere",
            "Check if items sold through alternate systems",
            "Consider manual date entry based on receiving date as proxy",
            "Prioritize physical review of high-value items without dates",
        ],
        color="#6b7280",  # Gray
    ),
    PrimitiveType.HIGH_VALUE_DEAD_STOCK: PrimitiveDefinition(
        name="High-Value Dead Stock",
        primitive_type=PrimitiveType.HIGH_VALUE_DEAD_STOCK,
        severity=Severity.CRITICAL,
        short_description="Dead stock items with significant inventory value at risk ($1,000+).",
        full_definition="""
High-value dead stock are items that meet dead stock criteria (no sales 12+ months)
AND have significant inventory value on hand (typically $1,000+). These represent
major working capital at risk of write-off and deserve urgent executive attention.
        """.strip(),
        how_it_occurs="""
• Large initial orders of items that didn't sell as expected
• Bulk discount purchases that turned out to be poor decisions
• Seasonal items that didn't clear and were held over
• Product line discontinuation without liquidation
• Special orders for projects that got cancelled
• Vendor closeout buys that didn't move
        """.strip(),
        cogs_impact="""
Significant COGS will need to be recognized as inventory write-down or write-off.
A $10,000 high-value dead stock position may require a $10,000 expense recognition
when properly accounted for under GAAP.
        """.strip(),
        gross_profit_impact="""
Large negative impact to gross profit when write-down is recorded. A single
high-value dead stock item could wipe out a week's worth of gross profit.
        """.strip(),
        gross_margin_impact="""
When write-down hits P&L, gross margin drops significantly for that period.
May cause missed profit targets or covenant issues.
        """.strip(),
        additional_risks=[
            "Major cash tied up with no return pathway",
            "Balance sheet overstated until write-down recorded",
            "Audit finding likely if not properly reserved",
            "Opportunity cost: this capital could be working elsewhere",
            "May need to pay for disposal if items can't be sold or donated",
        ],
        recommendations=[
            "Prioritize for immediate liquidation strategy",
            "Get quotes from liquidators - any recovery is better than zero",
            "Contact vendor about return or credit programs",
            "Consider donation for tax deduction (consult accountant)",
            "Record inventory write-down per accounting standards",
            "Present to ownership/management for decision",
        ],
        color="#dc2626",  # Red
    ),
    PrimitiveType.PRICE_DISCREPANCY: PrimitiveDefinition(
        name="Price Discrepancy",
        primitive_type=PrimitiveType.PRICE_DISCREPANCY,
        severity=Severity.MEDIUM,
        short_description="Unusual pricing patterns that may indicate data errors.",
        full_definition="""
Price discrepancy identifies items with unusual pricing characteristics that may
indicate data entry errors, system issues, or pricing that needs review. This
includes items with 100% margin (suspicious zero cost), negative margins, or
significant deviation from suggested retail.
        """.strip(),
        how_it_occurs="""
• Zero cost recorded but positive retail price
• Retail set far below suggested retail without documentation
• System defaults creating unusual values
• Import errors from vendor price files
• Manual entry mistakes
        """.strip(),
        cogs_impact="""
If cost data is wrong, COGS calculations are unreliable. Zero-cost items
understate COGS; inflated cost items overstate COGS.
        """.strip(),
        gross_profit_impact="""
Profit calculations are unreliable with bad pricing data. May show profit
where loss exists or vice versa.
        """.strip(),
        gross_margin_impact="""
Margin percentages are meaningless with incorrect pricing data. 100% margin
almost always indicates a data problem, not a great deal.
        """.strip(),
        additional_risks=[
            "Customer confusion if prices are wildly off",
            "Lost profit if selling too low",
            "Lost sales if priced too high",
            "Financial reporting unreliable",
        ],
        recommendations=[
            "Review items showing 100% margin - verify cost data",
            "Check items priced far below suggested retail",
            "Verify data import processes for vendor pricing",
            "Set up validation alerts for unusual margin ranges",
        ],
        color="#8b5cf6",  # Violet
    ),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    message: str
    details: list[str] = field(default_factory=list)
    severity: Severity = Severity.INFO


@dataclass
class ValidationReport:
    """Complete validation report for input file(s)."""

    filename: str
    timestamp: datetime
    checks: list[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True

    def add_check(self, check: ValidationResult) -> None:
        """Add a check result and update overall status."""
        self.checks.append(check)
        if not check.passed and check.severity in (Severity.CRITICAL, Severity.HIGH):
            self.overall_passed = False


@dataclass
class FlaggedItem:
    """An individual item flagged for an inventory issue."""

    sku: str
    description: str
    quantity: float
    cost: float | None
    retail: float | None
    sub_total: float | None
    margin: float | None
    sold: float | None
    last_sale: datetime | None
    vendor: str | None
    category: str | None
    issue_context: str
    raw_data: dict[str, Any]

    @property
    def inventory_value(self) -> float | None:
        """Calculate inventory value."""
        if self.sub_total is not None and self.sub_total > 0:
            return self.sub_total
        if self.quantity and self.cost:
            return abs(self.quantity) * self.cost
        return None

    @property
    def last_sale_display(self) -> str:
        """Format last sale date for display."""
        if self.last_sale is None or pd.isna(self.last_sale):
            return "Never / Unknown"
        return self.last_sale.strftime("%Y-%m-%d")

    @property
    def days_since_sold(self) -> int | None:
        """Calculate days since last sale."""
        if self.last_sale is None or pd.isna(self.last_sale):
            return None
        return (datetime.now() - self.last_sale).days


@dataclass
class PrimitiveFinding:
    """A finding for a specific primitive type."""

    primitive_type: PrimitiveType
    items: list[FlaggedItem] = field(default_factory=list)

    @property
    def definition(self) -> PrimitiveDefinition:
        """Get the definition for this primitive type."""
        return PRIMITIVE_DEFINITIONS[self.primitive_type]

    @property
    def total_items(self) -> int:
        """Count of items flagged."""
        return len(self.items)

    @property
    def total_quantity(self) -> float:
        """Sum of all quantities."""
        return sum(abs(item.quantity) for item in self.items)

    @property
    def total_value(self) -> float:
        """Sum of all inventory values (where calculable)."""
        return sum(item.inventory_value or 0 for item in self.items)


@dataclass
class AnalysisReport:
    """Complete analysis report."""

    filename: str
    analysis_timestamp: datetime
    total_rows: int
    total_skus: int
    validation_report: ValidationReport
    findings: dict[PrimitiveType, PrimitiveFinding]
    total_inventory_value: float | None
    total_inventory_quantity: float

    @property
    def total_issues(self) -> int:
        """Total number of issues found across all primitives."""
        return sum(f.total_items for f in self.findings.values())

    @property
    def total_value_at_risk(self) -> float:
        """Total inventory value across all flagged items."""
        return sum(f.total_value for f in self.findings.values())

    @property
    def critical_findings(self) -> list[PrimitiveFinding]:
        """Findings with critical severity."""
        return [
            f
            for f in self.findings.values()
            if f.definition.severity == Severity.CRITICAL and f.total_items > 0
        ]

    @property
    def high_findings(self) -> list[PrimitiveFinding]:
        """Findings with high severity."""
        return [
            f
            for f in self.findings.values()
            if f.definition.severity == Severity.HIGH and f.total_items > 0
        ]


@dataclass
class AnalysisConfig:
    """Configuration for analysis thresholds and behavior."""

    # Time-based thresholds (in days)
    dead_stock_days: int = 365
    slow_moving_days: int = 180

    # Stock level thresholds
    low_stock_threshold: int = 5
    high_qty_threshold: int = 500  # For overstock detection

    # Margin thresholds
    low_margin_threshold: float = 20.0  # Below this = margin leak
    margin_erosion_threshold: float = 15.0  # Below this = margin erosion

    # Value thresholds
    high_value_threshold: float = 1000.0  # For high-value dead stock

    @classmethod
    def from_yaml(cls, path: Path) -> AnalysisConfig:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed. Using default configuration.")
            return cls()

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            thresholds = data.get("thresholds", {})
            return cls(
                dead_stock_days=thresholds.get("dead_stock_days", 365),
                slow_moving_days=thresholds.get("slow_moving_days", 180),
                low_stock_threshold=thresholds.get("low_stock_threshold", 5),
                high_qty_threshold=thresholds.get("high_qty_threshold", 500),
                low_margin_threshold=thresholds.get("low_margin_threshold", 20.0),
                margin_erosion_threshold=thresholds.get("margin_erosion_threshold", 15.0),
                high_value_threshold=thresholds.get("high_value_threshold", 1000.0),
            )
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}. Using defaults.")
            return cls()


# =============================================================================
# FILE VALIDATION
# =============================================================================


class FileValidator:
    """Validates input files before processing."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.report = ValidationReport(
            filename=filepath.name,
            timestamp=datetime.now(),
        )
        self.df: pd.DataFrame | None = None
        self.column_mapping: dict[str, str] = {}

    def validate(self) -> ValidationReport:
        """Run all validation checks and return report."""
        logger.info(f"Starting validation for: {self.filepath.name}")

        # Step 1: File existence and type
        if not self._check_file_exists():
            return self.report

        if not self._check_file_type():
            return self.report

        # Step 2: Load file
        if not self._load_file():
            return self.report

        # Step 3: Basic structure checks
        self._check_not_empty()
        self._check_no_duplicate_headers()
        self._check_no_blank_rows()

        # Step 4: Column validation
        self._check_required_columns()
        self._check_recommended_columns()

        # Step 5: Data type validation
        self._check_quantity_data_type()
        self._check_cost_data_type()
        self._check_date_formatting()

        logger.info(
            f"Validation complete: {'PASSED' if self.report.overall_passed else 'FAILED'}"
        )
        return self.report

    def _check_file_exists(self) -> bool:
        """Check if file exists."""
        exists = self.filepath.exists()
        self.report.add_check(
            ValidationResult(
                check_name="File Exists",
                passed=exists,
                message=(
                    f"File found at {self.filepath}"
                    if exists
                    else f"File not found: {self.filepath}"
                ),
                severity=Severity.CRITICAL,
            )
        )
        return exists

    def _check_file_type(self) -> bool:
        """Check if file is CSV or Excel."""
        suffix = self.filepath.suffix.lower()
        valid_types = {".csv", ".xlsx", ".xls"}
        is_valid = suffix in valid_types

        self.report.add_check(
            ValidationResult(
                check_name="File Type",
                passed=is_valid,
                message=(
                    f"Valid file type: {suffix}"
                    if is_valid
                    else f"Invalid file type: {suffix}. Expected: {valid_types}"
                ),
                severity=Severity.CRITICAL,
            )
        )
        return is_valid

    def _load_file(self) -> bool:
        """Attempt to load the file into a DataFrame."""
        try:
            suffix = self.filepath.suffix.lower()
            if suffix == ".csv":
                # Try multiple encodings common in hardware store POS exports
                encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
                for encoding in encodings:
                    try:
                        self.df = pd.read_csv(
                            self.filepath, low_memory=False, encoding=encoding
                        )
                        logger.debug(f"Successfully read CSV with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, try with errors='replace'
                    self.df = pd.read_csv(
                        self.filepath,
                        low_memory=False,
                        encoding="utf-8",
                        errors="replace",
                    )
            else:
                self.df = pd.read_excel(self.filepath)

            self.report.add_check(
                ValidationResult(
                    check_name="File Readable",
                    passed=True,
                    message=f"Successfully loaded {len(self.df):,} rows",
                )
            )
            return True
        except Exception as e:
            self.report.add_check(
                ValidationResult(
                    check_name="File Readable",
                    passed=False,
                    message=f"Failed to read file: {str(e)}",
                    severity=Severity.CRITICAL,
                )
            )
            return False

    def _check_not_empty(self) -> None:
        """Check that file has data rows."""
        if self.df is None:
            return

        has_data = len(self.df) > 0
        self.report.add_check(
            ValidationResult(
                check_name="Has Data Rows",
                passed=has_data,
                message=(
                    f"File contains {len(self.df):,} data rows"
                    if has_data
                    else "File is empty"
                ),
                severity=Severity.CRITICAL,
            )
        )

    def _check_no_duplicate_headers(self) -> None:
        """Check for duplicate column headers."""
        if self.df is None:
            return

        columns = self.df.columns.tolist()
        duplicates = list(set(col for col in columns if columns.count(col) > 1))

        no_duplicates = len(duplicates) == 0
        self.report.add_check(
            ValidationResult(
                check_name="No Duplicate Headers",
                passed=no_duplicates,
                message=(
                    "All column headers are unique"
                    if no_duplicates
                    else f"Duplicate headers found: {duplicates}"
                ),
                details=duplicates,
                severity=Severity.HIGH,
            )
        )

    def _check_no_blank_rows(self) -> None:
        """Check for completely blank rows."""
        if self.df is None:
            return

        blank_rows = self.df[self.df.isna().all(axis=1)]
        blank_indices = blank_rows.index.tolist()

        no_blanks = len(blank_indices) == 0
        self.report.add_check(
            ValidationResult(
                check_name="No Blank Rows",
                passed=no_blanks,
                message=(
                    "No completely blank rows found"
                    if no_blanks
                    else f"Found {len(blank_indices)} blank rows"
                ),
                details=[f"Row {i+2}" for i in blank_indices[:10]],
                severity=Severity.MEDIUM,
            )
        )

    def _find_column(self, target: str) -> str | None:
        """Find a column by checking aliases."""
        if self.df is None:
            return None

        aliases = COLUMN_ALIASES.get(target, [])
        df_columns_lower = {col.lower().strip(): col for col in self.df.columns}

        for alias in aliases:
            if alias.lower() in df_columns_lower:
                actual_col = df_columns_lower[alias.lower()]
                self.column_mapping[target] = actual_col
                return actual_col
        return None

    def _check_required_columns(self) -> None:
        """Check that all required columns are present."""
        if self.df is None:
            return

        missing = []
        found = []

        for col in REQUIRED_COLUMNS:
            actual = self._find_column(col)
            if actual:
                found.append(f"{col} (as '{actual}')")
            else:
                missing.append(col)

        all_present = len(missing) == 0
        self.report.add_check(
            ValidationResult(
                check_name="Required Columns Present",
                passed=all_present,
                message=(
                    "All required columns found"
                    if all_present
                    else f"Missing required columns: {missing}"
                ),
                details=found + [f"MISSING: {m}" for m in missing],
                severity=Severity.CRITICAL,
            )
        )

    def _check_recommended_columns(self) -> None:
        """Check for recommended columns."""
        if self.df is None:
            return

        missing = []
        found = []

        for col in RECOMMENDED_COLUMNS:
            actual = self._find_column(col)
            if actual:
                found.append(f"{col} (as '{actual}')")
            else:
                missing.append(col)

        all_present = len(missing) == 0
        self.report.add_check(
            ValidationResult(
                check_name="Recommended Columns Present",
                passed=all_present,
                message=(
                    "All recommended columns found"
                    if all_present
                    else f"Missing recommended columns: {missing}"
                ),
                details=found + [f"MISSING: {m}" for m in missing],
                severity=Severity.LOW if all_present else Severity.MEDIUM,
            )
        )

    def _check_quantity_data_type(self) -> None:
        """Check that quantity column contains numeric data."""
        if self.df is None:
            return

        qty_col = self.column_mapping.get("quantity")
        if not qty_col:
            return

        non_numeric = []
        for idx, val in self.df[qty_col].items():
            if pd.isna(val):
                continue
            try:
                float(str(val).replace(",", ""))
            except (ValueError, TypeError):
                non_numeric.append(f"Row {idx+2}: '{val}'")

        all_numeric = len(non_numeric) == 0
        self.report.add_check(
            ValidationResult(
                check_name="Quantity Data Type",
                passed=all_numeric,
                message=(
                    "All quantity values are numeric"
                    if all_numeric
                    else f"Found {len(non_numeric)} non-numeric quantity values"
                ),
                details=non_numeric[:10],
                severity=Severity.HIGH,
            )
        )

    def _check_cost_data_type(self) -> None:
        """Check that cost column contains numeric data."""
        if self.df is None:
            return

        cost_col = self.column_mapping.get("cost")
        if not cost_col:
            return

        non_numeric = []
        for idx, val in self.df[cost_col].items():
            if pd.isna(val):
                continue
            if isinstance(val, str):
                val = val.replace("$", "").replace(",", "").strip()
            try:
                float(val)
            except (ValueError, TypeError):
                non_numeric.append(f"Row {idx+2}: '{val}'")

        all_numeric = len(non_numeric) == 0
        self.report.add_check(
            ValidationResult(
                check_name="Cost Data Type",
                passed=all_numeric,
                message=(
                    "All cost values are numeric or blank"
                    if all_numeric
                    else f"Found {len(non_numeric)} non-numeric cost values"
                ),
                details=non_numeric[:10],
                severity=Severity.HIGH,
            )
        )

    def _check_date_formatting(self) -> None:
        """Check that date columns can be parsed."""
        if self.df is None:
            return

        date_col = self.column_mapping.get("last_sale")
        if not date_col:
            self.report.add_check(
                ValidationResult(
                    check_name="Date Format",
                    passed=True,
                    message="No date column found (will mark as unknown)",
                    severity=Severity.INFO,
                )
            )
            return

        unparseable = []
        for idx, val in self.df[date_col].items():
            if pd.isna(val) or val == "" or val == 0:
                continue
            try:
                self._parse_date(val)
            except (ValueError, TypeError):
                unparseable.append(f"Row {idx+2}: '{val}'")

        all_parseable = len(unparseable) == 0
        self.report.add_check(
            ValidationResult(
                check_name="Date Format",
                passed=all_parseable,
                message=(
                    "All date values can be parsed"
                    if all_parseable
                    else f"Found {len(unparseable)} unparseable dates"
                ),
                details=unparseable[:10],
                severity=Severity.MEDIUM,
            )
        )

    def _parse_date(self, val: Any) -> datetime | None:
        """Parse various date formats."""
        if pd.isna(val) or val == "" or val == 0:
            return None

        # Handle YYYYMMDD format (common in hardware POS exports)
        if isinstance(val, (int, float)):
            val_str = str(int(val))
            if len(val_str) == 8:
                return datetime.strptime(val_str, "%Y%m%d")

        # Try pandas parsing
        return pd.to_datetime(val)


# =============================================================================
# INVENTORY ANALYZER
# =============================================================================


class HardwareInventoryAnalyzer:
    """Analyzes hardware store inventory data for various primitive issues."""

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig()
        self.column_mapping: dict[str, str] = {}

    def analyze(self, filepath: Path) -> AnalysisReport:
        """Run complete analysis on an inventory file."""
        logger.info(f"Starting analysis: {filepath.name}")

        # Validate first
        validator = FileValidator(filepath)
        validation_report = validator.validate()

        if not validation_report.overall_passed:
            logger.error("Validation failed. Cannot proceed with analysis.")
            return AnalysisReport(
                filename=filepath.name,
                analysis_timestamp=datetime.now(),
                total_rows=0,
                total_skus=0,
                validation_report=validation_report,
                findings={pt: PrimitiveFinding(pt) for pt in PrimitiveType},
                total_inventory_value=None,
                total_inventory_quantity=0,
            )

        # Use validated data
        df = validator.df
        self.column_mapping = validator.column_mapping

        # Clean and prepare data
        df = self._prepare_data(df)

        # Calculate totals
        total_value = self._calculate_total_value(df)
        total_qty = self._calculate_total_quantity(df)
        total_skus = len(df)

        # Run all primitive checks
        findings = self._run_all_checks(df)

        report = AnalysisReport(
            filename=filepath.name,
            analysis_timestamp=datetime.now(),
            total_rows=len(df),
            total_skus=total_skus,
            validation_report=validation_report,
            findings=findings,
            total_inventory_value=total_value,
            total_inventory_quantity=total_qty,
        )

        logger.info(
            f"Analysis complete. Found {report.total_issues:,} issues "
            f"across {len([f for f in findings.values() if f.total_items > 0])} primitives."
        )
        return report

    def _get_col(self, target: str) -> str | None:
        """Get actual column name from mapping."""
        return self.column_mapping.get(target)

    def _safe_float(self, val: Any) -> float:
        """Safely convert to float."""
        if pd.isna(val) or val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            return 0.0

    def _parse_date(self, val: Any) -> datetime | None:
        """Parse various date formats."""
        if pd.isna(val) or val == "" or val == 0 or val is None:
            return None
        try:
            # Handle YYYYMMDD format
            if isinstance(val, (int, float)):
                val_str = str(int(val))
                if len(val_str) == 8 and val_str.startswith("20"):
                    return datetime.strptime(val_str, "%Y%m%d")
            return pd.to_datetime(val)
        except Exception:
            return None

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame for analysis."""
        df = df.copy()

        # Convert quantity to numeric
        qty_col = self._get_col("quantity")
        if qty_col:
            df["_quantity"] = df[qty_col].apply(self._safe_float)
        else:
            df["_quantity"] = 0

        # Convert cost to numeric
        cost_col = self._get_col("cost")
        if cost_col:
            df["_cost"] = df[cost_col].apply(self._safe_float)
        else:
            df["_cost"] = None

        # Convert retail to numeric
        retail_col = self._get_col("retail")
        if retail_col:
            df["_retail"] = df[retail_col].apply(self._safe_float)
        else:
            df["_retail"] = None

        # Convert sub_total to numeric
        sub_total_col = self._get_col("sub_total")
        if sub_total_col:
            df["_sub_total"] = df[sub_total_col].apply(self._safe_float)
        else:
            df["_sub_total"] = None

        # Convert margin to numeric
        margin_col = self._get_col("margin")
        if margin_col:
            df["_margin"] = df[margin_col].apply(self._safe_float)
        else:
            df["_margin"] = None

        # Convert sold to numeric
        sold_col = self._get_col("sold")
        if sold_col:
            df["_sold"] = df[sold_col].apply(self._safe_float)
        else:
            df["_sold"] = None

        # Parse dates
        last_sale_col = self._get_col("last_sale")
        if last_sale_col:
            df["_last_sale"] = df[last_sale_col].apply(self._parse_date)
        else:
            df["_last_sale"] = None

        # Calculate inventory value if not present
        if "_sub_total" in df.columns and df["_sub_total"].notna().any():
            df["_inventory_value"] = df["_sub_total"]
        elif "_quantity" in df.columns and "_cost" in df.columns:
            df["_inventory_value"] = df["_quantity"].abs() * df["_cost"].fillna(0)
        else:
            df["_inventory_value"] = None

        return df

    def _calculate_total_value(self, df: pd.DataFrame) -> float | None:
        """Calculate total inventory value."""
        if "_inventory_value" in df.columns:
            return df["_inventory_value"].sum()
        return None

    def _calculate_total_quantity(self, df: pd.DataFrame) -> float:
        """Calculate total inventory quantity."""
        if "_quantity" in df.columns:
            return df["_quantity"].sum()
        return 0

    def _create_flagged_item(self, row: pd.Series, context: str) -> FlaggedItem:
        """Create a FlaggedItem from a DataFrame row."""
        return FlaggedItem(
            sku=str(row.get(self._get_col("sku"), "N/A")).strip(),
            description=str(row.get(self._get_col("description"), "N/A"))[:100],
            quantity=row.get("_quantity", 0),
            cost=row.get("_cost") if pd.notna(row.get("_cost")) else None,
            retail=row.get("_retail") if pd.notna(row.get("_retail")) else None,
            sub_total=row.get("_sub_total") if pd.notna(row.get("_sub_total")) else None,
            margin=row.get("_margin") if pd.notna(row.get("_margin")) else None,
            sold=row.get("_sold") if pd.notna(row.get("_sold")) else None,
            last_sale=row.get("_last_sale"),
            vendor=row.get(self._get_col("vendor")),
            category=row.get(self._get_col("category")),
            issue_context=context,
            raw_data=row.to_dict(),
        )

    def _run_all_checks(self, df: pd.DataFrame) -> dict[PrimitiveType, PrimitiveFinding]:
        """Run all primitive checks and return findings."""
        findings = {}

        # Critical checks
        findings[PrimitiveType.NEGATIVE_INVENTORY] = self._check_negative_inventory(df)
        findings[PrimitiveType.MISSING_COST] = self._check_missing_cost(df)
        findings[PrimitiveType.NEGATIVE_COST] = self._check_negative_cost(df)
        findings[PrimitiveType.COST_EXCEEDS_PRICE] = self._check_cost_exceeds_price(df)
        findings[PrimitiveType.HIGH_VALUE_DEAD_STOCK] = self._check_high_value_dead_stock(df)

        # High severity checks
        findings[PrimitiveType.DEAD_STOCK] = self._check_dead_stock(df)
        findings[PrimitiveType.ZERO_COST_WITH_QUANTITY] = self._check_zero_cost_with_quantity(df)
        findings[PrimitiveType.HIGH_MARGIN_LEAK] = self._check_high_margin_leak(df)
        findings[PrimitiveType.MARGIN_EROSION] = self._check_margin_erosion(df)
        findings[PrimitiveType.LOW_STOCK] = self._check_low_stock(df)
        findings[PrimitiveType.SHRINKAGE_PATTERN] = self._check_shrinkage_pattern(df)

        # Medium severity checks
        findings[PrimitiveType.SLOW_MOVING] = self._check_slow_moving(df)
        findings[PrimitiveType.OVERSTOCK] = self._check_overstock(df)
        findings[PrimitiveType.MISSING_LAST_SOLD] = self._check_missing_last_sold(df)
        findings[PrimitiveType.PRICE_DISCREPANCY] = self._check_price_discrepancy(df)

        return findings

    def _check_negative_inventory(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for negative inventory quantities."""
        finding = PrimitiveFinding(PrimitiveType.NEGATIVE_INVENTORY)

        negative = df[df["_quantity"] < 0]
        for _, row in negative.iterrows():
            qty = row["_quantity"]
            value = abs(qty) * (row.get("_cost", 0) or 0)
            context = f"QOH is {qty:,.0f} units (NEGATIVE). Value: ${value:,.2f}. Audit immediately."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Negative Inventory: {finding.total_items:,} items found")
        return finding

    def _check_missing_cost(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items missing cost data."""
        finding = PrimitiveFinding(PrimitiveType.MISSING_COST)

        # Find rows with null/blank/zero cost but positive quantity
        mask = (
            (df["_cost"].isna() | (df["_cost"] == 0))
            & (df["_quantity"] > 0)
        )
        missing = df[mask]

        for _, row in missing.iterrows():
            context = f"Cost is blank/zero but has {row['_quantity']:,.0f} units. COGS cannot be calculated."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Missing Cost: {finding.total_items:,} items found")
        return finding

    def _check_negative_cost(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for negative cost values."""
        finding = PrimitiveFinding(PrimitiveType.NEGATIVE_COST)

        negative = df[df["_cost"] < 0]
        for _, row in negative.iterrows():
            context = f"Cost is ${row['_cost']:.2f} (NEGATIVE) - impossible value, corrupts financials."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Negative Cost: {finding.total_items:,} items found")
        return finding

    def _check_cost_exceeds_price(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items where cost exceeds selling price."""
        finding = PrimitiveFinding(PrimitiveType.COST_EXCEEDS_PRICE)

        mask = (
            (df["_cost"] > df["_retail"])
            & (df["_cost"] > 0)
            & (df["_retail"] > 0)
            & (df["_quantity"] > 0)
        )
        negative_margin = df[mask]

        for _, row in negative_margin.iterrows():
            cost = row["_cost"]
            price = row["_retail"]
            loss = cost - price
            context = f"Cost ${cost:.2f} > Price ${price:.2f} = ${loss:.2f} LOSS per sale. Fix pricing immediately."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Cost Exceeds Price: {finding.total_items:,} items found")
        return finding

    def _check_dead_stock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for dead stock (no sales in 12+ months)."""
        finding = PrimitiveFinding(PrimitiveType.DEAD_STOCK)

        cutoff = datetime.now() - timedelta(days=self.config.dead_stock_days)

        # Items with inventory that haven't sold since cutoff
        mask = (df["_quantity"] > 0) & (df["_last_sale"].notna()) & (df["_last_sale"] < cutoff)
        dead = df[mask]

        for _, row in dead.iterrows():
            days_ago = (datetime.now() - row["_last_sale"]).days
            months = days_ago // 30
            value = row.get("_inventory_value", 0) or 0
            context = f"No sales in {days_ago} days ({months} months). Value at risk: ${value:,.2f}."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Dead Stock: {finding.total_items:,} items found")
        return finding

    def _check_high_value_dead_stock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for high-value dead stock."""
        finding = PrimitiveFinding(PrimitiveType.HIGH_VALUE_DEAD_STOCK)

        cutoff = datetime.now() - timedelta(days=self.config.dead_stock_days)

        mask = (
            (df["_quantity"] > 0)
            & (df["_last_sale"].notna())
            & (df["_last_sale"] < cutoff)
            & (df["_inventory_value"] >= self.config.high_value_threshold)
        )
        high_value_dead = df[mask]

        for _, row in high_value_dead.iterrows():
            days_ago = (datetime.now() - row["_last_sale"]).days
            value = row.get("_inventory_value", 0) or 0
            context = f"${value:,.2f} at risk - No sales in {days_ago} days. Prioritize for liquidation."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"High-Value Dead Stock: {finding.total_items:,} items found")
        return finding

    def _check_zero_cost_with_quantity(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items with zero cost but positive quantity."""
        finding = PrimitiveFinding(PrimitiveType.ZERO_COST_WITH_QUANTITY)

        mask = (df["_cost"] == 0) & (df["_quantity"] > 0) & (df["_cost"].notna())
        zero_cost = df[mask]

        for _, row in zero_cost.iterrows():
            context = f"Cost is $0.00 but has {row['_quantity']:,.0f} units. Verify cost data."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Zero Cost with Quantity: {finding.total_items:,} items found")
        return finding

    def _check_high_margin_leak(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items with margin below threshold."""
        finding = PrimitiveFinding(PrimitiveType.HIGH_MARGIN_LEAK)

        mask = (
            (df["_margin"].notna())
            & (df["_margin"] > 0)
            & (df["_margin"] < self.config.low_margin_threshold)
            & (df["_quantity"] > 0)
        )
        low_margin = df[mask]

        for _, row in low_margin.iterrows():
            margin = row["_margin"]
            cost = row.get("_cost", 0) or 0
            retail = row.get("_retail", 0) or 0
            context = f"Margin is only {margin:.1f}% (Cost: ${cost:.2f}, Retail: ${retail:.2f}). Review pricing."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"High Margin Leak: {finding.total_items:,} items found")
        return finding

    def _check_margin_erosion(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items where cost is approaching revenue."""
        finding = PrimitiveFinding(PrimitiveType.MARGIN_EROSION)

        mask = (
            (df["_margin"].notna())
            & (df["_margin"] > 0)
            & (df["_margin"] < self.config.margin_erosion_threshold)
            & (df["_quantity"] > 0)
            & (df["_retail"] > 0)
            & (df["_cost"] > 0)
        )
        eroded = df[mask]

        for _, row in eroded.iterrows():
            margin = row["_margin"]
            cost = row.get("_cost", 0)
            retail = row.get("_retail", 0)
            cost_pct = (cost / retail * 100) if retail > 0 else 0
            context = f"Margin eroded to {margin:.1f}%. Cost is {cost_pct:.0f}% of retail. Consider repricing."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Margin Erosion: {finding.total_items:,} items found")
        return finding

    def _check_low_stock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for low stock items."""
        finding = PrimitiveFinding(PrimitiveType.LOW_STOCK)

        # Low stock with recent sales activity
        recent_cutoff = datetime.now() - timedelta(days=90)
        mask = (
            (df["_quantity"] > 0)
            & (df["_quantity"] <= self.config.low_stock_threshold)
            & (
                (df["_sold"].notna() & (df["_sold"] > 50))
                | (df["_last_sale"].notna() & (df["_last_sale"] >= recent_cutoff))
            )
        )
        low = df[mask]

        for _, row in low.iterrows():
            sold = row.get("_sold", 0) or 0
            context = f"Only {row['_quantity']:.0f} units left, sold {sold:.0f} recently. Reorder soon."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Low Stock: {finding.total_items:,} items found")
        return finding

    def _check_shrinkage_pattern(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for potential shrinkage patterns."""
        finding = PrimitiveFinding(PrimitiveType.SHRINKAGE_PATTERN)

        # High value, low margin, minimal sales
        mask = (
            (df["_inventory_value"] > self.config.high_value_threshold)
            & (df["_margin"].notna())
            & (df["_margin"] < 15)
            & ((df["_sold"].isna()) | (df["_sold"] < 10))
        )
        shrinkage = df[mask]

        for _, row in shrinkage.iterrows():
            value = row.get("_inventory_value", 0) or 0
            margin = row.get("_margin", 0) or 0
            sold = row.get("_sold", 0) or 0
            context = f"High value (${value:,.2f}), low margin ({margin:.1f}%), minimal sales ({sold:.0f}). Investigate."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Shrinkage Pattern: {finding.total_items:,} items found")
        return finding

    def _check_slow_moving(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for slow-moving inventory."""
        finding = PrimitiveFinding(PrimitiveType.SLOW_MOVING)

        dead_cutoff = datetime.now() - timedelta(days=self.config.dead_stock_days)
        slow_cutoff = datetime.now() - timedelta(days=self.config.slow_moving_days)

        mask = (
            (df["_quantity"] > 0)
            & (df["_last_sale"].notna())
            & (df["_last_sale"] >= dead_cutoff)
            & (df["_last_sale"] < slow_cutoff)
        )
        slow = df[mask]

        for _, row in slow.iterrows():
            days_ago = (datetime.now() - row["_last_sale"]).days
            context = f"No sales in {days_ago} days. Trending toward dead stock."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Slow Moving: {finding.total_items:,} items found")
        return finding

    def _check_overstock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for overstocked items."""
        finding = PrimitiveFinding(PrimitiveType.OVERSTOCK)

        # High quantity with low or no recent sales
        mask = (
            (df["_quantity"] > self.config.high_qty_threshold)
            & ((df["_sold"].isna()) | (df["_sold"] < 10))
        )
        overstock = df[mask]

        for _, row in overstock.iterrows():
            qty = row["_quantity"]
            sold = row.get("_sold", 0) or 0
            months_supply = qty / sold if sold > 0 else float("inf")
            value = row.get("_inventory_value", 0) or 0
            context = f"{qty:,.0f} units on hand vs {sold:.0f} sold = {months_supply:.0f}+ months supply. Value: ${value:,.2f}."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Overstock: {finding.total_items:,} items found")
        return finding

    def _check_missing_last_sold(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items missing last sold date."""
        finding = PrimitiveFinding(PrimitiveType.MISSING_LAST_SOLD)

        mask = (df["_last_sale"].isna()) & (df["_quantity"] > 0)
        missing = df[mask]

        for _, row in missing.iterrows():
            context = "No last sale date - cannot assess inventory age or velocity."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Missing Last Sold: {finding.total_items:,} items found")
        return finding

    def _check_price_discrepancy(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for unusual pricing patterns."""
        finding = PrimitiveFinding(PrimitiveType.PRICE_DISCREPANCY)

        # 100% margin (zero cost but positive price) - suspicious
        mask = (
            (df["_margin"] == 100)
            & (df["_cost"] == 0)
            & (df["_retail"] > 0)
            & (df["_quantity"] > 0)
        )
        discrepancy = df[mask]

        for _, row in discrepancy.iterrows():
            retail = row.get("_retail", 0)
            context = f"Shows 100% margin (zero cost) but retail is ${retail:.2f}. Verify cost data."
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Price Discrepancy: {finding.total_items:,} items found")
        return finding


# =============================================================================
# REPORT GENERATOR
# =============================================================================


class ReportGenerator:
    """Generates HTML and console reports from analysis results."""

    def __init__(self, report: AnalysisReport):
        self.report = report

    def print_console_summary(self) -> None:
        """Print summary to console."""
        r = self.report

        print("\n" + "=" * 70)
        print("HARDWARE STORE INVENTORY HEALTH ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nFile: {r.filename}")
        print(f"Analyzed: {r.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total SKUs: {r.total_skus:,}")
        print(f"Total Quantity: {r.total_inventory_quantity:,.0f} units")
        if r.total_inventory_value:
            print(f"Total Value: ${r.total_inventory_value:,.2f}")

        print("\n" + "-" * 70)
        print("VALIDATION STATUS")
        print("-" * 70)
        print(
            f"Overall: {'✓ PASSED' if r.validation_report.overall_passed else '✗ FAILED'}"
        )
        for check in r.validation_report.checks:
            status = "✓" if check.passed else "✗"
            print(f"  {status} {check.check_name}: {check.message}")

        print("\n" + "-" * 70)
        print("FINDINGS SUMMARY")
        print("-" * 70)

        # Group by severity
        critical = [
            f
            for f in r.findings.values()
            if f.definition.severity == Severity.CRITICAL and f.total_items > 0
        ]
        high = [
            f
            for f in r.findings.values()
            if f.definition.severity == Severity.HIGH and f.total_items > 0
        ]
        medium = [
            f
            for f in r.findings.values()
            if f.definition.severity == Severity.MEDIUM and f.total_items > 0
        ]

        print(f"\n🔴 CRITICAL Issues: {sum(f.total_items for f in critical):,}")
        for f in critical:
            print(f"   • {f.definition.name}: {f.total_items:,} items (${f.total_value:,.2f})")

        print(f"\n🟠 HIGH Issues: {sum(f.total_items for f in high):,}")
        for f in high:
            print(f"   • {f.definition.name}: {f.total_items:,} items (${f.total_value:,.2f})")

        print(f"\n🟡 MEDIUM Issues: {sum(f.total_items for f in medium):,}")
        for f in medium:
            print(f"   • {f.definition.name}: {f.total_items:,} items (${f.total_value:,.2f})")

        print(f"\n{'=' * 70}")
        print(f"TOTAL ISSUES: {r.total_issues:,}")
        print(f"TOTAL VALUE AT RISK: ${r.total_value_at_risk:,.2f}")
        print("=" * 70 + "\n")

    def generate_html(self, output_path: Path) -> None:
        """Generate comprehensive HTML report."""
        # Build the HTML content
        html = self._build_html_report()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report generated: {output_path}")

    def _build_html_report(self) -> str:
        """Build the complete HTML report."""
        r = self.report

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Health Report - {r.filename}</title>
    {self._get_styles()}
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_executive_summary()}
        {self._generate_findings_sections()}
        {self._generate_recommendations_section()}
        {self._generate_validation_section()}
        {self._generate_footer()}
    </div>
</body>
</html>"""

    def _get_styles(self) -> str:
        """Return CSS styles for the report."""
        return """
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1, h2, h3, h4 { color: #f1f5f9; }
        .header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 16px;
            margin-bottom: 30px;
            border: 1px solid #334155;
        }
        .header h1 { font-size: 2.5em; color: #10b981; margin-bottom: 10px; }
        .header .subtitle { color: #94a3b8; font-size: 1.2em; }
        .meta-info {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .meta-item {
            background: #1e293b;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
        }
        .meta-item .label { color: #94a3b8; font-size: 0.9em; }
        .meta-item .value { color: #f1f5f9; font-size: 1.4em; font-weight: bold; }
        .section {
            background: #1e293b;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #334155;
        }
        .section h2 {
            color: #f1f5f9;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #334155;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: #0f172a;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #334155;
        }
        .summary-card.critical { border-color: #dc2626; }
        .summary-card.high { border-color: #f97316; }
        .summary-card .number { font-size: 2.5em; font-weight: bold; }
        .summary-card.critical .number { color: #dc2626; }
        .summary-card.high .number { color: #f97316; }
        .summary-card .label { color: #94a3b8; margin-top: 5px; }
        .primitive-definition {
            background: #0f172a;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #334155;
            border-left: 4px solid;
        }
        .primitive-definition h3 {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        .severity-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .severity-badge.critical { background: #dc262633; color: #dc2626; }
        .severity-badge.high { background: #f9731633; color: #f97316; }
        .severity-badge.medium { background: #eab30833; color: #eab308; }
        .definition-block {
            background: #1e293b;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .definition-block h4 {
            color: #10b981;
            margin-bottom: 10px;
            font-size: 0.95em;
            text-transform: uppercase;
        }
        .definition-block p, .definition-block ul { color: #cbd5e1; font-size: 0.95em; }
        .definition-block ul { margin-left: 20px; }
        .definition-block li { margin-bottom: 5px; }
        .impact-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .impact-card {
            background: #0f172a;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #334155;
        }
        .impact-card h5 { color: #f97316; margin-bottom: 8px; font-size: 0.9em; }
        .impact-card p { color: #94a3b8; font-size: 0.9em; }
        .findings-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
            flex-wrap: wrap;
            gap: 15px;
        }
        .findings-stats { display: flex; gap: 20px; }
        .findings-stats .stat {
            background: #0f172a;
            padding: 8px 15px;
            border-radius: 8px;
        }
        .findings-stats .stat-label { color: #94a3b8; font-size: 0.8em; }
        .findings-stats .stat-value { color: #f1f5f9; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #334155; }
        th {
            background: #0f172a;
            color: #10b981;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }
        tr:hover { background: #1e293b55; }
        .sku { font-family: monospace; color: #38bdf8; }
        .quantity { font-weight: bold; }
        .quantity.negative { color: #dc2626; }
        .quantity.low { color: #eab308; }
        .value { color: #10b981; font-weight: bold; }
        .context { color: #fbbf24; font-size: 0.9em; font-style: italic; }
        .date { color: #94a3b8; }
        .date.never { color: #6b7280; font-style: italic; }
        .validation-log { background: #0f172a; border-radius: 8px; padding: 20px; }
        .validation-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px 0;
            border-bottom: 1px solid #334155;
        }
        .validation-item:last-child { border-bottom: none; }
        .validation-icon { font-size: 1.2em; }
        .validation-icon.pass { color: #22c55e; }
        .validation-icon.fail { color: #dc2626; }
        .recommendations { background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, #0f172a 100%); }
        .recommendations h2 { color: #10b981; }
        .recommendation-list { list-style: none; }
        .recommendation-list li {
            padding: 15px;
            margin-bottom: 10px;
            background: #1e293b;
            border-radius: 8px;
            border-left: 3px solid #10b981;
        }
        .footer { text-align: center; padding: 30px; color: #64748b; font-size: 0.9em; }
        .footer a { color: #10b981; text-decoration: none; }
        @media (max-width: 768px) {
            .header h1 { font-size: 1.8em; }
            table { font-size: 0.85em; }
            th, td { padding: 8px 10px; }
        }
    </style>"""

    def _generate_header(self) -> str:
        """Generate the header section."""
        r = self.report
        value_html = f'<div class="meta-item"><div class="label">Total Value</div><div class="value">${r.total_inventory_value:,.2f}</div></div>' if r.total_inventory_value else ''

        return f"""
        <div class="header">
            <h1>📊 Inventory Health Analysis</h1>
            <p class="subtitle">Hardware Store Profit Leak Detection Report</p>
            <div class="meta-info">
                <div class="meta-item">
                    <div class="label">File Analyzed</div>
                    <div class="value">{r.filename}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Analysis Date</div>
                    <div class="value">{r.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Total SKUs</div>
                    <div class="value">{r.total_skus:,}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Total Quantity</div>
                    <div class="value">{r.total_inventory_quantity:,.0f}</div>
                </div>
                {value_html}
            </div>
        </div>"""

    def _generate_executive_summary(self) -> str:
        """Generate the executive summary section."""
        r = self.report

        breakdown_html = ""
        for finding in sorted(
            r.findings.values(),
            key=lambda f: (f.definition.severity.value, -f.total_items),
        ):
            if finding.total_items == 0:
                continue
            breakdown_html += f"""
            <div style="display: flex; justify-content: space-between; padding: 10px; background: #0f172a; border-radius: 8px; border-left: 4px solid {finding.definition.color}; margin-bottom: 8px;">
                <span>{finding.definition.name}</span>
                <span><strong>{finding.total_items:,}</strong> items (${finding.total_value:,.2f})</span>
            </div>"""

        return f"""
        <div class="section">
            <h2>📋 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card critical">
                    <div class="number">{sum(f.total_items for f in r.critical_findings):,}</div>
                    <div class="label">Critical Issues</div>
                </div>
                <div class="summary-card high">
                    <div class="number">{sum(f.total_items for f in r.high_findings):,}</div>
                    <div class="label">High Severity</div>
                </div>
                <div class="summary-card">
                    <div class="number" style="color: #10b981;">{r.total_issues:,}</div>
                    <div class="label">Total Issues</div>
                </div>
                <div class="summary-card">
                    <div class="number" style="color: #f97316;">${r.total_value_at_risk:,.0f}</div>
                    <div class="label">Value at Risk</div>
                </div>
            </div>
            <h3 style="margin-bottom: 15px; color: #94a3b8;">Issues by Category</h3>
            {breakdown_html if breakdown_html else "<p>No issues found! 🎉</p>"}
        </div>"""

    def _generate_findings_sections(self) -> str:
        """Generate all findings sections with definitions."""
        html = '<div class="section"><h2>🔍 Detailed Findings by Issue Type</h2>'

        sorted_findings = sorted(
            self.report.findings.values(),
            key=lambda f: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}[
                    f.definition.severity.value
                ],
                -f.total_items,
            ),
        )

        for finding in sorted_findings:
            if finding.total_items == 0:
                continue

            defn = finding.definition
            severity = defn.severity.value

            html += f"""
            <div class="primitive-definition" style="border-left-color: {defn.color};">
                <h3>
                    {defn.name}
                    <span class="severity-badge {severity}">{severity}</span>
                </h3>

                <div class="definition-block">
                    <h4>📖 Definition</h4>
                    <p>{defn.full_definition}</p>
                </div>

                <div class="definition-block">
                    <h4>⚙️ How This Occurs</h4>
                    <p style="white-space: pre-line;">{defn.how_it_occurs}</p>
                </div>

                <div class="impact-grid">
                    <div class="impact-card">
                        <h5>💰 COGS Impact</h5>
                        <p>{defn.cogs_impact}</p>
                    </div>
                    <div class="impact-card">
                        <h5>📈 Gross Profit Impact</h5>
                        <p>{defn.gross_profit_impact}</p>
                    </div>
                    <div class="impact-card">
                        <h5>📊 Gross Margin Impact</h5>
                        <p>{defn.gross_margin_impact}</p>
                    </div>
                </div>

                <div class="definition-block">
                    <h4>⚠️ Additional Business Risks</h4>
                    <ul>
                        {"".join(f"<li>{risk}</li>" for risk in defn.additional_risks)}
                    </ul>
                </div>

                <div class="findings-header">
                    <h3 style="color: {defn.color};">Affected Items</h3>
                    <div class="findings-stats">
                        <div class="stat">
                            <div class="stat-label">Items</div>
                            <div class="stat-value">{finding.total_items:,}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Total Qty</div>
                            <div class="stat-value">{finding.total_quantity:,.0f}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Value at Risk</div>
                            <div class="stat-value">${finding.total_value:,.2f}</div>
                        </div>
                    </div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>SKU</th>
                            <th>Description</th>
                            <th>QOH</th>
                            <th>Cost</th>
                            <th>Value</th>
                            <th>Last Sale</th>
                            <th>Issue Context</th>
                        </tr>
                    </thead>
                    <tbody>
                        {self._generate_items_rows(finding.items)}
                    </tbody>
                </table>
            </div>"""

        if not any(f.total_items > 0 for f in self.report.findings.values()):
            html += "<p>No issues found! Your inventory looks healthy! 🎉</p>"

        html += "</div>"
        return html

    def _generate_items_rows(self, items: list[FlaggedItem], max_items: int = 100) -> str:
        """Generate table rows for flagged items."""
        html = ""
        for item in items[:max_items]:
            qty_class = (
                "negative"
                if item.quantity < 0
                else "low"
                if item.quantity <= 5
                else ""
            )
            date_class = "never" if item.last_sale is None else ""

            html += f"""
            <tr>
                <td class="sku">{item.sku}</td>
                <td>{item.description[:50]}{'...' if len(item.description) > 50 else ''}</td>
                <td class="quantity {qty_class}">{item.quantity:,.0f}</td>
                <td>{f'${item.cost:.2f}' if item.cost is not None else 'N/A'}</td>
                <td class="value">{f'${item.inventory_value:,.2f}' if item.inventory_value else 'N/A'}</td>
                <td class="date {date_class}">{item.last_sale_display}</td>
                <td class="context">{item.issue_context}</td>
            </tr>"""

        if len(items) > max_items:
            html += f"""
            <tr>
                <td colspan="7" style="text-align: center; color: #94a3b8; padding: 20px;">
                    ... and {len(items) - max_items:,} more items (showing first {max_items})
                </td>
            </tr>"""

        return html

    def _generate_recommendations_section(self) -> str:
        """Generate prioritized recommendations."""
        html = '<div class="section recommendations"><h2>✅ Recommended Actions</h2><ul class="recommendation-list">'

        recommendations_by_priority: dict[Severity, list[str]] = {
            Severity.CRITICAL: [],
            Severity.HIGH: [],
            Severity.MEDIUM: [],
        }

        for finding in self.report.findings.values():
            if finding.total_items > 0:
                for rec in finding.definition.recommendations[:3]:
                    if rec not in recommendations_by_priority[finding.definition.severity]:
                        recommendations_by_priority[finding.definition.severity].append(rec)

        if recommendations_by_priority[Severity.CRITICAL]:
            html += '<li style="border-left-color: #dc2626;"><strong>🔴 CRITICAL - Address Immediately:</strong><ul>'
            for rec in recommendations_by_priority[Severity.CRITICAL][:5]:
                html += f"<li>{rec}</li>"
            html += "</ul></li>"

        if recommendations_by_priority[Severity.HIGH]:
            html += '<li style="border-left-color: #f97316;"><strong>🟠 HIGH PRIORITY - This Week:</strong><ul>'
            for rec in recommendations_by_priority[Severity.HIGH][:5]:
                html += f"<li>{rec}</li>"
            html += "</ul></li>"

        if recommendations_by_priority[Severity.MEDIUM]:
            html += '<li style="border-left-color: #eab308;"><strong>🟡 MEDIUM - Schedule Soon:</strong><ul>'
            for rec in recommendations_by_priority[Severity.MEDIUM][:5]:
                html += f"<li>{rec}</li>"
            html += "</ul></li>"

        html += "</ul></div>"
        return html

    def _generate_validation_section(self) -> str:
        """Generate validation log section."""
        html = '<div class="section"><h2>📝 Validation Log</h2><div class="validation-log">'

        for check in self.report.validation_report.checks:
            icon_class = "pass" if check.passed else "fail"
            icon = "✓" if check.passed else "✗"

            html += f"""
            <div class="validation-item">
                <span class="validation-icon {icon_class}">{icon}</span>
                <div>
                    <strong>{check.check_name}</strong>
                    <div style="color: #94a3b8; font-size: 0.9em;">{check.message}</div>
                    {"".join(f'<div style="color: #94a3b8; font-size: 0.85em;">• {d}</div>' for d in check.details[:5])}
                </div>
            </div>"""

        html += "</div></div>"
        return html

    def _generate_footer(self) -> str:
        """Generate footer section."""
        return """
        <div class="footer">
            <p>Generated by <strong>Profit Sentinel</strong> Hardware Inventory Analyzer</p>
            <p>For questions or support, visit <a href="https://profitsentinel.com">profitsentinel.com</a></p>
        </div>"""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """Main entry point for the inventory analyzer."""
    parser = argparse.ArgumentParser(
        description="Hardware Store Inventory Health Analyzer - Detect profit leaks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input file(s) to analyze (CSV or Excel)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("inventory_report.html"),
        help="Output HTML report path",
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = (
        AnalysisConfig.from_yaml(args.config)
        if args.config
        else AnalysisConfig()
    )
    logger.info(
        f"Configuration: dead_stock={config.dead_stock_days}d, "
        f"slow_moving={config.slow_moving_days}d, "
        f"high_value=${config.high_value_threshold}"
    )

    # Process each file
    analyzer = HardwareInventoryAnalyzer(config)

    for filepath in args.files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {filepath}")
        logger.info(f"{'='*60}")

        report = analyzer.analyze(filepath)

        # Generate reports
        generator = ReportGenerator(report)
        generator.print_console_summary()

        # Generate HTML
        output_path = args.output
        if len(args.files) > 1:
            output_path = args.output.parent / f"{filepath.stem}_{args.output.name}"

        generator.generate_html(output_path)
        print(f"\n📄 HTML report saved to: {output_path}")

    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
