#!/usr/bin/env python3
"""
================================================================================
SHOPIFY INVENTORY HEALTH ANALYZER
================================================================================

A comprehensive inventory analysis tool for Shopify merchants that identifies
inventory primitives (issue patterns) and generates detailed reports with
full accounting impact context.

USAGE:
------
    # Basic usage with default settings:
    python shopify_inventory_analyzer.py inventory_export.csv

    # With custom configuration:
    python shopify_inventory_analyzer.py inventory_export.csv --config config.yaml

    # Multiple files:
    python shopify_inventory_analyzer.py file1.csv file2.xlsx --output report.html

    # Specify output format:
    python shopify_inventory_analyzer.py inventory.csv --format html --output my_report.html

REQUIRED COLUMNS:
-----------------
The input file must contain these columns (case-insensitive matching):
    - SKU (or "Variant SKU")
    - Title (or "Variant Title", "Product Title")
    - Quantity (or "Inventory Quantity", "QOH", "In Stock Qty.", "Qty")
    - Cost (or "Cost per item", "Unit Cost")

OPTIONAL BUT RECOMMENDED COLUMNS:
---------------------------------
    - Last Sold Date (or "Last Sale Date", "Date Last Sold", "Sold")
    - Retail Price (or "Price", "Retail", "Selling Price")
    - Vendor (or "Supplier")
    - Product Type (or "Category")
    - Created At (or "Date Created")

OUTPUT:
-------
    - Console summary with key metrics
    - Detailed HTML report with:
        1. Executive Summary
        2. Primitive Definitions and Impacts
        3. Detailed Findings (grouped by primitive)
        4. Recommendations
        5. Validation Log

CONFIGURATION:
--------------
Create a config.yaml file to customize thresholds:

    thresholds:
        dead_stock_days: 365          # Days without sale = dead stock
        low_stock_threshold: 5        # Units below this = low stock
        overstock_months_supply: 12   # Months of supply = overstock
        slow_moving_days: 180         # Days without sale = slow moving

Author: Profit Sentinel
Version: 2.0.0
License: MIT
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import re
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
    """
    NEGATIVE_INVENTORY = "negative_inventory"
    DEAD_STOCK = "dead_stock"
    SLOW_MOVING = "slow_moving"
    OVERSTOCK = "overstock"
    LOW_STOCK = "low_stock"
    STOCKOUT = "stockout"
    MISSING_COST = "missing_cost"
    ZERO_COST_WITH_QUANTITY = "zero_cost_with_quantity"
    NEGATIVE_COST = "negative_cost"
    MISSING_LAST_SOLD = "missing_last_sold"
    DUPLICATE_SKU = "duplicate_sku"
    MISSING_SKU = "missing_sku"
    COST_EXCEEDS_PRICE = "cost_exceeds_price"
    HIGH_VALUE_DEAD_STOCK = "high_value_dead_stock"


# Column name aliases for flexible matching
COLUMN_ALIASES: dict[str, list[str]] = {
    "sku": ["sku", "variant sku", "item sku", "product sku", "item_sku", "variantsku"],
    "title": ["title", "variant title", "product title", "name", "description",
              "item name", "product name", "variant_title"],
    "quantity": ["quantity", "inventory quantity", "qoh", "qty", "in stock qty.",
                 "in stock qty", "stock", "units", "qty on hand", "quantity_on_hand",
                 "inventory_quantity"],
    "cost": ["cost", "cost per item", "unit cost", "cogs", "item cost",
             "cost_per_item", "unitcost"],
    "price": ["price", "retail", "retail price", "selling price", "sell price",
              "variant price", "retail_price"],
    "last_sold": ["last sold", "last sold date", "last sale date", "date last sold",
                  "sold", "last_sold_date", "lastsold", "last sale"],
    "vendor": ["vendor", "supplier", "manufacturer", "brand"],
    "product_type": ["product type", "type", "category", "product_type"],
    "created_at": ["created at", "date created", "created", "created_at", "date added"],
    "sub_total": ["sub total", "sub_total", "subtotal", "total value", "inventory value"],
    "margin": ["margin", "profit margin", "profit margin %", "margin %", "gross margin"],
}

# Required columns for validation
REQUIRED_COLUMNS = ["sku", "title", "quantity"]
RECOMMENDED_COLUMNS = ["cost", "last_sold", "price"]


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
        short_description="Inventory quantity on hand is less than zero for a variant.",
        full_definition="""
Negative inventory occurs when the recorded quantity on hand for a product variant
falls below zero. This represents a fundamental data integrity issue where the system
shows you owe inventory that doesn't exist, or where sales have been recorded without
corresponding stock being available.
        """.strip(),
        how_it_occurs="""
• Overselling: Sales recorded before inventory receipt is logged
• Sync delays: POS or marketplace sales not syncing with Shopify in real-time
• Manual adjustments: Incorrect inventory counts entered during cycle counts
• Returns processing: Returns not properly added back to inventory
• Multi-location issues: Stock transfers not completed correctly
• Shopify default settings: Many Shopify plans allow negative inventory by default
        """.strip(),
        cogs_impact="""
COGS is UNDERSTATED because when inventory goes negative, no inventory value is
relieved upon sale (or a negative value is recorded). The cost expense that should
be recognized doesn't exist because there's no inventory cost to relieve. This means
your reported cost of goods sold is artificially low.
        """.strip(),
        gross_profit_impact="""
Gross profit is OVERSTATED because revenue is recorded from the sale, but the
corresponding COGS expense is not fully recognized. The profit & loss statement
shows profit that doesn't actually exist when you account for the true cost of
goods that will eventually need to be purchased.
        """.strip(),
        gross_margin_impact="""
Gross margins appear ARTIFICIALLY INFLATED. If you're showing a 50% gross margin
but have significant negative inventory, your true margin could be substantially
lower (potentially 35-45%) once the missing COGS is properly accounted for. This
misleads stakeholders about actual business profitability.
        """.strip(),
        additional_risks=[
            "Tax underreporting: Lower COGS means higher taxable income is reported",
            "Fulfillment failures: Orders placed for items that don't exist",
            "Customer experience damage: Backorders, cancellations, delays",
            "Cash flow surprises: Future purchases to cover negative stock hit cash unexpectedly",
            "Audit red flags: Material discrepancies between physical and book inventory",
            "Valuation issues: Business valuation based on incorrect financials",
        ],
        recommendations=[
            "Enable 'Stop selling when out of stock' in Shopify settings immediately",
            "Conduct physical inventory count to reconcile actual vs. system quantities",
            "Review and fix all negative SKUs with adjustment entries and proper documentation",
            "Implement real-time inventory sync with all sales channels",
            "Set up low-stock alerts to prevent overselling",
            "Review POS and marketplace integration sync frequencies",
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
period (typically 12+ months). These items are sitting on shelves, tying up capital,
and likely losing value through obsolescence, spoilage, or market changes. Dead stock
represents a failed inventory investment.
        """.strip(),
        how_it_occurs="""
• Over-ordering: Purchasing too much of slow-selling items
• Trend changes: Fashion, technology, or market preferences shifted
• Seasonality misjudgment: Ordering seasonal items that didn't sell
• Poor demand forecasting: Inaccurate sales predictions
• Discontinued products: Items no longer promoted or relevant
• Quality or defect issues: Items customers don't want
• Pricing mistakes: Items priced too high for market
        """.strip(),
        cogs_impact="""
COGS is UNDERSTATED in the current period because these items haven't sold. However,
under proper accounting (GAAP/IFRS), dead stock should be written down to Net Realizable
Value (NRV). If NRV is lower than cost, an inventory write-down expense should be
recognized, increasing COGS or being recorded as a separate expense line.
        """.strip(),
        gross_profit_impact="""
Gross profit is OVERSTATED if dead stock hasn't been written down. The balance sheet
shows inventory at historical cost when its true recoverable value is much lower.
When you eventually sell at clearance prices or write off the inventory, you'll
recognize a large loss that should have been accrued gradually.
        """.strip(),
        gross_margin_impact="""
Gross margins are MISLEADING because they're calculated against sales that exclude
these problematic items. Your overall inventory investment's return is actually lower
than margins suggest. True portfolio margin should account for dead stock losses.
        """.strip(),
        additional_risks=[
            "Working capital locked: Cash tied up in unsellable inventory",
            "Storage costs: Ongoing warehousing expenses for items generating no revenue",
            "Opportunity cost: Space and capital not available for faster-moving items",
            "Obsolescence risk: Items may become completely worthless over time",
            "Insurance costs: Paying to insure items with little recovery value",
            "Year-end write-offs: Large, unexpected write-downs at fiscal year-end",
            "Investor/lender concerns: Overstated inventory affects financial ratios",
        ],
        recommendations=[
            "Implement aging analysis and review inventory age monthly",
            "Create a clearance strategy: bundle deals, flash sales, liquidation channels",
            "Write down inventory to NRV per accounting standards",
            "Donate for tax deduction if liquidation isn't viable",
            "Prevent future dead stock with better demand forecasting",
            "Set reorder points based on actual sales velocity, not gut feel",
            "Consider consignment or dropship for uncertain items",
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
        """.strip(),
        how_it_occurs="""
• Demand decline: Product lifecycle entering decline phase
• Competition: Better alternatives available in market
• Pricing issues: Price may not match perceived value
• Visibility problems: Items not being merchandised effectively
• Seasonal off-cycle: Items awaiting their selling season
• Market saturation: Customer base already owns the product
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
profitable ones. While current margin looks fine, Return on Inventory Investment (ROII)
is being dragged down by capital tied up in slow movers.
        """.strip(),
        additional_risks=[
            "Trending toward dead stock and complete write-off",
            "Tying up Open-to-Buy budget that could fund better items",
            "Risk of fashion/technology obsolescence increasing daily",
            "Carrying costs (storage, insurance, handling) eroding margins",
            "Cash flow constraint: money locked in slow inventory",
        ],
        recommendations=[
            "Flag for promotional activity before they become dead stock",
            "Analyze why items slowed - price, placement, or demand issue?",
            "Consider bundle deals with faster-moving items",
            "Move to clearance section or outlet channel",
            "Reduce reorder quantities or pause reorders entirely",
            "Track inventory age as a KPI in weekly reviews",
        ],
        color="#f59e0b",  # Amber
    ),

    PrimitiveType.OVERSTOCK: PrimitiveDefinition(
        name="Overstock / Excess Inventory",
        primitive_type=PrimitiveType.OVERSTOCK,
        severity=Severity.MEDIUM,
        short_description="Quantity on hand exceeds 12 months of sales velocity.",
        full_definition="""
Overstock occurs when you're holding significantly more inventory than needed to meet
demand. Measured as months of supply (Quantity on Hand ÷ Monthly Sales Velocity),
having more than 12 months of supply typically indicates over-purchasing. Even
fast-selling items can be overstocked if you've bought too much.
        """.strip(),
        how_it_occurs="""
• Bulk discount temptation: Buying more to get better unit pricing
• Demand overestimation: Expecting higher sales than materialized
• Vendor minimum orders: MOQs forcing larger purchases
• Fear of stockouts: Over-ordering as safety stock
• Discontinued notice: Panic buying when vendors discontinue items
• Failed promotions: Stocking up for sales that didn't perform
        """.strip(),
        cogs_impact="""
COGS per unit may actually be LOWER due to bulk pricing, but total COGS investment
is much higher than necessary. The carrying cost of excess inventory (typically
20-30% of value annually) erodes any unit cost savings achieved through bulk buying.
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
            "Obsolescence risk: Market may change before you sell through",
            "Markdown risk: May need to discount to move excess inventory",
            "Opportunity cost: Capital not available for new/better items",
            "Warehouse capacity: Space consumed that could hold better inventory",
        ],
        recommendations=[
            "Calculate true carrying cost per SKU (aim for 25% annual estimate)",
            "Set maximum months-of-supply thresholds by product category",
            "Negotiate smaller, more frequent deliveries with vendors",
            "Use promotional velocity to move excess before it ages",
            "Implement Open-to-Buy planning to control investment",
            "Review reorder points and quantities quarterly",
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
needed to maintain sales while awaiting replenishment. These items are at risk of
stocking out, which directly impacts revenue and customer satisfaction.
        """.strip(),
        how_it_occurs="""
• Demand spike: Unexpected increase in sales velocity
• Reorder delays: Late placing of purchase orders
• Supplier issues: Extended lead times or backorders from vendors
• Forecasting errors: Underestimating demand
• Cash constraints: Delaying orders due to cash flow
• System failures: Reorder alerts not triggering or being ignored
        """.strip(),
        cogs_impact="""
COGS itself isn't directly affected, but the OPPORTUNITY COST of lost sales is
significant. Every stockout represents revenue that would have generated COGS
expense and gross profit that now goes to competitors instead.
        """.strip(),
        gross_profit_impact="""
Gross profit is LOST for every sale that can't be fulfilled. If you stock out of
a $50 item with 50% margin for 2 weeks and normally sell 10/week, that's $500
in lost gross profit - often more than the cost of carrying safety stock.
        """.strip(),
        gross_margin_impact="""
While margin percentage isn't affected, TOTAL margin dollars are reduced. Additionally,
rush shipping to expedite restock orders often comes at premium cost, further eroding
margins on subsequent sales.
        """.strip(),
        additional_risks=[
            "Lost sales: Direct revenue loss when customers can't buy",
            "Customer defection: Buyers may switch to competitors permanently",
            "SEO/marketplace ranking damage: Out-of-stock hurts search placement",
            "Rush order costs: Expedited shipping to restock is expensive",
            "Bundle/upsell disruption: Missing one item breaks combo sales",
            "Reputation damage: Backorders frustrate customers",
        ],
        recommendations=[
            "Calculate safety stock levels based on demand variability",
            "Set up automated reorder point alerts in Shopify",
            "Review and adjust reorder quantities for high-velocity items",
            "Establish backup suppliers for critical items",
            "Monitor sales velocity trends weekly for fast movers",
            "Consider holding more safety stock for items with long lead times",
        ],
        color="#eab308",  # Yellow
    ),

    PrimitiveType.STOCKOUT: PrimitiveDefinition(
        name="Stockout (Zero Inventory)",
        primitive_type=PrimitiveType.STOCKOUT,
        severity=Severity.CRITICAL,
        short_description="Quantity on hand is zero - item cannot be sold.",
        full_definition="""
A stockout is the complete depletion of inventory for a product variant. The item
shows zero quantity on hand and cannot be sold (unless overselling is enabled,
which creates negative inventory). This represents immediate, ongoing revenue loss.
        """.strip(),
        how_it_occurs="""
• Reorder failure: Purchase order not placed in time
• Supply chain disruption: Vendor stockouts or shipping delays
• Demand surge: Viral moment or unexpected popularity
• Poor forecasting: Inadequate demand planning
• Working capital limits: Inability to fund inventory purchases
• Administrative errors: Orders lost or not processed
        """.strip(),
        cogs_impact="""
No COGS is recorded because no sales occur. While this sounds neutral, it means
the investment in building the sales opportunity (marketing, traffic acquisition)
generates no return. The capital that would have been deployed as COGS sits idle.
        """.strip(),
        gross_profit_impact="""
ZERO gross profit generated during stockout period for this SKU. If the item
normally contributes $1,000/week in gross profit and is out for 3 weeks, that's
$3,000 in lost profit - likely never recovered as customers buy elsewhere.
        """.strip(),
        gross_margin_impact="""
Overall store margin may actually appear better (bad items not dragging average),
but this is misleading. You're not achieving gross margin on zero sales. The right
metric is gross margin DOLLARS, which are severely impacted.
        """.strip(),
        additional_risks=[
            "Immediate revenue loss with each passing day",
            "Customer acquisition cost wasted: traffic comes but can't convert",
            "Search ranking drops: marketplaces penalize out-of-stock items",
            "Advertising waste: paid ads sending traffic to unavailable items",
            "Lost lifetime value: customers who leave may not return",
            "Competitor gain: every stockout is a gift to competition",
        ],
        recommendations=[
            "Implement emergency reorder process for critical SKUs",
            "Pause advertising spend on out-of-stock items immediately",
            "Enable back-in-stock notifications to capture demand",
            "Display 'Notify Me' instead of hiding out-of-stock items",
            "Expedite shipping from suppliers or transfer from other locations",
            "Post-mortem: analyze why stockout occurred to prevent recurrence",
        ],
        color="#dc2626",  # Red
    ),

    PrimitiveType.MISSING_COST: PrimitiveDefinition(
        name="Missing Cost Data",
        primitive_type=PrimitiveType.MISSING_COST,
        severity=Severity.CRITICAL,
        short_description="Cost per item field is blank - COGS cannot be calculated.",
        full_definition="""
Missing cost data means the 'Cost per item' field in Shopify is blank or null for
a product variant. Without this data, you cannot calculate Cost of Goods Sold,
gross profit, gross margin, inventory valuation, or any meaningful financial metric
for this item.
        """.strip(),
        how_it_occurs="""
• Incomplete product setup: Cost field skipped during product creation
• Dropship/consignment: Cost not entered for third-party fulfilled items
• Data migration issues: Cost field not mapped during platform migration
• Manual errors: Field accidentally cleared or never populated
• Free/promotional items: Assumption that free items don't need cost
• Bundles/kits: Component costs not rolled up to bundle level
        """.strip(),
        cogs_impact="""
COGS is COMPLETELY UNKNOWN for these items. Your P&L's cost of goods sold line is
understated by the entire cost value of sales from these items. If they represent
10% of sales, your reported COGS could be understated by 10%.
        """.strip(),
        gross_profit_impact="""
Gross profit is OVERSTATED by the missing COGS amount. You're recording revenue
but no cost, making profit appear higher than reality. This is a material
misstatement if the affected items represent significant sales volume.
        """.strip(),
        gross_margin_impact="""
Gross margin percentage is UNRELIABLE and likely overstated. You're dividing
profit by revenue, but the profit number excludes costs. Actual margin could be
significantly lower than reported - potentially by 5-15+ percentage points.
        """.strip(),
        additional_risks=[
            "Financial statements are materially misstated",
            "Tax liability may be incorrect (underreported COGS = overpaid taxes)",
            "Inventory valuation is wrong: balance sheet overstated or understated",
            "Pricing decisions made without knowing true costs",
            "Profitability analysis completely unreliable for these SKUs",
            "Audit findings: material weakness in internal controls",
            "Business valuation: investors can't trust your margins",
        ],
        recommendations=[
            "URGENT: Populate cost data for all items immediately",
            "Review supplier invoices to determine correct costs",
            "For bundles, calculate component costs and roll up",
            "Set up process requiring cost entry before item goes live",
            "Implement data validation rules in Shopify or ERP",
            "Run monthly report on items missing cost data",
            "Restate historical financials once costs are determined",
        ],
        color="#dc2626",  # Red
    ),

    PrimitiveType.ZERO_COST_WITH_QUANTITY: PrimitiveDefinition(
        name="Zero Cost with Positive Quantity",
        primitive_type=PrimitiveType.ZERO_COST_WITH_QUANTITY,
        severity=Severity.HIGH,
        short_description="Cost is $0.00 but quantity on hand is greater than zero.",
        full_definition="""
This primitive flags items where the cost field is explicitly zero (not blank) but
inventory exists. While some items legitimately have zero cost (free samples,
promotional items), most inventory has acquisition costs. A $0 cost is suspicious
and usually indicates a data entry error.
        """.strip(),
        how_it_occurs="""
• Data entry error: Accidentally entering 0 instead of actual cost
• Placeholder value: Using 0 as a placeholder that was never updated
• Free samples/promos: Legitimately free items (but should be documented)
• Received as gift: Donated inventory without recorded cost
• Bundled items: Components separated without cost allocation
• System default: Some systems default to 0 if not specified
        """.strip(),
        cogs_impact="""
COGS is recorded as $0 when these items sell, even though you likely paid for them.
This understates COGS. If you sell $10,000 of zero-cost items that actually cost
$6,000, your COGS is understated by $6,000.
        """.strip(),
        gross_profit_impact="""
Gross profit is overstated by the missing cost amount. Sales of these items appear
100% profitable when they're not. This inflates overall gross profit figures and
misleads decision-making.
        """.strip(),
        gross_margin_impact="""
These items show 100% gross margin (Revenue - $0 cost = 100%), which is almost
never accurate for purchased inventory. This artificially inflates blended margin
calculations.
        """.strip(),
        additional_risks=[
            "Tax implications: overstating profit, underpaying taxes correctly",
            "Pricing errors: thinking you can price lower because 'no cost'",
            "Inventory valuation: balance sheet shows $0 value for real assets",
            "Audit issues: zero-cost items are a red flag for auditors",
            "Insurance: if inventory is lost, claim would be for $0",
        ],
        recommendations=[
            "Review each zero-cost item - is $0 accurate or an error?",
            "Update costs from supplier invoices or purchase orders",
            "For truly free items, add a note/tag documenting why",
            "Implement validation: warn/block $0 cost entries",
            "Check if these were data migration errors",
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
should have a negative cost - suppliers don't pay you to take their products
(with rare exceptions like hazardous material disposal rebates).
        """.strip(),
        how_it_occurs="""
• Data entry error: Accidentally typing a minus sign
• Copy/paste error: Pasting from a spreadsheet with sign errors
• Return adjustments: Incorrectly adjusting costs for returns
• System bugs: Integration or calculation errors
• Rebate mishandling: Applying vendor rebates incorrectly to cost
        """.strip(),
        cogs_impact="""
COGS is calculated as a NEGATIVE value or credit when these items sell. Instead
of recognizing expense, you're recognizing negative expense (income). This makes
COGS significantly understated and may even show as negative COGS line in reports.
        """.strip(),
        gross_profit_impact="""
Gross profit is massively overstated. If an item has -$10 cost and sells for $50,
it appears to generate $60 profit ($50 - (-$10)), when real profit should be less.
This completely distorts profitability analysis.
        """.strip(),
        gross_margin_impact="""
Margin calculations become nonsensical - you can show margins over 100%, which
is a clear indicator of data problems. Any financial analysis using these figures
is unreliable.
        """.strip(),
        additional_risks=[
            "Complete corruption of financial reporting",
            "Tax reporting errors of significant magnitude",
            "Impossible to make rational pricing decisions",
            "Audit would identify this as material misstatement",
            "Inventory valuation shows negative asset values",
        ],
        recommendations=[
            "IMMEDIATE FIX: Correct all negative costs to accurate values",
            "Investigate how negative costs entered the system",
            "Add validation rules to reject negative cost entries",
            "Review integrations and imports for sign-handling bugs",
            "Restate any financial reports that included this data",
        ],
        color="#dc2626",  # Red
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
• Never sold: New products that haven't had first sale yet
• Data not tracked: Legacy system didn't capture sale dates
• Migration issue: Date field not mapped during platform switch
• Manual sales: Offline/manual sales not recorded in Shopify
• B2B/wholesale: Sales through other channels not reflected
        """.strip(),
        cogs_impact="""
COGS isn't directly affected, but you can't analyze COGS by product age or velocity.
Aged inventory should be evaluated for write-downs, but without dates, you can't
identify which items are aged.
        """.strip(),
        gross_profit_impact="""
Cannot determine if gross profit is being generated by fresh, healthy inventory
or old, problematic stock. Profit analysis by product lifecycle stage is impossible.
        """.strip(),
        gross_margin_impact="""
Can't correlate margin performance with inventory age. Understanding if old inventory
is sold at lower margins (clearance) vs. new inventory at full margin is blocked.
        """.strip(),
        additional_risks=[
            "Can't identify dead stock without sale date history",
            "Inventory aging reports are incomplete",
            "May be unknowingly holding obsolete inventory",
            "Demand forecasting lacks historical sales timing data",
            "Can't properly value inventory using market-based methods",
        ],
        recommendations=[
            "For new items: no action needed, date will populate on first sale",
            "For older items: research if sales occurred elsewhere",
            "Flag these items for physical review - are they sellable?",
            "Ensure all sales channels sync last-sold date to Shopify",
            "Consider using 'created date' as proxy if no sales history",
        ],
        color="#6b7280",  # Gray
    ),

    PrimitiveType.DUPLICATE_SKU: PrimitiveDefinition(
        name="Duplicate SKU",
        primitive_type=PrimitiveType.DUPLICATE_SKU,
        severity=Severity.HIGH,
        short_description="Same SKU appears on multiple product variants.",
        full_definition="""
Duplicate SKUs mean the same stock-keeping unit identifier is assigned to more than
one product variant. SKUs should be unique identifiers - duplicates cause inventory
tracking chaos, fulfillment errors, and make financial reporting unreliable.
        """.strip(),
        how_it_occurs="""
• Manual entry errors: Same SKU typed for multiple products
• Product duplication: Copying products without updating SKU
• Vendor SKU reuse: Using vendor SKUs that aren't unique
• Migration errors: Duplicates created during data imports
• Multi-variant confusion: Not understanding SKU-per-variant requirement
• Merged catalogs: Combining inventories without SKU rationalization
        """.strip(),
        cogs_impact="""
COGS may be doubled, split incorrectly, or untrackable. When a SKU sells, which
variant's cost is used? The system may pick arbitrarily, or apply cost to wrong
item, corrupting COGS accuracy.
        """.strip(),
        gross_profit_impact="""
Gross profit per SKU is unreliable because sales may be attributed to wrong variant.
If SKU "ABC123" exists twice at different costs, profit calculations are meaningless.
        """.strip(),
        gross_margin_impact="""
Cannot trust margin analysis at the SKU level. Reports may show certain SKUs as
highly profitable or unprofitable based on arbitrary cost assignment.
        """.strip(),
        additional_risks=[
            "Fulfillment errors: Wrong item shipped to customer",
            "Inventory counts impossible: Which one has which quantity?",
            "Reorder confusion: Which variant needs restocking?",
            "Returns processing: Which variant receives the return?",
            "Channel listing errors: Marketplaces may reject duplicate SKUs",
            "Lost traceability: Can't track product through supply chain",
        ],
        recommendations=[
            "Implement unique SKU policy: one SKU = one variant only",
            "Resolve duplicates: rename one or consolidate products",
            "Use SKU naming convention with variant codes (e.g., PROD-RED-LG)",
            "Add validation in Shopify or middleware to reject duplicate SKUs",
            "Audit SKU list quarterly for duplicates",
            "Consider using barcodes (UPC/EAN) as additional unique identifier",
        ],
        color="#f97316",  # Orange
    ),

    PrimitiveType.MISSING_SKU: PrimitiveDefinition(
        name="Missing SKU",
        primitive_type=PrimitiveType.MISSING_SKU,
        severity=Severity.HIGH,
        short_description="Product variant has no SKU assigned.",
        full_definition="""
Items without a SKU lack a unique identifier for tracking. While Shopify can function
without SKUs, their absence makes inventory management, warehouse operations,
reporting, and integrations significantly harder or impossible.
        """.strip(),
        how_it_occurs="""
• Incomplete product setup: SKU field skipped during creation
• Digital products: Assumption that non-physical goods don't need SKUs
• Services: Service items created without identifiers
• Quick add: Rushed product creation without proper data
• Migration: SKU field not included in import mapping
        """.strip(),
        cogs_impact="""
COGS tracking is compromised because you can't reliably track cost by product
identifier. Reports by SKU will exclude these items entirely.
        """.strip(),
        gross_profit_impact="""
Cannot analyze gross profit by SKU. These items become a black hole in profitability
analysis - you know they exist but can't report on them properly.
        """.strip(),
        gross_margin_impact="""
Margin analysis by SKU is incomplete. Overall margins include these items' revenue
but SKU-level reports exclude them.
        """.strip(),
        additional_risks=[
            "Warehouse picking errors: No identifier for pickers to scan",
            "Inventory counts: Can't scan or count without identifier",
            "Integration failures: Most systems require SKU for sync",
            "Marketplace listing issues: Many platforms require SKUs",
            "Reporting gaps: SKU-based reports exclude these items",
            "Fulfillment by Amazon: FBA requires SKUs for all items",
        ],
        recommendations=[
            "Assign unique SKU to every variant immediately",
            "Develop SKU naming convention for organization",
            "Make SKU a required field in product creation workflow",
            "Run weekly report on items missing SKUs",
            "For digital/service items: still use SKU for tracking",
        ],
        color="#f97316",  # Orange
    ),

    PrimitiveType.COST_EXCEEDS_PRICE: PrimitiveDefinition(
        name="Cost Exceeds Price (Negative Margin)",
        primitive_type=PrimitiveType.COST_EXCEEDS_PRICE,
        severity=Severity.CRITICAL,
        short_description="Cost per item is higher than the selling price.",
        full_definition="""
When an item's cost exceeds its selling price, every sale generates a loss. While
this may be intentional for loss leaders or clearance, it usually indicates a
pricing error, cost data error, or items that need repricing urgently.
        """.strip(),
        how_it_occurs="""
• Pricing errors: Price set incorrectly, especially after cost increases
• Cost increases: Supplier raised prices but retail not updated
• Clearance pricing: Intentional markdown below cost to clear inventory
• Loss leaders: Strategic items priced to drive traffic (should be tagged)
• Data errors: Cost or price field has wrong value
• Currency confusion: Cost in one currency, price in another
        """.strip(),
        cogs_impact="""
COGS exceeds revenue for these items. While technically correct (cost is recorded
as incurred), the relationship is inverted - every sale increases your losses
rather than generating profit.
        """.strip(),
        gross_profit_impact="""
Gross profit is NEGATIVE for these items. They're pulling down your overall gross
profit with every sale. A single high-volume negative-margin item can significantly
impact total gross profit.
        """.strip(),
        gross_margin_impact="""
Gross margin is negative (below 0%). A product costing $15 selling for $10 has
-50% margin. This drags down blended margin and indicates either error or
intentional (but documented) strategy.
        """.strip(),
        additional_risks=[
            "Losing money on every sale of this item",
            "If high volume, losses compound quickly",
            "May indicate vendor costs increased without your knowledge",
            "Competitors may be using similar pricing, indicating market shift",
            "Customer expectations set at unprofitable price point",
        ],
        recommendations=[
            "IMMEDIATE: Review if pricing or cost data is in error",
            "If intentional (clearance): document and set end date",
            "If error: correct price or cost immediately",
            "Consider discontinuing items that can't be sold profitably",
            "Set up alerts for any new items where cost > price",
            "Review margin reports weekly to catch issues early",
        ],
        color="#dc2626",  # Red
    ),

    PrimitiveType.HIGH_VALUE_DEAD_STOCK: PrimitiveDefinition(
        name="High-Value Dead Stock",
        primitive_type=PrimitiveType.HIGH_VALUE_DEAD_STOCK,
        severity=Severity.CRITICAL,
        short_description="Dead stock items with significant inventory value at risk.",
        full_definition="""
High-value dead stock are items that meet dead stock criteria (no sales 12+ months)
AND have significant inventory value on hand (typically $1,000+). These represent
major working capital at risk of write-off and deserve urgent attention.
        """.strip(),
        how_it_occurs="""
• Large initial orders of items that didn't sell as expected
• Bulk discount purchases that turned out to be poor decisions
• Seasonal items that didn't clear and were held over
• Product line discontinuation without liquidation
• Trend items that went out of style while in inventory
        """.strip(),
        cogs_impact="""
Significant COGS will need to be recognized as inventory write-down or write-off.
A $50,000 high-value dead stock position may require a $50,000 expense recognition
when properly accounted for.
        """.strip(),
        gross_profit_impact="""
Large negative impact to gross profit when write-down is recorded. Accountants may
require immediate write-down to Net Realizable Value, reducing current period profit.
        """.strip(),
        gross_margin_impact="""
When write-down hits P&L, gross margin drops significantly. A $50,000 write-down
on a business with $200,000 monthly gross profit could reduce margin by 25% for
that period.
        """.strip(),
        additional_risks=[
            "Major cash tied up with no return pathway",
            "Balance sheet overstated until write-down recorded",
            "Audit finding likely if not properly reserved",
            "Opportunity cost: this capital could be working elsewhere",
            "Storage costs continue to accumulate",
            "May need to pay for disposal if unsellable",
        ],
        recommendations=[
            "Prioritize for immediate liquidation strategy",
            "Get quotes from liquidators - any recovery better than zero",
            "Consider donation for tax deduction",
            "Record inventory write-down per accounting standards",
            "Analyze how this happened to prevent recurrence",
            "Set dollar-value triggers for senior management review",
        ],
        color="#dc2626",  # Red
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
    title: str
    quantity: float
    cost: float | None
    price: float | None
    last_sold: datetime | None
    inventory_value: float | None
    vendor: str | None
    product_type: str | None
    issue_context: str
    raw_data: dict[str, Any]

    @property
    def margin_pct(self) -> float | None:
        """Calculate margin percentage if cost and price are available."""
        if self.cost is not None and self.price is not None and self.price > 0:
            return ((self.price - self.cost) / self.price) * 100
        return None

    @property
    def last_sold_display(self) -> str:
        """Format last sold date for display."""
        if self.last_sold is None:
            return "Never / Unknown"
        return self.last_sold.strftime("%Y-%m-%d")

    @property
    def days_since_sold(self) -> int | None:
        """Calculate days since last sold."""
        if self.last_sold is None:
            return None
        return (datetime.now() - self.last_sold).days


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
        return sum(item.quantity for item in self.items)

    @property
    def total_value(self) -> float:
        """Sum of all inventory values (where calculable)."""
        return sum(item.inventory_value or 0 for item in self.items)

    @property
    def items_with_value(self) -> int:
        """Count of items with calculable inventory value."""
        return sum(1 for item in self.items if item.inventory_value is not None)


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    filename: str
    analysis_timestamp: datetime
    total_rows: int
    total_variants: int
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
        return [f for f in self.findings.values()
                if f.definition.severity == Severity.CRITICAL and f.total_items > 0]

    @property
    def high_findings(self) -> list[PrimitiveFinding]:
        """Findings with high severity."""
        return [f for f in self.findings.values()
                if f.definition.severity == Severity.HIGH and f.total_items > 0]


@dataclass
class AnalysisConfig:
    """Configuration for analysis thresholds and behavior."""
    # Time-based thresholds (in days)
    dead_stock_days: int = 365
    slow_moving_days: int = 180

    # Stock level thresholds
    low_stock_threshold: int = 5
    overstock_months_supply: int = 12

    # Value thresholds
    high_value_threshold: float = 1000.0

    # Analysis behavior
    include_zero_quantity: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "AnalysisConfig":
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
                overstock_months_supply=thresholds.get("overstock_months_supply", 12),
                high_value_threshold=thresholds.get("high_value_threshold", 1000.0),
                include_zero_quantity=thresholds.get("include_zero_quantity", True),
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

        logger.info(f"Validation complete: {'PASSED' if self.report.overall_passed else 'FAILED'}")
        return self.report

    def _check_file_exists(self) -> bool:
        """Check if file exists."""
        exists = self.filepath.exists()
        self.report.add_check(ValidationResult(
            check_name="File Exists",
            passed=exists,
            message=f"File found at {self.filepath}" if exists else f"File not found: {self.filepath}",
            severity=Severity.CRITICAL,
        ))
        return exists

    def _check_file_type(self) -> bool:
        """Check if file is CSV or Excel."""
        suffix = self.filepath.suffix.lower()
        valid_types = {".csv", ".xlsx", ".xls"}
        is_valid = suffix in valid_types

        self.report.add_check(ValidationResult(
            check_name="File Type",
            passed=is_valid,
            message=f"Valid file type: {suffix}" if is_valid else f"Invalid file type: {suffix}. Expected: {valid_types}",
            severity=Severity.CRITICAL,
        ))
        return is_valid

    def _load_file(self) -> bool:
        """Attempt to load the file into a DataFrame."""
        try:
            suffix = self.filepath.suffix.lower()
            if suffix == ".csv":
                self.df = pd.read_csv(self.filepath)
            else:
                self.df = pd.read_excel(self.filepath)

            self.report.add_check(ValidationResult(
                check_name="File Readable",
                passed=True,
                message=f"Successfully loaded {len(self.df)} rows",
            ))
            return True
        except Exception as e:
            self.report.add_check(ValidationResult(
                check_name="File Readable",
                passed=False,
                message=f"Failed to read file: {str(e)}",
                severity=Severity.CRITICAL,
            ))
            return False

    def _check_not_empty(self) -> None:
        """Check that file has data rows."""
        if self.df is None:
            return

        has_data = len(self.df) > 0
        self.report.add_check(ValidationResult(
            check_name="Has Data Rows",
            passed=has_data,
            message=f"File contains {len(self.df)} data rows" if has_data else "File is empty (no data rows)",
            severity=Severity.CRITICAL,
        ))

    def _check_no_duplicate_headers(self) -> None:
        """Check for duplicate column headers."""
        if self.df is None:
            return

        columns = self.df.columns.tolist()
        duplicates = [col for col in columns if columns.count(col) > 1]
        duplicates = list(set(duplicates))  # Unique duplicates

        no_duplicates = len(duplicates) == 0
        self.report.add_check(ValidationResult(
            check_name="No Duplicate Headers",
            passed=no_duplicates,
            message="All column headers are unique" if no_duplicates else f"Duplicate headers found: {duplicates}",
            details=duplicates if duplicates else [],
            severity=Severity.HIGH,
        ))

    def _check_no_blank_rows(self) -> None:
        """Check for completely blank rows."""
        if self.df is None:
            return

        blank_rows = self.df[self.df.isna().all(axis=1)]
        blank_indices = blank_rows.index.tolist()

        no_blanks = len(blank_indices) == 0
        self.report.add_check(ValidationResult(
            check_name="No Blank Rows",
            passed=no_blanks,
            message="No completely blank rows found" if no_blanks else f"Found {len(blank_indices)} blank rows",
            details=[f"Row {i+2}" for i in blank_indices[:10]],  # +2 for header and 0-index
            severity=Severity.MEDIUM,
        ))

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
        self.report.add_check(ValidationResult(
            check_name="Required Columns Present",
            passed=all_present,
            message="All required columns found" if all_present else f"Missing required columns: {missing}",
            details=found + [f"MISSING: {m}" for m in missing],
            severity=Severity.CRITICAL,
        ))

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
        self.report.add_check(ValidationResult(
            check_name="Recommended Columns Present",
            passed=all_present,
            message="All recommended columns found" if all_present else f"Missing recommended columns: {missing}",
            details=found + [f"MISSING: {m}" for m in missing],
            severity=Severity.LOW if all_present else Severity.MEDIUM,
        ))

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
                float(val)
            except (ValueError, TypeError):
                non_numeric.append(f"Row {idx+2}: '{val}'")

        all_numeric = len(non_numeric) == 0
        self.report.add_check(ValidationResult(
            check_name="Quantity Data Type",
            passed=all_numeric,
            message="All quantity values are numeric" if all_numeric else f"Found {len(non_numeric)} non-numeric quantity values",
            details=non_numeric[:10],
            severity=Severity.HIGH,
        ))

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
            # Handle currency symbols
            if isinstance(val, str):
                val = val.replace("$", "").replace(",", "").strip()
            try:
                float(val)
            except (ValueError, TypeError):
                non_numeric.append(f"Row {idx+2}: '{val}'")

        all_numeric = len(non_numeric) == 0
        self.report.add_check(ValidationResult(
            check_name="Cost Data Type",
            passed=all_numeric,
            message="All cost values are numeric or blank" if all_numeric else f"Found {len(non_numeric)} non-numeric cost values",
            details=non_numeric[:10],
            severity=Severity.HIGH,
        ))

    def _check_date_formatting(self) -> None:
        """Check that date columns can be parsed."""
        if self.df is None:
            return

        date_col = self.column_mapping.get("last_sold")
        if not date_col:
            self.report.add_check(ValidationResult(
                check_name="Date Format",
                passed=True,
                message="No date column found (will use created_at or mark as unknown)",
                severity=Severity.INFO,
            ))
            return

        unparseable = []
        for idx, val in self.df[date_col].items():
            if pd.isna(val) or val == "" or val == 0:
                continue
            try:
                pd.to_datetime(val)
            except (ValueError, TypeError):
                unparseable.append(f"Row {idx+2}: '{val}'")

        all_parseable = len(unparseable) == 0
        self.report.add_check(ValidationResult(
            check_name="Date Format",
            passed=all_parseable,
            message="All date values can be parsed" if all_parseable else f"Found {len(unparseable)} unparseable dates",
            details=unparseable[:10],
            severity=Severity.MEDIUM,
        ))


# =============================================================================
# INVENTORY ANALYZER
# =============================================================================

class InventoryAnalyzer:
    """Analyzes inventory data for various primitive issues."""

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
            # Return report with validation failure
            return AnalysisReport(
                filename=filepath.name,
                analysis_timestamp=datetime.now(),
                total_rows=0,
                total_variants=0,
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

        # Run all primitive checks
        findings = self._run_all_checks(df)

        report = AnalysisReport(
            filename=filepath.name,
            analysis_timestamp=datetime.now(),
            total_rows=len(df),
            total_variants=df[self._get_col("sku")].nunique() if self._get_col("sku") else len(df),
            validation_report=validation_report,
            findings=findings,
            total_inventory_value=total_value,
            total_inventory_quantity=total_qty,
        )

        logger.info(f"Analysis complete. Found {report.total_issues} issues across {len([f for f in findings.values() if f.total_items > 0])} primitives.")
        return report

    def _get_col(self, target: str) -> str | None:
        """Get actual column name from mapping."""
        return self.column_mapping.get(target)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame for analysis."""
        df = df.copy()

        # Convert quantity to numeric
        qty_col = self._get_col("quantity")
        if qty_col:
            df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0)

        # Convert cost to numeric (handle currency symbols)
        cost_col = self._get_col("cost")
        if cost_col:
            if df[cost_col].dtype == object:
                df[cost_col] = df[cost_col].astype(str).str.replace(r"[$,]", "", regex=True)
            df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

        # Convert price to numeric
        price_col = self._get_col("price")
        if price_col:
            if df[price_col].dtype == object:
                df[price_col] = df[price_col].astype(str).str.replace(r"[$,]", "", regex=True)
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

        # Convert dates
        date_col = self._get_col("last_sold")
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Calculate inventory value
        if qty_col and cost_col:
            df["_inventory_value"] = df[qty_col] * df[cost_col].fillna(0)
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
        qty_col = self._get_col("quantity")
        if qty_col:
            return df[qty_col].sum()
        return 0

    def _create_flagged_item(self, row: pd.Series, context: str) -> FlaggedItem:
        """Create a FlaggedItem from a DataFrame row."""
        qty_col = self._get_col("quantity")
        cost_col = self._get_col("cost")
        price_col = self._get_col("price")
        date_col = self._get_col("last_sold")

        quantity = row[qty_col] if qty_col else 0
        cost = row[cost_col] if cost_col and pd.notna(row.get(cost_col)) else None
        price = row[price_col] if price_col and pd.notna(row.get(price_col)) else None
        last_sold = row[date_col] if date_col and pd.notna(row.get(date_col)) else None

        inventory_value = None
        if quantity and cost:
            inventory_value = abs(quantity) * cost

        return FlaggedItem(
            sku=str(row.get(self._get_col("sku"), "N/A")),
            title=str(row.get(self._get_col("title"), "N/A")),
            quantity=quantity,
            cost=cost,
            price=price,
            last_sold=last_sold,
            inventory_value=inventory_value,
            vendor=row.get(self._get_col("vendor")),
            product_type=row.get(self._get_col("product_type")),
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
        findings[PrimitiveType.STOCKOUT] = self._check_stockout(df)
        findings[PrimitiveType.COST_EXCEEDS_PRICE] = self._check_cost_exceeds_price(df)

        # High severity checks
        findings[PrimitiveType.DEAD_STOCK] = self._check_dead_stock(df)
        findings[PrimitiveType.HIGH_VALUE_DEAD_STOCK] = self._check_high_value_dead_stock(df)
        findings[PrimitiveType.ZERO_COST_WITH_QUANTITY] = self._check_zero_cost_with_quantity(df)
        findings[PrimitiveType.DUPLICATE_SKU] = self._check_duplicate_sku(df)
        findings[PrimitiveType.MISSING_SKU] = self._check_missing_sku(df)
        findings[PrimitiveType.LOW_STOCK] = self._check_low_stock(df)

        # Medium severity checks
        findings[PrimitiveType.SLOW_MOVING] = self._check_slow_moving(df)
        findings[PrimitiveType.OVERSTOCK] = self._check_overstock(df)
        findings[PrimitiveType.MISSING_LAST_SOLD] = self._check_missing_last_sold(df)

        return findings

    def _check_negative_inventory(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for negative inventory quantities."""
        finding = PrimitiveFinding(PrimitiveType.NEGATIVE_INVENTORY)
        qty_col = self._get_col("quantity")

        if not qty_col:
            return finding

        negative = df[df[qty_col] < 0]
        for _, row in negative.iterrows():
            context = f"Quantity is {row[qty_col]:.0f} units (NEGATIVE)"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Negative Inventory: {finding.total_items} items found")
        return finding

    def _check_missing_cost(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items missing cost data."""
        finding = PrimitiveFinding(PrimitiveType.MISSING_COST)
        cost_col = self._get_col("cost")
        qty_col = self._get_col("quantity")

        if not cost_col:
            # If no cost column at all, flag everything with quantity
            if qty_col:
                has_qty = df[df[qty_col] > 0]
                for _, row in has_qty.iterrows():
                    context = "No cost column in file - cannot calculate COGS"
                    finding.items.append(self._create_flagged_item(row, context))
            return finding

        # Find rows with null/blank cost but positive quantity
        if qty_col:
            missing = df[(df[cost_col].isna()) & (df[qty_col] > 0)]
        else:
            missing = df[df[cost_col].isna()]

        for _, row in missing.iterrows():
            context = "Cost field is blank - COGS cannot be calculated"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Missing Cost: {finding.total_items} items found")
        return finding

    def _check_negative_cost(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for negative cost values."""
        finding = PrimitiveFinding(PrimitiveType.NEGATIVE_COST)
        cost_col = self._get_col("cost")

        if not cost_col:
            return finding

        negative = df[df[cost_col] < 0]
        for _, row in negative.iterrows():
            context = f"Cost is ${row[cost_col]:.2f} (NEGATIVE) - impossible value"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Negative Cost: {finding.total_items} items found")
        return finding

    def _check_stockout(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items with zero inventory (stockouts)."""
        finding = PrimitiveFinding(PrimitiveType.STOCKOUT)
        qty_col = self._get_col("quantity")
        date_col = self._get_col("last_sold")

        if not qty_col:
            return finding

        # Zero quantity items that have sold before (active products now out of stock)
        stockout = df[df[qty_col] == 0]

        if date_col:
            # Only flag items that have sold before (have a last_sold date)
            stockout = stockout[stockout[date_col].notna()]

        for _, row in stockout.iterrows():
            context = "STOCKOUT - Zero inventory, cannot fulfill orders"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Stockout: {finding.total_items} items found")
        return finding

    def _check_cost_exceeds_price(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items where cost exceeds selling price."""
        finding = PrimitiveFinding(PrimitiveType.COST_EXCEEDS_PRICE)
        cost_col = self._get_col("cost")
        price_col = self._get_col("price")
        qty_col = self._get_col("quantity")

        if not cost_col or not price_col:
            return finding

        # Find items where cost > price and both are valid
        mask = (df[cost_col] > df[price_col]) & (df[cost_col] > 0) & (df[price_col] > 0)
        if qty_col:
            mask = mask & (df[qty_col] > 0)  # Only items with inventory

        negative_margin = df[mask]

        for _, row in negative_margin.iterrows():
            cost = row[cost_col]
            price = row[price_col]
            loss_per_unit = cost - price
            context = f"Cost ${cost:.2f} > Price ${price:.2f} = ${loss_per_unit:.2f} LOSS per sale"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Cost Exceeds Price: {finding.total_items} items found")
        return finding

    def _check_dead_stock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for dead stock (no sales in 12+ months)."""
        finding = PrimitiveFinding(PrimitiveType.DEAD_STOCK)
        date_col = self._get_col("last_sold")
        qty_col = self._get_col("quantity")

        if not date_col or not qty_col:
            return finding

        cutoff = datetime.now() - timedelta(days=self.config.dead_stock_days)

        # Items with inventory that haven't sold since cutoff
        dead = df[
            (df[qty_col] > 0) &
            (df[date_col] < cutoff)
        ]

        for _, row in dead.iterrows():
            days_ago = (datetime.now() - row[date_col]).days
            context = f"No sales in {days_ago} days ({days_ago // 30} months)"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Dead Stock: {finding.total_items} items found")
        return finding

    def _check_high_value_dead_stock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for high-value dead stock (dead stock with value > threshold)."""
        finding = PrimitiveFinding(PrimitiveType.HIGH_VALUE_DEAD_STOCK)
        date_col = self._get_col("last_sold")
        qty_col = self._get_col("quantity")
        cost_col = self._get_col("cost")

        if not date_col or not qty_col or not cost_col:
            return finding

        cutoff = datetime.now() - timedelta(days=self.config.dead_stock_days)

        # High value dead stock
        mask = (
            (df[qty_col] > 0) &
            (df[date_col] < cutoff) &
            (df["_inventory_value"] >= self.config.high_value_threshold)
        )
        high_value_dead = df[mask]

        for _, row in high_value_dead.iterrows():
            days_ago = (datetime.now() - row[date_col]).days
            value = row["_inventory_value"]
            context = f"${value:,.2f} value at risk - No sales in {days_ago} days"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"High-Value Dead Stock: {finding.total_items} items found")
        return finding

    def _check_zero_cost_with_quantity(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items with zero cost but positive quantity."""
        finding = PrimitiveFinding(PrimitiveType.ZERO_COST_WITH_QUANTITY)
        cost_col = self._get_col("cost")
        qty_col = self._get_col("quantity")

        if not cost_col or not qty_col:
            return finding

        # Explicitly zero cost (not null) with positive quantity
        zero_cost = df[(df[cost_col] == 0) & (df[qty_col] > 0)]

        for _, row in zero_cost.iterrows():
            context = f"Cost is $0.00 but has {row[qty_col]:.0f} units in stock"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Zero Cost with Quantity: {finding.total_items} items found")
        return finding

    def _check_duplicate_sku(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for duplicate SKUs."""
        finding = PrimitiveFinding(PrimitiveType.DUPLICATE_SKU)
        sku_col = self._get_col("sku")

        if not sku_col:
            return finding

        # Find duplicate SKUs (excluding blank/null)
        sku_counts = df[df[sku_col].notna() & (df[sku_col] != "")][sku_col].value_counts()
        duplicates = sku_counts[sku_counts > 1].index.tolist()

        for sku in duplicates:
            dup_rows = df[df[sku_col] == sku]
            for _, row in dup_rows.iterrows():
                context = f"SKU '{sku}' appears {sku_counts[sku]} times"
                finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Duplicate SKU: {finding.total_items} items found")
        return finding

    def _check_missing_sku(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items without SKU."""
        finding = PrimitiveFinding(PrimitiveType.MISSING_SKU)
        sku_col = self._get_col("sku")
        qty_col = self._get_col("quantity")

        if not sku_col:
            return finding

        # Missing or blank SKU
        mask = df[sku_col].isna() | (df[sku_col] == "")
        if qty_col:
            mask = mask & (df[qty_col] != 0)  # Only items with quantity

        missing = df[mask]

        for _, row in missing.iterrows():
            context = "No SKU assigned - cannot track inventory properly"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Missing SKU: {finding.total_items} items found")
        return finding

    def _check_low_stock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for low stock items."""
        finding = PrimitiveFinding(PrimitiveType.LOW_STOCK)
        qty_col = self._get_col("quantity")
        date_col = self._get_col("last_sold")

        if not qty_col:
            return finding

        # Low stock: positive quantity below threshold, and has sold recently
        mask = (df[qty_col] > 0) & (df[qty_col] <= self.config.low_stock_threshold)

        if date_col:
            # Only flag items that have sold in the last 90 days (active items)
            recent_cutoff = datetime.now() - timedelta(days=90)
            mask = mask & (df[date_col] >= recent_cutoff)

        low = df[mask]

        for _, row in low.iterrows():
            context = f"Only {row[qty_col]:.0f} units remaining - reorder soon"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Low Stock: {finding.total_items} items found")
        return finding

    def _check_slow_moving(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for slow-moving inventory."""
        finding = PrimitiveFinding(PrimitiveType.SLOW_MOVING)
        date_col = self._get_col("last_sold")
        qty_col = self._get_col("quantity")

        if not date_col or not qty_col:
            return finding

        dead_cutoff = datetime.now() - timedelta(days=self.config.dead_stock_days)
        slow_cutoff = datetime.now() - timedelta(days=self.config.slow_moving_days)

        # Slow moving: between slow_moving_days and dead_stock_days
        slow = df[
            (df[qty_col] > 0) &
            (df[date_col] >= dead_cutoff) &
            (df[date_col] < slow_cutoff)
        ]

        for _, row in slow.iterrows():
            days_ago = (datetime.now() - row[date_col]).days
            context = f"No sales in {days_ago} days - trending toward dead stock"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Slow Moving: {finding.total_items} items found")
        return finding

    def _check_overstock(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for overstocked items."""
        finding = PrimitiveFinding(PrimitiveType.OVERSTOCK)
        qty_col = self._get_col("quantity")
        date_col = self._get_col("last_sold")

        if not qty_col:
            return finding

        # For overstock, we need sales velocity data
        # Without detailed sales data, use a simpler heuristic: very high quantity
        # with no recent sales indicates potential overstock
        high_qty_threshold = 100  # Items with >100 units

        if date_col:
            recent_cutoff = datetime.now() - timedelta(days=90)
            # High quantity items that haven't sold recently
            overstock = df[
                (df[qty_col] > high_qty_threshold) &
                ((df[date_col] < recent_cutoff) | df[date_col].isna())
            ]
        else:
            overstock = df[df[qty_col] > high_qty_threshold]

        for _, row in overstock.iterrows():
            context = f"{row[qty_col]:.0f} units on hand - potential overstock"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Overstock: {finding.total_items} items found")
        return finding

    def _check_missing_last_sold(self, df: pd.DataFrame) -> PrimitiveFinding:
        """Check for items missing last sold date."""
        finding = PrimitiveFinding(PrimitiveType.MISSING_LAST_SOLD)
        date_col = self._get_col("last_sold")
        qty_col = self._get_col("quantity")

        if not date_col:
            # No date column means all items missing date
            return finding

        # Missing date with positive quantity
        mask = df[date_col].isna()
        if qty_col:
            mask = mask & (df[qty_col] > 0)

        missing = df[mask]

        for _, row in missing.iterrows():
            context = "No last sold date - cannot assess inventory age"
            finding.items.append(self._create_flagged_item(row, context))

        logger.info(f"Missing Last Sold: {finding.total_items} items found")
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
        print("SHOPIFY INVENTORY HEALTH ANALYSIS REPORT")
        print("=" * 70)
        print(f"\nFile: {r.filename}")
        print(f"Analyzed: {r.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Variants: {r.total_variants:,}")
        print(f"Total Quantity: {r.total_inventory_quantity:,.0f} units")
        if r.total_inventory_value:
            print(f"Total Value: ${r.total_inventory_value:,.2f}")

        print("\n" + "-" * 70)
        print("VALIDATION STATUS")
        print("-" * 70)
        print(f"Overall: {'✓ PASSED' if r.validation_report.overall_passed else '✗ FAILED'}")
        for check in r.validation_report.checks:
            status = "✓" if check.passed else "✗"
            print(f"  {status} {check.check_name}: {check.message}")

        print("\n" + "-" * 70)
        print("FINDINGS SUMMARY")
        print("-" * 70)

        # Group by severity
        critical = [f for f in r.findings.values() if f.definition.severity == Severity.CRITICAL and f.total_items > 0]
        high = [f for f in r.findings.values() if f.definition.severity == Severity.HIGH and f.total_items > 0]
        medium = [f for f in r.findings.values() if f.definition.severity == Severity.MEDIUM and f.total_items > 0]

        print(f"\n🔴 CRITICAL Issues: {sum(f.total_items for f in critical)}")
        for f in critical:
            print(f"   • {f.definition.name}: {f.total_items} items (${f.total_value:,.2f} value)")

        print(f"\n🟠 HIGH Issues: {sum(f.total_items for f in high)}")
        for f in high:
            print(f"   • {f.definition.name}: {f.total_items} items (${f.total_value:,.2f} value)")

        print(f"\n🟡 MEDIUM Issues: {sum(f.total_items for f in medium)}")
        for f in medium:
            print(f"   • {f.definition.name}: {f.total_items} items (${f.total_value:,.2f} value)")

        print(f"\n{'=' * 70}")
        print(f"TOTAL ISSUES: {r.total_issues}")
        print(f"TOTAL VALUE AT RISK: ${r.total_value_at_risk:,.2f}")
        print("=" * 70 + "\n")

    def generate_html(self, output_path: Path) -> None:
        """Generate comprehensive HTML report."""
        r = self.report

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inventory Health Report - {r.filename}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1, h2, h3, h4 {{
            color: #f1f5f9;
        }}
        .header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 16px;
            margin-bottom: 30px;
            border: 1px solid #334155;
        }}
        .header h1 {{
            font-size: 2.5em;
            color: #10b981;
            margin-bottom: 10px;
        }}
        .header .subtitle {{
            color: #94a3b8;
            font-size: 1.2em;
        }}
        .meta-info {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .meta-item {{
            background: #1e293b;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
        }}
        .meta-item .label {{
            color: #94a3b8;
            font-size: 0.9em;
        }}
        .meta-item .value {{
            color: #f1f5f9;
            font-size: 1.4em;
            font-weight: bold;
        }}

        /* Executive Summary */
        .executive-summary {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #334155;
        }}
        .executive-summary h2 {{
            color: #10b981;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: #0f172a;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .summary-card.critical {{
            border-color: #dc2626;
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.1) 0%, #0f172a 100%);
        }}
        .summary-card.high {{
            border-color: #f97316;
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.1) 0%, #0f172a 100%);
        }}
        .summary-card.medium {{
            border-color: #eab308;
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.1) 0%, #0f172a 100%);
        }}
        .summary-card .number {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .summary-card.critical .number {{ color: #dc2626; }}
        .summary-card.high .number {{ color: #f97316; }}
        .summary-card.medium .number {{ color: #eab308; }}
        .summary-card .label {{
            color: #94a3b8;
            margin-top: 5px;
        }}

        /* Severity breakdown */
        .severity-breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .severity-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #0f172a;
            border-radius: 8px;
            border-left: 4px solid;
        }}
        .severity-item.critical {{ border-color: #dc2626; }}
        .severity-item.high {{ border-color: #f97316; }}
        .severity-item.medium {{ border-color: #eab308; }}

        /* Section styling */
        .section {{
            background: #1e293b;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid #334155;
        }}
        .section h2 {{
            color: #f1f5f9;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #334155;
        }}

        /* Primitive Definition Block */
        .primitive-definition {{
            background: #0f172a;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #334155;
            border-left: 4px solid;
        }}
        .primitive-definition h3 {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .severity-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .severity-badge.critical {{ background: #dc262633; color: #dc2626; }}
        .severity-badge.high {{ background: #f9731633; color: #f97316; }}
        .severity-badge.medium {{ background: #eab30833; color: #eab308; }}
        .severity-badge.low {{ background: #22c55e33; color: #22c55e; }}

        .definition-content {{
            display: grid;
            gap: 20px;
        }}
        .definition-block {{
            background: #1e293b;
            border-radius: 8px;
            padding: 15px;
        }}
        .definition-block h4 {{
            color: #10b981;
            margin-bottom: 10px;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .definition-block p, .definition-block ul {{
            color: #cbd5e1;
            font-size: 0.95em;
        }}
        .definition-block ul {{
            margin-left: 20px;
        }}
        .definition-block li {{
            margin-bottom: 5px;
        }}
        .impact-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }}
        .impact-card {{
            background: #0f172a;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #334155;
        }}
        .impact-card h5 {{
            color: #f97316;
            margin-bottom: 8px;
            font-size: 0.9em;
        }}
        .impact-card p {{
            color: #94a3b8;
            font-size: 0.9em;
        }}

        /* Findings Table */
        .findings-section {{
            margin-top: 30px;
        }}
        .findings-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .findings-header h3 {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .findings-stats {{
            display: flex;
            gap: 20px;
        }}
        .findings-stats .stat {{
            background: #0f172a;
            padding: 8px 15px;
            border-radius: 8px;
        }}
        .findings-stats .stat-label {{
            color: #94a3b8;
            font-size: 0.8em;
        }}
        .findings-stats .stat-value {{
            color: #f1f5f9;
            font-weight: bold;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{
            background: #0f172a;
            color: #10b981;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background: #1e293b55;
        }}
        .sku {{
            font-family: 'Monaco', 'Consolas', monospace;
            color: #38bdf8;
        }}
        .quantity {{
            font-weight: bold;
        }}
        .quantity.negative {{
            color: #dc2626;
        }}
        .quantity.low {{
            color: #eab308;
        }}
        .quantity.normal {{
            color: #22c55e;
        }}
        .value {{
            color: #10b981;
            font-weight: bold;
        }}
        .context {{
            color: #fbbf24;
            font-size: 0.9em;
            font-style: italic;
        }}
        .date {{
            color: #94a3b8;
        }}
        .date.never {{
            color: #6b7280;
            font-style: italic;
        }}

        /* Validation Log */
        .validation-log {{
            background: #0f172a;
            border-radius: 8px;
            padding: 20px;
        }}
        .validation-item {{
            display: flex;
            align-items: flex-start;
            gap: 10px;
            padding: 10px 0;
            border-bottom: 1px solid #334155;
        }}
        .validation-item:last-child {{
            border-bottom: none;
        }}
        .validation-icon {{
            font-size: 1.2em;
        }}
        .validation-icon.pass {{
            color: #22c55e;
        }}
        .validation-icon.fail {{
            color: #dc2626;
        }}
        .validation-details {{
            color: #94a3b8;
            font-size: 0.9em;
            margin-top: 5px;
        }}

        /* Recommendations */
        .recommendations {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, #0f172a 100%);
            border: 1px solid #10b98155;
        }}
        .recommendations h2 {{
            color: #10b981;
        }}
        .recommendation-list {{
            list-style: none;
        }}
        .recommendation-list li {{
            padding: 15px;
            margin-bottom: 10px;
            background: #1e293b;
            border-radius: 8px;
            border-left: 3px solid #10b981;
        }}
        .recommendation-list li::before {{
            content: "→ ";
            color: #10b981;
            font-weight: bold;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: #64748b;
            font-size: 0.9em;
        }}
        .footer a {{
            color: #10b981;
            text-decoration: none;
        }}

        /* Responsive */
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            .meta-info {{
                flex-direction: column;
                align-items: center;
            }}
            table {{
                font-size: 0.85em;
            }}
            th, td {{
                padding: 8px 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 Inventory Health Analysis</h1>
            <p class="subtitle">Comprehensive Profit Leak Detection Report</p>
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
                    <div class="label">Total Variants</div>
                    <div class="value">{r.total_variants:,}</div>
                </div>
                <div class="meta-item">
                    <div class="label">Total Quantity</div>
                    <div class="value">{r.total_inventory_quantity:,.0f}</div>
                </div>
                {f'<div class="meta-item"><div class="label">Total Value</div><div class="value">${r.total_inventory_value:,.2f}</div></div>' if r.total_inventory_value else ''}
            </div>
        </div>

        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2>📋 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card critical">
                    <div class="number">{sum(f.total_items for f in r.critical_findings)}</div>
                    <div class="label">Critical Issues</div>
                </div>
                <div class="summary-card high">
                    <div class="number">{sum(f.total_items for f in r.high_findings)}</div>
                    <div class="label">High Severity Issues</div>
                </div>
                <div class="summary-card">
                    <div class="number" style="color: #10b981;">{r.total_issues}</div>
                    <div class="label">Total Issues Found</div>
                </div>
                <div class="summary-card">
                    <div class="number" style="color: #f97316;">${r.total_value_at_risk:,.0f}</div>
                    <div class="label">Value at Risk</div>
                </div>
            </div>

            <h3 style="margin-bottom: 15px; color: #94a3b8;">Issues by Category</h3>
            <div class="severity-breakdown">
                {self._generate_severity_breakdown()}
            </div>
        </div>

        <!-- Primitive Definitions and Findings -->
        <div class="section">
            <h2>🔍 Detailed Findings by Issue Type</h2>
            {self._generate_findings_sections()}
        </div>

        <!-- Recommendations -->
        <div class="section recommendations">
            <h2>✅ Recommended Actions</h2>
            {self._generate_recommendations()}
        </div>

        <!-- Validation Log -->
        <div class="section">
            <h2>📝 Validation Log</h2>
            <div class="validation-log">
                {self._generate_validation_log()}
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Generated by <strong>Profit Sentinel</strong> Inventory Analyzer</p>
            <p>For questions or support, visit <a href="https://profitsentinel.com">profitsentinel.com</a></p>
        </div>
    </div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report generated: {output_path}")

    def _generate_severity_breakdown(self) -> str:
        """Generate HTML for severity breakdown."""
        html = ""
        for finding in sorted(self.report.findings.values(),
                             key=lambda f: (f.definition.severity.value, -f.total_items)):
            if finding.total_items == 0:
                continue
            severity = finding.definition.severity.value
            html += f"""
            <div class="severity-item {severity}">
                <span>{finding.definition.name}</span>
                <span><strong>{finding.total_items}</strong> items (${finding.total_value:,.2f})</span>
            </div>
            """
        return html if html else "<p>No issues found! 🎉</p>"

    def _generate_findings_sections(self) -> str:
        """Generate HTML for all findings sections with definitions."""
        html = ""

        # Sort by severity then by count
        sorted_findings = sorted(
            self.report.findings.values(),
            key=lambda f: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}[f.definition.severity.value],
                -f.total_items
            )
        )

        for finding in sorted_findings:
            if finding.total_items == 0:
                continue

            defn = finding.definition
            severity = defn.severity.value
            color = defn.color

            html += f"""
            <div class="primitive-definition" style="border-left-color: {color};">
                <h3>
                    {defn.name}
                    <span class="severity-badge {severity}">{severity}</span>
                </h3>

                <div class="definition-content">
                    <!-- Definition -->
                    <div class="definition-block">
                        <h4>📖 Definition</h4>
                        <p>{defn.full_definition}</p>
                    </div>

                    <!-- How It Occurs -->
                    <div class="definition-block">
                        <h4>⚙️ How This Occurs in Shopify</h4>
                        <p style="white-space: pre-line;">{defn.how_it_occurs}</p>
                    </div>

                    <!-- Financial Impact -->
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

                    <!-- Additional Risks -->
                    <div class="definition-block">
                        <h4>⚠️ Additional Business Risks</h4>
                        <ul>
                            {"".join(f"<li>{risk}</li>" for risk in defn.additional_risks)}
                        </ul>
                    </div>
                </div>

                <!-- Findings Table -->
                <div class="findings-section">
                    <div class="findings-header">
                        <h3 style="color: {color};">
                            Affected Items
                        </h3>
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
                                <th>Title</th>
                                <th>QOH</th>
                                <th>Cost</th>
                                <th>Value</th>
                                <th>Last Sold</th>
                                <th>Issue Context</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_items_rows(finding.items)}
                        </tbody>
                    </table>
                </div>
            </div>
            """

        return html if html else "<p>No issues found in this analysis. Your inventory looks healthy! 🎉</p>"

    def _generate_items_rows(self, items: list[FlaggedItem], max_items: int = 100) -> str:
        """Generate table rows for flagged items."""
        html = ""
        for item in items[:max_items]:
            qty_class = "negative" if item.quantity < 0 else "low" if item.quantity <= 5 else "normal"
            date_class = "never" if item.last_sold is None else ""

            html += f"""
            <tr>
                <td class="sku">{item.sku}</td>
                <td>{item.title[:50]}{'...' if len(item.title) > 50 else ''}</td>
                <td class="quantity {qty_class}">{item.quantity:,.0f}</td>
                <td>{f'${item.cost:.2f}' if item.cost is not None else 'N/A'}</td>
                <td class="value">{f'${item.inventory_value:,.2f}' if item.inventory_value else 'N/A'}</td>
                <td class="date {date_class}">{item.last_sold_display}</td>
                <td class="context">{item.issue_context}</td>
            </tr>
            """

        if len(items) > max_items:
            html += f"""
            <tr>
                <td colspan="7" style="text-align: center; color: #94a3b8; padding: 20px;">
                    ... and {len(items) - max_items} more items (showing first {max_items})
                </td>
            </tr>
            """

        return html

    def _generate_recommendations(self) -> str:
        """Generate prioritized recommendations."""
        html = "<ul class='recommendation-list'>"

        # Collect unique recommendations from findings with issues
        recommendations_by_priority = {
            Severity.CRITICAL: [],
            Severity.HIGH: [],
            Severity.MEDIUM: [],
        }

        for finding in self.report.findings.values():
            if finding.total_items > 0:
                for rec in finding.definition.recommendations:
                    if rec not in recommendations_by_priority[finding.definition.severity]:
                        recommendations_by_priority[finding.definition.severity].append(rec)

        # Output in priority order
        if recommendations_by_priority[Severity.CRITICAL]:
            html += "<li style='border-left-color: #dc2626;'><strong>🔴 CRITICAL - Address Immediately:</strong><ul>"
            for rec in recommendations_by_priority[Severity.CRITICAL][:5]:
                html += f"<li>{rec}</li>"
            html += "</ul></li>"

        if recommendations_by_priority[Severity.HIGH]:
            html += "<li style='border-left-color: #f97316;'><strong>🟠 HIGH PRIORITY - This Week:</strong><ul>"
            for rec in recommendations_by_priority[Severity.HIGH][:5]:
                html += f"<li>{rec}</li>"
            html += "</ul></li>"

        if recommendations_by_priority[Severity.MEDIUM]:
            html += "<li style='border-left-color: #eab308;'><strong>🟡 MEDIUM - Schedule Soon:</strong><ul>"
            for rec in recommendations_by_priority[Severity.MEDIUM][:5]:
                html += f"<li>{rec}</li>"
            html += "</ul></li>"

        html += "</ul>"
        return html

    def _generate_validation_log(self) -> str:
        """Generate validation log HTML."""
        html = ""
        for check in self.report.validation_report.checks:
            icon_class = "pass" if check.passed else "fail"
            icon = "✓" if check.passed else "✗"

            html += f"""
            <div class="validation-item">
                <span class="validation-icon {icon_class}">{icon}</span>
                <div>
                    <strong>{check.check_name}</strong>
                    <div class="validation-details">{check.message}</div>
                    {"".join(f"<div class='validation-details'>• {d}</div>" for d in check.details[:5])}
                </div>
            </div>
            """
        return html


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the inventory analyzer."""
    parser = argparse.ArgumentParser(
        description="Shopify Inventory Health Analyzer - Detect profit leaks and inventory issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python shopify_inventory_analyzer.py inventory.csv
  python shopify_inventory_analyzer.py inventory.xlsx --output report.html
  python shopify_inventory_analyzer.py file.csv --config config.yaml

For more information, see the documentation at the top of this script.
        """
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input file(s) to analyze (CSV or Excel)"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("inventory_report.html"),
        help="Output HTML report path (default: inventory_report.html)"
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = None
    if args.config:
        config = AnalysisConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = AnalysisConfig()
        logger.info("Using default configuration")

    # Process each file
    analyzer = InventoryAnalyzer(config)

    for filepath in args.files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {filepath}")
        logger.info(f"{'='*60}")

        report = analyzer.analyze(filepath)

        # Generate reports
        generator = ReportGenerator(report)
        generator.print_console_summary()

        # Generate HTML with filename in output
        output_path = args.output
        if len(args.files) > 1:
            # Multiple files: include filename in output
            output_path = args.output.parent / f"{filepath.stem}_{args.output.name}"

        generator.generate_html(output_path)
        print(f"\n📄 HTML report saved to: {output_path}")

    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
