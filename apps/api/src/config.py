"""
Application configuration and settings.

Loads environment variables and provides typed configuration objects.
"""

import logging
import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Profit Sentinel"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False)

    # Analysis thresholds (configurable via environment)
    # Margin leak detection: flag items with margin below this threshold
    margin_leak_threshold: float = Field(default=0.25, description="Margin below this triggers leak alert")
    # Margin below average: flag items 30% below category average
    margin_below_average_factor: float = Field(default=0.7, description="Flag if margin < avg * this factor")
    # Dead stock days: items not sold in this many days
    dead_stock_days: int = Field(default=90, description="Days without sale to flag as dead stock")
    # Low stock threshold: quantity below this is flagged
    low_stock_threshold: int = Field(default=10, description="Quantity below this is low stock")
    # Overstock threshold: days of supply above this is overstock
    overstock_days_supply: int = Field(default=180, description="Days supply above this is overstock")
    # Price discrepancy threshold: % difference from MSRP to flag
    price_discrepancy_threshold: float = Field(default=0.15, description="Price diff from MSRP to flag")
    # Shrinkage threshold: variance % to flag as shrinkage
    shrinkage_variance_threshold: float = Field(default=0.02, description="Inventory variance % to flag")

    # CORS - All allowed origins for frontend requests
    # Includes production domains, Vercel previews, and local development
    cors_origins: List[str] = Field(default=[
        # Production domains
        "https://profitsentinel.com",
        "https://www.profitsentinel.com",
        # Vercel deployments (main + preview)
        "https://profit-sentinel-saas.vercel.app",
        "https://profit-sentinel.vercel.app",
        # Vercel preview URLs pattern (handled via regex in middleware if needed)
        # Local development
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ])

    # AWS S3
    s3_bucket_name: str = Field(default="profitsentinel-dev-uploads")
    aws_region: str = Field(default="us-east-1")

    # Supabase
    supabase_url: Optional[str] = Field(default=None)
    supabase_service_key: Optional[str] = Field(default=None)

    # Grok AI (X.AI) - Get your key at https://x.ai/api
    # Accepts either XAI_API_KEY or GROK_API_KEY (XAI_API_KEY preferred)
    grok_api_key: Optional[str] = Field(default=None)
    xai_api_key: Optional[str] = Field(default=None)

    @property
    def ai_api_key(self) -> Optional[str]:
        """
        Get the AI API key (XAI or Grok).

        Prefers XAI_API_KEY over GROK_API_KEY for consistency with xAI SDK.
        Returns None if neither is set (AI features will be disabled).
        """
        return self.xai_api_key or self.grok_api_key

    @property
    def has_ai_key(self) -> bool:
        """Check if an AI API key is configured."""
        return self.ai_api_key is not None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra="allow" # Allows extra env vars 

@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()

    # Log warnings for missing optional but recommended keys
    if not settings.has_ai_key:
        logger.warning(
            "XAI_API_KEY not configured. AI-powered column mapping will use "
            "heuristic fallback. Get your API key at https://x.ai/api"
        )

    return settings


def require_ai_api_key() -> str:
    """
    Get the AI API key or raise an error if not configured.

    Use this for endpoints that require AI functionality.

    Raises:
        ValueError: If no AI API key is configured.

    Returns:
        The configured AI API key.
    """
    settings = get_settings()
    if not settings.ai_api_key:
        raise ValueError(
            "XAI_API_KEY environment variable is required for this operation. "
            "Get your API key at https://x.ai/api and add it to your .env file."
        )
    return settings.ai_api_key


# =============================================================================
# UNIVERSAL POS COLUMN MAPPING - Covers ALL Major Retail POS Systems
# Supports: Paladin, Spruce, Epicor, Square, Lightspeed, Clover, Shopify,
#           NCR Counterpoint, Microsoft Dynamics RMS, QuickBooks POS, Vend,
#           Toast, Revel, TouchBistro, and generic retail exports
# =============================================================================

STANDARD_FIELDS = {
    # -------------------------------------------------------------------------
    # SKU / Item Identifiers (Critical - Rating: 3)
    # -------------------------------------------------------------------------
    "sku": [
        # Generic
        "sku", "product_id", "item_id", "upc", "barcode", "plu", "item_number",
        "item_no", "itemno", "item#", "product_code", "prod_id", "article_code",
        # Paladin POS
        "partnumber", "altpartnumber", "mfgpartnumber", "part_number",
        # NCR Counterpoint
        "item_no", "barcod",
        # Microsoft Dynamics RMS
        "itemlookupcode", "item_lookup_code",
        # Lightspeed
        "system_id", "systemid", "custom_sku", "customsku", "manufacturer_sku",
        "product_sku", "product_handle", "internal_id", "internal_variant_id",
        # Shopify
        "handle", "variant_sku", "variantsku", "variant_barcode",
        # Square
        "token", "reference_handle", "gtin",
        # Clover
        "clover_id", "cloverid",
        # Epicor
        "ean",
        # Generic variations
        "stock_code", "stockcode", "item_code", "itemcode", "material_number",
        "part_no", "partno", "article_no", "articleno", "inventory_id",
    ],

    # -------------------------------------------------------------------------
    # Description (Rating: 2)
    # -------------------------------------------------------------------------
    "description": [
        # Generic
        "description", "desc", "item_name", "itemname", "product_name",
        "productname", "name", "item_description", "item_desc", "title",
        # Paladin
        "description1", "description2",
        # NCR Counterpoint
        "descr", "long_descr", "short_descr",
        # Microsoft Dynamics RMS
        "itemdescription", "itemsubdescription2",
        # Lightspeed eCom
        "title_short", "title_long", "description_short", "description_long",
        # Shopify
        "body_html", "body",
        # Generic variations
        "item", "product", "product_title", "item_title", "display_name",
    ],

    # -------------------------------------------------------------------------
    # Quantity on Hand (Critical - Rating: 5)
    # -------------------------------------------------------------------------
    "quantity": [
        # Generic - MOST IMPORTANT
        "qty", "quantity", "qoh", "on_hand", "onhand", "in_stock", "instock",
        "stock", "inventory", "inv_qty", "stock_qty", "stock_level",
        "current_stock", "available", "available_qty", "units_on_hand",
        "quantity_on_hand", "qty_on_hand", "in_stock_qty", "instockqty",
        # With dots/periods (common in exports)
        "qty.", "in stock qty.", "quantity on hand",
        # Paladin
        "stockonhand", "stock_on_hand",
        # NCR Counterpoint
        "qty_on_hnd", "qtyonhnd", "qty_avail", "qtyavail", "net_qty", "netqty",
        # Lightspeed
        "quantity_on_hand", "variant_inventory_qty", "variantinventoryqty",
        # Shopify
        "on_hand_current", "on_hand_new", "variant inventory qty",
        # Square
        "new_quantity", "newquantity",
        # Epicor
        "qty_oh", "qtyoh",
        # Clover
        "quantity",
        # Generic retail
        "bal", "balance", "stock_balance", "current_qty", "currentqty",
        "physical_qty", "physicalqty", "book_qty", "bookqty", "system_qty",
    ],

    # -------------------------------------------------------------------------
    # Quantity Difference / Variance (Critical - Rating: 5)
    # -------------------------------------------------------------------------
    "qty_difference": [
        "qty_difference", "qtydifference", "qty. difference", "difference",
        "variance", "qty_variance", "qtyvariance", "inventory_variance",
        "stock_variance", "count_variance", "adjustment", "qty_adjustment",
        "shrinkage", "shrink", "loss", "qty_loss", "discrepancy",
        "audit_variance", "auditvariance", "over_short", "overshort",
    ],

    # -------------------------------------------------------------------------
    # Cost (Critical - Rating: 5)
    # -------------------------------------------------------------------------
    "cost": [
        # Generic - MOST IMPORTANT
        "cost", "cogs", "cost_price", "costprice", "unit_cost", "unitcost",
        "avg_cost", "avgcost", "average_cost", "averagecost", "standard_cost",
        "standardcost", "item_cost", "itemcost", "product_cost", "productcost",
        # Paladin
        "unitcost", "mktcost", "market_cost", "marketcost",
        # NCR Counterpoint
        "lst_cost", "lstcost", "last_cost", "lastcost",
        # Lightspeed
        "default_cost", "defaultcost", "vendor_cost", "vendorcost",
        "per_item_supply_price", "supply_price", "supplyprice", "price_cost",
        # Shopify
        "cost_per_item", "costperitem",
        # Epicor
        "posting_cost", "postingcost",
        # Microsoft Dynamics RMS
        "cost",
        # Generic retail
        "landed_cost", "landedcost", "purchase_cost", "purchasecost",
        "acquisition_cost", "wholesale_cost", "wholesalecost", "buy_price",
        "buyprice", "invoice_cost", "invoicecost",
    ],

    # -------------------------------------------------------------------------
    # Retail Price / Revenue (Critical - Rating: 5)
    # -------------------------------------------------------------------------
    "revenue": [
        # Generic - MOST IMPORTANT
        "retail", "retail_price", "retailprice", "price", "sell_price",
        "sellprice", "selling_price", "sellingprice", "sale_price", "saleprice",
        "unit_price", "unitprice", "msrp", "list_price", "listprice",
        # With dots/periods
        "sug. retail", "sug retail", "suggested_retail", "suggestedretail",
        # Revenue totals
        "revenue", "total_sale", "totalsale", "ext_price", "extprice",
        "line_total", "linetotal", "amount", "gross_sales", "grosssales",
        "extended_price", "extendedprice", "total_price", "totalprice",
        # Paladin
        "netprice", "net_price", "plvl_price1", "plvlprice1", "saleprice",
        # NCR Counterpoint
        "prc_1", "prc1", "reg_prc", "regprc", "regular_price", "regularprice",
        # Microsoft Dynamics RMS
        "pricea", "priceb", "pricec", "price_a", "price_b", "price_c",
        # Lightspeed
        "default_price", "defaultprice",
        # Shopify
        "variant_price", "variantprice", "variant_compare_at_price",
        "compare_at_price", "compareatprice",
        # Square/Clover
        "price",
        # Generic retail
        "pos_price", "posprice", "current_price", "currentprice",
        "active_price", "activeprice", "regular", "reg_price", "regprice",
    ],

    # -------------------------------------------------------------------------
    # Units Sold (Critical - Rating: 5)
    # -------------------------------------------------------------------------
    "sold": [
        # Generic - MOST IMPORTANT
        "sold", "units_sold", "unitssold", "qty_sold", "qtysold",
        "quantity_sold", "quantitysold", "sales_qty", "salesqty",
        "total_sold", "totalsold", "sold_qty", "soldqty",
        # Time-based
        "sold_last_week", "sold_last_month", "sold_30_days", "sold30days",
        "sold_7_days", "sold7days", "sold_ytd", "soldytd", "ytd_sold",
        "mtd_sold", "mtdsold", "wtd_sold", "wtdsold",
        # Generic retail
        "sales_units", "salesunits", "unit_sales", "unitsales",
        "total_units_sold", "movement", "turns", "turnover",
    ],

    # -------------------------------------------------------------------------
    # Last Sale Date (Critical - Rating: 4)
    # -------------------------------------------------------------------------
    "last_sale_date": [
        # Generic
        "last_sale", "lastsale", "last_sold", "lastsold", "last_sale_date",
        "lastsaledate", "last_sold_date", "lastsolddate", "date_last_sold",
        "datelastsold", "last_transaction", "lasttransaction",
        # With dots/periods
        "last sale", "last sold",
        # NCR Counterpoint
        "lst_maint_dt", "lstmaintdt",
        # Generic retail
        "last_activity", "lastactivity", "last_movement", "lastmovement",
        "recent_sale", "recentsale", "most_recent_sale", "last_pos_date",
    ],

    # -------------------------------------------------------------------------
    # Last Purchase/Receiving Date (Rating: 4)
    # -------------------------------------------------------------------------
    "last_purchase_date": [
        "last_purchase", "lastpurchase", "last_pur", "lastpur", "last_pur.",
        "last_received", "lastreceived", "last_recv", "lastrecv",
        "last_receiving", "lastreceiving", "date_last_received",
        "datelastrecieved", "last_receipt", "lastreceipt", "po_date",
        "podate", "last_po_date", "lastpodate", "receiving_date",
        # NCR Counterpoint
        "lst_recv_dat", "lstrecvdat",
    ],

    # -------------------------------------------------------------------------
    # Vendor / Supplier (Rating: 3)
    # -------------------------------------------------------------------------
    "vendor": [
        # Generic
        "vendor", "supplier", "vendor_name", "vendorname", "supplier_name",
        "suppliername", "manufacturer", "mfg", "mfr", "brand",
        # Paladin
        "supplier_number1", "suppliernumber1",
        # NCR Counterpoint
        "item_vend_no", "itemvendno", "vend_no", "vendno",
        # Lightspeed
        "vendor_id", "vendorid",
        # Square
        "default_vendor_name", "defaultvendorname",
        # Generic retail
        "primary_vendor", "primaryvendor", "main_vendor", "mainvendor",
        "distributor", "wholesaler", "source",
    ],

    # -------------------------------------------------------------------------
    # Category / Department (Rating: 3)
    # -------------------------------------------------------------------------
    "category": [
        # Generic
        "category", "cat", "department", "dept", "dpt", "product_type",
        "producttype", "class", "group", "product_group", "productgroup",
        "item_class", "itemclass", "item_type", "itemtype",
        # With dots/periods
        "dpt.",
        # Paladin
        "deptid", "dept_id", "classid", "class_id",
        # NCR Counterpoint
        "categ_cod", "categcod", "category_code", "categorycode",
        "subcat_cod", "subcatcod", "subcategory", "sub_category",
        # Lightspeed
        "subcategory",
        # Shopify
        "type", "tags", "product_tags", "producttags",
        # Square
        "reporting_category", "reportingcategory",
        # Clover
        "labels",
        # Generic retail
        "merchandise_class", "merchandiseclass", "product_category",
        "productcategory", "item_category", "itemcategory", "division",
        "sub_department", "subdepartment", "family", "subfamily",
    ],

    # -------------------------------------------------------------------------
    # Profit Margin % (Critical - Rating: 5)
    # -------------------------------------------------------------------------
    "margin": [
        "margin", "profit_margin", "profitmargin", "margin_pct", "marginpct",
        "margin_percent", "marginpercent", "margin_%", "margin%", "gp",
        "gross_profit", "grossprofit", "gp_pct", "gppct", "gp_percent",
        "gppercent", "gp_%", "gp%", "profit_pct", "profitpct",
        "profit_percent", "profitpercent", "markup", "markup_pct",
        "markuppct", "markup_percent", "markuppercent",
    ],

    # -------------------------------------------------------------------------
    # Inventoried Quantity (Rating: 4)
    # -------------------------------------------------------------------------
    "inventoried_qty": [
        "inventoried_qty", "inventoriedqty", "inventoried", "counted_qty",
        "countedqty", "physical_count", "physicalcount", "count",
        "actual_qty", "actualqty", "actual_count", "actualcount",
        "audit_count", "auditcount", "cycle_count", "cyclecount",
        # With dots/periods
        "inventoried qty.",
    ],

    # -------------------------------------------------------------------------
    # Reorder / Safety Stock (Rating: 3)
    # -------------------------------------------------------------------------
    "reorder_point": [
        "reorder_point", "reorderpoint", "reorder_level", "reorderlevel",
        "min_qty", "minqty", "minimum_qty", "minimumqty", "safety_stock",
        "safetystock", "par_level", "parlevel", "low_stock_alert",
        "lowstockalert", "stock_alert", "stockalert", "reorder_trigger",
        "reordertrigger", "rec_order", "recorder",
        # Paladin
        "minorderqty", "min_order_qty",
        # Lightspeed
        "reorder_point", "stock_min", "stockmin", "stock_alert",
        # Square
        "stock_alert_count", "stockalertcount",
    ],

    # -------------------------------------------------------------------------
    # Status Fields (Rating: 3)
    # -------------------------------------------------------------------------
    "status": [
        "status", "item_status", "itemstatus", "product_status",
        "productstatus", "active", "is_active", "isactive", "enabled",
        "is_enabled", "isenabled", "available", "is_available", "isavailable",
        "discontinued", "is_discontinued", "isdiscontinued", "archived",
        "is_archived", "isarchived", "visible", "is_visible", "isvisible",
        # NCR Counterpoint
        "stat",
    ],

    # -------------------------------------------------------------------------
    # Transaction / Date Fields (Rating: 3)
    # -------------------------------------------------------------------------
    "date": [
        "date", "transaction_date", "transactiondate", "sale_date", "saledate",
        "timestamp", "posted_date", "posteddate", "created", "created_date",
        "createddate", "modified", "modified_date", "modifieddate",
        "changed", "last_changed", "lastchanged", "update_date", "updatedate",
    ],

    "transaction_id": [
        "transaction_id", "transactionid", "order_id", "orderid", "invoice",
        "receipt_id", "receiptid", "ticket", "ticket_id", "ticketid",
        "sale_id", "saleid", "doc_id", "docid",
    ],

    # -------------------------------------------------------------------------
    # Customer Fields (Rating: 2)
    # -------------------------------------------------------------------------
    "customer_id": [
        "customer", "customer_id", "customerid", "client_id", "clientid",
        "member_id", "memberid", "loyalty_id", "loyaltyid", "account",
        "account_id", "accountid",
    ],

    # -------------------------------------------------------------------------
    # Discount / Promotion (Rating: 2)
    # -------------------------------------------------------------------------
    "discount": [
        "discount", "promo", "coupon", "markdown", "discount_amt",
        "discountamt", "discount_amount", "discountamount", "discount_pct",
        "discountpct", "discount_percent", "discountpercent", "promo_code",
        "promocode", "coupon_code", "couponcode",
    ],

    # -------------------------------------------------------------------------
    # Tax (Rating: 2)
    # -------------------------------------------------------------------------
    "tax": [
        "tax", "sales_tax", "salestax", "vat", "tax_amount", "taxamount",
        "tax_rate", "taxrate", "tax_pct", "taxpct", "is_taxable", "istaxable",
        "taxable",
    ],

    # -------------------------------------------------------------------------
    # Return / Refund Flag (Rating: 2)
    # -------------------------------------------------------------------------
    "return_flag": [
        "return", "refund", "is_return", "isreturn", "negative_qty",
        "negativeqty", "return_flag", "returnflag", "void", "is_void",
        "isvoid", "credit", "is_credit", "iscredit",
    ],

    # -------------------------------------------------------------------------
    # Location / Store (Rating: 3)
    # -------------------------------------------------------------------------
    "location": [
        "location", "store", "store_id", "storeid", "store_name", "storename",
        "warehouse", "warehouse_id", "warehouseid", "outlet", "outlet_id",
        "outletid", "branch", "branch_id", "branchid", "site", "site_id",
        "siteid", "loc_id", "locid", "bin", "bin_location", "binlocation",
    ],

    # -------------------------------------------------------------------------
    # Package / Unit of Measure (Rating: 2)
    # -------------------------------------------------------------------------
    "package_qty": [
        "pkg_qty", "pkgqty", "package_qty", "packageqty", "pack_size",
        "packsize", "case_qty", "caseqty", "unit_of_measure", "unitofmeasure",
        "uom", "stk_unit", "stkunit", "stock_unit", "stockunit", "inner_pack",
        "innerpack", "outer_pack", "outerpack",
    ],

    # -------------------------------------------------------------------------
    # Sub Total / Extended Values (Rating: 4)
    # -------------------------------------------------------------------------
    "sub_total": [
        "sub_total", "subtotal", "total", "ext_total", "exttotal",
        "extended_total", "extendedtotal", "line_amount", "lineamount",
        "gl_val", "glval", "inventory_value", "inventoryvalue", "stock_value",
        "stockvalue", "on_hand_value", "onhandvalue",
    ],
}

# Field importance rankings for leak detection (1-5, 5 = most critical)
FIELD_IMPORTANCE = {
    "margin": 5,
    "qty_difference": 5,
    "quantity": 5,
    "cost": 5,
    "revenue": 5,
    "sold": 5,
    "inventoried_qty": 4,
    "sub_total": 4,
    "last_sale_date": 4,
    "last_purchase_date": 4,
    "sku": 3,
    "vendor": 3,
    "category": 3,
    "status": 3,
    "reorder_point": 3,
    "package_qty": 3,
    "location": 3,
    "description": 2,
    "discount": 2,
    "tax": 2,
    "return_flag": 2,
    "date": 2,
    "transaction_id": 2,
    "customer_id": 2,
}

# Supported POS Systems (for documentation/UI)
SUPPORTED_POS_SYSTEMS = [
    "Paladin POS",
    "Spruce POS",
    "Epicor Eagle",
    "Square POS",
    "Lightspeed Retail (R-Series, X-Series, S-Series)",
    "Lightspeed eCom (C-Series)",
    "Clover POS",
    "Shopify POS",
    "NCR Counterpoint",
    "Microsoft Dynamics RMS",
    "QuickBooks POS",
    "Vend POS",
    "Toast POS",
    "Revel Systems",
    "TouchBistro",
    "Generic CSV/Excel exports",
]
