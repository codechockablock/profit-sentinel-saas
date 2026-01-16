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


# Standard fields for column mapping
STANDARD_FIELDS = {
    "date": ["date", "transaction_date", "sale_date", "timestamp", "posted_date"],
    "sku": ["sku", "product_id", "item_id", "upc", "barcode", "plu"],
    "quantity": ["qty", "quantity", "units_sold", "qty_sold", "units", "on_hand"],
    "revenue": [
        "sale_price", "price", "total_sale", "revenue", "ext_price",
        "line_total", "amount", "gross_sales"
    ],
    "cost": ["cost", "cogs", "cost_price", "unit_cost", "avg_cost", "standard_cost"],
    "vendor": ["vendor", "supplier", "vendor_name", "manufacturer"],
    "category": ["category", "department", "product_type", "class", "group"],
    "transaction_id": ["transaction_id", "order_id", "invoice", "receipt_id"],
    "customer_id": ["customer", "client_id", "member_id"],
    "discount": ["discount", "promo", "coupon", "markdown"],
    "tax": ["tax", "sales_tax", "vat"],
    "return_flag": ["return", "refund", "is_return", "negative_qty"]
}
