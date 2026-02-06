"""
Application configuration and settings.

Loads environment variables and provides typed configuration objects.
"""

import logging
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings

from .utils.column_mappings import (
    get_field_aliases,
    get_field_importance,
    load_field_importance,
    load_standard_fields,
    load_supported_systems,
)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Profit Sentinel"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False)
    env: str = Field(
        default="development",
        description="Environment: development, staging, production",
    )

    # Analysis thresholds (configurable via environment)
    # Margin leak detection: flag items with margin below this threshold
    margin_leak_threshold: float = Field(
        default=0.25, description="Margin below this triggers leak alert"
    )
    # Margin below average: flag items 30% below category average
    margin_below_average_factor: float = Field(
        default=0.7, description="Flag if margin < avg * this factor"
    )
    # Dead stock days: items not sold in this many days
    dead_stock_days: int = Field(
        default=90, description="Days without sale to flag as dead stock"
    )
    # Low stock threshold: quantity below this is flagged
    low_stock_threshold: int = Field(
        default=10, description="Quantity below this is low stock"
    )
    # Overstock threshold: days of supply above this is overstock
    overstock_days_supply: int = Field(
        default=180, description="Days supply above this is overstock"
    )
    # Price discrepancy threshold: % difference from MSRP to flag
    price_discrepancy_threshold: float = Field(
        default=0.15, description="Price diff from MSRP to flag"
    )
    # Shrinkage threshold: variance % to flag as shrinkage
    shrinkage_variance_threshold: float = Field(
        default=0.02, description="Inventory variance % to flag"
    )

    # VSA Evidence Grounding (v4.0.0)
    # Enables evidence-based cause attribution for 0% hallucination
    use_vsa_grounding: bool = Field(
        default=True,
        description="Enable VSA-grounded evidence retrieval for cause diagnosis",
    )
    # Hot path confidence threshold - below this routes to LLM cold path
    vsa_confidence_threshold: float = Field(
        default=0.6, description="Confidence below this routes to cold path"
    )
    # Ambiguity threshold - above this routes to LLM cold path
    vsa_ambiguity_threshold: float = Field(
        default=0.5, description="Ambiguity above this routes to cold path"
    )
    # Whether to include cause diagnosis in analysis results
    include_cause_diagnosis: bool = Field(
        default=True, description="Include VSA cause diagnosis in results"
    )

    # New Engine (M1-M6 Migration)
    # When True, uses Rust pipeline via sentinel-server subprocess
    # When False, uses legacy Python heuristic engine
    use_new_engine: bool = Field(
        default=False,
        description="Enable Rust analysis engine (M1 migration). "
        "Falls back to legacy engine on error.",
    )
    sentinel_bin: str = Field(
        default="sentinel-server",
        description="Path to sentinel-server binary. "
        "Set to absolute path in Docker (e.g., /app/sentinel-server).",
    )
    sentinel_default_store: str = Field(
        default="default",
        description="Default store_id for single-store analysis.",
    )
    sentinel_top_k: int = Field(
        default=20,
        description="Number of top issues per type for Rust pipeline output.",
    )

    # CORS - All allowed origins for frontend requests
    # Includes production domains, Vercel previews, and local development
    cors_origins: list[str] = Field(
        default=[
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
        ]
    )

    # AWS S3
    s3_bucket_name: str = Field(default="profitsentinel-dev-uploads")
    aws_region: str = Field(default="us-east-1")

    # Supabase
    supabase_url: str | None = Field(default=None)
    supabase_service_key: str | None = Field(default=None)

    # Grok AI (X.AI) - Get your key at https://x.ai/api
    # Accepts either XAI_API_KEY or GROK_API_KEY (XAI_API_KEY preferred)
    grok_api_key: str | None = Field(default=None)
    xai_api_key: str | None = Field(default=None)

    # Stripe Payment Configuration
    stripe_secret_key: str | None = Field(default=None)
    stripe_webhook_secret: str | None = Field(default=None)
    stripe_price_id: str | None = Field(default=None)
    stripe_success_url: str | None = Field(
        default=None, description="URL to redirect after successful payment"
    )
    stripe_cancel_url: str | None = Field(
        default=None, description="URL to redirect after cancelled payment"
    )

    # Session Management
    session_ttl_hours: int = Field(
        default=24, description="Hours before diagnostic sessions expire"
    )

    # Redis for distributed rate limiting (H6-H7)
    redis_url: str | None = Field(
        default=None,
        description="Redis URL for distributed rate limiting (e.g., redis://localhost:6379)",
    )

    # File Validation (C6) - v3.7: Replaced GuardDuty with lightweight validation
    # GuardDuty settings are deprecated but kept for backward compatibility
    guardduty_scan_enabled: bool = Field(
        default=False,  # v3.7: Disabled by default, using FileValidator instead
        description="[Deprecated] Enable GuardDuty malware scanning on S3 uploads",
    )
    guardduty_scan_timeout: int = Field(
        default=30,
        description="[Deprecated] Max seconds to wait for GuardDuty scan to complete",
    )
    # New lightweight file validation (v3.7)
    file_validation_enabled: bool = Field(
        default=True,
        description="Enable lightweight file validation (replaces GuardDuty)",
    )
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum file size in MB for uploads",
    )

    @property
    def has_redis(self) -> bool:
        """Check if Redis is configured for distributed rate limiting."""
        return self.redis_url is not None

    @property
    def has_stripe(self) -> bool:
        """Check if Stripe is configured."""
        return self.stripe_secret_key is not None

    @property
    def ai_api_key(self) -> str | None:
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
        extra = "allow"  # Allows extra env vars


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
# POS Column Mappings - Loaded from YAML config files
# See: config/pos_mappings/
# =============================================================================

# Lazy-loaded from YAML for backward compatibility
# These are loaded on first access via the utility functions
STANDARD_FIELDS = load_standard_fields()
FIELD_IMPORTANCE = load_field_importance()
SUPPORTED_POS_SYSTEMS = load_supported_systems()

# Re-export utility functions for convenience
__all__ = [
    "Settings",
    "get_settings",
    "require_ai_api_key",
    "STANDARD_FIELDS",
    "FIELD_IMPORTANCE",
    "SUPPORTED_POS_SYSTEMS",
    "get_field_aliases",
    "get_field_importance",
]
