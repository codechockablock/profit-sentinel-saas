"""Sidecar API configuration.

Mirrors the pydantic-settings pattern from apps/api/src/config.py.
Loads from environment variables and .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class SidecarSettings(BaseSettings):
    """Configuration for the Sentinel sidecar API.

    All values can be set via environment variables or .env file.
    Prefix: SIDECAR_ for sidecar-specific settings.
    """

    # ----- Pipeline -----
    sentinel_bin: str | None = Field(
        default=None,
        description="Path to sentinel-server binary. Auto-detected if not set.",
    )
    csv_path: str = Field(
        default="fixtures/sample_inventory.csv",
        description="Default CSV data path for pipeline execution.",
    )
    default_store: str = Field(
        default="store-7",
        description="Default store ID when none specified.",
    )
    top_k: int = Field(
        default=5,
        description="Number of top issues to surface.",
    )

    # ----- Server -----
    sidecar_host: str = Field(default="0.0.0.0", description="Bind host.")
    sidecar_port: int = Field(default=8001, description="Bind port.")
    sidecar_dev_mode: bool = Field(
        default=False,
        description="Dev mode: bypasses auth, enables CORS wildcard.",
    )

    # ----- Auth (Supabase JWT) -----
    supabase_url: str = Field(
        default="",
        description="Supabase project URL for JWT validation.",
    )
    supabase_service_key: str = Field(
        default="",
        description="Supabase service role key.",
    )

    # ----- LLM (Anthropic Claude) -----
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for Claude-powered AI features.",
    )

    # ----- AWS S3 -----
    s3_bucket_name: str = Field(
        default="",
        description="S3 bucket name for file uploads.",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 and other services.",
    )

    # ----- Analysis -----
    sentinel_default_store: str = Field(
        default="default",
        description="Default store ID for analysis pipeline.",
    )
    sentinel_top_k: int = Field(
        default=20,
        description="Number of top issues per leak type to return.",
    )

    # ----- Email (Resend) -----
    resend_api_key: str = Field(
        default="",
        description="Resend API key for email delivery.",
    )
    digest_email_enabled: bool = Field(
        default=False,
        description="Enable scheduled morning digest emails.",
    )
    digest_send_hour: int = Field(
        default=6,
        description="Hour (0-23) to send morning digest (in subscriber timezone).",
    )

    # ----- Cache -----
    digest_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL for cached digests (seconds). Default 5 minutes.",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> SidecarSettings:
    """Get cached settings singleton."""
    return SidecarSettings()
