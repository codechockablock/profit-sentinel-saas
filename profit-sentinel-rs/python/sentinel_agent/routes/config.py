"""Configuration endpoint — user-configurable dead stock thresholds.

GET  /api/v1/config — fetch current config (or defaults)
PUT  /api/v1/config — save config preset + overrides

STUB: Uses in-memory storage. Production should persist to Supabase
      user_preferences table keyed by user_id.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, model_validator

from ..dual_auth import UserContext

# Import with graceful fallback — world_model/__init__.py imports pipeline/core
# which need numpy. config.py itself only uses stdlib, but the package __init__
# triggers the full import chain. If it fails, we still need config types.
try:
    from ..world_model.config import ConfigPresets, DeadStockConfig
except ImportError:
    import importlib.util
    import os

    _config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "world_model",
        "config.py",
    )
    _spec = importlib.util.spec_from_file_location("world_model_config", _config_path)
    _config_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    _spec.loader.exec_module(_config_mod)  # type: ignore[union-attr]
    ConfigPresets = _config_mod.ConfigPresets  # type: ignore[misc]
    DeadStockConfig = _config_mod.DeadStockConfig  # type: ignore[misc]
from .state import AppState

logger = logging.getLogger("sentinel.routes.config")


# ---------------------------------------------------------------------------
# Pydantic request model for PUT /config
# ---------------------------------------------------------------------------


class ThresholdOverrides(BaseModel):
    """Optional overrides for dead stock day thresholds."""

    watchlist_days: int | None = Field(None, ge=1, le=3650)
    attention_days: int | None = Field(None, ge=1, le=3650)
    action_days: int | None = Field(None, ge=1, le=3650)
    writeoff_days: int | None = Field(None, ge=1, le=3650)

    @model_validator(mode="after")
    def _validate_ordering(self) -> ThresholdOverrides:
        """Ensure thresholds are in ascending order when all are provided."""
        vals = [
            ("watchlist_days", self.watchlist_days),
            ("attention_days", self.attention_days),
            ("action_days", self.action_days),
            ("writeoff_days", self.writeoff_days),
        ]
        # Only validate ordering among explicitly provided values
        provided = [(name, v) for name, v in vals if v is not None]
        for i in range(1, len(provided)):
            if provided[i][1] <= provided[i - 1][1]:
                raise ValueError(
                    f"{provided[i][0]} ({provided[i][1]}) must be greater "
                    f"than {provided[i - 1][0]} ({provided[i - 1][1]})"
                )
        return self


class ConfigOverrides(BaseModel):
    """Override block within a config update request."""

    global_thresholds: ThresholdOverrides | None = None
    min_capital_threshold: float | None = Field(None, ge=0.0, le=1_000_000.0)
    min_healthy_velocity: float | None = Field(None, ge=0.0, le=10_000.0)


class ConfigUpdateRequest(BaseModel):
    """Request body for PUT /api/v1/config."""

    preset: str | None = None
    overrides: ConfigOverrides | None = None


# In-memory config store per user (production: Supabase user_preferences)
_user_configs: dict[str, dict] = {}


def create_config_router(state: AppState, require_auth) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["config"])

    @router.get("/config")
    async def get_config(
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Fetch current dead stock configuration.

        Returns the user's saved config or defaults.
        Includes available presets for the frontend preset selector.
        """
        saved = _user_configs.get(ctx.user_id)

        if saved:
            config = DeadStockConfig.from_dict(saved)
        else:
            config = DeadStockConfig()  # defaults: 60/120/180/360

        return {
            "preset": saved.get("_preset") if saved else None,
            "global_thresholds": {
                "watchlist_days": config.global_thresholds.watchlist_days,
                "attention_days": config.global_thresholds.attention_days,
                "action_days": config.global_thresholds.action_days,
                "writeoff_days": config.global_thresholds.writeoff_days,
            },
            "category_overrides": {
                name: {
                    "watchlist_days": t.watchlist_days,
                    "attention_days": t.attention_days,
                    "action_days": t.action_days,
                    "writeoff_days": t.writeoff_days,
                }
                for name, t in config.category_overrides.items()
            },
            "min_capital_threshold": config.min_capital_threshold,
            "min_healthy_velocity": config.min_healthy_velocity,
            "available_presets": list(ConfigPresets.all_presets().keys()),
        }

    @router.put("/config")
    async def update_config(
        body: ConfigUpdateRequest,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Save dead stock configuration.

        Accepts a preset name and/or manual overrides.
        Pydantic validates field bounds and threshold ordering.
        """
        preset_name = body.preset

        # Start from preset or current defaults
        if preset_name:
            all_presets = ConfigPresets.all_presets()
            if preset_name not in all_presets:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset: {preset_name}. "
                    f"Available: {list(all_presets.keys())}",
                )
            config = all_presets[preset_name]
        else:
            config = DeadStockConfig()

        # Apply overrides (already validated by Pydantic)
        if body.overrides is not None:
            gt = body.overrides.global_thresholds
            if gt is not None:
                if gt.watchlist_days is not None:
                    config.global_thresholds.watchlist_days = gt.watchlist_days
                if gt.attention_days is not None:
                    config.global_thresholds.attention_days = gt.attention_days
                if gt.action_days is not None:
                    config.global_thresholds.action_days = gt.action_days
                if gt.writeoff_days is not None:
                    config.global_thresholds.writeoff_days = gt.writeoff_days

            if body.overrides.min_capital_threshold is not None:
                config.min_capital_threshold = body.overrides.min_capital_threshold

            if body.overrides.min_healthy_velocity is not None:
                config.min_healthy_velocity = body.overrides.min_healthy_velocity

        # Validate
        errors = config.validate()
        if errors:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid configuration: {errors}",
            )

        # Save — keyed by authenticated user_id
        config_dict = config.to_dict()
        config_dict["_preset"] = preset_name
        _user_configs[ctx.user_id] = config_dict

        logger.info(
            "Config saved for user %s: preset=%s, thresholds=%d/%d/%d/%d",
            ctx.user_id,
            preset_name,
            config.global_thresholds.watchlist_days,
            config.global_thresholds.attention_days,
            config.global_thresholds.action_days,
            config.global_thresholds.writeoff_days,
        )

        return {
            "saved": True,
            "preset": preset_name,
            "global_thresholds": {
                "watchlist_days": config.global_thresholds.watchlist_days,
                "attention_days": config.global_thresholds.attention_days,
                "action_days": config.global_thresholds.action_days,
                "writeoff_days": config.global_thresholds.writeoff_days,
            },
        }

    return router
