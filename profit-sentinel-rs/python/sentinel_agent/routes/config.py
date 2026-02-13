"""Configuration endpoint — user-configurable dead stock thresholds.

GET  /api/v1/config — fetch current config (or defaults)
PUT  /api/v1/config — save config preset + overrides

STUB: Uses in-memory storage. Production should persist to Supabase
      user_preferences table keyed by user_id.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

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
        request: Request,
        ctx: UserContext = Depends(require_auth),
    ) -> dict:
        """Save dead stock configuration.

        Accepts a preset name and/or manual overrides.

        Request body:
            preset: str (optional) — "hardware_store", "garden_center", etc.
            overrides: dict (optional) — manual threshold overrides
        """
        body = await request.json()

        preset_name = body.get("preset")
        overrides = body.get("overrides", {})

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

        # Apply overrides
        if "global_thresholds" in overrides:
            gt = overrides["global_thresholds"]
            if "watchlist_days" in gt:
                config.global_thresholds.watchlist_days = int(gt["watchlist_days"])
            if "attention_days" in gt:
                config.global_thresholds.attention_days = int(gt["attention_days"])
            if "action_days" in gt:
                config.global_thresholds.action_days = int(gt["action_days"])
            if "writeoff_days" in gt:
                config.global_thresholds.writeoff_days = int(gt["writeoff_days"])

        if "min_capital_threshold" in overrides:
            config.min_capital_threshold = float(overrides["min_capital_threshold"])

        if "min_healthy_velocity" in overrides:
            config.min_healthy_velocity = float(overrides["min_healthy_velocity"])

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
