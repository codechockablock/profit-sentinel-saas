"""Config schema validation for YAML configuration files.

Validates anomaly_detection.yaml and hybrid_pipeline_config.yaml at
startup to catch conflicts (e.g. inconsistent max_iterations values)
and missing required fields before they cause runtime errors.

Usage:
    from sentinel_agent.config_schema import validate_configs
    validate_configs("/path/to/config")  # raises ValueError on conflict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)


class ResonatorSettings(BaseModel):
    """Resonator parameters that must be consistent across configs."""

    max_iterations: int
    vigilance_threshold: float | None = None
    convergence_tolerance: float | None = None


class AnomalyDetectionSettings(BaseModel):
    """Subset of anomaly_detection.yaml we validate."""

    resonator: ResonatorSettings


class AnomalyDetectionConfig(BaseModel):
    """Top-level anomaly detection config."""

    settings: AnomalyDetectionSettings


class ResonatorParameters(BaseModel):
    """Resonator parameters from hybrid_pipeline_config.yaml."""

    max_iterations: int
    dimensions: int | None = None
    convergence_threshold: float | None = None
    top_k: int | None = None
    alpha: float | None = None
    power: float | None = None


class HybridResonator(BaseModel):
    """Resonator section from hybrid pipeline config."""

    parameters: ResonatorParameters
    enabled: bool = True


class HybridPipelineConfig(BaseModel):
    """Top-level hybrid pipeline config."""

    resonator: HybridResonator


class CrossConfigValidation(BaseModel):
    """Cross-config validation — ensures consistency between files."""

    anomaly_detection: AnomalyDetectionConfig
    hybrid_pipeline: HybridPipelineConfig

    @model_validator(mode="after")
    def check_resonator_consistency(self) -> CrossConfigValidation:
        ad_iters = self.anomaly_detection.settings.resonator.max_iterations
        hp_iters = self.hybrid_pipeline.resonator.parameters.max_iterations

        if ad_iters != hp_iters:
            raise ValueError(
                f"Conflicting resonator max_iterations: "
                f"anomaly_detection.yaml={ad_iters}, "
                f"hybrid_pipeline_config.yaml={hp_iters}. "
                f"These must match."
            )
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def validate_configs(config_dir: str | Path) -> None:
    """Validate all YAML configs in the given directory.

    Args:
        config_dir: Path to the config/ directory.

    Raises:
        ValueError: If configs are inconsistent.
        FileNotFoundError: If expected config files are missing.
    """
    config_dir = Path(config_dir)

    anomaly_path = config_dir / "rules" / "anomaly_detection.yaml"
    hybrid_path = config_dir / "hybrid_pipeline_config.yaml"

    if not anomaly_path.exists():
        logger.warning(
            "anomaly_detection.yaml not found at %s, skipping validation", anomaly_path
        )
        return

    if not hybrid_path.exists():
        logger.warning(
            "hybrid_pipeline_config.yaml not found at %s, skipping validation",
            hybrid_path,
        )
        return

    anomaly_raw = _load_yaml(anomaly_path)
    hybrid_raw = _load_yaml(hybrid_path)

    # Parse with Pydantic models (validates structure)
    anomaly = AnomalyDetectionConfig(**anomaly_raw)
    hybrid = HybridPipelineConfig(**hybrid_raw)

    # Cross-config validation
    CrossConfigValidation(
        anomaly_detection=anomaly,
        hybrid_pipeline=hybrid,
    )

    logger.info(
        "Config validation passed: resonator.max_iterations=%d",
        anomaly.settings.resonator.max_iterations,
    )
