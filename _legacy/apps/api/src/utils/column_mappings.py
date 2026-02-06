"""
Column mapping loader.

Loads POS column mappings from YAML configuration files.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml


def _find_config_dir() -> Path:
    """Find the config directory, handling both local and Docker environments."""
    # Docker: config is copied to /app/config/
    docker_path = Path("/app/config/pos_mappings")
    if docker_path.exists():
        return docker_path

    # Local development: relative to repo root
    # From apps/api/src/utils/column_mappings.py -> repo root
    local_path = (
        Path(__file__).parent.parent.parent.parent.parent / "config" / "pos_mappings"
    )
    if local_path.exists():
        return local_path

    # Environment variable override
    env_path = os.environ.get("POS_MAPPINGS_DIR")
    if env_path:
        return Path(env_path)

    # Return local path and let caller handle FileNotFoundError
    return local_path


CONFIG_DIR = _find_config_dir()


@lru_cache(maxsize=1)
def load_standard_fields() -> dict[str, list[str]]:
    """Load standard field mappings from YAML."""
    yaml_path = CONFIG_DIR / "standard_fields.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Column mappings not found: {yaml_path}")

    with open(yaml_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_field_importance() -> dict[str, int]:
    """Load field importance rankings from YAML."""
    yaml_path = CONFIG_DIR / "field_importance.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Field importance not found: {yaml_path}")

    with open(yaml_path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_supported_systems() -> list[str]:
    """Load list of supported POS systems from YAML."""
    yaml_path = CONFIG_DIR / "supported_systems.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"Supported systems not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
        return data.get("systems", [])


def get_field_aliases(field_name: str) -> list[str]:
    """Get all known aliases for a standard field."""
    fields = load_standard_fields()
    return fields.get(field_name, [])


def get_field_importance(field_name: str) -> int:
    """Get importance ranking for a field (1-5, 5 = most critical)."""
    importance = load_field_importance()
    return importance.get(field_name, 1)
