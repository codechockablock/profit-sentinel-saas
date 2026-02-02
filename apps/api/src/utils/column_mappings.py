"""
Column mapping loader.

Loads POS column mappings from YAML configuration files.
"""

from functools import lru_cache
from pathlib import Path

import yaml

# Path to config directory (relative to repo root)
CONFIG_DIR = (
    Path(__file__).parent.parent.parent.parent.parent / "config" / "pos_mappings"
)


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
