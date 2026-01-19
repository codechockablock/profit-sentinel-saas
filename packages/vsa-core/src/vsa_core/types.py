"""
vsa_core/types.py - Pydantic type definitions for VSA operations

Uses Pydantic v2 for validation with JSON Schema generation.
All configuration and data structures are strictly typed.
"""
from __future__ import annotations

import math
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# VECTOR CONFIGURATION
# =============================================================================

class VectorConfig(BaseModel):
    """Configuration for hyperdimensional vectors."""

    dimensions: int = Field(
        default=16384,
        ge=1024,
        le=131072,
        description="Vector dimensionality (power of 2 recommended)"
    )
    dtype: Literal["complex64", "complex128"] = Field(
        default="complex64",
        description="Complex data type for phasor representation"
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Compute device"
    )

    @field_validator('dimensions')
    @classmethod
    def validate_power_of_two(cls, v: int) -> int:
        """Round up to nearest power of 2 if needed."""
        if v & (v - 1) != 0:
            v = 2 ** math.ceil(math.log2(v))
        return v

    def get_device(self) -> torch.device:
        """Get torch device based on config."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def get_dtype(self) -> torch.dtype:
        """Get torch dtype based on config."""
        return torch.complex64 if self.dtype == "complex64" else torch.complex128

    model_config = {"frozen": True}


# =============================================================================
# RESONATOR CONFIGURATION
# =============================================================================

class ResonatorConfig(BaseModel):
    """Configuration for the convergence-lock resonator."""

    iterations: int = Field(default=450, ge=1, le=10000)
    multi_steps: int = Field(default=3, ge=1, le=10)
    alpha: float = Field(default=0.85, ge=0.0, le=1.0, description="Momentum decay")
    power: float = Field(default=0.64, ge=0.0, le=2.0, description="Similarity amplification")
    top_k: int = Field(default=64, ge=1, le=1000, description="Sparse attention width")
    phase_bins: int = Field(default=512, ge=1, le=4096, description="Phase quantization bins")

    convergence_threshold: float = Field(default=0.0001, ge=0.0)
    early_exit: bool = Field(default=True, description="Exit when converged")

    model_config = {"frozen": True}


# =============================================================================
# PRIMITIVE DEFINITIONS
# =============================================================================

class RootCause(BaseModel):
    """Possible root cause for an anomaly."""
    code: str = Field(..., min_length=1, pattern=r"^[A-Z_]+$")
    description: str
    likelihood: float = Field(default=0.5, ge=0.0, le=1.0)


class DetectionHint(BaseModel):
    """Hints for automatic primitive detection."""
    field: str | None = None
    operator: Literal["<", "<=", ">", ">=", "==", "!="] = "<"
    threshold: float | None = None
    computed_field: str | None = None
    computation: str | None = None
    time_window_days: int | None = None


class Primitive(BaseModel):
    """A single VSA primitive definition."""

    seed: str = Field(..., min_length=1, description="Deterministic seed string")
    description: str
    category: Literal["anomaly", "state", "operator", "temporal", "role", "metric", "pattern"]
    severity: Literal["critical", "high", "medium", "warning", "info", "normal"] = "medium"

    root_causes: list[RootCause] = Field(default_factory=list)
    detection_hints: DetectionHint | None = None
    related_primitives: list[str] = Field(default_factory=list)
    investigation_steps: list[str] = Field(default_factory=list)
    algebraic_note: str | None = None


class CompositePattern(BaseModel):
    """Pre-defined composite primitive pattern."""
    description: str
    composition: dict[str, Any]
    severity: Literal["critical", "high", "medium", "warning", "info"] = "medium"


class PrimitiveSetMetadata(BaseModel):
    """Metadata for a primitive set."""
    author: str | None = None
    created: str | None = None
    version: str | None = None
    checksum_algorithm: Literal["sha256", "md5", "sha1"] | None = None


class PrimitiveSet(BaseModel):
    """A domain-specific set of primitives."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    domain: str
    description: str | None = None
    dimensions: int = 16384

    metadata: PrimitiveSetMetadata | None = None
    primitives: dict[str, dict[str, Primitive]]  # category -> name -> Primitive
    composite_patterns: dict[str, CompositePattern] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)

    def get_primitive(self, path: str) -> Primitive | None:
        """Get primitive by dot-notation path (e.g., 'inventory.low_stock')."""
        parts = path.split('.')
        if len(parts) == 2:
            category, name = parts
            return self.primitives.get(category, {}).get(name)
        return None

    def list_all_primitives(self) -> list[str]:
        """List all primitive paths."""
        paths = []
        for category, prims in self.primitives.items():
            for name in prims:
                paths.append(f"{category}.{name}")
        return paths


# =============================================================================
# MAGNITUDE BUCKETS
# =============================================================================

class MagnitudeBucket(BaseModel):
    """A single magnitude bucket for quantization."""
    name: str
    min_value: float | None = None
    max_value: float | None = None
    seed_suffix: str  # Appended to entity seed for bucket binding


class MagnitudeField(BaseModel):
    """Buckets for a specific field."""
    field: str
    buckets: list[MagnitudeBucket]

    def get_bucket(self, value: float) -> MagnitudeBucket | None:
        """Get bucket for a value."""
        for bucket in self.buckets:
            min_ok = bucket.min_value is None or value >= bucket.min_value
            max_ok = bucket.max_value is None or value < bucket.max_value
            if min_ok and max_ok:
                return bucket
        return None

    def get_bucket_name(self, value: float) -> str | None:
        """Get bucket name for a value."""
        bucket = self.get_bucket(value)
        return bucket.name if bucket else None


class MagnitudeConfig(BaseModel):
    """Full magnitude bucket configuration."""
    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    domain: str
    fields: dict[str, MagnitudeField]

    def get_bucket(self, field: str, value: float) -> MagnitudeBucket | None:
        """Get bucket for field value."""
        if field not in self.fields:
            return None
        return self.fields[field].get_bucket(value)


# =============================================================================
# RULE DEFINITIONS
# =============================================================================

class RuleCondition(BaseModel):
    """A condition in a detection rule."""
    primitive: str  # Dot-notation path
    magnitude_bucket: str | None = None
    required: bool = True
    weight: float = 1.0


class RuleDetection(BaseModel):
    """Detection pattern for a rule."""
    type: Literal["primitive", "compound", "aggregate"]
    operator: Literal["bind", "bundle", "permute"] = "bind"
    conditions: list[RuleCondition]
    exclude_if: list[str] = Field(default_factory=list)


class AlertConfig(BaseModel):
    """Alert configuration for a rule."""
    notify: list[dict[str, str]] = Field(default_factory=list)
    escalate_after_hours: int = 24
    suppress_duplicate_hours: int = 4
    confidential: bool = False


class RecommendedAction(BaseModel):
    """A recommended action for a rule."""
    action: str
    description: str
    priority: int = 1
    condition: str | None = None
    calculation: str | None = None


class Rule(BaseModel):
    """A complete anomaly detection rule."""

    id: str = Field(..., pattern=r"^[a-z_]+$")
    name: str
    description: str
    enabled: bool = True

    severity: Literal["critical", "high", "medium", "warning"] = "medium"
    priority: int = Field(default=5, ge=1, le=10)

    detection: RuleDetection
    thresholds: dict[str, float] = Field(default_factory=dict)
    entity_context: dict[str, Any] = Field(default_factory=dict)

    root_cause_analysis: dict[str, Any] | None = None
    recommended_actions: dict[str, list[RecommendedAction]] = Field(default_factory=dict)
    alert_config: AlertConfig | None = None
    documentation: dict[str, str] = Field(default_factory=dict)


class InferenceStep(BaseModel):
    """A step in an inference chain."""
    step: int
    hypothesis: str | None = None
    check: str | None = None
    test: dict[str, Any] | None = None
    if_true: str | None = None
    if_false: str | None = None
    if_match: dict[str, Any] | None = None
    default: bool = False
    confidence: float = 0.5
    conclude: str | None = None
    action: str | None = None


class InferenceChain(BaseModel):
    """Multi-step reasoning chain for root cause analysis."""
    id: str
    name: str
    description: str | None = None
    trigger: dict[str, str]
    chain: list[InferenceStep]


class RuleGroup(BaseModel):
    """Logical grouping of rules."""
    description: str
    rules: list[str]
    schedule: str | None = None


class RuleSetSettings(BaseModel):
    """Global settings for a rule set."""
    resonator: dict[str, Any] | None = None
    alerts: dict[str, Any] | None = None
    confidence: dict[str, float] | None = None


class RuleSet(BaseModel):
    """A complete rule configuration."""

    schema_version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    domain: str
    description: str | None = None
    metadata: dict[str, Any] | None = None

    rules: list[Rule]
    inference_chains: list[InferenceChain] = Field(default_factory=list)
    rule_groups: dict[str, RuleGroup] = Field(default_factory=dict)
    settings: RuleSetSettings | None = None

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def get_enabled_rules(self) -> list[Rule]:
        """Get all enabled rules."""
        return [r for r in self.rules if r.enabled]

    def get_rules_by_severity(self, severity: str) -> list[Rule]:
        """Get rules by severity level."""
        return [r for r in self.rules if r.severity == severity and r.enabled]
