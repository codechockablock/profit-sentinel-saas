"""
vsa_core/loader.py - YAML Configuration Loaders

This module provides loaders for declarative VSA configurations:
- PrimitiveLoader: Load primitive definitions from YAML
- MagnitudeLoader: Load magnitude bucket definitions
- RuleCompiler: Compile rules to VSA query patterns

All loaders validate against Pydantic schemas and generate vectors lazily.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .operators import bind, bind_many
from .types import (
    CompositePattern,
    DetectionHint,
    MagnitudeBucket,
    MagnitudeConfig,
    MagnitudeField,
    Primitive,
    PrimitiveSet,
    PrimitiveSetMetadata,
    RootCause,
    Rule,
    RuleCondition,
    RuleDetection,
    RuleSet,
)
from .vectors import get_device, get_dtype, seed_hash


class PrimitiveLoader:
    """Load and manage VSA primitives from YAML definitions.

    The loader parses YAML primitive files, validates them, and provides
    efficient access to primitive vectors.

    Example:
        loader = PrimitiveLoader()
        loader.load_file("primitives/retail_inventory.yaml")

        # Get primitive vector
        low_stock = loader.get_vector("inventory.low_stock")

        # Get primitive definition
        prim = loader.get_primitive("inventory.low_stock")
        print(prim.root_causes)

        # Build codebook
        labels, vectors = loader.build_codebook()
    """

    def __init__(self, dimensions: int = 16384):
        """Initialize loader.

        Args:
            dimensions: Vector dimensionality
        """
        self.dimensions = dimensions
        self.primitive_sets: dict[str, PrimitiveSet] = {}
        self._vector_cache: dict[str, torch.Tensor] = {}
        self._device = get_device()
        self._dtype = get_dtype()

    def load_file(self, path: str | Path) -> PrimitiveSet:
        """Load primitives from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed PrimitiveSet
        """
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Parse into PrimitiveSet
        pset = self._parse_primitive_set(raw)
        self.primitive_sets[pset.domain] = pset

        # Update dimensions if specified
        if pset.dimensions:
            self.dimensions = pset.dimensions

        return pset

    def _parse_primitive_set(self, raw: dict) -> PrimitiveSet:
        """Parse raw YAML into PrimitiveSet."""
        # Parse metadata
        metadata = None
        if "metadata" in raw:
            metadata = PrimitiveSetMetadata(**raw["metadata"])

        # Parse primitives by category
        primitives: dict[str, dict[str, Primitive]] = {}
        for category, prims in raw.get("primitives", {}).items():
            primitives[category] = {}
            for name, prim_data in prims.items():
                # Parse root causes
                root_causes = []
                for rc in prim_data.get("root_causes", []):
                    root_causes.append(RootCause(**rc))

                # Parse detection hints
                detection_hints = None
                if "detection_hints" in prim_data:
                    detection_hints = DetectionHint(**prim_data["detection_hints"])

                primitives[category][name] = Primitive(
                    seed=prim_data["seed"],
                    description=prim_data["description"],
                    category=prim_data["category"],
                    severity=prim_data.get("severity", "medium"),
                    root_causes=root_causes,
                    detection_hints=detection_hints,
                    related_primitives=prim_data.get("related_primitives", []),
                    investigation_steps=prim_data.get("investigation_steps", []),
                    algebraic_note=prim_data.get("algebraic_note"),
                )

        # Parse composite patterns
        composite_patterns: dict[str, CompositePattern] = {}
        for name, pattern in raw.get("composite_patterns", {}).items():
            composite_patterns[name] = CompositePattern(
                description=pattern["description"],
                composition=pattern["composition"],
                severity=pattern.get("severity", "medium")
            )

        return PrimitiveSet(
            schema_version=raw["schema_version"],
            domain=raw["domain"],
            description=raw.get("description"),
            dimensions=raw.get("dimensions", 16384),
            metadata=metadata,
            primitives=primitives,
            composite_patterns=composite_patterns,
            aliases=raw.get("aliases", {})
        )

    def get_primitive(self, path: str) -> Primitive | None:
        """Get primitive definition by path.

        Args:
            path: Dot-notation path (e.g., "inventory.low_stock")

        Returns:
            Primitive definition or None
        """
        # Check aliases
        for pset in self.primitive_sets.values():
            if path in pset.aliases:
                path = pset.aliases[path]

        # Search all loaded sets
        for pset in self.primitive_sets.values():
            prim = pset.get_primitive(path)
            if prim is not None:
                return prim
        return None

    def get_vector(self, path: str) -> torch.Tensor:
        """Get primitive vector by path.

        Args:
            path: Dot-notation path (e.g., "inventory.low_stock")

        Returns:
            Primitive vector

        Raises:
            KeyError: If primitive not found
        """
        # Check cache
        if path in self._vector_cache:
            return self._vector_cache[path]

        # Resolve alias
        resolved = path
        for pset in self.primitive_sets.values():
            if path in pset.aliases:
                resolved = pset.aliases[path]
                break

        # Get primitive
        prim = self.get_primitive(resolved)
        if prim is None:
            raise KeyError(f"Primitive not found: {path}")

        # Generate vector
        vec = seed_hash(
            prim.seed,
            dimensions=self.dimensions,
            device=self._device,
            dtype=self._dtype
        )

        self._vector_cache[path] = vec
        self._vector_cache[resolved] = vec  # Cache both alias and resolved
        return vec

    def get_composite_vector(self, name: str) -> torch.Tensor:
        """Get composite pattern vector.

        Args:
            name: Composite pattern name

        Returns:
            Composed vector

        Raises:
            KeyError: If pattern not found
        """
        # Find pattern
        pattern = None
        for pset in self.primitive_sets.values():
            if name in pset.composite_patterns:
                pattern = pset.composite_patterns[name]
                break

        if pattern is None:
            raise KeyError(f"Composite pattern not found: {name}")

        # Get operand vectors
        operands = [self.get_vector(op) for op in pattern.composition["operands"]]

        # Apply operator
        op = pattern.composition["operator"]
        if op == "bind":
            return bind_many(*operands)
        elif op == "bundle":
            from .operators import bundle_many
            return bundle_many(*operands)
        else:
            raise ValueError(f"Unknown operator: {op}")

    def list_primitives(self) -> list[str]:
        """List all available primitive paths."""
        paths = []
        for pset in self.primitive_sets.values():
            paths.extend(pset.list_all_primitives())
        return paths

    def build_codebook(
        self,
        include_composites: bool = True
    ) -> tuple[list[str], torch.Tensor]:
        """Build codebook from all loaded primitives.

        Args:
            include_composites: Include composite patterns

        Returns:
            (labels, vectors) tuple
        """
        labels = []
        vectors = []

        # Add primitives
        for path in self.list_primitives():
            labels.append(path)
            vectors.append(self.get_vector(path))

        # Add composites
        if include_composites:
            for pset in self.primitive_sets.values():
                for name in pset.composite_patterns:
                    labels.append(f"composite:{name}")
                    vectors.append(self.get_composite_vector(name))

        return labels, torch.stack(vectors)


class MagnitudeLoader:
    """Load and manage magnitude bucket definitions.

    Magnitude buckets discretize continuous values into semantic ranges,
    enabling continuous-to-discrete binding.

    Example:
        loader = MagnitudeLoader()
        loader.load_file("magnitude_buckets/retail_inventory.yaml")

        # Get bucket name
        bucket = loader.get_bucket_name("quantity", 5)  # "critical_low"

        # Bind magnitude to entity
        binding = loader.bind_magnitude(sku_vec, "quantity", 5)
    """

    def __init__(self, dimensions: int = 16384):
        """Initialize loader."""
        self.dimensions = dimensions
        self.configs: dict[str, MagnitudeConfig] = {}
        self._vector_cache: dict[str, torch.Tensor] = {}
        self._device = get_device()
        self._dtype = get_dtype()

    def load_file(self, path: str | Path) -> MagnitudeConfig:
        """Load magnitude config from YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        config = self._parse_config(raw)
        self.configs[config.domain] = config
        return config

    def _parse_config(self, raw: dict) -> MagnitudeConfig:
        """Parse raw YAML into MagnitudeConfig."""
        fields: dict[str, MagnitudeField] = {}

        for field_name, field_data in raw.get("fields", {}).items():
            buckets = []
            for bucket in field_data.get("buckets", []):
                buckets.append(MagnitudeBucket(
                    name=bucket["name"],
                    min_value=bucket.get("min_value"),
                    max_value=bucket.get("max_value"),
                    seed_suffix=bucket["seed_suffix"]
                ))

            fields[field_name] = MagnitudeField(
                field=field_data["field"],
                buckets=buckets
            )

        return MagnitudeConfig(
            schema_version=raw["schema_version"],
            domain=raw["domain"],
            fields=fields
        )

    def get_bucket(self, field: str, value: float) -> MagnitudeBucket | None:
        """Get bucket for field value.

        Args:
            field: Field name (e.g., "quantity", "margin")
            value: Numeric value

        Returns:
            Matching bucket or None
        """
        for config in self.configs.values():
            bucket = config.get_bucket(field, value)
            if bucket is not None:
                return bucket
        return None

    def get_bucket_name(self, field: str, value: float) -> str | None:
        """Get bucket name for field value."""
        bucket = self.get_bucket(field, value)
        return bucket.name if bucket else None

    def get_bucket_vector(
        self,
        field: str,
        value: float,
        entity_seed: str = ""
    ) -> torch.Tensor | None:
        """Get vector for magnitude bucket.

        Args:
            field: Field name
            value: Numeric value
            entity_seed: Optional entity seed to combine with

        Returns:
            Bucket vector (optionally bound with entity)
        """
        bucket = self.get_bucket(field, value)
        if bucket is None:
            return None

        # Generate bucket vector
        cache_key = f"{field}:{bucket.name}:{entity_seed}"
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]

        full_seed = f"{entity_seed}{bucket.seed_suffix}"
        vec = seed_hash(
            full_seed,
            dimensions=self.dimensions,
            device=self._device,
            dtype=self._dtype
        )

        self._vector_cache[cache_key] = vec
        return vec

    def bind_magnitude(
        self,
        entity_vec: torch.Tensor,
        field: str,
        value: float
    ) -> torch.Tensor | None:
        """Bind magnitude bucket to entity vector.

        This creates a representation like:
            sku_vec ⊗ bucket_vec

        Args:
            entity_vec: Entity vector
            field: Field name
            value: Numeric value

        Returns:
            Bound vector or None if no bucket matches
        """
        bucket = self.get_bucket(field, value)
        if bucket is None:
            return None

        bucket_vec = seed_hash(
            bucket.seed_suffix,
            dimensions=self.dimensions,
            device=self._device,
            dtype=self._dtype
        )

        return bind(entity_vec, bucket_vec)

    def list_fields(self) -> list[str]:
        """List all available fields."""
        fields = set()
        for config in self.configs.values():
            fields.update(config.fields.keys())
        return list(fields)


class RuleCompiler:
    """Compile declarative rules into VSA query patterns.

    Rules defined in YAML are compiled into VSA vectors that can be
    used to query for matching anomalies.

    Example:
        compiler = RuleCompiler(primitive_loader, magnitude_loader)
        compiler.load_file("rules/retail/anomaly_rules.yaml")

        # Compile rule to query vector
        query = compiler.compile_rule("margin_erosion")

        # Use with resonator
        result = resonator.resonate(query)
    """

    def __init__(
        self,
        primitive_loader: PrimitiveLoader,
        magnitude_loader: MagnitudeLoader | None = None
    ):
        """Initialize compiler.

        Args:
            primitive_loader: Loader for primitives
            magnitude_loader: Loader for magnitude buckets
        """
        self.primitives = primitive_loader
        self.magnitudes = magnitude_loader
        self.rule_sets: dict[str, RuleSet] = {}
        self._compiled_cache: dict[str, torch.Tensor] = {}

    def load_file(self, path: str | Path) -> RuleSet:
        """Load rules from YAML file."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        rule_set = self._parse_rule_set(raw)
        self.rule_sets[rule_set.domain] = rule_set
        return rule_set

    def _parse_rule_set(self, raw: dict) -> RuleSet:
        """Parse raw YAML into RuleSet."""
        rules = []
        for rule_data in raw.get("rules", []):
            # Parse detection
            detection_data = rule_data["detection"]
            conditions = []
            for cond in detection_data.get("conditions", []):
                conditions.append(RuleCondition(
                    primitive=cond["primitive"],
                    magnitude_bucket=cond.get("magnitude_bucket"),
                    required=cond.get("required", True),
                    weight=cond.get("weight", 1.0)
                ))

            detection = RuleDetection(
                type=detection_data["type"],
                operator=detection_data.get("operator", "bind"),
                conditions=conditions,
                exclude_if=detection_data.get("exclude_if", [])
            )

            rules.append(Rule(
                id=rule_data["id"],
                name=rule_data["name"],
                description=rule_data["description"],
                enabled=rule_data.get("enabled", True),
                severity=rule_data.get("severity", "medium"),
                priority=rule_data.get("priority", 5),
                detection=detection,
                thresholds=rule_data.get("thresholds", {}),
                entity_context=rule_data.get("entity_context", {}),
                root_cause_analysis=rule_data.get("root_cause_analysis"),
                documentation=rule_data.get("documentation", {})
            ))

        return RuleSet(
            schema_version=raw["schema_version"],
            domain=raw["domain"],
            description=raw.get("description"),
            rules=rules,
            settings=None
        )

    def compile_rule(self, rule_id: str) -> torch.Tensor:
        """Compile rule to query vector.

        Args:
            rule_id: Rule identifier

        Returns:
            Query vector for rule
        """
        if rule_id in self._compiled_cache:
            return self._compiled_cache[rule_id]

        # Find rule
        rule = None
        for rs in self.rule_sets.values():
            rule = rs.get_rule(rule_id)
            if rule is not None:
                break

        if rule is None:
            raise KeyError(f"Rule not found: {rule_id}")

        # Compile based on detection type
        detection = rule.detection

        if detection.type == "primitive":
            # Single primitive
            vec = self.primitives.get_vector(detection.conditions[0].primitive)

        elif detection.type in ("compound", "aggregate"):
            # Multiple conditions
            condition_vecs = []
            for cond in detection.conditions:
                if not cond.required:
                    continue  # Skip optional conditions for base query
                vec = self.primitives.get_vector(cond.primitive)
                condition_vecs.append(vec)

            # Apply operator
            if detection.operator == "bind":
                vec = bind_many(*condition_vecs)
            else:
                from .operators import bundle_many
                vec = bundle_many(*condition_vecs)

        else:
            raise ValueError(f"Unknown detection type: {detection.type}")

        self._compiled_cache[rule_id] = vec
        return vec

    def compile_all_enabled(self) -> dict[str, torch.Tensor]:
        """Compile all enabled rules.

        Returns:
            Dict mapping rule_id to compiled vector
        """
        compiled = {}
        for rs in self.rule_sets.values():
            for rule in rs.get_enabled_rules():
                compiled[rule.id] = self.compile_rule(rule.id)
        return compiled

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get rule definition."""
        for rs in self.rule_sets.values():
            rule = rs.get_rule(rule_id)
            if rule is not None:
                return rule
        return None
