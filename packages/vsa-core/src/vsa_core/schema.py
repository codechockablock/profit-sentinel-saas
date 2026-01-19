"""
vsa_core/schema.py - Schema Evolution Binding (SE-Bind) for VSA

Handles evolving data schemas without breaking existing VSA representations.
When POS systems upgrade, column names change, or new fields appear, SE-Bind
ensures backward compatibility and seamless migration.

Key Concepts:
    - SchemaRegistry: Central registry of field mappings across versions
    - Alias Resolution: Multiple field names → canonical semantic slot
    - Migration: Transform bundles from old schema to new
    - Version Tracking: Know which schema version encoded each bundle

Use Cases:
    - POS system upgrade changes 'qty' → 'quantity_on_hand'
    - Multi-store integration with different column naming
    - Historical data remains queryable after schema changes
    - Gradual schema migration without downtime

Example:
    # Register schema v1
    registry = SchemaRegistry("v1")
    registry.add_field("quantity", aliases=["qty", "on_hand", "qoh"])

    # Encode with schema-aware binding
    bound = se_bind(qty_vec, "qty", registry)  # Resolves to "quantity"

    # Migrate to v2 when needed
    new_bundle = migrate_bundle(old_bundle, v1_registry, v2_registry)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from .operators import bind, bundle, unbind
from .vectors import normalize, random_vector, similarity


@dataclass
class FieldSpec:
    """Specification for a schema field."""
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    vector: torch.Tensor | None = None
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SchemaRegistry:
    """Registry of field mappings across schema versions.

    Provides:
    - Canonical field name resolution from aliases
    - Deterministic vector generation for fields
    - Version tracking for migration
    - Transformation hooks for value normalization

    Example:
        registry = SchemaRegistry("v2.1", dimensions=8192)
        registry.add_field("quantity", aliases=["qty", "on_hand"])
        registry.add_field("margin", aliases=["profit_margin", "gross_margin"])

        # Later, resolve aliases
        canonical = registry.resolve("qty")  # Returns "quantity"
    """

    def __init__(
        self,
        version: str,
        dimensions: int = 8192,
        seed: int | None = None
    ):
        """Initialize schema registry.

        Args:
            version: Schema version identifier
            dimensions: VSA dimensionality
            seed: Random seed for deterministic vector generation
        """
        self.version = version
        self.dimensions = dimensions
        self.seed = seed or hash(version) % (2**31)

        self.fields: dict[str, FieldSpec] = {}
        self._alias_map: dict[str, str] = {}  # alias -> canonical
        self._vectors: dict[str, torch.Tensor] = {}

    def add_field(
        self,
        canonical_name: str,
        aliases: list[str] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a field to the schema.

        Args:
            canonical_name: The canonical/standard name for this field
            aliases: Alternative names that map to this field
            transform: Optional transformation function for values
            metadata: Optional metadata about the field
        """
        aliases = aliases or []
        metadata = metadata or {}

        # Generate deterministic vector for this field
        field_seed = self.seed + hash(canonical_name)
        vector = self._generate_field_vector(canonical_name, field_seed)

        spec = FieldSpec(
            canonical_name=canonical_name,
            aliases=aliases,
            vector=vector,
            transform=transform,
            metadata=metadata
        )

        self.fields[canonical_name] = spec
        self._vectors[canonical_name] = vector

        # Register aliases
        self._alias_map[canonical_name] = canonical_name
        for alias in aliases:
            self._alias_map[alias.lower()] = canonical_name

    def _generate_field_vector(self, name: str, seed: int) -> torch.Tensor:
        """Generate deterministic vector for field name."""
        # Use hash-based seeding for reproducibility
        torch.manual_seed(seed)
        return random_vector(self.dimensions)

    def resolve(self, field_name: str) -> str:
        """Resolve field name to canonical name.

        Args:
            field_name: Any field name (canonical or alias)

        Returns:
            Canonical field name

        Raises:
            KeyError: If field not found in registry
        """
        key = field_name.lower().strip()
        if key in self._alias_map:
            return self._alias_map[key]
        raise KeyError(f"Unknown field: {field_name}")

    def get_vector(self, field_name: str) -> torch.Tensor:
        """Get vector for field (resolves aliases).

        Args:
            field_name: Any field name

        Returns:
            Field vector

        Raises:
            KeyError: If field not found
        """
        canonical = self.resolve(field_name)
        return self._vectors[canonical]

    def get_transform(
        self,
        field_name: str
    ) -> Callable[[torch.Tensor], torch.Tensor] | None:
        """Get transformation function for field.

        Args:
            field_name: Any field name

        Returns:
            Transform function or None
        """
        canonical = self.resolve(field_name)
        return self.fields[canonical].transform

    def has_field(self, field_name: str) -> bool:
        """Check if field exists in registry."""
        try:
            self.resolve(field_name)
            return True
        except KeyError:
            return False

    def add_alias(self, canonical_name: str, alias: str) -> None:
        """Add additional alias to existing field.

        Args:
            canonical_name: Existing canonical field name
            alias: New alias to add
        """
        if canonical_name not in self.fields:
            raise KeyError(f"Unknown canonical field: {canonical_name}")

        self.fields[canonical_name].aliases.append(alias)
        self._alias_map[alias.lower()] = canonical_name

    def list_fields(self) -> list[str]:
        """List all canonical field names."""
        return list(self.fields.keys())

    def list_aliases(self, canonical_name: str) -> list[str]:
        """List all aliases for a canonical field."""
        if canonical_name not in self.fields:
            raise KeyError(f"Unknown field: {canonical_name}")
        return self.fields[canonical_name].aliases

    def to_dict(self) -> dict[str, Any]:
        """Export registry as dictionary."""
        return {
            "version": self.version,
            "dimensions": self.dimensions,
            "fields": {
                name: {
                    "aliases": spec.aliases,
                    "has_transform": spec.transform is not None,
                    "metadata": spec.metadata
                }
                for name, spec in self.fields.items()
            }
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        transforms: dict[str, Callable] | None = None
    ) -> SchemaRegistry:
        """Create registry from dictionary.

        Args:
            data: Dictionary from to_dict()
            transforms: Optional mapping of field names to transform functions
        """
        transforms = transforms or {}
        registry = cls(data["version"], data["dimensions"])

        for name, spec_data in data["fields"].items():
            registry.add_field(
                canonical_name=name,
                aliases=spec_data.get("aliases", []),
                transform=transforms.get(name),
                metadata=spec_data.get("metadata", {})
            )

        return registry


def se_bind(
    value_vector: torch.Tensor,
    field_name: str,
    schema: SchemaRegistry,
    auto_add: bool = False
) -> torch.Tensor:
    """Schema-evolution-aware binding.

    Resolves field aliases to canonical names before binding, ensuring
    consistent semantic representation regardless of field naming.

    Args:
        value_vector: The value to bind
        field_name: Field name (canonical or alias)
        schema: Schema registry
        auto_add: If True, auto-create unknown fields

    Returns:
        Bound vector (field ⊗ value)

    Example:
        # These all produce equivalent bindings:
        se_bind(qty_vec, "qty", schema)
        se_bind(qty_vec, "quantity", schema)
        se_bind(qty_vec, "on_hand", schema)
    """
    try:
        canonical = schema.resolve(field_name)
    except KeyError:
        if auto_add:
            schema.add_field(field_name)
            canonical = field_name
        else:
            raise

    field_vec = schema.get_vector(canonical)

    # Apply transformation if defined
    transform = schema.get_transform(canonical)
    if transform is not None:
        value_vector = transform(value_vector)

    return bind(field_vec, value_vector)


def se_unbind(
    bound_vector: torch.Tensor,
    field_name: str,
    schema: SchemaRegistry
) -> torch.Tensor:
    """Schema-aware unbinding.

    Args:
        bound_vector: The bound vector
        field_name: Field to unbind
        schema: Schema registry

    Returns:
        Unbound value vector
    """
    field_vec = schema.get_vector(field_name)
    return unbind(bound_vector, field_vec)


def create_schema_record(
    field_values: dict[str, torch.Tensor],
    schema: SchemaRegistry,
    auto_add: bool = False
) -> torch.Tensor:
    """Create bundled record from field-value pairs with schema awareness.

    Args:
        field_values: Dict of field_name -> value_vector
        schema: Schema registry
        auto_add: Auto-create unknown fields

    Returns:
        Bundled record vector

    Example:
        record = create_schema_record({
            "qty": quantity_vec,
            "sku": sku_vec,
            "margin": margin_vec,
        }, schema)
    """
    bindings = []
    for field_name, value_vec in field_values.items():
        bound = se_bind(value_vec, field_name, schema, auto_add=auto_add)
        bindings.append(bound)

    if len(bindings) == 0:
        raise ValueError("At least one field required")

    result = bindings[0]
    for b in bindings[1:]:
        result = bundle(result, b)

    return result


def migrate_bundle(
    old_bundle: torch.Tensor,
    old_schema: SchemaRegistry,
    new_schema: SchemaRegistry,
    field_mapping: dict[str, str] | None = None
) -> torch.Tensor:
    """Migrate a bundle from old schema to new schema.

    For fields that exist in both schemas with the same canonical name,
    the binding is updated to use the new schema's field vector.

    For renamed fields (via field_mapping), unbind with old name and
    rebind with new name.

    Args:
        old_bundle: Bundle encoded with old schema
        old_schema: The schema used to encode old_bundle
        new_schema: Target schema for migration
        field_mapping: Optional explicit mapping of old_field -> new_field

    Returns:
        Bundle re-encoded with new schema

    Example:
        # Field was renamed from "qty" to "quantity_on_hand"
        new_bundle = migrate_bundle(
            old_bundle, v1_schema, v2_schema,
            field_mapping={"qty": "quantity_on_hand"}
        )
    """
    field_mapping = field_mapping or {}
    result = old_bundle.clone()

    # Process explicit mappings first
    for old_field, new_field in field_mapping.items():
        if old_schema.has_field(old_field) and new_schema.has_field(new_field):
            # Unbind with old field vector
            old_vec = old_schema.get_vector(old_field)
            value = unbind(result, old_vec)

            # Rebind with new field vector
            new_vec = new_schema.get_vector(new_field)
            rebound = bind(new_vec, value)

            # Update result (bundle to add new, unbind to remove old)
            result = bundle(unbind(result, old_vec), rebound)

    # Handle fields that exist in both schemas with same canonical name
    # but potentially different vectors (e.g., re-seeded)
    for canonical in old_schema.list_fields():
        if canonical in field_mapping:
            continue  # Already handled

        if new_schema.has_field(canonical):
            old_vec = old_schema.get_vector(canonical)
            new_vec = new_schema.get_vector(canonical)

            # Only migrate if vectors differ
            sim_val = similarity(old_vec, new_vec)
            sim_val = sim_val.abs() if hasattr(sim_val, 'abs') else abs(sim_val)
            if float(sim_val) < 0.99:
                value = unbind(result, old_vec)
                rebound = bind(new_vec, value)
                result = bundle(unbind(result, old_vec), rebound)

    return normalize(result)


def schema_compatibility_check(
    schema_a: SchemaRegistry,
    schema_b: SchemaRegistry
) -> dict[str, Any]:
    """Check compatibility between two schemas.

    Args:
        schema_a: First schema
        schema_b: Second schema

    Returns:
        Compatibility report with:
        - common_fields: Fields in both schemas
        - only_a: Fields only in schema_a
        - only_b: Fields only in schema_b
        - vector_similarity: Similarity of common field vectors
    """
    fields_a = set(schema_a.list_fields())
    fields_b = set(schema_b.list_fields())

    common = fields_a & fields_b
    only_a = fields_a - fields_b
    only_b = fields_b - fields_a

    # Check vector similarity for common fields
    vector_sims = {}
    for field in common:
        vec_a = schema_a.get_vector(field)
        vec_b = schema_b.get_vector(field)
        sim_val = similarity(vec_a, vec_b)
        sim_val = sim_val.abs() if hasattr(sim_val, 'abs') else abs(sim_val)
        vector_sims[field] = float(sim_val)

    return {
        "common_fields": list(common),
        "only_a": list(only_a),
        "only_b": list(only_b),
        "vector_similarity": vector_sims,
        "compatible": len(only_a) == 0 and len(only_b) == 0,
        "migration_required": any(s < 0.99 for s in vector_sims.values())
    }


# =============================================================================
# Standard Retail Schema Templates
# =============================================================================

def create_retail_schema(
    version: str = "retail-v1",
    dimensions: int | None = None
) -> SchemaRegistry:
    """Create standard retail inventory schema.

    Includes common POS field names and their aliases based on
    real-world retail systems (IdoSoft, Lightspeed, Square, etc.)

    Args:
        version: Schema version string
        dimensions: VSA dimensionality (default: from global config)

    Returns:
        Pre-configured SchemaRegistry for retail data
    """
    # Import here to avoid circular dependency
    from .vectors import get_config

    if dimensions is None:
        dimensions = get_config().dimensions

    schema = SchemaRegistry(version, dimensions)

    # SKU/Item identification
    schema.add_field("sku", aliases=[
        "item", "product_id", "productid", "item_id", "itemid",
        "upc", "barcode", "item_number", "item_no"
    ])

    # Quantity fields
    schema.add_field("quantity", aliases=[
        "qty", "on_hand", "qoh", "quantity_on_hand", "stock",
        "inventory", "inv_qty", "count"
    ])

    # Financial fields
    schema.add_field("cost", aliases=[
        "unit_cost", "cost_price", "wholesale", "purchase_price",
        "cogs", "cost_of_goods"
    ])

    schema.add_field("price", aliases=[
        "retail_price", "sell_price", "selling_price", "unit_price",
        "retail", "srp", "msrp"
    ])

    schema.add_field("margin", aliases=[
        "profit_margin", "gross_margin", "markup", "margin_pct",
        "gm", "gp_margin"
    ])

    schema.add_field("revenue", aliases=[
        "sales", "total_sales", "sales_revenue", "gross_sales"
    ])

    # Sales activity
    schema.add_field("sold", aliases=[
        "quantity_sold", "qty_sold", "units_sold", "sold_qty",
        "sales_qty", "sold_count"
    ])

    # Categorization
    schema.add_field("category", aliases=[
        "cat", "product_category", "item_category", "class",
        "product_class"
    ])

    schema.add_field("department", aliases=[
        "dept", "division", "dept_name", "department_name"
    ])

    schema.add_field("vendor", aliases=[
        "supplier", "manufacturer", "brand", "vendor_name",
        "supplier_name", "mfr"
    ])

    # Descriptions
    schema.add_field("description", aliases=[
        "desc", "item_desc", "item_description", "product_name",
        "name", "item_name"
    ])

    return schema
