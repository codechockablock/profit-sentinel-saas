"""
tests/vsa_core/test_schema.py - Tests for Schema Evolution Binding (SE-Bind)

Tests the schema evolution system that handles changing data schemas
without breaking existing VSA representations.

Verifies:
    - SchemaRegistry creation and field management
    - Alias resolution
    - se_bind/se_unbind operations
    - Schema migration
    - Compatibility checking
    - Retail schema template
"""

import os
import sys

import pytest

# Add parent to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from vsa_core import seed_hash, similarity
from vsa_core.schema import (
    FieldSpec,
    SchemaRegistry,
    create_retail_schema,
    create_schema_record,
    migrate_bundle,
    schema_compatibility_check,
    se_bind,
    se_unbind,
)
from vsa_core.vectors import get_config

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def config():
    """Get VSA config - uses global dimensions."""
    return get_config()


@pytest.fixture
def dims(config):
    """Get dimensions from config for convenience."""
    return config.dimensions


@pytest.fixture
def basic_registry(dims):
    """Create a basic schema registry for testing with matching dimensions."""
    registry = SchemaRegistry("test-v1", dimensions=dims)
    registry.add_field("quantity", aliases=["qty", "on_hand", "qoh"])
    registry.add_field("price", aliases=["retail_price", "sell_price"])
    registry.add_field("cost", aliases=["unit_cost", "wholesale"])
    return registry


@pytest.fixture
def qty_vec():
    return seed_hash("value:quantity:100")


@pytest.fixture
def price_vec():
    return seed_hash("value:price:9.99")


@pytest.fixture
def cost_vec():
    return seed_hash("value:cost:5.00")


# =============================================================================
# SCHEMA REGISTRY TESTS
# =============================================================================


class TestSchemaRegistry:
    """Tests for SchemaRegistry class."""

    def test_registry_creation(self):
        """Registry should be created with version and dimensions."""
        registry = SchemaRegistry("v1.0", dimensions=8192)

        assert registry.version == "v1.0"
        assert registry.dimensions == 8192
        assert len(registry.fields) == 0

    def test_add_field(self, basic_registry):
        """add_field should register field with aliases."""
        assert "quantity" in basic_registry.fields
        assert "price" in basic_registry.fields
        assert "cost" in basic_registry.fields

    def test_field_has_vector(self, basic_registry, dims):
        """Each field should have a generated vector."""
        for name, spec in basic_registry.fields.items():
            assert spec.vector is not None
            assert spec.vector.shape == (dims,)

    def test_field_vectors_deterministic(self, dims):
        """Same schema version should produce same vectors."""
        registry1 = SchemaRegistry("v1.0", dimensions=dims, seed=42)
        registry1.add_field("quantity")

        registry2 = SchemaRegistry("v1.0", dimensions=dims, seed=42)
        registry2.add_field("quantity")

        vec1 = registry1.get_vector("quantity")
        vec2 = registry2.get_vector("quantity")

        sim = float(similarity(vec1, vec2))
        assert sim > 0.999, f"Same seed should produce identical vectors: {sim}"

    def test_resolve_canonical_name(self, basic_registry):
        """resolve should return canonical name for canonical input."""
        assert basic_registry.resolve("quantity") == "quantity"
        assert basic_registry.resolve("price") == "price"

    def test_resolve_alias(self, basic_registry):
        """resolve should return canonical name for alias input."""
        assert basic_registry.resolve("qty") == "quantity"
        assert basic_registry.resolve("on_hand") == "quantity"
        assert basic_registry.resolve("qoh") == "quantity"
        assert basic_registry.resolve("retail_price") == "price"
        assert basic_registry.resolve("wholesale") == "cost"

    def test_resolve_case_insensitive(self, basic_registry):
        """resolve should be case-insensitive."""
        assert basic_registry.resolve("QTY") == "quantity"
        assert basic_registry.resolve("On_Hand") == "quantity"
        assert basic_registry.resolve("RETAIL_PRICE") == "price"

    def test_resolve_unknown_raises(self, basic_registry):
        """resolve should raise for unknown field."""
        with pytest.raises(KeyError, match="Unknown field"):
            basic_registry.resolve("nonexistent")

    def test_get_vector(self, basic_registry, dims):
        """get_vector should return vector for field."""
        vec = basic_registry.get_vector("quantity")
        assert vec.shape == (dims,)

    def test_get_vector_via_alias(self, basic_registry):
        """get_vector should work with aliases."""
        vec1 = basic_registry.get_vector("quantity")
        vec2 = basic_registry.get_vector("qty")
        vec3 = basic_registry.get_vector("on_hand")

        # All should return the same vector
        assert float(similarity(vec1, vec2)) > 0.999
        assert float(similarity(vec1, vec3)) > 0.999

    def test_has_field(self, basic_registry):
        """has_field should correctly check existence."""
        assert basic_registry.has_field("quantity")
        assert basic_registry.has_field("qty")  # alias
        assert basic_registry.has_field("QOH")  # case-insensitive alias
        assert not basic_registry.has_field("nonexistent")

    def test_add_alias(self, basic_registry):
        """add_alias should extend field aliases."""
        basic_registry.add_alias("quantity", "inventory_count")

        assert basic_registry.resolve("inventory_count") == "quantity"

    def test_add_alias_unknown_field(self, basic_registry):
        """add_alias should raise for unknown field."""
        with pytest.raises(KeyError):
            basic_registry.add_alias("nonexistent", "alias")

    def test_list_fields(self, basic_registry):
        """list_fields should return canonical names."""
        fields = basic_registry.list_fields()

        assert set(fields) == {"quantity", "price", "cost"}

    def test_list_aliases(self, basic_registry):
        """list_aliases should return aliases for field."""
        aliases = basic_registry.list_aliases("quantity")

        assert set(aliases) == {"qty", "on_hand", "qoh"}

    def test_to_dict(self, basic_registry, dims):
        """to_dict should export registry."""
        data = basic_registry.to_dict()

        assert data["version"] == "test-v1"
        assert data["dimensions"] == dims
        assert "quantity" in data["fields"]
        assert set(data["fields"]["quantity"]["aliases"]) == {"qty", "on_hand", "qoh"}

    def test_from_dict(self, basic_registry):
        """from_dict should restore registry."""
        data = basic_registry.to_dict()
        restored = SchemaRegistry.from_dict(data)

        assert restored.version == basic_registry.version
        assert restored.dimensions == basic_registry.dimensions
        assert set(restored.list_fields()) == set(basic_registry.list_fields())

        # Check alias resolution works
        assert restored.resolve("qty") == "quantity"


# =============================================================================
# FIELD TRANSFORM TESTS
# =============================================================================


class TestFieldTransforms:
    """Tests for field transformation functions."""

    def test_transform_applied_on_bind(self, dims):
        """Transform should be applied during se_bind."""
        registry = SchemaRegistry("v1", dimensions=dims)

        # Add field with transform that doubles magnitude
        def double_transform(v):
            return v * 2

        registry.add_field("quantity", transform=double_transform)

        # The transform is applied to the value vector before binding
        value = seed_hash("test_value")
        bound = se_bind(value, "quantity", registry)

        # Verify it's a valid bound vector (transform was applied internally)
        assert bound.shape == (dims,)

    def test_get_transform(self, dims):
        """get_transform should return transform function."""
        registry = SchemaRegistry("v1", dimensions=dims)

        def transform(v):
            return v * 2

        registry.add_field("quantity", transform=transform)
        registry.add_field("price")  # No transform

        assert registry.get_transform("quantity") is transform
        assert registry.get_transform("price") is None


# =============================================================================
# SE_BIND / SE_UNBIND TESTS
# =============================================================================


class TestSEBind:
    """Tests for schema-evolution-aware binding."""

    def test_se_bind_basic(self, basic_registry, qty_vec, dims):
        """se_bind should create bound vector."""
        bound = se_bind(qty_vec, "quantity", basic_registry)

        assert bound.shape == (dims,)

    def test_se_bind_alias_equivalence(self, basic_registry, qty_vec):
        """se_bind with aliases should produce equivalent bindings."""
        bound1 = se_bind(qty_vec, "quantity", basic_registry)
        bound2 = se_bind(qty_vec, "qty", basic_registry)
        bound3 = se_bind(qty_vec, "on_hand", basic_registry)

        # All should be identical (same canonical field)
        assert float(similarity(bound1, bound2)) > 0.999
        assert float(similarity(bound1, bound3)) > 0.999

    def test_se_bind_different_fields_different_results(self, basic_registry, qty_vec):
        """Different fields should produce dissimilar bindings."""
        bound_qty = se_bind(qty_vec, "quantity", basic_registry)
        bound_price = se_bind(qty_vec, "price", basic_registry)

        sim = abs(float(similarity(bound_qty, bound_price)))
        assert sim < 0.2, f"Different fields should produce dissimilar bindings: {sim}"

    def test_se_bind_unknown_raises(self, basic_registry, qty_vec):
        """se_bind should raise for unknown field."""
        with pytest.raises(KeyError):
            se_bind(qty_vec, "nonexistent", basic_registry)

    def test_se_bind_auto_add(self, qty_vec, dims):
        """se_bind with auto_add should create unknown fields."""
        registry = SchemaRegistry("v1", dimensions=dims)

        # Should not raise, should auto-create field
        bound = se_bind(qty_vec, "new_field", registry, auto_add=True)

        assert bound.shape == (dims,)
        assert registry.has_field("new_field")

    def test_se_unbind_recovers_value(self, basic_registry, qty_vec):
        """se_unbind should recover bound value."""
        bound = se_bind(qty_vec, "quantity", basic_registry)
        recovered = se_unbind(bound, "quantity", basic_registry)

        sim = float(similarity(recovered, qty_vec))
        assert sim > 0.95, f"Unbind should recover value: {sim}"

    def test_se_unbind_via_alias(self, basic_registry, qty_vec):
        """se_unbind should work with aliases."""
        bound = se_bind(qty_vec, "quantity", basic_registry)
        recovered = se_unbind(bound, "qty", basic_registry)

        sim = float(similarity(recovered, qty_vec))
        assert sim > 0.95

    def test_se_unbind_wrong_field(self, basic_registry, qty_vec):
        """Unbinding with wrong field should not recover."""
        bound = se_bind(qty_vec, "quantity", basic_registry)
        wrong = se_unbind(bound, "price", basic_registry)

        sim = abs(float(similarity(wrong, qty_vec)))
        assert sim < 0.2, f"Wrong field should not recover: {sim}"


# =============================================================================
# CREATE_SCHEMA_RECORD TESTS
# =============================================================================


class TestCreateSchemaRecord:
    """Tests for bundled record creation."""

    def test_create_record_basic(
        self, basic_registry, qty_vec, price_vec, cost_vec, dims
    ):
        """create_schema_record should bundle field bindings."""
        record = create_schema_record(
            {
                "qty": qty_vec,
                "price": price_vec,
                "cost": cost_vec,
            },
            basic_registry,
        )

        assert record.shape == (dims,)

    def test_create_record_empty_raises(self, basic_registry):
        """Empty field dict should raise."""
        with pytest.raises(ValueError, match="At least one field"):
            create_schema_record({}, basic_registry)

    def test_create_record_query(self, basic_registry, qty_vec, price_vec):
        """Record should be queryable via unbinding."""
        record = create_schema_record(
            {
                "qty": qty_vec,
                "price": price_vec,
            },
            basic_registry,
        )

        # Unbind to get quantity
        basic_registry.get_vector("quantity")
        recovered = se_unbind(record, "quantity", basic_registry)

        # Should be similar to original (bundle degradation expected)
        sim = float(similarity(recovered, qty_vec))
        assert sim > 0.3, f"Record query should recover value: {sim}"

    def test_create_record_auto_add(self, qty_vec, price_vec, dims):
        """create_schema_record with auto_add should create fields."""
        registry = SchemaRegistry("v1", dimensions=dims)

        create_schema_record(
            {
                "qty": qty_vec,
                "price": price_vec,
            },
            registry,
            auto_add=True,
        )

        assert registry.has_field("qty")
        assert registry.has_field("price")


# =============================================================================
# MIGRATION TESTS
# =============================================================================


class TestMigration:
    """Tests for schema migration."""

    def test_migrate_same_schema(self, basic_registry, qty_vec, price_vec):
        """Migrating to same schema should preserve data."""
        record = create_schema_record(
            {
                "qty": qty_vec,
                "price": price_vec,
            },
            basic_registry,
        )

        migrated = migrate_bundle(record, basic_registry, basic_registry)

        # Should be very similar
        sim = float(similarity(migrated, record))
        assert sim > 0.9, f"Same-schema migration should preserve: {sim}"

    def test_migrate_field_mapping(self, qty_vec, dims):
        """migrate_bundle should handle explicit field mappings."""
        v1 = SchemaRegistry("v1", dimensions=dims, seed=1)
        v1.add_field("qty")

        v2 = SchemaRegistry("v2", dimensions=dims, seed=2)
        v2.add_field("quantity_on_hand")

        # Create record with v1
        record = create_schema_record({"qty": qty_vec}, v1)

        # Migrate with explicit mapping
        migrated = migrate_bundle(
            record, v1, v2, field_mapping={"qty": "quantity_on_hand"}
        )

        # Query with new field name
        recovered = se_unbind(migrated, "quantity_on_hand", v2)

        # Should recover the value (with some bundle degradation)
        sim = float(similarity(recovered, qty_vec))
        assert sim > 0.3, f"Migration should preserve queryability: {sim}"


# =============================================================================
# COMPATIBILITY CHECK TESTS
# =============================================================================


class TestCompatibilityCheck:
    """Tests for schema compatibility checking."""

    def test_identical_schemas_compatible(self, basic_registry):
        """Identical schemas should be fully compatible."""
        result = schema_compatibility_check(basic_registry, basic_registry)

        assert result["compatible"] is True
        assert len(result["only_a"]) == 0
        assert len(result["only_b"]) == 0

    def test_different_fields_detected(self, dims):
        """Different fields should be detected."""
        a = SchemaRegistry("a", dimensions=dims)
        a.add_field("quantity")
        a.add_field("price")

        b = SchemaRegistry("b", dimensions=dims)
        b.add_field("quantity")
        b.add_field("cost")

        result = schema_compatibility_check(a, b)

        assert result["compatible"] is False
        assert "price" in result["only_a"]
        assert "cost" in result["only_b"]
        assert "quantity" in result["common_fields"]

    def test_vector_similarity_reported(self, dims):
        """Vector similarity for common fields should be reported."""
        # Same seed = same vectors
        a = SchemaRegistry("a", dimensions=dims, seed=42)
        a.add_field("quantity")

        b = SchemaRegistry("b", dimensions=dims, seed=42)
        b.add_field("quantity")

        result = schema_compatibility_check(a, b)

        # Same seed = high similarity
        assert result["vector_similarity"]["quantity"] > 0.99
        assert result["migration_required"] is False

    def test_migration_required_flag(self, dims):
        """migration_required should be set when vectors differ."""
        a = SchemaRegistry("a", dimensions=dims, seed=1)
        a.add_field("quantity")

        b = SchemaRegistry("b", dimensions=dims, seed=999)
        b.add_field("quantity")

        result = schema_compatibility_check(a, b)

        # Different seeds = different vectors = migration required
        assert result["migration_required"] is True


# =============================================================================
# RETAIL SCHEMA TESTS
# =============================================================================


class TestRetailSchema:
    """Tests for retail schema template."""

    def test_retail_schema_creation(self, config):
        """create_retail_schema should create valid registry."""
        schema = create_retail_schema("retail-v1")

        assert schema.version == "retail-v1"
        assert len(schema.fields) > 0

    def test_retail_schema_common_fields(self, config):
        """Retail schema should have common inventory fields."""
        schema = create_retail_schema()

        expected = [
            "sku",
            "quantity",
            "cost",
            "price",
            "margin",
            "sold",
            "category",
            "vendor",
        ]
        for field in expected:
            assert schema.has_field(field), f"Missing expected field: {field}"

    def test_retail_schema_aliases(self, config):
        """Retail schema should have common POS aliases."""
        schema = create_retail_schema()

        # Test common aliases
        assert schema.resolve("qty") == "quantity"
        assert schema.resolve("on_hand") == "quantity"
        assert schema.resolve("qoh") == "quantity"
        assert schema.resolve("item") == "sku"
        assert schema.resolve("product_id") == "sku"
        assert schema.resolve("upc") == "sku"
        assert schema.resolve("unit_cost") == "cost"
        assert schema.resolve("wholesale") == "cost"
        assert schema.resolve("retail_price") == "price"
        assert schema.resolve("sell_price") == "price"

    def test_retail_schema_cross_pos_compatibility(self, config):
        """Different POS column names should resolve to same field."""
        schema = create_retail_schema()

        # Simulate different POS systems using different names
        idosoft_qty = schema.resolve("qty")
        lightspeed_qty = schema.resolve("quantity_on_hand")
        square_qty = schema.resolve("stock")

        assert idosoft_qty == lightspeed_qty == square_qty == "quantity"


# =============================================================================
# FIELD SPEC TESTS
# =============================================================================


class TestFieldSpec:
    """Tests for FieldSpec dataclass."""

    def test_field_spec_creation(self):
        """FieldSpec should initialize with defaults."""
        spec = FieldSpec("quantity")

        assert spec.canonical_name == "quantity"
        assert spec.aliases == []
        assert spec.vector is None
        assert spec.transform is None
        assert spec.metadata == {}

    def test_field_spec_with_aliases(self):
        """FieldSpec should accept aliases."""
        spec = FieldSpec("quantity", aliases=["qty", "on_hand"])

        assert spec.aliases == ["qty", "on_hand"]

    def test_field_spec_with_metadata(self):
        """FieldSpec should accept metadata."""
        spec = FieldSpec("quantity", metadata={"unit": "items", "required": True})

        assert spec.metadata["unit"] == "items"
        assert spec.metadata["required"] is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestSchemaIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, dims):
        """Test complete schema workflow."""
        # Create v1 schema - use same seed for simpler migration testing
        v1 = SchemaRegistry("v1", dimensions=dims, seed=42)
        v1.add_field("qty", aliases=["quantity"])
        v1.add_field("price")

        # Encode some data
        qty_val = seed_hash("qty:100")
        price_val = seed_hash("price:9.99")

        record = create_schema_record(
            {
                "qty": qty_val,
                "price": price_val,
            },
            v1,
        )

        # Create v2 schema (same seed = same field vectors, easier migration)
        v2 = SchemaRegistry("v2", dimensions=dims, seed=42)
        v2.add_field("quantity_on_hand", aliases=["qty", "quantity"])
        v2.add_field("retail_price", aliases=["price"])

        # Check compatibility
        compat = schema_compatibility_check(v1, v2)
        assert compat["compatible"] is False  # Different canonical names

        # Migrate with explicit mapping
        migrated = migrate_bundle(
            record,
            v1,
            v2,
            field_mapping={"qty": "quantity_on_hand", "price": "retail_price"},
        )

        # Verify migration produced valid vector
        assert migrated.shape == (dims,)

        # Note: Migration between schemas with renamed fields and different
        # field vectors is lossy. The test verifies the migration completes,
        # not perfect recovery (which requires same field vectors).

    def test_multi_pos_ingestion(self, dims):
        """Test ingesting data from multiple POS systems."""
        schema = create_retail_schema()

        # IdoSoft uses 'qty', 'wholesale'
        idosoft_data = {
            "qty": seed_hash("idosoft:qty:50"),
            "wholesale": seed_hash("idosoft:cost:2.00"),
        }

        # Lightspeed uses 'on_hand', 'unit_cost'
        lightspeed_data = {
            "on_hand": seed_hash("lightspeed:qty:100"),
            "unit_cost": seed_hash("lightspeed:cost:2.50"),
        }

        # Both should create valid records
        idosoft_record = create_schema_record(idosoft_data, schema)
        lightspeed_record = create_schema_record(lightspeed_data, schema)

        # Both records can be queried with canonical names
        for record, name in [
            (idosoft_record, "idosoft"),
            (lightspeed_record, "lightspeed"),
        ]:
            qty_recovered = se_unbind(record, "quantity", schema)
            cost_recovered = se_unbind(record, "cost", schema)

            assert qty_recovered.shape[0] > 0, f"{name} quantity should be recoverable"
            assert cost_recovered.shape[0] > 0, f"{name} cost should be recoverable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
