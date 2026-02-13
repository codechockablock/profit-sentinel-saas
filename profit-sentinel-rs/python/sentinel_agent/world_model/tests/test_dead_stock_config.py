"""
Tests for dead stock configuration and lifecycle tracking.
"""


def test_config_presets():
    """All presets validate cleanly."""
    from ..config import ConfigPresets

    for name, preset in ConfigPresets.all_presets().items():
        errors = preset.validate()
        assert len(errors) == 0, f"Preset '{name}' has errors: {errors}"


def test_tier_classification():
    """Items are classified into correct tiers."""
    from ..config import DeadStockConfig, DeadStockTier

    config = DeadStockConfig()  # Default 60/120/180/360

    assert config.classify(30).tier == DeadStockTier.ACTIVE
    assert config.classify(60).tier == DeadStockTier.WATCHLIST
    assert config.classify(120).tier == DeadStockTier.ATTENTION
    assert config.classify(180).tier == DeadStockTier.ACTION_REQUIRED
    assert config.classify(360).tier == DeadStockTier.WRITEOFF


def test_category_overrides():
    """Per-category thresholds override globals."""
    from ..config import ConfigPresets, DeadStockTier

    config = ConfigPresets.hardware_store()

    # Regular item at 45 days = ACTIVE (global watchlist is 60)
    assert config.classify(45, category="Hardware").tier == DeadStockTier.ACTIVE

    # Seasonal item at 45 days = WATCHLIST (seasonal watchlist is 30)
    assert config.classify(45, category="Seasonal").tier == DeadStockTier.WATCHLIST

    # Commercial item at 100 days = WATCHLIST (commercial watchlist is 90)
    assert (
        config.classify(100, category="Commercial Hardware").tier
        == DeadStockTier.WATCHLIST
    )


def test_capital_threshold():
    """Items below capital threshold don't alert."""
    from ..config import DeadStockConfig

    config = DeadStockConfig(min_capital_threshold=50.0)

    # $2.50 at risk — below threshold
    result = config.classify(200, current_stock=10, unit_cost=0.25)
    assert not result.should_alert

    # $500 at risk — above threshold
    result = config.classify(200, current_stock=10, unit_cost=50.0)
    assert result.should_alert


def test_serialization_roundtrip():
    """Config survives JSON serialization."""
    from ..config import ConfigPresets, DeadStockConfig

    original = ConfigPresets.hardware_store()
    serialized = original.to_dict()
    restored = DeadStockConfig.from_dict(serialized)

    assert restored.global_thresholds.watchlist_days == 60
    assert "Seasonal" in restored.category_overrides
    assert restored.category_overrides["Seasonal"].watchlist_days == 30


def test_validation_catches_bad_config():
    """Validation rejects structurally broken configs."""
    from ..config import DeadStockThresholds

    bad = DeadStockThresholds(
        watchlist_days=100,
        attention_days=50,  # Lower than watchlist — invalid
        action_days=200,
        writeoff_days=300,
    )
    errors = bad.validate()
    assert len(errors) > 0


def test_lifecycle_tracker():
    """Lifecycle tracker records tier transitions."""
    from ..config import DeadStockConfig, DeadStockTier, InventoryLifecycleTracker

    config = DeadStockConfig()
    tracker = InventoryLifecycleTracker(config)

    # Item enters watchlist (ACTIVE → WATCHLIST = 1 escalation)
    result_watch = config.classify(65, current_stock=10, unit_cost=50.0)
    tracker.update_item("sku_001", result_watch, timestamp=1000.0)
    assert tracker.current_tiers["sku_001"] == DeadStockTier.WATCHLIST
    assert tracker.escalations == 1

    # Item escalates to attention (WATCHLIST → ATTENTION = 2 escalations total)
    result_attn = config.classify(125, current_stock=10, unit_cost=50.0)
    tracker.update_item("sku_001", result_attn, timestamp=2000.0)
    assert tracker.current_tiers["sku_001"] == DeadStockTier.ATTENTION
    assert tracker.escalations == 2

    # Item recovers (sold!)
    result_active = config.classify(5, current_stock=3, unit_cost=50.0)
    tracker.update_item("sku_001", result_active, timestamp=3000.0)
    assert tracker.current_tiers["sku_001"] == DeadStockTier.ACTIVE
    assert tracker.recoveries == 1


def test_full_integration():
    """Hardware store scenario: classify items, track lifecycle, round-trip."""
    import time

    from ..config import (
        ConfigPresets,
        DeadStockConfig,
        DeadStockTier,
        InventoryLifecycleTracker,
    )

    config = ConfigPresets.hardware_store()
    assert len(config.validate()) == 0

    test_items = [
        ("Active paint brush", 15, 30, 6.50, 2.5, "Paint Supplies"),
        ("Slow cabinet pulls", 50, 47, 12.50, 0.3, "Hardware"),
        ("Watchlist deadbolt", 65, 23, 34.00, 0.0, "Hardware"),
        ("Attention copper pipe", 125, 85, 8.75, 0.0, "Plumbing"),
        ("Action needed anchors", 185, 30, 18.00, 0.0, "Fasteners"),
        ("Writeoff smart home", 400, 12, 85.00, 0.0, "Electrical"),
        ("Dead xmas lights", 45, 50, 3.00, 0.0, "Seasonal"),
        ("Slow commercial lock", 100, 5, 250.00, 0.1, "Commercial Hardware"),
        ("Cheap washers", 200, 10, 0.25, 0.0, "Fasteners"),
    ]

    tracker = InventoryLifecycleTracker(config)
    base_time = time.time()

    for desc, days, stock, cost, velocity, category in test_items:
        result = config.classify(
            days_since_last_sale=days,
            current_stock=stock,
            unit_cost=cost,
            velocity=velocity,
            category=category,
        )
        entity_key = desc.replace(" ", "_").lower()
        tracker.update_item(entity_key, result, base_time)

    # Seasonal items flag earlier
    assert (
        config.classify(45, 50, 3.00, 0.0, "Seasonal").tier == DeadStockTier.WATCHLIST
    )
    assert (
        config.classify(65, 50, 3.00, 0.0, "Seasonal").tier == DeadStockTier.ATTENTION
    )
    # Regular items use global thresholds
    assert config.classify(45, 50, 3.00, 0.0, "Hardware").tier == DeadStockTier.ACTIVE
    # Commercial hardware is relaxed
    assert (
        config.classify(100, 5, 250.00, 0.1, "Commercial Hardware").tier
        == DeadStockTier.WATCHLIST
    )
    # Below capital threshold = no alert
    assert not config.classify(200, 10, 0.25, 0.0, "Fasteners").should_alert

    # Lifecycle report
    report = tracker.lifecycle_report()
    assert report["total_tracked"] > 0

    # Serialization round-trip
    serialized = config.to_dict()
    restored = DeadStockConfig.from_dict(serialized)
    assert restored.global_thresholds.watchlist_days == 60
    assert "Seasonal" in restored.category_overrides
    assert restored.category_overrides["Seasonal"].watchlist_days == 30

    # All presets exist
    assert len(ConfigPresets.all_presets()) >= 3
