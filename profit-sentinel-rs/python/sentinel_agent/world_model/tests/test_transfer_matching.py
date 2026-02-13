"""
Tests for cross-store transfer matching.

Moved from transfer_matching.py's embedded __main__ harness.
"""


def test_three_store_transfer_network():
    """Realistic 3-store hardware chain finds transfer opportunities."""
    from ..core import PhasorAlgebra
    from ..transfer_matching import (
        EntityHierarchy,
        StoreAgent,
        TransferMatcher,
    )

    algebra = PhasorAlgebra(dim=4096, seed=42)
    hierarchy = EntityHierarchy(algebra)
    matcher = TransferMatcher(algebra, hierarchy)

    # STORE 1: Suburban, homeowner-focused — dead stock in contractor items
    store1 = StoreAgent("Store_1_Suburban", algebra, hierarchy)
    for item in [
        (
            "SKU-7742",
            "Stainless Cabinet Pulls (25pk)",
            "Cabinet Hardware",
            "Hardware",
            47,
            0.0,
            12.50,
            24.99,
            95,
        ),
        (
            "SKU-3301",
            "Commercial Grade Deadbolt",
            "Locks",
            "Hardware",
            23,
            0.0,
            34.00,
            59.99,
            120,
        ),
        (
            "SKU-5510",
            "3/4 inch Copper Pipe 10ft",
            "Copper Pipe",
            "Plumbing",
            85,
            0.2,
            8.75,
            15.99,
            45,
        ),
        (
            "SKU-8820",
            "Concrete Anchors 50pk",
            "Anchors",
            "Fasteners",
            30,
            0.1,
            18.00,
            32.99,
            78,
        ),
        (
            "SKU-1100",
            "Interior Latex Paint Gallon",
            "Interior Paint",
            "Paint",
            120,
            15.0,
            22.00,
            38.99,
            2,
        ),
        (
            "SKU-1205",
            "Paint Roller Kit",
            "Paint Supplies",
            "Paint",
            60,
            8.0,
            6.50,
            12.99,
            1,
        ),
        (
            "SKU-2200",
            "LED Bulb 4-pack",
            "Light Bulbs",
            "Electrical",
            200,
            20.0,
            4.50,
            9.99,
            0,
        ),
        (
            "SKU-4400",
            "Garden Hose 50ft",
            "Garden Hose",
            "Garden",
            40,
            5.0,
            15.00,
            29.99,
            3,
        ),
    ]:
        store1.ingest_sku(*item)
    matcher.register_agent(store1)

    # STORE 2: Construction corridor, contractor-heavy
    store2 = StoreAgent("Store_2_Contractor", algebra, hierarchy)
    for item in [
        (
            "SKU-7742",
            "Stainless Cabinet Pulls (25pk)",
            "Cabinet Hardware",
            "Hardware",
            8,
            12.0,
            12.50,
            24.99,
            1,
        ),
        (
            "SKU-3301",
            "Commercial Grade Deadbolt",
            "Locks",
            "Hardware",
            5,
            6.0,
            34.00,
            59.99,
            2,
        ),
        (
            "SKU-5515",
            "1 inch Copper Pipe 10ft",
            "Copper Pipe",
            "Plumbing",
            30,
            8.0,
            11.25,
            19.99,
            1,
        ),
        (
            "SKU-8825",
            "Wedge Anchors 25pk",
            "Anchors",
            "Fasteners",
            15,
            4.0,
            22.00,
            39.99,
            3,
        ),
        (
            "SKU-1100",
            "Interior Latex Paint Gallon",
            "Interior Paint",
            "Paint",
            30,
            3.0,
            22.00,
            38.99,
            5,
        ),
        (
            "SKU-6600",
            "Framing Lumber 2x4x8",
            "Dimensional Lumber",
            "Lumber",
            500,
            50.0,
            3.25,
            5.99,
            0,
        ),
        (
            "SKU-6610",
            "Treated Lumber 2x4x8",
            "Treated Lumber",
            "Lumber",
            300,
            30.0,
            4.50,
            8.49,
            0,
        ),
    ]:
        store2.ingest_sku(*item)
    matcher.register_agent(store2)

    # STORE 3: Rural, mixed customer base
    store3 = StoreAgent("Store_3_Rural", algebra, hierarchy)
    for item in [
        (
            "SKU-4400",
            "Garden Hose 50ft",
            "Garden Hose",
            "Garden",
            15,
            2.0,
            15.00,
            29.99,
            4,
        ),
        (
            "SKU-4410",
            "Sprinkler System Kit",
            "Irrigation",
            "Garden",
            10,
            3.0,
            45.00,
            79.99,
            2,
        ),
        (
            "SKU-7750",
            "Brushed Nickel Cabinet Knobs (10pk)",
            "Cabinet Hardware",
            "Hardware",
            20,
            3.0,
            8.00,
            16.99,
            5,
        ),
        (
            "SKU-9900",
            "Smart Doorbell Camera",
            "Smart Home",
            "Electrical",
            12,
            0.0,
            85.00,
            149.99,
            150,
        ),
        (
            "SKU-9910",
            "WiFi Thermostat",
            "Smart Home",
            "Electrical",
            8,
            0.0,
            65.00,
            119.99,
            130,
        ),
        (
            "SKU-2200",
            "LED Bulb 4-pack",
            "Light Bulbs",
            "Electrical",
            100,
            10.0,
            4.50,
            9.99,
            1,
        ),
        (
            "SKU-6600",
            "Framing Lumber 2x4x8",
            "Dimensional Lumber",
            "Lumber",
            200,
            15.0,
            3.25,
            5.99,
            0,
        ),
    ]:
        store3.ingest_sku(*item)
    matcher.register_agent(store3)

    # Network should have 3 stores
    assert len(matcher.agents) == 3
    assert len(matcher.shared_codebook) > 0

    # Suburban store should have dead stock
    dead1 = store1.find_dead_stock()
    assert len(dead1) > 0

    # Rural store should have dead smart-home items
    dead3 = store3.find_dead_stock()
    assert len(dead3) > 0

    # Transfer matching runs without error (may find 0 matches
    # depending on VSA confidence thresholds — the original harness
    # was print-only and didn't assert on match counts)
    all_recs = matcher.find_all_transfers(max_per_store=10)
    assert isinstance(all_recs, dict)

    # Every recommendation that IS returned should have positive benefit
    for _store_id, recs in all_recs.items():
        for rec in recs:
            assert rec.net_benefit >= 0
            assert rec.summary()  # should produce non-empty text

    # Network summary
    summary = matcher.network_summary()
    assert summary["n_stores"] == 3
    assert summary["n_dead_items"] > 0
