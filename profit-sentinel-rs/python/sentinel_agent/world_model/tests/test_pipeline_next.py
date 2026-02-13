"""
Integration test for the next-generation pipeline.
Tests all five systems: feedback, temporal, vendor, predictive, moat.
"""

import time

import numpy as np

from ..pipeline import Outcome, OutcomeType, SentinelPipeline, TimeScale


def _run_integrated_pipeline_scenario() -> SentinelPipeline:
    """Run a realistic multi-store scenario through the full pipeline."""
    pipeline = SentinelPipeline(dim=4096, seed=42)
    rng = np.random.RandomState(42)

    base_time = 1704067200  # Jan 1, 2024
    day = 86400
    stores = ["Store_1_Suburban", "Store_2_Contractor", "Store_3_Rural"]

    # 8 weeks of observations with both velocity decay and vendor drift.
    for week in range(8):
        for store_id in stores:
            timestamp = base_time + week * 7 * day

            velocity_decline = max(0.1, 8.0 - week * 1.2)
            pipeline.record_observation(
                store_id=store_id,
                entity_id="SKU-7742",
                observation={
                    "velocity": velocity_decline + rng.normal(0, 0.5),
                    "stock": 47 - week * 2,
                    "margin": 0.25,
                    "cost": 12.50,
                    "price": 24.99,
                    "vendor_id": "VENDOR_A",
                    "vendor_name": "Cabinet Supply Co",
                },
                timestamp=timestamp,
            )

            cost_increase = 8.75 + week * 0.15
            pipeline.record_observation(
                store_id=store_id,
                entity_id="SKU-5510",
                observation={
                    "velocity": 4.0 + rng.normal(0, 0.3),
                    "stock": 85,
                    "margin": (15.99 - cost_increase) / 15.99,
                    "cost": cost_increase,
                    "price": 15.99,
                    "vendor_id": "VENDOR_B",
                    "vendor_name": "Pipe & Fittings Inc",
                },
                timestamp=timestamp,
            )

            fill_rate = max(0.7, 0.96 - week * 0.03 + rng.normal(0, 0.02))
            pipeline.vendor_intel.record_delivery(
                "VENDOR_B", store_id, 100, int(100 * fill_rate), timestamp
            )

    for vendor_id in pipeline.vendor_intel.vendors:
        pipeline.vendor_intel.encode_vendor_behavior(vendor_id)

    for scale in [TimeScale.WEEKLY, TimeScale.MONTHLY]:
        pipeline.temporal.learn_patterns(scale)

    sim_time = base_time + 8 * 7 * day
    for store_id in stores:
        for entity_id in ["SKU-7742", "SKU-5510"]:
            pipeline.predict_interventions(store_id, entity_id, current_time=sim_time)

    pipeline.record_outcome(
        Outcome(
            outcome_id="OUT-001",
            outcome_type=OutcomeType.TRANSFER_SOLD,
            source_finding_id="FIND-001",
            timestamp=time.time(),
            entity_id="SKU-7742",
            store_id="Store_1_Suburban",
            dest_store_id="Store_2_Contractor",
            units_transferred=20,
            units_sold=18,
            days_to_sell=21,
            actual_recovery=449.82,
            predicted_recovery=499.80,
        )
    )
    pipeline.record_outcome(
        Outcome(
            outcome_id="OUT-002",
            outcome_type=OutcomeType.PREDICTION_WRONG,
            source_finding_id="PRED-001",
            timestamp=time.time(),
            entity_id="SKU-5510",
            store_id="Store_3_Rural",
            dollar_impact_actual=0,
            dollar_impact_predicted=500,
        )
    )

    pipeline.moat_snapshot(n_stores=len(stores), n_skus=2)
    pipeline.pipeline_status()
    return pipeline


def test_integrated_pipeline():
    """Run the full integrated pipeline test."""
    pipeline = _run_integrated_pipeline_scenario()

    # Verify feedback engine recorded outcomes
    assert (
        pipeline.feedback.total_value_recovered > 0
    ), "Feedback engine should have recorded value"
    assert (
        pipeline.feedback.transfer_success_rate > 0
    ), "Should have successful transfers"

    # Verify temporal patterns were learned
    total_patterns = sum(len(p) for p in pipeline.temporal.primitives.values())
    assert total_patterns > 0, "Should have learned temporal patterns"

    # Verify vendor intelligence detected alerts
    assert len(pipeline.vendor_intel.alerts) > 0, "Should have vendor alerts"
    assert len(pipeline.vendor_intel.vendors) == 2, "Should track 2 vendors"

    # Verify moat snapshot was captured
    assert len(pipeline.moat.snapshots) > 0, "Should have moat snapshots"
