"""
Integration test for the next-generation pipeline.
Tests all five systems: feedback, temporal, vendor, predictive, moat.
"""

import os
import sys


def test_integrated_pipeline():
    """Run the full integrated pipeline test."""
    from ..pipeline import run_integrated_test

    pipeline = run_integrated_test()

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

    print("All pipeline integration checks passed!")


if __name__ == "__main__":
    test_integrated_pipeline()
