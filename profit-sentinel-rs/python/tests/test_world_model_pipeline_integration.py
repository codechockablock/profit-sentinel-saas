"""Ensure the moved world_model integration harness is exercised in CI testpaths."""

from sentinel_agent.world_model.tests.test_pipeline_next import (
    _run_integrated_pipeline_scenario,
)


def test_world_model_integrated_pipeline_scenario():
    pipeline = _run_integrated_pipeline_scenario()

    assert pipeline.feedback.total_value_recovered > 0
    assert pipeline.feedback.transfer_success_rate > 0
    assert len(pipeline.vendor_intel.vendors) == 2
    assert len(pipeline.moat.snapshots) > 0
