"""Tests for the VendorRebateTracker."""

from datetime import date

import pytest
from sentinel_agent.coop_models import (
    CoopAlertType,
    RebateTier,
    VendorRebateProgram,
)
from sentinel_agent.vendor_rebates import (
    SAMPLE_REBATE_PROGRAMS,
    VendorRebateTracker,
)


def _make_program(
    vendor_id: str = "DMG",
    vendor_name: str = "Martin's Supply",
    tiers: list[dict] | None = None,
) -> VendorRebateProgram:
    if tiers is None:
        tiers = [
            {"tier_name": "Bronze", "threshold": 10000, "rebate_pct": 0.02},
            {"tier_name": "Silver", "threshold": 25000, "rebate_pct": 0.035},
            {"tier_name": "Gold", "threshold": 50000, "rebate_pct": 0.05},
        ]
    return VendorRebateProgram(
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        program_name="Volume Incentive",
        program_type="volume",
        tiers=[RebateTier(**t) for t in tiers],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        category="General Hardware",
    )


class TestProgramEvaluation:
    def setup_method(self):
        self.tracker = VendorRebateTracker()
        self.program = _make_program()

    def test_no_tier_when_below_first(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=5000,
            as_of=date(2026, 6, 15),
        )
        assert status.current_tier is None
        assert status.next_tier is not None
        assert status.next_tier.tier_name == "Bronze"
        assert status.shortfall == 5000

    def test_bronze_tier_reached(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=12000,
            as_of=date(2026, 6, 15),
        )
        assert status.current_tier is not None
        assert status.current_tier.tier_name == "Bronze"
        assert status.next_tier.tier_name == "Silver"
        assert status.shortfall == 13000  # 25000 - 12000

    def test_silver_tier_reached(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=30000,
            as_of=date(2026, 6, 15),
        )
        assert status.current_tier.tier_name == "Silver"
        assert status.next_tier.tier_name == "Gold"

    def test_top_tier_reached(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=60000,
            as_of=date(2026, 6, 15),
        )
        assert status.current_tier.tier_name == "Gold"
        assert status.next_tier is None
        assert status.shortfall == 0
        assert status.on_track  # At top tier = always on track

    def test_on_track_calculation(self):
        # 6 months in, 30000 spent, pace = 60000/year => hits Gold
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=30000,
            as_of=date(2026, 7, 1),
        )
        assert status.on_track  # Pace to hit Gold

    def test_not_on_track(self):
        # 10 months in, only 8000 spent, pace = ~9600/year => misses Bronze
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=8000,
            as_of=date(2026, 11, 1),
        )
        assert not status.on_track

    def test_daily_run_rate(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=18000,
            as_of=date(2026, 7, 1),
        )
        # Jan 1 to Jul 1 = 181 days
        days_elapsed = (date(2026, 7, 1) - date(2026, 1, 1)).days
        expected_rate = 18000 / days_elapsed
        assert abs(status.daily_run_rate - expected_rate) < 1

    def test_projected_total(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=18000,
            as_of=date(2026, 7, 1),
        )
        # Projected should be ytd + daily_rate × days_remaining
        assert status.projected_total > 18000

    def test_current_rebate_value(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=15000,
            as_of=date(2026, 6, 15),
        )
        # At Bronze (2%), value = 15000 × 0.02 = 300
        assert abs(status.current_rebate_value - 300) < 0.01

    def test_incremental_value(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=15000,
            as_of=date(2026, 6, 15),
        )
        # Next tier: Silver at 25000, 3.5%
        # Next rebate value: 25000 × 0.035 = 875
        # Current: 15000 × 0.02 = 300
        # Incremental: 875 - 300 = 575
        assert abs(status.incremental_value - 575) < 1


class TestRecommendations:
    def setup_method(self):
        self.tracker = VendorRebateTracker()
        self.program = _make_program()

    def test_on_track_recommendation(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=30000,
            as_of=date(2026, 7, 1),
        )
        assert "On track" in status.recommendation

    def test_at_risk_recommendation(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=8000,
            as_of=date(2026, 11, 1),
        )
        assert "At risk" in status.recommendation
        assert "ROI" in status.recommendation

    def test_top_tier_recommendation(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=60000,
            as_of=date(2026, 6, 15),
        )
        assert "top tier" in status.recommendation.lower()

    def test_recommendation_has_dollars(self):
        status = self.tracker.evaluate_program(
            self.program,
            "store-7",
            ytd_purchases=20000,
            as_of=date(2026, 11, 1),
        )
        assert "$" in status.recommendation


class TestEvaluateAll:
    def setup_method(self):
        self.tracker = VendorRebateTracker()

    def test_evaluates_matching_programs(self):
        vendor_ytd = {"DMG": 18000, "PNT": 25000}
        statuses = self.tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 6, 15),
        )
        # Should match DMG and PNT programs from SAMPLE_REBATE_PROGRAMS
        vendor_ids = [s.program.vendor_id for s in statuses]
        assert "DMG" in vendor_ids
        assert "PNT" in vendor_ids

    def test_skips_unmatched_vendors(self):
        vendor_ytd = {"XYZ": 10000}  # No matching program
        statuses = self.tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 6, 15),
        )
        assert len(statuses) == 0


class TestAlertGeneration:
    def setup_method(self):
        self.tracker = VendorRebateTracker()

    def test_generates_at_risk_alerts(self):
        # Late in year, low purchases => at risk
        vendor_ytd = {"DMG": 8000, "PNT": 5000}
        statuses = self.tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 11, 1),
        )
        alerts = self.tracker.generate_alerts(statuses)
        assert len(alerts) >= 1
        assert all(a.alert_type == CoopAlertType.REBATE_THRESHOLD_RISK for a in alerts)

    def test_no_alerts_when_on_track(self):
        # Strong pace => on track
        vendor_ytd = {"DMG": 40000, "PNT": 50000}
        statuses = self.tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 7, 1),
        )
        alerts = self.tracker.generate_alerts(statuses)
        # Should have few or no alerts since on track
        at_risk = [
            a for a in alerts if a.alert_type == CoopAlertType.REBATE_THRESHOLD_RISK
        ]
        # All on track => no threshold risk alerts
        assert len(at_risk) == 0

    def test_alert_has_vendor_name(self):
        vendor_ytd = {"DMG": 5000}
        statuses = self.tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 11, 1),
        )
        alerts = self.tracker.generate_alerts(statuses)
        for alert in alerts:
            assert "Martin" in alert.title or "Supply" in alert.title

    def test_alerts_sorted_by_value(self):
        vendor_ytd = {"DMG": 5000, "PNT": 8000, "ELC": 3000}
        statuses = self.tracker.evaluate_all(
            "store-7",
            vendor_ytd,
            as_of=date(2026, 11, 1),
        )
        alerts = self.tracker.generate_alerts(statuses)
        for i in range(len(alerts) - 1):
            assert alerts[i].dollar_impact >= alerts[i + 1].dollar_impact


class TestSamplePrograms:
    def test_sample_programs_exist(self):
        assert len(SAMPLE_REBATE_PROGRAMS) >= 3

    def test_all_have_tiers(self):
        for prog in SAMPLE_REBATE_PROGRAMS:
            assert len(prog.tiers) >= 2

    def test_tiers_ascending(self):
        for prog in SAMPLE_REBATE_PROGRAMS:
            for i in range(len(prog.tiers) - 1):
                assert prog.tiers[i].threshold < prog.tiers[i + 1].threshold
                assert prog.tiers[i].rebate_pct < prog.tiers[i + 1].rebate_pct
