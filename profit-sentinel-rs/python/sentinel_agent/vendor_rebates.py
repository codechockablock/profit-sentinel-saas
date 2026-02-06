"""Vendor Rebate Tracker.

Tracks progress toward vendor rebate thresholds, calculates the ROI
of accelerating purchases to hit the next tier, and alerts when a
store is at risk of missing a threshold.

Example alert:
  "You're $4,200 short of hitting Gold tier with Martin's Supply.
   Spend $4,200 more in 47 days → unlock $1,890 additional rebate.
   ROI: 45% on the incremental spend."
"""

from __future__ import annotations

from datetime import date, timedelta

from .coop_models import (
    CoopAlert,
    CoopAlertType,
    RebateTier,
    VendorRebateProgram,
    VendorRebateStatus,
)

# ---------------------------------------------------------------------------
# Stub rebate programs (production would load from vendor data)
# ---------------------------------------------------------------------------

SAMPLE_REBATE_PROGRAMS: list[VendorRebateProgram] = [
    VendorRebateProgram(
        vendor_id="DMG",
        vendor_name="Martin's Supply Co.",
        program_name="Volume Incentive",
        program_type="volume",
        tiers=[
            RebateTier(
                tier_name="Bronze",
                threshold=10000,
                rebate_pct=0.02,
                description="2% on all purchases",
            ),
            RebateTier(
                tier_name="Silver",
                threshold=25000,
                rebate_pct=0.035,
                description="3.5% on all purchases",
            ),
            RebateTier(
                tier_name="Gold",
                threshold=50000,
                rebate_pct=0.05,
                description="5% on all purchases",
            ),
        ],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        category="General Hardware",
    ),
    VendorRebateProgram(
        vendor_id="PNT",
        vendor_name="ColorMax Paint Supply",
        program_name="Paint Volume Rebate",
        program_type="volume",
        tiers=[
            RebateTier(
                tier_name="Standard",
                threshold=15000,
                rebate_pct=0.03,
                description="3% on paint purchases",
            ),
            RebateTier(
                tier_name="Preferred",
                threshold=35000,
                rebate_pct=0.05,
                description="5% on paint purchases",
            ),
            RebateTier(
                tier_name="Premium",
                threshold=60000,
                rebate_pct=0.07,
                description="7% on paint purchases",
            ),
        ],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        category="Paint",
    ),
    VendorRebateProgram(
        vendor_id="ELC",
        vendor_name="National Electrical Distributors",
        program_name="Growth Rebate",
        program_type="growth",
        tiers=[
            RebateTier(
                tier_name="Base",
                threshold=8000,
                rebate_pct=0.02,
                description="2% on electrical",
            ),
            RebateTier(
                tier_name="Growth",
                threshold=20000,
                rebate_pct=0.04,
                description="4% on electrical",
            ),
        ],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
        category="Electrical",
    ),
]


class VendorRebateTracker:
    """Track and optimize vendor rebate programs.

    Usage:
        tracker = VendorRebateTracker()
        status = tracker.evaluate_program(
            program=program,
            store_id="store-7",
            ytd_purchases=18500.0,
            as_of=date(2026, 6, 15),
        )
        print(f"Shortfall to next tier: {status.shortfall_display}")
    """

    def __init__(
        self,
        programs: list[VendorRebateProgram] | None = None,
    ):
        self.programs = programs or SAMPLE_REBATE_PROGRAMS

    # -----------------------------------------------------------------
    # Program Evaluation
    # -----------------------------------------------------------------

    def evaluate_program(
        self,
        program: VendorRebateProgram,
        store_id: str,
        ytd_purchases: float,
        as_of: date | None = None,
    ) -> VendorRebateStatus:
        """Evaluate current progress toward rebate tiers.

        Calculates current tier, next tier shortfall, projected total,
        and whether the store is on track to hit the next tier.

        Args:
            program: The vendor rebate program.
            store_id: Store ID.
            ytd_purchases: Year-to-date purchases from this vendor.
            as_of: Date for projection. Defaults to today.

        Returns:
            VendorRebateStatus with full tier analysis.
        """
        if as_of is None:
            as_of = date.today()

        # Find current and next tier
        current_tier = None
        next_tier = None
        sorted_tiers = sorted(program.tiers, key=lambda t: t.threshold)

        for tier in sorted_tiers:
            if ytd_purchases >= tier.threshold:
                current_tier = tier
            elif next_tier is None:
                next_tier = tier

        # If we've hit all tiers, next_tier stays None
        # If we haven't hit any, current_tier is None

        # Calculate shortfall
        shortfall = 0.0
        if next_tier is not None:
            shortfall = next_tier.threshold - ytd_purchases

        # Days remaining in the program period
        days_remaining = max((program.period_end - as_of).days, 1)
        days_elapsed = max((as_of - program.period_start).days, 1)

        # Daily run rate and projection
        daily_run_rate = ytd_purchases / days_elapsed
        projected_total = ytd_purchases + (daily_run_rate * days_remaining)

        # On track?
        on_track = (
            next_tier is None  # Already at top tier
            or projected_total >= next_tier.threshold
        )

        # Current rebate value
        current_rate = current_tier.rebate_pct if current_tier else 0.0
        current_rebate_value = ytd_purchases * current_rate

        # Next tier rebate value (projected)
        next_rate = next_tier.rebate_pct if next_tier else current_rate
        next_tier_threshold = next_tier.threshold if next_tier else ytd_purchases
        next_tier_rebate_value = next_tier_threshold * next_rate

        # Incremental value of hitting next tier
        incremental_value = next_tier_rebate_value - current_rebate_value

        # Build recommendation
        recommendation = self._build_recommendation(
            program,
            current_tier,
            next_tier,
            shortfall,
            days_remaining,
            daily_run_rate,
            on_track,
            incremental_value,
        )

        return VendorRebateStatus(
            program=program,
            store_id=store_id,
            ytd_purchases=ytd_purchases,
            current_tier=current_tier,
            next_tier=next_tier,
            shortfall=shortfall,
            days_remaining=days_remaining,
            daily_run_rate=daily_run_rate,
            projected_total=projected_total,
            on_track=on_track,
            current_rebate_value=current_rebate_value,
            next_tier_rebate_value=next_tier_rebate_value,
            incremental_value=incremental_value,
            recommendation=recommendation,
        )

    def _build_recommendation(
        self,
        program: VendorRebateProgram,
        current_tier: RebateTier | None,
        next_tier: RebateTier | None,
        shortfall: float,
        days_remaining: int,
        daily_run_rate: float,
        on_track: bool,
        incremental_value: float,
    ) -> str:
        """Build a specific, dollar-quantified recommendation."""
        if next_tier is None:
            return (
                f"Already at top tier ({current_tier.tier_name}) with "
                f"{program.vendor_name}. Earning {current_tier.rebate_pct * 100:.1f}%."
            )

        if on_track:
            return (
                f"On track to hit {next_tier.tier_name} tier "
                f"({next_tier.rebate_pct * 100:.1f}%) with {program.vendor_name}. "
                f"${shortfall:,.0f} remaining. "
                f"Projected value: ${incremental_value:,.0f} additional rebate."
            )

        # At risk — calculate the ROI of accelerating
        required_daily = shortfall / days_remaining if days_remaining > 0 else shortfall
        roi = (incremental_value / shortfall * 100) if shortfall > 0 else 0

        return (
            f"At risk of missing {next_tier.tier_name} tier with "
            f"{program.vendor_name}. Need ${shortfall:,.0f} in {days_remaining} days "
            f"(${required_daily:,.0f}/day vs current ${daily_run_rate:,.0f}/day). "
            f"ROI on incremental spend: {roi:.0f}%. "
            f"Unlocks ${incremental_value:,.0f} additional rebate."
        )

    # -----------------------------------------------------------------
    # Evaluate All Programs for a Store
    # -----------------------------------------------------------------

    def evaluate_all(
        self,
        store_id: str,
        vendor_purchases: dict[str, float],
        as_of: date | None = None,
    ) -> list[VendorRebateStatus]:
        """Evaluate all rebate programs for a store.

        Args:
            store_id: Store ID.
            vendor_purchases: Dict of vendor_id → YTD purchases.
            as_of: Date for projection.

        Returns:
            List of VendorRebateStatus for each matching program.
        """
        statuses: list[VendorRebateStatus] = []
        for program in self.programs:
            ytd = vendor_purchases.get(program.vendor_id, 0.0)
            if ytd > 0 or program.vendor_id in vendor_purchases:
                status = self.evaluate_program(
                    program,
                    store_id,
                    ytd,
                    as_of,
                )
                statuses.append(status)

        return statuses

    # -----------------------------------------------------------------
    # Alert Generation
    # -----------------------------------------------------------------

    def generate_alerts(
        self,
        statuses: list[VendorRebateStatus],
    ) -> list[CoopAlert]:
        """Generate alerts for at-risk rebate thresholds.

        Args:
            statuses: List of VendorRebateStatus from evaluate_all().

        Returns:
            List of CoopAlert for at-risk programs, sorted by value.
        """
        alerts: list[CoopAlert] = []

        for status in statuses:
            if not status.is_at_risk:
                continue

            # Only alert if the incremental value justifies action
            if status.incremental_value < 50:
                continue

            alerts.append(
                CoopAlert(
                    alert_type=CoopAlertType.REBATE_THRESHOLD_RISK,
                    store_id=status.store_id,
                    title=(
                        f"Rebate Risk: {status.program.vendor_name} "
                        f"{status.next_tier.tier_name} tier "
                        f"(${status.shortfall:,.0f} short)"
                    ),
                    dollar_impact=status.incremental_value,
                    detail=(
                        f"${status.shortfall:,.0f} short of "
                        f"{status.next_tier.tier_name} tier "
                        f"({status.next_tier.rebate_pct * 100:.1f}%) with "
                        f"{status.program.vendor_name}. "
                        f"{status.days_remaining} days remaining. "
                        f"Current daily rate: ${status.daily_run_rate:,.0f}."
                    ),
                    recommendation=status.recommendation,
                    confidence=0.85,
                )
            )

        # Sort by dollar impact descending
        alerts.sort(key=lambda a: a.dollar_impact, reverse=True)
        return alerts
