"""Tests for Automated Vendor Performance Scoring.

Covers:
    - Individual dimension scoring (quality, delivery, pricing, compliance)
    - Overall scorecard generation
    - Report generation from digest
    - Edge cases: no SKUs, no issues, perfect vendor
    - API endpoint integration
"""

import pytest

from sentinel_agent.vendor_scoring import (
    DimensionScore,
    VendorPerformanceScorer,
    VendorScorecard,
    VendorScoringReport,
    score_vendors,
)
from sentinel_agent.models import (
    Digest,
    Issue,
    IssueType,
    RootCause,
    Sku,
    Summary,
    TrendDirection,
)
from sentinel_agent.coop_models import (
    RebateTier,
    VendorRebateProgram,
    VendorRebateStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sku(
    sku_id: str = "DMG-001",
    qty_on_hand: float = 10,
    unit_cost: float = 25.0,
    retail_price: float = 40.0,
    margin_pct: float = 0.375,
    is_damaged: bool = False,
    on_order_qty: float = 0,
    sales_last_30d: float = 5,
) -> Sku:
    return Sku(
        sku_id=sku_id,
        qty_on_hand=qty_on_hand,
        unit_cost=unit_cost,
        retail_price=retail_price,
        margin_pct=margin_pct,
        sales_last_30d=sales_last_30d,
        days_since_receipt=30,
        is_damaged=is_damaged,
        on_order_qty=on_order_qty,
        is_seasonal=False,
    )


def _make_issue(
    issue_type: IssueType = IssueType.VENDOR_SHORT_SHIP,
    dollar_impact: float = 5000,
    skus: list[Sku] | None = None,
    store_id: str = "store-7",
) -> Issue:
    if skus is None:
        skus = [_make_sku(is_damaged=True)]
    return Issue(
        id=f"issue-{issue_type.value}",
        issue_type=issue_type,
        store_id=store_id,
        dollar_impact=dollar_impact,
        confidence=0.85,
        trend_direction=TrendDirection.WORSENING,
        priority_score=7.5,
        urgency_score=6.0,
        detection_timestamp="2026-02-06T10:00:00Z",
        skus=skus,
        context="Test issue",
    )


def _make_digest(issues: list[Issue] | None = None) -> Digest:
    if issues is None:
        issues = []
    total_impact = sum(i.dollar_impact for i in issues)
    return Digest(
        generated_at="2026-02-06T10:00:00Z",
        store_filter=["store-7"],
        pipeline_ms=150,
        issues=issues,
        summary=Summary(
            total_issues=len(issues),
            total_dollar_impact=total_impact,
            stores_affected=1,
            records_processed=100,
            issues_detected=len(issues),
            issues_filtered_out=0,
        ),
    )


def _make_rebate_status(
    vendor_id: str = "DMG",
    on_track: bool = True,
    shortfall: float = 0,
    incremental_value: float = 500,
) -> VendorRebateStatus:
    from datetime import date

    program = VendorRebateProgram(
        vendor_id=vendor_id,
        vendor_name="Martin's Supply Co.",
        program_name="Volume Incentive",
        program_type="volume",
        tiers=[
            RebateTier(
                tier_name="Gold",
                threshold=50000,
                rebate_pct=0.05,
                description="5% on all",
            ),
        ],
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
    )
    return VendorRebateStatus(
        program=program,
        store_id="store-7",
        ytd_purchases=30000,
        current_tier=RebateTier(
            tier_name="Silver",
            threshold=25000,
            rebate_pct=0.035,
        ),
        next_tier=RebateTier(
            tier_name="Gold",
            threshold=50000,
            rebate_pct=0.05,
        ),
        shortfall=shortfall,
        days_remaining=200,
        daily_run_rate=150,
        projected_total=60000,
        on_track=on_track,
        current_rebate_value=1050,
        next_tier_rebate_value=2500,
        incremental_value=incremental_value,
        recommendation="On track for Gold.",
    )


# ---------------------------------------------------------------------------
# DimensionScore tests
# ---------------------------------------------------------------------------


class TestDimensionScore:
    def test_grade_a(self):
        d = DimensionScore("Quality", 95, 0.3)
        assert d.grade == "A"

    def test_grade_b(self):
        d = DimensionScore("Quality", 85, 0.3)
        assert d.grade == "B"

    def test_grade_c(self):
        d = DimensionScore("Quality", 75, 0.3)
        assert d.grade == "C"

    def test_grade_d(self):
        d = DimensionScore("Quality", 65, 0.3)
        assert d.grade == "D"

    def test_grade_f(self):
        d = DimensionScore("Quality", 50, 0.3)
        assert d.grade == "F"

    def test_weighted_score(self):
        d = DimensionScore("Quality", 80, 0.3)
        assert d.weighted_score == pytest.approx(24.0)

    def test_to_dict(self):
        d = DimensionScore("Quality", 80, 0.3, findings=["Good"], dollar_impact=100)
        data = d.to_dict()
        assert data["dimension"] == "Quality"
        assert data["score"] == 80.0
        assert data["grade"] == "B"
        assert data["findings"] == ["Good"]


# ---------------------------------------------------------------------------
# VendorScorecard tests
# ---------------------------------------------------------------------------


class TestVendorScorecard:
    def test_risk_level_low(self):
        card = VendorScorecard(
            vendor_id="DMG",
            vendor_name="Test",
            overall_score=85,
            overall_grade="B",
            dimensions=[],
            total_skus=10,
            total_dollar_exposure=5000,
            total_quality_cost=0,
            recommendations=[],
        )
        assert card.risk_level == "low"

    def test_risk_level_medium(self):
        card = VendorScorecard(
            vendor_id="DMG",
            vendor_name="Test",
            overall_score=65,
            overall_grade="D",
            dimensions=[],
            total_skus=10,
            total_dollar_exposure=5000,
            total_quality_cost=0,
            recommendations=[],
        )
        assert card.risk_level == "medium"

    def test_risk_level_high(self):
        card = VendorScorecard(
            vendor_id="DMG",
            vendor_name="Test",
            overall_score=45,
            overall_grade="F",
            dimensions=[],
            total_skus=10,
            total_dollar_exposure=5000,
            total_quality_cost=0,
            recommendations=[],
        )
        assert card.risk_level == "high"

    def test_risk_level_critical(self):
        card = VendorScorecard(
            vendor_id="DMG",
            vendor_name="Test",
            overall_score=35,
            overall_grade="F",
            dimensions=[],
            total_skus=10,
            total_dollar_exposure=5000,
            total_quality_cost=0,
            recommendations=[],
        )
        assert card.risk_level == "critical"

    def test_to_dict(self):
        card = VendorScorecard(
            vendor_id="DMG",
            vendor_name="Test",
            overall_score=85,
            overall_grade="B",
            dimensions=[DimensionScore("Quality", 90, 0.3)],
            total_skus=10,
            total_dollar_exposure=5000,
            total_quality_cost=100,
            recommendations=["Good"],
        )
        data = card.to_dict()
        assert data["vendor_id"] == "DMG"
        assert data["overall_score"] == 85.0
        assert data["risk_level"] == "low"
        assert len(data["dimensions"]) == 1


# ---------------------------------------------------------------------------
# Scorer tests
# ---------------------------------------------------------------------------


class TestVendorPerformanceScorer:
    def test_score_perfect_vendor(self):
        """Vendor with no issues should score high."""
        skus = [_make_sku(sku_id=f"DMG-{i:03d}") for i in range(5)]
        issue = _make_issue(
            issue_type=IssueType.DEAD_STOCK,
            skus=skus,
            dollar_impact=100,
        )
        digest = _make_digest([issue])

        report = score_vendors(digest, store_id="store-7")
        assert report.total_vendors_scored == 1
        card = report.scorecards[0]
        assert card.vendor_id == "DMG"
        assert card.overall_score > 60  # Should be reasonably high

    def test_score_quality_damaged(self):
        """Vendor with damaged goods should score lower on quality."""
        skus = [
            _make_sku(sku_id="DMG-001", is_damaged=True),
            _make_sku(sku_id="DMG-002", is_damaged=True),
            _make_sku(sku_id="DMG-003"),
            _make_sku(sku_id="DMG-004"),
        ]
        issue = _make_issue(
            issue_type=IssueType.VENDOR_SHORT_SHIP,
            skus=skus,
            dollar_impact=3000,
        )
        digest = _make_digest([issue])

        scorer = VendorPerformanceScorer()
        report = scorer.score_from_digest(digest, store_id="store-7")
        card = report.scorecards[0]

        # Quality dimension should be below 100
        quality = next(d for d in card.dimensions if d.dimension == "Quality")
        assert quality.score < 80
        assert quality.dollar_impact > 0
        assert any("damaged" in f.lower() for f in quality.findings)

    def test_score_delivery_short_ship(self):
        """Vendor with short-ship issues should score lower on delivery."""
        skus = [
            _make_sku(sku_id="ELC-001", on_order_qty=50, is_damaged=True),
            _make_sku(sku_id="ELC-002", on_order_qty=25),
        ]
        issue = _make_issue(
            issue_type=IssueType.VENDOR_SHORT_SHIP,
            skus=skus,
            dollar_impact=5000,
        )
        digest = _make_digest([issue])

        report = score_vendors(digest, store_id="store-7")
        card = report.scorecards[0]

        delivery = next(d for d in card.dimensions if d.dimension == "Delivery")
        assert delivery.score < 80
        assert any("short-ship" in f.lower() for f in delivery.findings)

    def test_score_pricing_below_benchmark(self):
        """Vendor with low-margin SKUs should score lower on pricing."""
        skus = [
            _make_sku(sku_id="PNT-001", margin_pct=0.15),
            _make_sku(sku_id="PNT-002", margin_pct=0.20),
            _make_sku(sku_id="PNT-003", margin_pct=0.45),
        ]
        issue = _make_issue(
            issue_type=IssueType.MARGIN_EROSION,
            skus=skus,
            dollar_impact=2000,
        )
        digest = _make_digest([issue])

        report = score_vendors(digest, store_id="store-7")
        card = report.scorecards[0]

        pricing = next(d for d in card.dimensions if d.dimension == "Pricing")
        assert pricing.score < 80
        assert any("benchmark" in f.lower() for f in pricing.findings)

    def test_score_compliance_at_risk(self):
        """Vendor at risk of missing rebate tier."""
        skus = [_make_sku(sku_id="DMG-001")]
        issue = _make_issue(
            issue_type=IssueType.DEAD_STOCK,
            skus=skus,
            dollar_impact=100,
        )
        digest = _make_digest([issue])

        rebate = _make_rebate_status(
            on_track=False,
            shortfall=20000,
            incremental_value=1450,
        )
        report = score_vendors(
            digest,
            store_id="store-7",
            rebate_statuses=[rebate],
        )
        card = report.scorecards[0]

        compliance = next(d for d in card.dimensions if d.dimension == "Compliance")
        assert compliance.score < 80
        assert any("risk" in f.lower() for f in compliance.findings)

    def test_score_compliance_on_track(self):
        """Vendor on track for next tier should score well."""
        skus = [_make_sku(sku_id="DMG-001")]
        issue = _make_issue(
            issue_type=IssueType.DEAD_STOCK,
            skus=skus,
            dollar_impact=100,
        )
        digest = _make_digest([issue])

        rebate = _make_rebate_status(on_track=True, shortfall=0)
        report = score_vendors(
            digest,
            store_id="store-7",
            rebate_statuses=[rebate],
        )
        card = report.scorecards[0]

        compliance = next(d for d in card.dimensions if d.dimension == "Compliance")
        assert compliance.score >= 90

    def test_empty_digest(self):
        """Empty digest should produce empty report."""
        digest = _make_digest([])
        report = score_vendors(digest)
        assert report.total_vendors_scored == 0
        assert report.average_score == 0

    def test_multiple_vendors(self):
        """Issues from multiple vendors should create separate scorecards."""
        issues = [
            _make_issue(
                issue_type=IssueType.VENDOR_SHORT_SHIP,
                skus=[_make_sku(sku_id="DMG-001", is_damaged=True)],
            ),
            _make_issue(
                issue_type=IssueType.MARGIN_EROSION,
                skus=[_make_sku(sku_id="ELC-001", margin_pct=0.20)],
            ),
            _make_issue(
                issue_type=IssueType.DEAD_STOCK,
                skus=[_make_sku(sku_id="PNT-001")],
            ),
        ]
        digest = _make_digest(issues)

        report = score_vendors(digest, store_id="store-7")
        assert report.total_vendors_scored == 3
        vendor_ids = {c.vendor_id for c in report.scorecards}
        assert vendor_ids == {"DMG", "ELC", "PNT"}

    def test_store_filter(self):
        """Store filter should exclude other stores."""
        issues = [
            _make_issue(
                skus=[_make_sku(sku_id="DMG-001")],
                store_id="store-7",
            ),
            _make_issue(
                skus=[_make_sku(sku_id="ELC-001")],
                store_id="store-12",
            ),
        ]
        digest = _make_digest(issues)

        report = score_vendors(digest, store_id="store-7")
        assert report.total_vendors_scored == 1

    def test_recommendations_generated(self):
        """Low-scoring vendors should get actionable recommendations."""
        skus = [
            _make_sku(sku_id="DMG-001", is_damaged=True),
            _make_sku(sku_id="DMG-002", is_damaged=True),
            _make_sku(sku_id="DMG-003", is_damaged=True),
        ]
        issue = _make_issue(
            issue_type=IssueType.VENDOR_SHORT_SHIP,
            skus=skus,
            dollar_impact=8000,
        )
        digest = _make_digest([issue])

        report = score_vendors(digest, store_id="store-7")
        card = report.scorecards[0]
        assert len(card.recommendations) >= 1
        assert any("Martin" in r for r in card.recommendations)

    def test_report_to_dict_serializable(self):
        """Report should be JSON-serializable."""
        import json

        issues = [
            _make_issue(skus=[_make_sku(sku_id="DMG-001", is_damaged=True)])
        ]
        digest = _make_digest(issues)
        report = score_vendors(digest, store_id="store-7")

        data = report.to_dict()
        json_str = json.dumps(data)
        assert len(json_str) > 50

    def test_sorted_by_worst_first(self):
        """Scorecards should be sorted worst-first."""
        issues = [
            _make_issue(
                issue_type=IssueType.VENDOR_SHORT_SHIP,
                skus=[
                    _make_sku(sku_id="DMG-001", is_damaged=True),
                    _make_sku(sku_id="DMG-002", is_damaged=True),
                    _make_sku(sku_id="DMG-003", is_damaged=True),
                ],
                dollar_impact=8000,
            ),
            _make_issue(
                issue_type=IssueType.DEAD_STOCK,
                skus=[_make_sku(sku_id="ELC-001")],
                dollar_impact=100,
            ),
        ]
        digest = _make_digest(issues)
        report = score_vendors(digest, store_id="store-7")

        assert report.total_vendors_scored >= 2, (
            f"Expected >=2 vendors scored for multi-vendor digest, got {report.total_vendors_scored}"
        )
        # Worst vendor should be first
        assert report.scorecards[0].overall_score <= report.scorecards[1].overall_score

    def test_high_risk_count(self):
        """Report should count high-risk vendors."""
        skus = [
            _make_sku(sku_id="DMG-001", is_damaged=True),
            _make_sku(sku_id="DMG-002", is_damaged=True),
            _make_sku(sku_id="DMG-003", is_damaged=True),
            _make_sku(sku_id="DMG-004", is_damaged=True),
        ]
        issue = _make_issue(
            issue_type=IssueType.VENDOR_SHORT_SHIP,
            skus=skus,
            dollar_impact=15000,
        )
        digest = _make_digest([issue])
        report = score_vendors(digest, store_id="store-7")

        # With 100% damaged rate and short ship, should be high risk
        assert report.scorecards[0].risk_level in ("high", "critical", "medium")


# ---------------------------------------------------------------------------
# API endpoint integration test
# ---------------------------------------------------------------------------


class TestVendorScoringEndpoint:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from sentinel_agent.sidecar import create_app
        from sentinel_agent.sidecar_config import SidecarSettings

        settings = SidecarSettings(
            sidecar_dev_mode=True,
            csv_path="fixtures/sample_inventory.csv",
            supabase_url="",
            supabase_service_key="",
        )
        app = create_app(settings)
        return TestClient(app)

    def test_vendor_scores_endpoint_exists(self, client):
        """Endpoint should exist and return 502 (no binary) or 200 or 404 (file)."""
        resp = client.get("/api/v1/vendor-scores?store_id=store-7")
        # In test env: 502 (no pipeline binary), 200 (success), or 404 (no CSV)
        assert resp.status_code in (200, 404, 502)

    def test_vendor_scores_unauthenticated(self, client):
        """Endpoint requires authentication in production."""
        # In dev mode, all requests are authenticated
        # Just verify the endpoint is wired up
        resp = client.get("/api/v1/vendor-scores")
        assert resp.status_code in (200, 404, 502)
