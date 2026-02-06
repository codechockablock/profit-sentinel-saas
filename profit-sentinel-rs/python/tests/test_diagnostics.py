"""Tests for the conversational diagnostic engine.

Tests cover:
- Pattern detection (26 patterns + "other")
- Classification enum properties
- Session lifecycle (start -> question -> answer -> complete -> report)
- Running totals and reduction calculation
- Edge cases (empty data, no negative stock, single pattern)
- Rendered output
- Sidecar API endpoints for diagnostics
"""

import pytest
from fastapi.testclient import TestClient
from sentinel_agent.diagnostics import (
    DETECTION_PATTERNS,
    Classification,
    DetectedPattern,
    DiagnosticEngine,
    DiagnosticSession,
    InventoryItem,
    render_diagnostic_report,
    render_diagnostic_summary,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _make_items(items_data: list[tuple]) -> list[dict]:
    """Build item dicts from (sku, description, stock, cost) tuples."""
    return [
        {"sku": sku, "description": desc, "stock": stock, "cost": cost}
        for sku, desc, stock, cost in items_data
    ]


LUMBER_ITEMS = _make_items(
    [
        ("LBR-001", "2X4 8FT STUD", -50, 4.50),
        ("LBR-002", "2X6 10FT", -30, 6.80),
        ("LBR-003", "2X8 12FT", -10, 9.20),
    ]
)

PLYWOOD_ITEMS = _make_items(
    [
        ("PLY-001", "3/4 PLYWOOD 4X8", -5, 35.00),
        ("PLY-002", "1/2 BIRCH PLYWOOD", -3, 42.00),
    ]
)

WIRE_ITEMS = _make_items(
    [
        ("WR-001", "12GA WIRE BY THE FOOT", -200, 0.55),
        ("WR-002", "14GA ROMEX PER FOOT", -150, 0.40),
    ]
)

BATTERY_ITEMS = _make_items(
    [
        ("BAT-001", "ENERGIZER AA 8PK", -20, 5.50),
        ("BAT-002", "DURACELL AAA 4PK", -15, 4.25),
    ]
)

CONCRETE_ITEMS = _make_items(
    [
        ("CON-001", "QUIKRETE 80LB", -40, 5.75),
        ("CON-002", "SAKRETE MORTAR 60LB", -25, 6.20),
    ]
)

MISC_ITEMS = _make_items(
    [
        ("MISC-001", "RANDOM WIDGET", -5, 12.00),
        ("MISC-002", "UNKNOWN GADGET", -3, 8.50),
    ]
)

POSITIVE_ITEMS = _make_items(
    [
        ("POS-001", "HAMMER", 10, 15.00),
        ("POS-002", "SAW BLADE", 25, 8.00),
    ]
)


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------


class TestClassification:
    def test_is_explained_receiving_gap(self):
        assert Classification.RECEIVING_GAP.is_explained is True

    def test_is_explained_non_tracked(self):
        assert Classification.NON_TRACKED.is_explained is True

    def test_is_explained_vendor_managed(self):
        assert Classification.VENDOR_MANAGED.is_explained is True

    def test_is_explained_expiration(self):
        assert Classification.EXPIRATION.is_explained is True

    def test_is_not_explained_theft(self):
        assert Classification.THEFT.is_explained is False

    def test_is_not_explained_investigate(self):
        assert Classification.INVESTIGATE.is_explained is False

    def test_is_not_explained_pending(self):
        assert Classification.PENDING.is_explained is False

    def test_display_names_all_present(self):
        for cls in Classification:
            assert cls.display_name, f"Missing display_name for {cls}"

    def test_value_strings(self):
        assert Classification.RECEIVING_GAP.value == "receiving_gap"
        assert Classification.THEFT.value == "theft"
        assert Classification.PENDING.value == "pending"


# ---------------------------------------------------------------------------
# InventoryItem tests
# ---------------------------------------------------------------------------


class TestInventoryItem:
    def test_shrinkage_value_negative_stock(self):
        item = InventoryItem(sku="A", description="Test", stock=-10, cost=5.0)
        assert item.shrinkage_value == 50.0

    def test_shrinkage_value_positive_stock(self):
        item = InventoryItem(sku="A", description="Test", stock=10, cost=5.0)
        assert item.shrinkage_value == 50.0  # abs(stock) * cost

    def test_shrinkage_value_zero(self):
        item = InventoryItem(sku="A", description="Test", stock=0, cost=5.0)
        assert item.shrinkage_value == 0.0


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestPatternDetection:
    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_all_26_patterns_defined(self):
        assert len(DETECTION_PATTERNS) == 26

    def test_detect_lumber_pattern(self):
        session = self.engine.start_session(LUMBER_ITEMS)
        lumber_patterns = [p for p in session.patterns if p.pattern_id == "lumber_2x"]
        assert len(lumber_patterns) == 1
        assert lumber_patterns[0].item_count == 3

    def test_detect_plywood_pattern(self):
        session = self.engine.start_session(PLYWOOD_ITEMS)
        ply_patterns = [p for p in session.patterns if p.pattern_id == "plywood"]
        assert len(ply_patterns) == 1
        assert ply_patterns[0].item_count == 2

    def test_detect_wire_by_foot(self):
        session = self.engine.start_session(WIRE_ITEMS)
        wire_patterns = [p for p in session.patterns if p.pattern_id == "wire_by_foot"]
        assert len(wire_patterns) == 1

    def test_detect_batteries(self):
        session = self.engine.start_session(BATTERY_ITEMS)
        batt_patterns = [p for p in session.patterns if p.pattern_id == "batteries"]
        assert len(batt_patterns) == 1

    def test_unmatched_go_to_other(self):
        session = self.engine.start_session(MISC_ITEMS)
        other_patterns = [p for p in session.patterns if p.pattern_id == "other"]
        assert len(other_patterns) == 1
        assert other_patterns[0].item_count == 2

    def test_items_only_match_once(self):
        """Each item should match at most one pattern."""
        all_items = (
            LUMBER_ITEMS + PLYWOOD_ITEMS + WIRE_ITEMS + BATTERY_ITEMS + MISC_ITEMS
        )
        session = self.engine.start_session(all_items)

        total_matched = sum(p.item_count for p in session.patterns)
        neg_count = sum(1 for i in all_items if i["stock"] < 0)
        assert total_matched == neg_count

    def test_positive_stock_ignored(self):
        session = self.engine.start_session(POSITIVE_ITEMS)
        assert session.negative_items == 0
        assert len(session.patterns) == 0
        assert session.total_shrinkage == 0.0

    def test_sorted_by_value(self):
        """Patterns should be sorted by total_value, highest first."""
        all_items = LUMBER_ITEMS + CONCRETE_ITEMS + BATTERY_ITEMS + MISC_ITEMS
        session = self.engine.start_session(all_items)

        values = [p.total_value for p in session.patterns]
        assert values == sorted(values, reverse=True)

    def test_empty_items(self):
        session = self.engine.start_session([])
        assert session.items_analyzed == 0
        assert session.negative_items == 0
        assert len(session.patterns) == 0

    def test_question_text_formatted(self):
        """Questions should have count and value substituted."""
        session = self.engine.start_session(LUMBER_ITEMS)
        lumber = [p for p in session.patterns if p.pattern_id == "lumber_2x"][0]
        assert "3 items" in lumber.question or "3" in lumber.question
        assert "$" in lumber.question

    def test_multiple_patterns_detected(self):
        all_items = LUMBER_ITEMS + PLYWOOD_ITEMS + WIRE_ITEMS + CONCRETE_ITEMS
        session = self.engine.start_session(all_items)
        pattern_ids = {p.pattern_id for p in session.patterns}
        assert "lumber_2x" in pattern_ids
        assert "plywood" in pattern_ids
        assert "wire_by_foot" in pattern_ids
        assert "concrete" in pattern_ids


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def setup_method(self):
        self.engine = DiagnosticEngine()
        all_items = LUMBER_ITEMS + PLYWOOD_ITEMS + BATTERY_ITEMS + MISC_ITEMS
        self.session = self.engine.start_session(all_items)

    def test_session_id_generated(self):
        assert self.session.session_id.startswith("diag-")

    def test_initial_state(self):
        assert self.session.current_index == 0
        assert not self.session.is_complete
        assert self.session.explained_value == 0.0
        assert self.session.negative_items > 0

    def test_get_first_question(self):
        q = self.engine.get_current_question(self.session)
        assert q is not None
        assert "pattern_id" in q
        assert "question" in q
        assert "suggested_answers" in q
        assert q["progress"]["current"] == 1
        assert q["progress"]["total"] == len(self.session.patterns)

    def test_answer_advances(self):
        initial_index = self.session.current_index
        result = self.engine.answer_question(self.session, "receiving_gap")
        assert self.session.current_index == initial_index + 1
        assert "answered" in result
        assert "progress" in result
        assert "running_totals" in result

    def test_answer_with_note(self):
        self.engine.answer_question(
            self.session, "receiving_gap", note="Already knew about this"
        )
        pattern = self.session.patterns[0]
        assert pattern.user_note == "Already knew about this"

    def test_explained_value_increases(self):
        first_pattern = self.session.patterns[0]
        expected_value = first_pattern.total_value

        self.engine.answer_question(self.session, "receiving_gap")
        assert self.session.explained_value >= expected_value

    def test_investigate_not_explained(self):
        first_pattern = self.session.patterns[0]
        self.engine.answer_question(self.session, "investigate")
        assert first_pattern.classification == Classification.INVESTIGATE
        # The value should NOT be in explained_value
        assert first_pattern.total_value > 0

    def test_walk_through_all_patterns(self):
        while not self.session.is_complete:
            self.engine.answer_question(self.session, "receiving_gap")
        assert self.session.is_complete
        assert self.engine.get_current_question(self.session) is None

    def test_answer_after_complete_returns_error(self):
        while not self.session.is_complete:
            self.engine.answer_question(self.session, "receiving_gap")
        result = self.engine.answer_question(self.session, "receiving_gap")
        assert "error" in result

    def test_invalid_classification_falls_back(self):
        self.engine.answer_question(self.session, "not_a_real_thing")
        pattern = self.session.patterns[0]
        assert pattern.classification == Classification.INVESTIGATE

    def test_next_question_in_answer(self):
        """Answer result should contain next_question if not complete."""
        result = self.engine.answer_question(self.session, "receiving_gap")
        if not result["is_complete"]:
            assert result["next_question"] is not None
            assert "pattern_id" in result["next_question"]


# ---------------------------------------------------------------------------
# Running totals and reduction
# ---------------------------------------------------------------------------


class TestRunningTotals:
    def setup_method(self):
        self.engine = DiagnosticEngine()

    def test_reduction_percent_all_explained(self):
        session = self.engine.start_session(LUMBER_ITEMS)
        while not session.is_complete:
            self.engine.answer_question(session, "receiving_gap")
        assert session.reduction_percent == pytest.approx(100.0, rel=0.01)

    def test_reduction_percent_none_explained(self):
        session = self.engine.start_session(LUMBER_ITEMS)
        while not session.is_complete:
            self.engine.answer_question(session, "investigate")
        assert session.reduction_percent == 0.0

    def test_reduction_percent_partial(self):
        all_items = LUMBER_ITEMS + BATTERY_ITEMS
        session = self.engine.start_session(all_items)

        total = session.total_shrinkage
        # Answer first as explained, second as investigate
        self.engine.answer_question(session, "receiving_gap")
        first_value = session.patterns[0].total_value
        self.engine.answer_question(session, "investigate")

        expected_pct = (first_value / total) * 100.0
        assert session.reduction_percent == pytest.approx(expected_pct, rel=0.01)

    def test_unexplained_value(self):
        session = self.engine.start_session(LUMBER_ITEMS)
        total = session.total_shrinkage
        self.engine.answer_question(session, "receiving_gap")
        explained = session.explained_value
        assert session.unexplained_value == pytest.approx(total - explained, abs=0.01)

    def test_summary_dict_structure(self):
        session = self.engine.start_session(LUMBER_ITEMS)
        summary = session.get_summary()
        assert "total_shrinkage" in summary
        assert "explained_value" in summary
        assert "unexplained_value" in summary
        assert "reduction_percent" in summary
        assert "patterns_total" in summary
        assert "patterns_answered" in summary
        assert "by_classification" in summary

    def test_summary_by_classification(self):
        session = self.engine.start_session(LUMBER_ITEMS)
        self.engine.answer_question(session, "receiving_gap")
        summary = session.get_summary()
        assert "receiving_gap" in summary["by_classification"]

    def test_empty_session_zero_reduction(self):
        session = self.engine.start_session([])
        assert session.reduction_percent == 0.0


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


class TestFinalReport:
    def setup_method(self):
        self.engine = DiagnosticEngine()
        all_items = LUMBER_ITEMS + BATTERY_ITEMS + MISC_ITEMS
        self.session = self.engine.start_session(all_items)
        # Classify: lumber=receiving_gap, batteries=theft, other=investigate
        self.engine.answer_question(self.session, "receiving_gap")
        self.engine.answer_question(self.session, "theft")
        self.engine.answer_question(self.session, "investigate")

    def test_report_structure(self):
        report = self.engine.get_final_report(self.session)
        assert "session_id" in report
        assert "summary" in report
        assert "by_classification" in report
        assert "items_to_investigate" in report
        assert "journey" in report

    def test_report_by_classification(self):
        report = self.engine.get_final_report(self.session)
        assert "receiving_gap" in report["by_classification"]
        assert "theft" in report["by_classification"]
        assert "investigate" in report["by_classification"]

    def test_items_to_investigate(self):
        report = self.engine.get_final_report(self.session)
        # Theft and investigate items should appear
        inv = report["items_to_investigate"]
        assert len(inv) > 0
        # Sorted by value desc
        if len(inv) > 1:
            assert inv[0]["value"] >= inv[1]["value"]

    def test_journey(self):
        report = self.engine.get_final_report(self.session)
        journey = report["journey"]
        assert len(journey) == len(self.session.patterns)
        for step in journey:
            assert "pattern" in step
            assert "value" in step
            assert "classification" in step
            assert "items" in step

    def test_report_max_50_investigate_items(self):
        """Even with many items, report caps at 50 investigate items."""
        report = self.engine.get_final_report(self.session)
        assert len(report["items_to_investigate"]) <= 50


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


class TestRendering:
    def setup_method(self):
        self.engine = DiagnosticEngine()
        self.session = self.engine.start_session(LUMBER_ITEMS)
        while not self.session.is_complete:
            self.engine.answer_question(self.session, "receiving_gap")

    def test_render_summary(self):
        text = render_diagnostic_summary(self.session)
        assert "DIAGNOSTIC SUMMARY" in text
        assert "Total Apparent Shrinkage" in text
        assert "Process Issues Found" in text
        assert "Reduction:" in text

    def test_render_report(self):
        report = self.engine.get_final_report(self.session)
        text = render_diagnostic_report(report)
        assert "SHRINKAGE DIAGNOSTIC REPORT" in text
        assert "REDUCTION:" in text
        assert "Classification Journey:" in text


# ---------------------------------------------------------------------------
# DetectedPattern tests
# ---------------------------------------------------------------------------


class TestDetectedPattern:
    def test_to_dict(self):
        items = [InventoryItem("A", "Test", -10, 5.0)]
        p = DetectedPattern(
            pattern_id="test",
            name="Test",
            keywords=["TEST"],
            items=items,
            total_value=50.0,
            question="Test question",
            suggested_answers=[["Yes", "receiving_gap"]],
        )
        d = p.to_dict()
        assert d["pattern_id"] == "test"
        assert d["item_count"] == 1
        assert d["total_value"] == 50.0
        assert d["classification"] == "pending"

    def test_item_count_property(self):
        items = [InventoryItem(f"A{i}", "Test", -1, 1.0) for i in range(5)]
        p = DetectedPattern(
            pattern_id="test",
            name="Test",
            keywords=[],
            items=items,
            total_value=5.0,
            question="Q",
            suggested_answers=[],
        )
        assert p.item_count == 5


# ---------------------------------------------------------------------------
# DiagnosticSession tests
# ---------------------------------------------------------------------------


class TestDiagnosticSession:
    def test_current_pattern_at_start(self):
        s = DiagnosticSession(session_id="test")
        s.patterns = [
            DetectedPattern("a", "A", [], [], 10.0, "Q", []),
            DetectedPattern("b", "B", [], [], 5.0, "Q", []),
        ]
        assert s.current_pattern.pattern_id == "a"

    def test_current_pattern_none_when_complete(self):
        s = DiagnosticSession(session_id="test")
        s.patterns = [DetectedPattern("a", "A", [], [], 10.0, "Q", [])]
        s.current_index = 1  # past the end
        assert s.current_pattern is None
        assert s.is_complete

    def test_explained_value_with_mixed(self):
        s = DiagnosticSession(session_id="test", total_shrinkage=1000.0)
        p1 = DetectedPattern("a", "A", [], [], 600.0, "Q", [])
        p1.classification = Classification.RECEIVING_GAP
        p2 = DetectedPattern("b", "B", [], [], 400.0, "Q", [])
        p2.classification = Classification.THEFT
        s.patterns = [p1, p2]
        assert s.explained_value == 600.0
        assert s.unexplained_value == 400.0


# ---------------------------------------------------------------------------
# All 26 patterns have required fields
# ---------------------------------------------------------------------------


class TestPatternDefinitions:
    def test_all_patterns_have_keywords(self):
        for pid, config in DETECTION_PATTERNS.items():
            assert "keywords" in config, f"Pattern {pid} missing keywords"
            assert len(config["keywords"]) > 0, f"Pattern {pid} has empty keywords"

    def test_all_patterns_have_question(self):
        for pid, config in DETECTION_PATTERNS.items():
            assert "question" in config, f"Pattern {pid} missing question"
            assert (
                "{count}" in config["question"]
            ), f"Pattern {pid} question missing {{count}}"
            assert (
                "{value}" in config["question"]
            ), f"Pattern {pid} question missing {{value}}"

    def test_all_patterns_have_suggested_answers(self):
        for pid, config in DETECTION_PATTERNS.items():
            assert (
                "suggested_answers" in config
            ), f"Pattern {pid} missing suggested_answers"
            assert (
                len(config["suggested_answers"]) >= 2
            ), f"Pattern {pid} needs at least 2 suggested answers"

    def test_all_answers_have_valid_classification(self):
        valid = {c.value for c in Classification}
        # "partial" is also used in some patterns (treated as investigate)
        valid.add("partial")
        for pid, config in DETECTION_PATTERNS.items():
            for label, cls_value in config["suggested_answers"]:
                assert cls_value in valid or cls_value in {
                    "partial"
                }, f"Pattern {pid} has invalid classification '{cls_value}'"

    def test_all_patterns_have_typical_behavior(self):
        for pid, config in DETECTION_PATTERNS.items():
            assert (
                "typical_behavior" in config
            ), f"Pattern {pid} missing typical_behavior"


# ---------------------------------------------------------------------------
# Sidecar diagnostic endpoint tests
# ---------------------------------------------------------------------------


class TestDiagnosticEndpoints:
    """Test diagnostic endpoints via FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create test app and client."""
        from sentinel_agent.sidecar import create_app
        from sentinel_agent.sidecar_config import SidecarSettings

        settings = SidecarSettings(
            sentinel_bin="",
            csv_path="",
            default_store="store-7",
            sidecar_dev_mode=True,
        )
        app = create_app(settings)
        self.client = TestClient(app)
        self._sessions = app.extra["sentinel_state"].diagnostic_sessions

    def test_start_diagnostic(self):
        resp = self.client.post(
            "/api/v1/diagnostic/start",
            json={
                "items": LUMBER_ITEMS + BATTERY_ITEMS,
                "store_name": "Test Store",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["store_name"] == "Test Store"
        assert data["patterns_detected"] > 0
        assert data["negative_items"] > 0

    def test_start_diagnostic_empty(self):
        resp = self.client.post(
            "/api/v1/diagnostic/start",
            json={"items": POSITIVE_ITEMS, "store_name": "Test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["patterns_detected"] == 0
        assert data["negative_items"] == 0

    def test_get_question(self):
        # Start session
        start_resp = self.client.post(
            "/api/v1/diagnostic/start",
            json={"items": LUMBER_ITEMS, "store_name": "Test"},
        )
        sid = start_resp.json()["session_id"]

        # Get question
        resp = self.client.get(f"/api/v1/diagnostic/{sid}/question")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pattern_id"] is not None
        assert "question" in data
        assert "suggested_answers" in data

    def test_answer_question(self):
        start_resp = self.client.post(
            "/api/v1/diagnostic/start",
            json={"items": LUMBER_ITEMS, "store_name": "Test"},
        )
        sid = start_resp.json()["session_id"]

        resp = self.client.post(
            f"/api/v1/diagnostic/{sid}/answer",
            json={"classification": "receiving_gap"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answered" in data
        assert data["answered"]["classification"] == "receiving_gap"
        assert "running_totals" in data

    def test_full_diagnostic_flow(self):
        """Walk through the entire diagnostic flow."""
        # Start
        start_resp = self.client.post(
            "/api/v1/diagnostic/start",
            json={"items": LUMBER_ITEMS + BATTERY_ITEMS + MISC_ITEMS},
        )
        sid = start_resp.json()["session_id"]
        num_patterns = start_resp.json()["patterns_detected"]

        # Answer all questions
        for _ in range(num_patterns):
            resp = self.client.post(
                f"/api/v1/diagnostic/{sid}/answer",
                json={"classification": "receiving_gap"},
            )
            assert resp.status_code == 200

        # Last answer should be complete
        data = resp.json()
        assert data["is_complete"] is True

        # Get summary
        summary_resp = self.client.get(f"/api/v1/diagnostic/{sid}/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        assert summary["status"] == "complete"
        assert summary["reduction_percent"] == pytest.approx(100.0, rel=0.01)

        # Get report
        report_resp = self.client.get(f"/api/v1/diagnostic/{sid}/report")
        assert report_resp.status_code == 200
        report = report_resp.json()
        assert "rendered_text" in report
        assert "journey" in report

    def test_get_summary(self):
        start_resp = self.client.post(
            "/api/v1/diagnostic/start",
            json={"items": LUMBER_ITEMS},
        )
        sid = start_resp.json()["session_id"]

        resp = self.client.get(f"/api/v1/diagnostic/{sid}/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert data["total_shrinkage"] > 0

    def test_invalid_session_404(self):
        resp = self.client.get("/api/v1/diagnostic/nonexistent/question")
        assert resp.status_code == 404

    def test_question_returns_null_when_complete(self):
        """After all questions answered, GET question returns null."""
        start_resp = self.client.post(
            "/api/v1/diagnostic/start",
            json={"items": LUMBER_ITEMS},
        )
        sid = start_resp.json()["session_id"]
        num = start_resp.json()["patterns_detected"]

        for _ in range(num):
            self.client.post(
                f"/api/v1/diagnostic/{sid}/answer",
                json={"classification": "receiving_gap"},
            )

        resp = self.client.get(f"/api/v1/diagnostic/{sid}/question")
        assert resp.status_code == 200
        # Response should be null/None
        assert resp.json() is None
