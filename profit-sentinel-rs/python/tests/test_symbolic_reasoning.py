"""Tests for Phase 13 — VSA-to-Symbolic Bridge.

Tests the SymbolicReasoner, proof tree generation, forward/backward chaining,
domain rules, evidence rule mirroring, and API endpoints.
"""

from __future__ import annotations

import pytest
from pydantic import Field
from sentinel_agent.models import (
    CauseScoreDetail,
    Digest,
    Issue,
    IssueType,
    RootCause,
    Sku,
    Summary,
    TrendDirection,
)
from sentinel_agent.symbolic_reasoning import (
    DOMAIN_RULES,
    EVIDENCE_RULES,
    SIGNAL_DESCRIPTIONS,
    CompetingHypothesis,
    DomainRule,
    EvidenceRuleSpec,
    Fact,
    FactSource,
    ProofNode,
    ProofTree,
    SignalContribution,
    SymbolicReasoner,
)

# ---------------------------------------------------------------------------
# Fixtures — sample issues with detailed cause scores
# ---------------------------------------------------------------------------


def _make_sku(sku_id: str = "SKU-001", **kwargs) -> Sku:
    defaults = dict(
        sku_id=sku_id,
        qty_on_hand=-47.0,
        unit_cost=23.50,
        retail_price=31.73,
        margin_pct=0.26,
        sales_last_30d=10.0,
        days_since_receipt=30.0,
        is_damaged=False,
        on_order_qty=0.0,
        is_seasonal=False,
    )
    defaults.update(kwargs)
    return Sku(**defaults)


def _make_theft_issue() -> Issue:
    """An issue with strong theft signals."""
    return Issue(
        id="test-store-NegativeInventory-001",
        issue_type=IssueType.NEGATIVE_INVENTORY,
        store_id="test-store",
        dollar_impact=1104.50,
        confidence=0.85,
        trend_direction=TrendDirection.WORSENING,
        priority_score=8.5,
        urgency_score=9.0,
        detection_timestamp="2025-01-15T00:00:00Z",
        skus=[_make_sku()],
        context="47 units short at $23.50/unit.",
        root_cause=RootCause.THEFT,
        root_cause_confidence=0.68,
        cause_scores=[
            CauseScoreDetail(cause="Theft", score=1.230, evidence_count=3),
            CauseScoreDetail(cause="InventoryDrift", score=0.950, evidence_count=2),
            CauseScoreDetail(cause="DemandShift", score=0.420, evidence_count=1),
            CauseScoreDetail(cause="MarginLeak", score=0.310, evidence_count=1),
            CauseScoreDetail(cause="VendorIncrease", score=0.200, evidence_count=1),
            CauseScoreDetail(cause="RebateTiming", score=0.100, evidence_count=0),
            CauseScoreDetail(cause="QualityIssue", score=0.050, evidence_count=0),
            CauseScoreDetail(cause="PricingError", score=0.020, evidence_count=0),
        ],
        root_cause_ambiguity=0.77,
        active_signals=["negative_qty", "zero_sales", "high_cost"],
    )


def _make_pricing_error_issue() -> Issue:
    """An issue with strong pricing error signals."""
    return Issue(
        id="store-7-PriceDiscrepancy-001",
        issue_type=IssueType.PRICE_DISCREPANCY,
        store_id="store-7",
        dollar_impact=500.0,
        confidence=0.85,
        trend_direction=TrendDirection.WORSENING,
        priority_score=7.0,
        urgency_score=7.5,
        detection_timestamp="2025-01-15T00:00:00Z",
        skus=[
            _make_sku(
                sku_id="SKU-PRICE",
                qty_on_hand=10.0,
                unit_cost=80.0,
                retail_price=60.0,
                margin_pct=-0.33,
            )
        ],
        context="Cost exceeds retail.",
        root_cause=RootCause.PRICING_ERROR,
        root_cause_confidence=0.92,
        cause_scores=[
            CauseScoreDetail(cause="PricingError", score=2.100, evidence_count=2),
            CauseScoreDetail(cause="MarginLeak", score=1.800, evidence_count=2),
            CauseScoreDetail(cause="VendorIncrease", score=1.500, evidence_count=2),
            CauseScoreDetail(cause="Theft", score=0.300, evidence_count=1),
            CauseScoreDetail(cause="RebateTiming", score=0.200, evidence_count=1),
            CauseScoreDetail(cause="DemandShift", score=0.100, evidence_count=0),
            CauseScoreDetail(cause="QualityIssue", score=0.050, evidence_count=0),
            CauseScoreDetail(cause="InventoryDrift", score=0.020, evidence_count=0),
        ],
        root_cause_ambiguity=0.86,
        active_signals=["zero_cost", "cost_exceeds_retail", "low_margin"],
    )


def _make_quality_issue() -> Issue:
    """An issue with damaged goods and on-order signals."""
    return Issue(
        id="test-store-VendorShortShip-001",
        issue_type=IssueType.VENDOR_SHORT_SHIP,
        store_id="test-store",
        dollar_impact=3250.0,
        confidence=0.80,
        trend_direction=TrendDirection.WORSENING,
        priority_score=7.5,
        urgency_score=8.0,
        detection_timestamp="2025-01-15T00:00:00Z",
        skus=[
            _make_sku(
                sku_id="SKU-DMG",
                qty_on_hand=10.0,
                unit_cost=200.0,
                retail_price=260.0,
                margin_pct=0.23,
                is_damaged=True,
                on_order_qty=25.0,
            )
        ],
        context="Damaged goods with active purchase orders.",
        root_cause=RootCause.QUALITY_ISSUE,
        root_cause_confidence=0.85,
        cause_scores=[
            CauseScoreDetail(cause="QualityIssue", score=1.500, evidence_count=2),
            CauseScoreDetail(cause="VendorIncrease", score=0.300, evidence_count=1),
            CauseScoreDetail(cause="Theft", score=0.100, evidence_count=0),
            CauseScoreDetail(cause="InventoryDrift", score=0.050, evidence_count=0),
            CauseScoreDetail(cause="MarginLeak", score=0.000, evidence_count=0),
            CauseScoreDetail(cause="DemandShift", score=0.000, evidence_count=0),
            CauseScoreDetail(cause="RebateTiming", score=0.000, evidence_count=0),
            CauseScoreDetail(cause="PricingError", score=0.000, evidence_count=0),
        ],
        root_cause_ambiguity=0.20,
        active_signals=["damaged", "on_order"],
    )


def _make_no_cause_issue() -> Issue:
    """An issue with no root cause attribution."""
    return Issue(
        id="store-12-DeadStock-001",
        issue_type=IssueType.DEAD_STOCK,
        store_id="store-12",
        dollar_impact=5000.0,
        confidence=0.70,
        trend_direction=TrendDirection.STABLE,
        priority_score=5.0,
        urgency_score=4.0,
        detection_timestamp="2025-01-15T00:00:00Z",
        skus=[
            _make_sku(
                sku_id="SKU-DEAD",
                qty_on_hand=100.0,
                unit_cost=50.0,
                retail_price=67.5,
                margin_pct=0.35,
                sales_last_30d=0.0,
                days_since_receipt=180.0,
            )
        ],
        context="Zero sales for 180 days.",
        root_cause=None,
        root_cause_confidence=None,
        cause_scores=[],
        root_cause_ambiguity=None,
        active_signals=["zero_sales", "old_receipt"],
    )


def _make_demand_shift_issue() -> Issue:
    """An issue with demand shift signals and high dollar impact."""
    return Issue(
        id="test-store-Overstock-001",
        issue_type=IssueType.OVERSTOCK,
        store_id="test-store",
        dollar_impact=15000.0,
        confidence=0.75,
        trend_direction=TrendDirection.WORSENING,
        priority_score=7.0,
        urgency_score=6.5,
        detection_timestamp="2025-01-15T00:00:00Z",
        skus=[
            _make_sku(
                sku_id="SKU-OVER",
                qty_on_hand=600.0,
                unit_cost=25.0,
                retail_price=33.75,
                margin_pct=0.35,
                sales_last_30d=0.0,
                days_since_receipt=120.0,
            )
        ],
        context="600 units with no sales.",
        root_cause=RootCause.DEMAND_SHIFT,
        root_cause_confidence=0.82,
        cause_scores=[
            CauseScoreDetail(cause="DemandShift", score=1.800, evidence_count=3),
            CauseScoreDetail(cause="InventoryDrift", score=0.600, evidence_count=1),
            CauseScoreDetail(cause="Theft", score=0.400, evidence_count=1),
            CauseScoreDetail(cause="MarginLeak", score=0.000, evidence_count=0),
            CauseScoreDetail(cause="VendorIncrease", score=0.000, evidence_count=0),
            CauseScoreDetail(cause="RebateTiming", score=0.000, evidence_count=0),
            CauseScoreDetail(cause="QualityIssue", score=0.000, evidence_count=0),
            CauseScoreDetail(cause="PricingError", score=0.000, evidence_count=0),
        ],
        root_cause_ambiguity=0.33,
        active_signals=["zero_sales", "old_receipt", "high_qty"],
    )


# ---------------------------------------------------------------------------
# Tests: Evidence Rules
# ---------------------------------------------------------------------------


class TestEvidenceRules:
    """Test the evidence rule set mirrors Rust's evidence.rs."""

    def test_rule_count(self):
        """Should have 31 rules matching Rust's build_evidence_rules()."""
        # Note: evidence.rs has 31 rules (not 38 as in the original Python spec)
        assert len(EVIDENCE_RULES) == 31

    def test_all_causes_covered(self):
        """All 8 root causes should have at least one rule."""
        causes = {r.cause for r in EVIDENCE_RULES}
        expected = {
            "Theft",
            "VendorIncrease",
            "RebateTiming",
            "MarginLeak",
            "DemandShift",
            "QualityIssue",
            "PricingError",
            "InventoryDrift",
        }
        assert causes == expected

    def test_all_signals_have_descriptions(self):
        """Every signal used in rules should have a description."""
        signals = {r.signal for r in EVIDENCE_RULES}
        for signal in signals:
            assert (
                signal in SIGNAL_DESCRIPTIONS
            ), f"Missing description for signal: {signal}"

    def test_weights_in_range(self):
        """All rule weights should be in [-1.0, 1.0]."""
        for rule in EVIDENCE_RULES:
            assert (
                -1.0 <= rule.weight <= 1.0
            ), f"Rule {rule.signal}->{rule.cause} has weight {rule.weight} outside [-1, 1]"

    def test_counter_evidence_rules_exist(self):
        """Should have negative-weight rules (counter-evidence)."""
        counter = [r for r in EVIDENCE_RULES if r.weight < 0]
        assert len(counter) >= 4, "Expected at least 4 counter-evidence rules"

    def test_theft_rules(self):
        """Theft cause should have strong positive and counter-evidence."""
        theft = [r for r in EVIDENCE_RULES if r.cause == "Theft"]
        positive = [r for r in theft if r.weight > 0]
        negative = [r for r in theft if r.weight < 0]
        assert len(positive) >= 3
        assert len(negative) >= 1  # damaged is counter-evidence


# ---------------------------------------------------------------------------
# Tests: Domain Rules
# ---------------------------------------------------------------------------


class TestDomainRules:
    """Test the domain rule set for hardware retail inference."""

    def test_rule_count(self):
        """Should have 23+ domain rules."""
        assert len(DOMAIN_RULES) >= 23

    def test_all_rules_have_names(self):
        """Every rule should have a unique name."""
        names = [r.name for r in DOMAIN_RULES]
        assert len(names) == len(set(names)), "Duplicate rule names found"

    def test_severity_levels(self):
        """Rules should use valid severity levels."""
        valid = {"critical", "high", "medium", "low"}
        for rule in DOMAIN_RULES:
            assert (
                rule.severity in valid
            ), f"Rule {rule.name} has invalid severity: {rule.severity}"

    def test_confidence_decay_range(self):
        """Confidence decay should be in [0.0, 0.5]."""
        for rule in DOMAIN_RULES:
            assert (
                0.0 <= rule.confidence_decay <= 0.5
            ), f"Rule {rule.name} has decay {rule.confidence_decay} outside [0, 0.5]"

    def test_action_rules_exist(self):
        """Should have rules that derive action() conclusions."""
        actions = [r for r in DOMAIN_RULES if r.conclusion.startswith("action(")]
        assert len(actions) >= 5, "Expected at least 5 action rules"

    def test_suspect_rules_exist(self):
        """Should have rules that derive suspect() conclusions."""
        suspects = [r for r in DOMAIN_RULES if r.conclusion.startswith("suspect(")]
        assert len(suspects) >= 10, "Expected at least 10 suspect rules"

    def test_compound_rules_reference_suspects(self):
        """Compound/escalation rules should reference suspect premises."""
        compound = [r for r in DOMAIN_RULES if r.conclusion.startswith("action(")]
        for rule in compound:
            has_suspect = any(p.startswith("suspect(") for p in rule.premises)
            has_detected = any(p.startswith("detected(") for p in rule.premises)
            assert (
                has_suspect or has_detected
            ), f"Rule {rule.name} should reference suspect() or detected() premises"


# ---------------------------------------------------------------------------
# Tests: SymbolicReasoner
# ---------------------------------------------------------------------------


class TestSymbolicReasoner:
    """Test the main reasoning engine."""

    def test_explain_theft_issue(self):
        """Should produce a valid proof tree for theft issue."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)

        assert proof.issue_id == "test-store-NegativeInventory-001"
        assert proof.root_cause == "Theft"
        assert proof.root_cause_confidence == 0.68
        assert proof.root_cause_ambiguity == 0.77
        assert len(proof.active_signals) == 3
        assert "negative_qty" in proof.active_signals

    def test_explain_pricing_error(self):
        """Should produce a valid proof tree for pricing error."""
        reasoner = SymbolicReasoner()
        issue = _make_pricing_error_issue()
        proof = reasoner.explain(issue)

        assert proof.root_cause == "PricingError"
        assert proof.root_cause_confidence == 0.92
        assert len(proof.cause_scores) == 8

    def test_explain_no_cause(self):
        """Should handle issues with no root cause."""
        reasoner = SymbolicReasoner()
        issue = _make_no_cause_issue()
        proof = reasoner.explain(issue)

        assert proof.root_cause is None
        assert proof.root_cause_display == "Unknown"
        assert proof.root_cause_confidence == 0.0

    def test_signal_contributions_traced(self):
        """Should trace which rules fire for each signal."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)

        assert len(proof.signal_contributions) == 3
        # negative_qty should fire rules for Theft and InventoryDrift
        neg_qty = [s for s in proof.signal_contributions if s.signal == "negative_qty"]
        assert len(neg_qty) == 1
        causes_hit = {r["cause"] for r in neg_qty[0].rules_fired}
        assert "Theft" in causes_hit
        assert "InventoryDrift" in causes_hit

    def test_competing_hypotheses(self):
        """Should explain why alternative causes scored lower."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)

        assert len(proof.competing_hypotheses) >= 1
        # InventoryDrift was #2
        drift = [h for h in proof.competing_hypotheses if h.cause == "InventoryDrift"]
        assert len(drift) == 1
        assert drift[0].rank == 2
        assert len(drift[0].why_lower) > 0

    def test_recommendations_from_root_cause(self):
        """Should include standard recommendations from root cause."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)

        assert len(proof.recommendations) == 4
        assert "security footage" in proof.recommendations[0].lower()

    def test_cause_scores_ranked(self):
        """Cause scores should be ranked by score descending."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)

        scores = [cs["score"] for cs in proof.cause_scores]
        assert scores == sorted(scores, reverse=True)
        assert proof.cause_scores[0]["cause"] == "Theft"
        assert proof.cause_scores[0]["rank"] == 1


# ---------------------------------------------------------------------------
# Tests: Forward Chaining
# ---------------------------------------------------------------------------


class TestForwardChaining:
    """Test forward-chaining domain rule inference."""

    def test_theft_signals_derive_suspects(self):
        """Theft signals should derive suspect facts."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)

        inferred_statements = {f["statement"] for f in proof.inferred_facts}
        # negative_qty + high_cost -> suspect(theft_high_value_items)
        assert "suspect(theft_high_value_items)" in inferred_statements
        # negative_qty + zero_sales -> suspect(theft_no_sales_shrinkage)
        assert "suspect(theft_no_sales_shrinkage)" in inferred_statements

    def test_systematic_shrinkage_triple_signal(self):
        """Triple signal should derive systematic shrinkage."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        # Issue has negative_qty, zero_sales, high_cost — but also needs low_margin
        # Let's make an issue with all four
        issue.active_signals = ["negative_qty", "low_margin", "zero_sales", "high_cost"]
        proof = reasoner.explain(issue)

        inferred = {f["statement"] for f in proof.inferred_facts}
        assert "suspect(systematic_shrinkage)" in inferred

    def test_escalation_rules_chain(self):
        """Compound rules should fire when suspects are established."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        issue.active_signals = ["negative_qty", "low_margin", "zero_sales", "high_cost"]
        proof = reasoner.explain(issue)

        action_names = {a["action"] for a in proof.suggested_actions}
        # systematic_shrinkage + theft_high_value_items -> escalate_security_investigation
        assert "action(escalate_security_investigation)" in action_names

    def test_pricing_error_derives_missing_cost(self):
        """Pricing error signals should derive data quality suspects."""
        reasoner = SymbolicReasoner()
        issue = _make_pricing_error_issue()
        proof = reasoner.explain(issue)

        inferred = {f["statement"] for f in proof.inferred_facts}
        # zero_cost -> suspect(missing_cost_data)
        assert "suspect(missing_cost_data)" in inferred

    def test_quality_issue_derives_vendor_claim(self):
        """Quality issue with reorder should suggest vendor claim."""
        reasoner = SymbolicReasoner()
        issue = _make_quality_issue()
        proof = reasoner.explain(issue)

        inferred = {f["statement"] for f in proof.inferred_facts}
        assert "suspect(vendor_quality_repeat_issue)" in inferred

        action_names = {a["action"] for a in proof.suggested_actions}
        assert "action(file_vendor_claim)" in action_names

    def test_demand_shift_derives_overstock(self):
        """Demand shift signals should derive overstock suspects."""
        reasoner = SymbolicReasoner()
        issue = _make_demand_shift_issue()
        proof = reasoner.explain(issue)

        inferred = {f["statement"] for f in proof.inferred_facts}
        # zero_sales + old_receipt -> suspect(demand_dried_up)
        assert "suspect(demand_dried_up)" in inferred
        # high_qty + zero_sales -> suspect(overstock_no_demand)
        assert "suspect(overstock_no_demand)" in inferred

    def test_forward_chain_terminates(self):
        """Forward chaining should terminate (fixed-point)."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        issue.active_signals = [
            "negative_qty",
            "low_margin",
            "zero_sales",
            "high_cost",
            "damaged",
            "on_order",
            "old_receipt",
            "high_qty",
            "seasonal",
        ]
        # Even with many signals, should terminate
        proof = reasoner.explain(issue)
        assert proof is not None
        # Check that we didn't hit infinite loop
        assert len(proof.inferred_facts) + len(proof.suggested_actions) < 50

    def test_confidence_decay(self):
        """Inferred facts should have decayed confidence."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)

        for fact in proof.inferred_facts:
            # Signal facts have confidence 1.0, so derived should be < 1.0
            assert fact["confidence"] <= 1.0
            assert fact["confidence"] > 0.0

    def test_high_dollar_impact_fact(self):
        """Issues with >$10K impact should have high_dollar_impact fact."""
        reasoner = SymbolicReasoner()
        issue = _make_demand_shift_issue()  # $15,000 impact
        fact_base = reasoner._build_fact_base(issue)
        predicates = {f.predicate for f in fact_base}
        assert "high_dollar_impact" in predicates


# ---------------------------------------------------------------------------
# Tests: Backward Chaining
# ---------------------------------------------------------------------------


class TestBackwardChaining:
    """Test backward-chaining goal explanation."""

    def test_backward_chain_detected_signal(self):
        """Should find signal facts directly."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        steps = reasoner.backward_chain(issue, "detected(negative_qty)")

        satisfied = [s for s in steps if s["satisfied"]]
        assert len(satisfied) >= 1
        assert satisfied[0]["method"] == "fact"

    def test_backward_chain_root_cause(self):
        """Should explain root cause via VSA scoring fact."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        steps = reasoner.backward_chain(issue, "root_cause(Theft)")

        satisfied = [s for s in steps if s["satisfied"]]
        assert len(satisfied) >= 1

    def test_backward_chain_domain_rule(self):
        """Should find inferred facts that were derived by domain rules."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        steps = reasoner.backward_chain(issue, "suspect(theft_high_value_items)")

        # Forward chaining already derived this fact, so backward chain
        # finds it as a fact (from inference). Either method is valid.
        satisfied = [s for s in steps if s["satisfied"]]
        assert len(satisfied) >= 1
        # Verify the fact came from inference (forward chaining)
        for s in satisfied:
            if s["method"] == "fact":
                assert s["supporting"][0]["source"] == "inference"

    def test_backward_chain_unproven_goal(self):
        """Should report when a goal cannot be proved."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        steps = reasoner.backward_chain(issue, "suspect(nonexistent_conclusion)")

        unsatisfied = [s for s in steps if not s["satisfied"]]
        assert len(unsatisfied) >= 1

    def test_backward_chain_action(self):
        """Should trace action conclusions through suspects or facts."""
        reasoner = SymbolicReasoner()
        issue = _make_quality_issue()
        steps = reasoner.backward_chain(issue, "action(file_vendor_claim)")

        # Forward chaining already derived this action, so backward chain
        # finds it as a fact. Verify it was proven.
        satisfied = [s for s in steps if s["satisfied"]]
        assert len(satisfied) >= 1


# ---------------------------------------------------------------------------
# Tests: ProofTree Serialization
# ---------------------------------------------------------------------------


class TestProofTreeSerialization:
    """Test proof tree JSON serialization."""

    def test_to_dict_has_all_fields(self):
        """to_dict should include all required fields."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)
        d = proof.to_dict()

        required_keys = {
            "issue_id",
            "issue_type",
            "store_id",
            "dollar_impact",
            "root_cause",
            "root_cause_display",
            "root_cause_confidence",
            "root_cause_ambiguity",
            "active_signals",
            "signal_contributions",
            "cause_scores",
            "proof_tree",
            "inferred_facts",
            "competing_hypotheses",
            "recommendations",
            "suggested_actions",
        }
        assert required_keys.issubset(d.keys())

    def test_to_dict_proof_tree_is_nested(self):
        """proof_tree should have nested children."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        d = reasoner.explain(issue).to_dict()

        tree = d["proof_tree"]
        assert "statement" in tree
        assert "children" in tree
        assert len(tree["children"]) >= 1

    def test_to_dict_signal_contributions(self):
        """Signal contributions should be serializable."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        d = reasoner.explain(issue).to_dict()

        for sc in d["signal_contributions"]:
            assert "signal" in sc
            assert "description" in sc
            assert "rules_fired" in sc
            for rule in sc["rules_fired"]:
                assert "cause" in rule
                assert "weight" in rule

    def test_to_dict_competing_hypotheses(self):
        """Competing hypotheses should include why_lower."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        d = reasoner.explain(issue).to_dict()

        for hyp in d["competing_hypotheses"]:
            assert "cause" in hyp
            assert "cause_display" in hyp
            assert "rank" in hyp
            assert "why_lower" in hyp

    def test_render_produces_text(self):
        """render() should produce human-readable text."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        proof = reasoner.explain(issue)
        text = proof.render()

        assert "ROOT CAUSE" in text
        assert "Theft" in text
        assert "OBSERVED SIGNALS" in text
        assert "negative_qty" in text
        assert "CAUSE SCORES" in text
        assert "WINNER" in text


# ---------------------------------------------------------------------------
# Tests: Fact and ProofNode
# ---------------------------------------------------------------------------


class TestFactModel:
    """Test Fact dataclass."""

    def test_fact_equality(self):
        """Facts with same predicate should be equal."""
        f1 = Fact("detected(low_margin)", 0.8, FactSource.SIGNAL)
        f2 = Fact("detected(low_margin)", 1.0, FactSource.VSA_SCORING)
        assert f1 == f2

    def test_fact_hash(self):
        """Facts with same predicate should hash the same."""
        f1 = Fact("detected(low_margin)", 0.8, FactSource.SIGNAL)
        f2 = Fact("detected(low_margin)", 1.0, FactSource.SIGNAL)
        assert hash(f1) == hash(f2)
        assert len({f1, f2}) == 1  # deduplicates in sets

    def test_fact_inequality(self):
        """Facts with different predicates should not be equal."""
        f1 = Fact("detected(low_margin)", 0.8, FactSource.SIGNAL)
        f2 = Fact("detected(high_cost)", 0.8, FactSource.SIGNAL)
        assert f1 != f2


class TestProofNodeModel:
    """Test ProofNode dataclass."""

    def test_to_dict_basic(self):
        """to_dict should serialize basic fields."""
        node = ProofNode(
            statement="test statement",
            confidence=0.85,
            explanation="test explanation",
            source="signal",
        )
        d = node.to_dict()
        assert d["statement"] == "test statement"
        assert d["confidence"] == 0.85
        assert "children" not in d  # No children

    def test_to_dict_with_children(self):
        """to_dict should include children recursively."""
        child = ProofNode("child", 0.5, "child explanation", "signal")
        parent = ProofNode("parent", 0.9, "parent explanation", "conclusion", [child])
        d = parent.to_dict()
        assert len(d["children"]) == 1
        assert d["children"][0]["statement"] == "child"


# ---------------------------------------------------------------------------
# Tests: CauseScoreDetail Model
# ---------------------------------------------------------------------------


class TestCauseScoreDetail:
    """Test the new CauseScoreDetail model."""

    def test_model_creation(self):
        csd = CauseScoreDetail(cause="Theft", score=1.23, evidence_count=3)
        assert csd.cause == "Theft"
        assert csd.score == 1.23
        assert csd.evidence_count == 3

    def test_model_serialization(self):
        csd = CauseScoreDetail(cause="MarginLeak", score=0.456, evidence_count=1)
        d = csd.model_dump()
        assert d["cause"] == "MarginLeak"
        assert d["score"] == 0.456

    def test_issue_with_cause_scores(self):
        """Issue should accept cause_scores field."""
        issue = _make_theft_issue()
        assert len(issue.cause_scores) == 8
        assert issue.cause_scores[0].cause == "Theft"
        assert issue.root_cause_ambiguity == 0.77

    def test_issue_defaults_empty(self):
        """Issue should default to empty cause_scores."""
        issue = Issue(
            id="test",
            issue_type=IssueType.DEAD_STOCK,
            store_id="store-1",
            dollar_impact=100.0,
            confidence=0.5,
            trend_direction=TrendDirection.STABLE,
            priority_score=5.0,
            urgency_score=5.0,
            detection_timestamp="2025-01-01T00:00:00Z",
            skus=[],
            context="test",
        )
        assert issue.cause_scores == []
        assert issue.root_cause_ambiguity is None
        assert issue.active_signals == []


# ---------------------------------------------------------------------------
# Tests: Sidecar API (Phase 13 endpoints)
# ---------------------------------------------------------------------------


class TestSidecarExplainEndpoints:
    """Test the /api/v1/explain endpoints via TestClient."""

    @pytest.fixture
    def client(self):
        """Create test client with dev mode enabled."""
        from fastapi.testclient import TestClient
        from sentinel_agent.sidecar import create_app
        from sentinel_agent.sidecar_config import SidecarSettings

        settings = SidecarSettings(
            sidecar_dev_mode=True,
            csv_path="nonexistent.csv",  # Don't need real pipeline
        )
        app = create_app(settings)
        return TestClient(app)

    @pytest.fixture
    def client_with_digest(self, client):
        """Populate the digest cache with a sample issue."""
        from sentinel_agent.routes.state import DigestCacheEntry

        # Build a sample digest and cache it
        issue = _make_theft_issue()
        digest = Digest(
            generated_at="2025-01-15T06:00:00Z",
            store_filter=["test-store"],
            pipeline_ms=100,
            issues=[issue],
            summary=Summary(
                total_issues=1,
                total_dollar_impact=1104.50,
                stores_affected=1,
                records_processed=1000,
                issues_detected=5,
                issues_filtered_out=4,
            ),
        )
        # Inject into cache via app state (keyed by user_id for tenant isolation)
        state = client.app.extra["sentinel_state"]
        entry = DigestCacheEntry(digest, 3600)
        state.digest_cache["dev-user"] = {
            "test-store:5": entry,
            ":5": entry,  # Default cache key
        }

        yield client

        # Cleanup
        state.digest_cache.clear()

    def test_health_version(self, client):
        """Health endpoint should report version 0.13.0."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["version"] == "0.13.0"

    def test_explain_endpoint(self, client_with_digest):
        """GET /api/v1/explain/{issue_id} should return proof tree."""
        resp = client_with_digest.get(
            "/api/v1/explain/test-store-NegativeInventory-001"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["issue_id"] == "test-store-NegativeInventory-001"
        assert "proof_tree" in data
        assert "rendered_text" in data

        # Verify proof tree structure
        tree = data["proof_tree"]
        assert "root_cause" in tree
        assert "active_signals" in tree
        assert "signal_contributions" in tree
        assert "cause_scores" in tree
        assert "competing_hypotheses" in tree

    def test_explain_not_found(self, client):
        """Should 404 for non-existent issue."""
        resp = client.get("/api/v1/explain/nonexistent-issue")
        assert resp.status_code == 404

    def test_backward_chain_endpoint(self, client_with_digest):
        """POST /api/v1/explain/{id}/why should return reasoning steps."""
        resp = client_with_digest.post(
            "/api/v1/explain/test-store-NegativeInventory-001/why",
            json={"goal": "detected(negative_qty)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["issue_id"] == "test-store-NegativeInventory-001"
        assert data["goal"] == "detected(negative_qty)"
        assert len(data["reasoning_steps"]) >= 1

    def test_backward_chain_unproven(self, client_with_digest):
        """Backward chain for impossible goal should still return."""
        resp = client_with_digest.post(
            "/api/v1/explain/test-store-NegativeInventory-001/why",
            json={"goal": "suspect(impossible)"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert any(not s["satisfied"] for s in data["reasoning_steps"])


# ---------------------------------------------------------------------------
# Tests: Signal Descriptions
# ---------------------------------------------------------------------------


class TestSignalDescriptions:
    """Test the signal description lookup table."""

    def test_all_classifier_signals_covered(self):
        """All 13 signals from issue_classifier.rs should have descriptions."""
        expected_signals = {
            "negative_qty",
            "high_cost",
            "low_margin",
            "zero_sales",
            "high_qty",
            "recent_receipt",
            "old_receipt",
            "negative_retail",
            "damaged",
            "on_order",
            "seasonal",
            "zero_cost",
            "cost_exceeds_retail",
        }
        assert expected_signals.issubset(SIGNAL_DESCRIPTIONS.keys())

    def test_descriptions_are_non_empty(self):
        """All descriptions should be non-empty strings."""
        for signal, desc in SIGNAL_DESCRIPTIONS.items():
            assert isinstance(desc, str)
            assert len(desc) > 5, f"Description for {signal} is too short"


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases in the symbolic reasoning system."""

    def test_empty_signals(self):
        """Issue with no active signals should still produce valid proof."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        issue.active_signals = []
        issue.cause_scores = []
        proof = reasoner.explain(issue)

        assert proof.issue_id == issue.id
        assert len(proof.signal_contributions) == 0
        assert len(proof.inferred_facts) == 0

    def test_custom_rules(self):
        """Should accept custom evidence and domain rules."""
        custom_evidence = [
            EvidenceRuleSpec("test_signal", "Theft", 0.9, "Test rule"),
        ]
        custom_domain = [
            DomainRule(
                name="test_rule",
                premises=["detected(test_signal)"],
                conclusion="suspect(test_conclusion)",
                confidence_decay=0.1,
                explanation="Test domain rule",
            ),
        ]
        reasoner = SymbolicReasoner(
            evidence_rules=custom_evidence,
            domain_rules=custom_domain,
        )
        assert len(reasoner.evidence_rules) == 1
        assert len(reasoner.domain_rules) == 1

    def test_high_ambiguity_note_in_render(self):
        """High ambiguity should produce a note in rendered output."""
        reasoner = SymbolicReasoner()
        issue = _make_theft_issue()
        issue.root_cause_ambiguity = 0.95
        proof = reasoner.explain(issue)
        text = proof.render()
        assert "ambiguity" in text.lower()

    def test_low_ambiguity_no_note(self):
        """Low ambiguity should not produce an ambiguity note."""
        reasoner = SymbolicReasoner()
        issue = _make_quality_issue()  # ambiguity=0.20
        proof = reasoner.explain(issue)
        text = proof.render()
        # Should not contain the high ambiguity warning
        assert "competing hypotheses are close" not in text

    def test_multiple_issues_independent(self):
        """Explaining different issues should be independent."""
        reasoner = SymbolicReasoner()
        proof1 = reasoner.explain(_make_theft_issue())
        proof2 = reasoner.explain(_make_pricing_error_issue())

        assert proof1.root_cause == "Theft"
        assert proof2.root_cause == "PricingError"
        assert proof1.issue_id != proof2.issue_id
