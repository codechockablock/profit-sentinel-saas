"""
Tests for evidence rules module.

Validates:
- Pattern matching works correctly
- Rules apply to facts correctly
- Retail evidence rules cover all cases
"""

import pytest
from sentinel_engine.vsa_evidence.rules import (
    RETAIL_EVIDENCE_RULES,
    EvidenceRule,
    _pattern_matches,
    create_rule_engine,
    extract_evidence_facts,
)


class TestPatternMatching:
    """Test suite for pattern matching."""

    def test_greater_than(self):
        """Test > pattern."""
        assert _pattern_matches(">0", 1) is True
        assert _pattern_matches(">0", 0) is False
        assert _pattern_matches(">0", -1) is False

    def test_less_than(self):
        """Test < pattern."""
        assert _pattern_matches("<10", 5) is True
        assert _pattern_matches("<10", 10) is False
        assert _pattern_matches("<10", 15) is False

    def test_greater_equal(self):
        """Test >= pattern."""
        assert _pattern_matches(">=0.5", 0.5) is True
        assert _pattern_matches(">=0.5", 0.6) is True
        assert _pattern_matches(">=0.5", 0.4) is False

    def test_less_equal(self):
        """Test <= pattern."""
        assert _pattern_matches("<=100", 100) is True
        assert _pattern_matches("<=100", 99) is True
        assert _pattern_matches("<=100", 101) is False

    def test_boolean_true(self):
        """Test true pattern."""
        assert _pattern_matches("true", True) is True
        assert _pattern_matches("true", "true") is True
        assert _pattern_matches("true", "yes") is True
        assert _pattern_matches("true", "1") is True
        assert _pattern_matches("true", False) is False

    def test_boolean_false(self):
        """Test false pattern."""
        assert _pattern_matches("false", False) is True
        assert _pattern_matches("false", "false") is True
        assert _pattern_matches("false", "no") is True
        assert _pattern_matches("false", "0") is True

    def test_string_equality(self):
        """Test string equality (case-insensitive)."""
        assert _pattern_matches("net-60", "net-60") is True
        assert _pattern_matches("net-60", "NET-60") is True
        assert _pattern_matches("high", "HIGH") is True

    def test_regex_pattern(self):
        """Test regex pattern."""
        assert _pattern_matches("/price.*/", "price_increase") is True
        assert _pattern_matches("/^theft/", "theft_suspected") is True
        assert _pattern_matches("/^theft/", "suspected_theft") is False

    def test_invalid_pattern_fails_gracefully(self):
        """Test that invalid patterns fail gracefully."""
        assert _pattern_matches(">abc", "not_a_number") is False
        assert _pattern_matches("/[invalid/", "text") is False


class TestEvidenceRule:
    """Test suite for EvidenceRule class."""

    def test_rule_matches_fact(self):
        """Test rule matching against facts."""
        rule = EvidenceRule(
            attribute="shrinkage_rate",
            pattern=">0.05",
            cause="theft",
            weight=1.0,
        )

        assert rule.matches({"shrinkage_rate": 0.1}) is True
        assert rule.matches({"shrinkage_rate": 0.03}) is False
        assert rule.matches({"other_attr": 0.1}) is False

    def test_rule_with_missing_attribute(self):
        """Test rule when attribute is missing from fact."""
        rule = EvidenceRule(
            attribute="rebate_pending",
            pattern=">0",
            cause="rebate_timing",
            weight=0.8,
        )

        assert rule.matches({}) is False
        assert rule.matches({"rebate_pending": None}) is False


class TestRuleEngine:
    """Test suite for RuleEngine class."""

    @pytest.fixture
    def engine(self):
        """Create rule engine with default retail rules."""
        return create_rule_engine()

    def test_engine_loads_retail_rules(self, engine):
        """Test that engine loads retail rules by default."""
        assert len(engine.rules) > 0
        assert len(engine.rules) == len(RETAIL_EVIDENCE_RULES)

    def test_apply_returns_cause_weights(self, engine):
        """Test that apply returns cause weights."""
        fact = {"shrinkage_rate": 0.1, "qty_difference": -15}

        weights = engine.apply(fact)

        assert isinstance(weights, dict)
        assert "theft" in weights
        assert weights["theft"] > 0

    def test_apply_with_no_matches(self, engine):
        """Test apply with fact that matches no rules."""
        fact = {"random_attr": "random_value"}

        weights = engine.apply(fact)

        assert weights == {}

    def test_explain_match(self, engine):
        """Test rule match explanation."""
        fact = {"cost_delta": 0.15, "margin_compression": True}

        explanations = engine.explain_match(fact)

        assert len(explanations) > 0
        assert all("attribute" in e for e in explanations)
        assert all("cause" in e for e in explanations)
        assert all("description" in e for e in explanations)

    def test_get_rules_for_cause(self, engine):
        """Test getting rules for a specific cause."""
        theft_rules = engine.get_rules_for_cause("theft")

        assert len(theft_rules) > 0
        assert all(r.cause == "theft" for r in theft_rules)

    def test_add_rule(self, engine):
        """Test adding a new rule."""
        initial_count = len(engine.rules)

        engine.add_rule(
            EvidenceRule(
                attribute="custom_attr",
                pattern=">0",
                cause="theft",
                weight=0.5,
                description="Custom test rule",
            )
        )

        assert len(engine.rules) == initial_count + 1

    def test_remove_rule(self, engine):
        """Test removing a rule."""
        # Add a rule first
        engine.add_rule(
            EvidenceRule(
                attribute="test_remove",
                pattern=">0",
                cause="theft",
                weight=0.5,
            )
        )

        initial_count = len(engine.rules)

        # Remove it
        removed = engine.remove_rule("test_remove", ">0", "theft")

        assert removed is True
        assert len(engine.rules) == initial_count - 1


class TestExtractEvidenceFacts:
    """Test suite for fact extraction from POS rows."""

    def test_extract_basic_facts(self):
        """Test extracting basic facts from POS row."""
        row = {
            "sku": "TEST-001",
            "quantity": 100,
            "cost": 10.0,
            "Retail": 25.0,
            "Sold": 50,
        }

        facts = extract_evidence_facts(row)

        assert facts["quantity"] == 100
        assert facts["cost"] == 10.0
        assert facts["revenue"] == 25.0
        assert facts["sold"] == 50
        assert facts["margin"] == 0.6  # (25-10)/25

    def test_extract_negative_inventory(self):
        """Test negative inventory detection."""
        row = {"quantity": -10, "cost": 5.0, "Retail": 10.0}

        facts = extract_evidence_facts(row)

        assert facts["negative_inventory"] is True

    def test_extract_shrinkage_rate(self):
        """Test shrinkage rate calculation."""
        row = {
            "quantity": 100,
            "qty_difference": -10,
            "cost": 5.0,
            "Retail": 10.0,
        }

        facts = extract_evidence_facts(row)

        assert facts["shrinkage_rate"] == 0.1  # 10/100

    def test_extract_price_below_cost(self):
        """Test price below cost detection."""
        row = {"quantity": 50, "cost": 20.0, "Retail": 15.0}

        facts = extract_evidence_facts(row)

        assert facts["price_below_cost"] is True

    def test_extract_high_value_item(self):
        """Test high value item detection."""
        row = {"quantity": 10, "cost": 150.0, "Retail": 300.0}

        facts = extract_evidence_facts(row)

        assert facts["high_value"] is True

    def test_extract_with_context(self):
        """Test extraction with dataset context."""
        row = {"quantity": 50, "cost": 10.0, "Retail": 20.0}  # 50% margin

        context = {"avg_margin": 0.4}  # 40% average
        facts = extract_evidence_facts(row, context)

        # 50% is not below 50% of 40% (20%), so should be False
        assert facts["margin_below_category_avg"] is False

        context = {"avg_margin": 0.8}  # 80% average
        facts = extract_evidence_facts(row, context)

        # 50% is NOT below 50% of 80% (40%), 0.5 > 0.4, so should be False
        assert facts["margin_below_category_avg"] is False


class TestRetailRulesCoverage:
    """Test that retail rules cover expected scenarios."""

    @pytest.fixture
    def engine(self):
        """Create rule engine."""
        return create_rule_engine()

    def test_theft_scenario(self, engine):
        """Test theft detection scenario."""
        fact = {
            "shrinkage_rate": 0.1,  # 10% shrinkage
            "qty_difference": -20,
            "negative_inventory": True,
        }

        weights = engine.apply(fact)

        assert "theft" in weights
        assert weights["theft"] > 1.0  # Multiple rules should fire

    def test_vendor_increase_scenario(self, engine):
        """Test vendor cost increase scenario."""
        fact = {
            "cost_delta": 0.15,  # 15% cost increase
            "margin_compression": True,
        }

        weights = engine.apply(fact)

        assert "vendor_increase" in weights

    def test_rebate_timing_scenario(self, engine):
        """Test rebate timing scenario."""
        fact = {
            "payment_terms": "net-60",
            "rebate_pending": 1000,
        }

        weights = engine.apply(fact)

        assert "rebate_timing" in weights

    def test_margin_leak_scenario(self, engine):
        """Test margin leak scenario."""
        fact = {
            "margin_delta": -0.15,  # 15% margin drop
            "promo_stuck": True,
        }

        weights = engine.apply(fact)

        assert "margin_leak" in weights

    def test_quality_issue_scenario(self, engine):
        """Test quality issue scenario."""
        fact = {
            "return_rate": 0.15,  # 15% returns
            "customer_complaints": 5,
        }

        weights = engine.apply(fact)

        assert "quality_issue" in weights
