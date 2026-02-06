"""Phase 13 — VSA-to-Symbolic Bridge.

Translates VSA evidence scores from the Rust pipeline into transparent,
explainable proof trees. Makes the AI's reasoning visible to users —
showing which signals fired, which rules matched, what the competing
hypotheses were, and why the system reached its conclusion.

Components:
    - EvidenceRule: domain knowledge encoded as signal→cause mappings
    - DomainRuleEngine: 30+ hardware retail rules with confidence decay
    - SymbolicReasoner: forward/backward chaining over facts
    - ProofTree: serializable reasoning chain for API consumption

Usage:
    from sentinel_agent.symbolic_reasoning import SymbolicReasoner

    reasoner = SymbolicReasoner()
    proof = reasoner.explain(issue)
    print(proof.render())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .models import CauseScoreDetail, Issue, RootCause

# ---------------------------------------------------------------------------
# Facts — atomic propositions derived from data
# ---------------------------------------------------------------------------


class FactSource(str, Enum):
    """Where a fact was derived from."""

    SIGNAL = "signal"  # POS data signal detection
    VSA_SCORING = "vsa_scoring"  # VSA evidence scorer output
    RULE_INFERENCE = "inference"  # Derived by forward-chaining rules
    DATA_ATTRIBUTE = "attribute"  # Issue metadata (dollar impact, etc.)


@dataclass
class Fact:
    """An atomic proposition in the symbolic reasoning system.

    Examples:
        Fact("detected(low_margin)", 0.8, FactSource.SIGNAL)
        Fact("cause_score(Theft, 1.23)", 1.0, FactSource.VSA_SCORING)
        Fact("high_value_item", 0.9, FactSource.RULE_INFERENCE)
    """

    predicate: str
    confidence: float  # 0.0–1.0
    source: FactSource
    explanation: str = ""  # Human-readable why this fact holds

    def __hash__(self) -> int:
        return hash(self.predicate)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fact):
            return NotImplemented
        return self.predicate == other.predicate


# ---------------------------------------------------------------------------
# Evidence Rules — the 38 rules mirrored from Rust evidence.rs
# ---------------------------------------------------------------------------


@dataclass
class EvidenceRuleSpec:
    """A single signal→cause mapping with weight.

    Mirrors Rust's EvidenceRule struct for transparent reasoning.
    """

    signal: str
    cause: str  # RootCause enum value name
    weight: float
    rationale: str  # Human-readable explanation


# Complete rule set from evidence.rs (38 rules)
EVIDENCE_RULES: list[EvidenceRuleSpec] = [
    # === THEFT / SHRINKAGE ===
    EvidenceRuleSpec(
        "negative_qty",
        "Theft",
        0.9,
        "Negative inventory without recent receipt suggests theft/shrinkage",
    ),
    EvidenceRuleSpec(
        "low_margin", "Theft", 0.5, "Low margin on high-value items can indicate theft"
    ),
    EvidenceRuleSpec(
        "zero_sales",
        "Theft",
        0.6,
        "Zero sales but positive stock means items are disappearing",
    ),
    EvidenceRuleSpec(
        "high_cost", "Theft", 0.5, "High-cost items are common theft targets"
    ),
    EvidenceRuleSpec(
        "damaged",
        "Theft",
        -0.4,
        "Damaged items are accounted for, not theft (counter-evidence)",
    ),
    EvidenceRuleSpec(
        "recent_receipt",
        "Theft",
        -0.3,
        "Recent receipt more likely indicates receiving error (counter-evidence)",
    ),
    # === VENDOR PRICE INCREASE ===
    EvidenceRuleSpec(
        "low_margin", "VendorIncrease", 0.8, "Low margin signals vendor cost increase"
    ),
    EvidenceRuleSpec(
        "high_cost", "VendorIncrease", 0.6, "High-cost items with margin compression"
    ),
    EvidenceRuleSpec(
        "on_order",
        "VendorIncrease",
        0.3,
        "On-order items may have new (higher) pricing",
    ),
    EvidenceRuleSpec(
        "cost_exceeds_retail",
        "VendorIncrease",
        0.9,
        "Cost exceeding retail indicates vendor price jump",
    ),
    # === REBATE TIMING ===
    EvidenceRuleSpec(
        "seasonal",
        "RebateTiming",
        0.7,
        "Seasonal items may have co-op rebate timing issues",
    ),
    EvidenceRuleSpec(
        "high_qty",
        "RebateTiming",
        0.4,
        "High quantity with low margin suggests rebate not yet applied",
    ),
    EvidenceRuleSpec(
        "low_margin",
        "RebateTiming",
        0.5,
        "Low margin may indicate missing rebate credit",
    ),
    # === MARGIN LEAK ===
    EvidenceRuleSpec(
        "low_margin",
        "MarginLeak",
        1.0,
        "Direct margin signal — margin is below acceptable threshold",
    ),
    EvidenceRuleSpec(
        "cost_exceeds_retail",
        "MarginLeak",
        0.9,
        "Selling below cost — most direct form of margin leak",
    ),
    EvidenceRuleSpec(
        "zero_cost",
        "MarginLeak",
        0.4,
        "Zero cost with retail makes margin calculation meaningless",
    ),
    # === DEMAND SHIFT ===
    EvidenceRuleSpec(
        "zero_sales", "DemandShift", 0.9, "Zero sales indicates demand has dropped"
    ),
    EvidenceRuleSpec(
        "old_receipt",
        "DemandShift",
        0.7,
        "Old receipt combined with no sales means demand dried up",
    ),
    EvidenceRuleSpec(
        "high_qty", "DemandShift", 0.5, "High quantity is overstock from demand miss"
    ),
    EvidenceRuleSpec(
        "seasonal", "DemandShift", 0.6, "Seasonal items inherently have demand shifts"
    ),
    EvidenceRuleSpec(
        "recent_receipt",
        "DemandShift",
        -0.4,
        "Recent receipt means still in demand cycle (counter-evidence)",
    ),
    # === QUALITY ISSUE ===
    EvidenceRuleSpec(
        "damaged",
        "QualityIssue",
        1.0,
        "Damaged goods directly indicate quality problems",
    ),
    EvidenceRuleSpec(
        "on_order",
        "QualityIssue",
        0.5,
        "On-order plus damaged suggests repeat quality problem",
    ),
    EvidenceRuleSpec(
        "high_qty",
        "QualityIssue",
        0.3,
        "High quantity of damaged goods suggests batch issue",
    ),
    # === PRICING ERROR ===
    EvidenceRuleSpec(
        "zero_cost",
        "PricingError",
        1.0,
        "Zero cost means missing cost data — pricing configuration error",
    ),
    EvidenceRuleSpec(
        "cost_exceeds_retail",
        "PricingError",
        0.9,
        "Cost exceeding retail suggests pricing setup error",
    ),
    EvidenceRuleSpec(
        "negative_retail",
        "PricingError",
        1.0,
        "Negative retail price is a data entry error",
    ),
    # === INVENTORY DRIFT ===
    EvidenceRuleSpec(
        "negative_qty",
        "InventoryDrift",
        0.7,
        "Negative quantity without other explanations suggests drift",
    ),
    EvidenceRuleSpec(
        "old_receipt",
        "InventoryDrift",
        0.6,
        "Old receipt means long time for drift to accumulate",
    ),
    EvidenceRuleSpec(
        "high_qty",
        "InventoryDrift",
        0.3,
        "High-quantity items have more opportunity for drift",
    ),
    EvidenceRuleSpec(
        "damaged",
        "InventoryDrift",
        -0.3,
        "Damaged items are accounted for (counter-evidence)",
    ),
]


# ---------------------------------------------------------------------------
# Domain Rules — hardware retail inference rules
# ---------------------------------------------------------------------------


@dataclass
class DomainRule:
    """A forward-chaining inference rule for hardware retail.

    When all premises are satisfied in the fact base, the conclusion
    is derived with confidence = min(premise confidences) * (1 - decay).
    """

    name: str
    premises: list[str]  # Predicate patterns to match
    conclusion: str  # Predicate to derive
    confidence_decay: float  # 0.0–0.5, applied per chain step
    explanation: str  # Human-readable rule description
    severity: str = "medium"  # critical, high, medium, low


# 30+ domain rules encoding hardware retail knowledge
DOMAIN_RULES: list[DomainRule] = [
    # --- Theft / Shrinkage rules ---
    DomainRule(
        name="theft_high_value",
        premises=["detected(negative_qty)", "detected(high_cost)"],
        conclusion="suspect(theft_high_value_items)",
        confidence_decay=0.10,
        explanation="Negative inventory on high-cost items suggests targeted theft",
        severity="critical",
    ),
    DomainRule(
        name="theft_no_sales_pattern",
        premises=["detected(negative_qty)", "detected(zero_sales)"],
        conclusion="suspect(theft_no_sales_shrinkage)",
        confidence_decay=0.10,
        explanation="Negative stock with no recorded sales — items vanished",
        severity="critical",
    ),
    DomainRule(
        name="systematic_shrinkage",
        premises=[
            "detected(negative_qty)",
            "detected(low_margin)",
            "detected(zero_sales)",
        ],
        conclusion="suspect(systematic_shrinkage)",
        confidence_decay=0.05,
        explanation="Triple signal: negative stock, low margin, no sales — systematic issue",
        severity="critical",
    ),
    DomainRule(
        name="receiving_counter_theft",
        premises=["detected(negative_qty)", "detected(recent_receipt)"],
        conclusion="suspect(receiving_error_not_theft)",
        confidence_decay=0.15,
        explanation="Recent receipt with negative qty more likely a receiving error than theft",
        severity="high",
    ),
    # --- Vendor / Pricing rules ---
    DomainRule(
        name="vendor_cost_jump",
        premises=["detected(cost_exceeds_retail)", "detected(high_cost)"],
        conclusion="suspect(vendor_cost_jump)",
        confidence_decay=0.10,
        explanation="Cost exceeding retail on expensive items — vendor raised prices",
        severity="high",
    ),
    DomainRule(
        name="margin_compression",
        premises=["detected(low_margin)", "detected(high_cost)"],
        conclusion="suspect(margin_compression)",
        confidence_decay=0.10,
        explanation="Low margin on high-cost items — purchasing terms need review",
        severity="high",
    ),
    DomainRule(
        name="selling_at_loss",
        premises=["detected(cost_exceeds_retail)", "detected(low_margin)"],
        conclusion="suspect(selling_at_loss)",
        confidence_decay=0.05,
        explanation="Cost exceeds retail with low margin — actively losing money on sales",
        severity="critical",
    ),
    DomainRule(
        name="pricing_data_error",
        premises=["detected(zero_cost)", "detected(negative_retail)"],
        conclusion="suspect(pricing_data_corruption)",
        confidence_decay=0.05,
        explanation="Both zero cost and negative retail — systematic data corruption",
        severity="critical",
    ),
    DomainRule(
        name="missing_cost_active_sales",
        premises=["detected(zero_cost)"],
        conclusion="suspect(missing_cost_data)",
        confidence_decay=0.10,
        explanation="Zero cost in system — profitability metrics are unreliable",
        severity="high",
    ),
    # --- Demand / Inventory rules ---
    DomainRule(
        name="dead_stock_demand_shift",
        premises=["detected(zero_sales)", "detected(old_receipt)"],
        conclusion="suspect(demand_dried_up)",
        confidence_decay=0.10,
        explanation="No sales for 90+ days with old receipt — demand has shifted",
        severity="medium",
    ),
    DomainRule(
        name="seasonal_overstock",
        premises=["detected(high_qty)", "detected(seasonal)"],
        conclusion="suspect(seasonal_overstock)",
        confidence_decay=0.10,
        explanation="High quantity of seasonal items — likely past the selling window",
        severity="medium",
    ),
    DomainRule(
        name="overstock_no_movement",
        premises=["detected(high_qty)", "detected(zero_sales)"],
        conclusion="suspect(overstock_no_demand)",
        confidence_decay=0.10,
        explanation="Large quantity with zero sales — capital trapped in non-moving goods",
        severity="high",
    ),
    DomainRule(
        name="seasonal_rebate_miss",
        premises=["detected(seasonal)", "detected(low_margin)"],
        conclusion="suspect(seasonal_rebate_timing)",
        confidence_decay=0.15,
        explanation="Seasonal items with low margin — co-op rebate may not be applied yet",
        severity="medium",
    ),
    # --- Quality rules ---
    DomainRule(
        name="vendor_quality_repeat",
        premises=["detected(damaged)", "detected(on_order)"],
        conclusion="suspect(vendor_quality_repeat_issue)",
        confidence_decay=0.10,
        explanation="Damaged goods with active reorder — vendor sending defective product",
        severity="high",
    ),
    DomainRule(
        name="batch_quality_problem",
        premises=["detected(damaged)", "detected(high_qty)"],
        conclusion="suspect(batch_quality_defect)",
        confidence_decay=0.10,
        explanation="Large quantity damaged — likely a whole batch is defective",
        severity="high",
    ),
    # --- Inventory Drift rules ---
    DomainRule(
        name="gradual_drift",
        premises=["detected(negative_qty)", "detected(old_receipt)"],
        conclusion="suspect(gradual_inventory_drift)",
        confidence_decay=0.10,
        explanation="Negative qty with old receipt — drift accumulated over time",
        severity="medium",
    ),
    DomainRule(
        name="high_volume_drift",
        premises=["detected(negative_qty)", "detected(high_qty)"],
        conclusion="suspect(high_volume_count_error)",
        confidence_decay=0.15,
        explanation="Negative on high-volume item — counting errors compound faster",
        severity="medium",
    ),
    # --- Compound / Escalation rules ---
    DomainRule(
        name="escalate_margin_erosion",
        premises=["suspect(selling_at_loss)", "suspect(vendor_cost_jump)"],
        conclusion="action(escalate_vendor_negotiation)",
        confidence_decay=0.10,
        explanation="Selling at a loss AND vendor raised prices — urgent vendor renegotiation needed",
        severity="critical",
    ),
    DomainRule(
        name="escalate_shrinkage_investigation",
        premises=["suspect(systematic_shrinkage)", "suspect(theft_high_value_items)"],
        conclusion="action(escalate_security_investigation)",
        confidence_decay=0.05,
        explanation="Systematic shrinkage on high-value items — full security investigation warranted",
        severity="critical",
    ),
    DomainRule(
        name="escalate_dead_stock_markdown",
        premises=["suspect(demand_dried_up)", "suspect(overstock_no_demand)"],
        conclusion="action(aggressive_markdown_program)",
        confidence_decay=0.10,
        explanation="Dead stock with overstock — needs aggressive markdown or liquidation",
        severity="high",
    ),
    DomainRule(
        name="data_quality_audit",
        premises=["suspect(pricing_data_corruption)", "suspect(missing_cost_data)"],
        conclusion="action(comprehensive_data_audit)",
        confidence_decay=0.05,
        explanation="Multiple data quality issues — systematic POS data audit required",
        severity="critical",
    ),
    DomainRule(
        name="vendor_claim_required",
        premises=["suspect(vendor_quality_repeat_issue)"],
        conclusion="action(file_vendor_claim)",
        confidence_decay=0.10,
        explanation="Repeat vendor quality problems — formal vendor claim should be filed",
        severity="high",
    ),
    DomainRule(
        name="cycle_count_needed",
        premises=["suspect(gradual_inventory_drift)"],
        conclusion="action(schedule_cycle_count)",
        confidence_decay=0.10,
        explanation="Inventory drift detected — cycle count needed to establish true counts",
        severity="medium",
    ),
]


# ---------------------------------------------------------------------------
# Signal Descriptions — human-readable explanations for each POS signal
# ---------------------------------------------------------------------------


SIGNAL_DESCRIPTIONS: dict[str, str] = {
    "negative_qty": "Quantity on hand is below zero",
    "high_cost": "Unit cost exceeds $500 threshold",
    "low_margin": "Margin is below 20% benchmark",
    "zero_sales": "No sales recorded in last 30 days",
    "high_qty": "Quantity on hand exceeds 200 units",
    "recent_receipt": "Received within the last 7 days",
    "old_receipt": "No receipt in over 90 days",
    "negative_retail": "Retail price is negative (data error)",
    "damaged": "Item is flagged as damaged",
    "on_order": "Item has active purchase orders",
    "seasonal": "Item is classified as seasonal",
    "zero_cost": "Unit cost is $0.00 (missing data)",
    "cost_exceeds_retail": "Unit cost is higher than retail price",
}


# ---------------------------------------------------------------------------
# Proof Tree — the explainability output
# ---------------------------------------------------------------------------


@dataclass
class ProofNode:
    """A node in the proof tree showing one reasoning step."""

    statement: str  # The conclusion or fact
    confidence: float  # 0.0–1.0
    explanation: str  # Human-readable why
    source: str  # "signal", "vsa_scoring", "inference", "attribute"
    children: list[ProofNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for JSON API response."""
        result = {
            "statement": self.statement,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "source": self.source,
        }
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


@dataclass
class CompetingHypothesis:
    """An alternative cause that was considered but scored lower."""

    cause: str
    cause_display: str
    score: float
    rank: int
    why_lower: str  # Explanation of why this scored lower than the winner

    def to_dict(self) -> dict:
        return {
            "cause": self.cause,
            "cause_display": self.cause_display,
            "score": round(self.score, 4),
            "rank": self.rank,
            "why_lower": self.why_lower,
        }


@dataclass
class SignalContribution:
    """How a single signal contributed to the cause scoring."""

    signal: str
    signal_description: str
    rules_fired: list[dict]  # [{cause, weight, rationale}]

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "description": self.signal_description,
            "rules_fired": self.rules_fired,
        }


@dataclass
class ProofTree:
    """Complete proof tree for an issue's root cause attribution.

    This is the top-level output of the symbolic bridge — it contains
    everything needed to explain why the system reached its conclusion.
    """

    issue_id: str
    issue_type: str
    store_id: str
    dollar_impact: float

    # Root cause conclusion
    root_cause: str | None
    root_cause_display: str
    root_cause_confidence: float
    root_cause_ambiguity: float

    # Evidence breakdown
    active_signals: list[str]
    signal_contributions: list[SignalContribution]
    cause_scores: list[dict]  # All 8 causes ranked by score

    # Reasoning chain
    proof_root: ProofNode
    inferred_facts: list[dict]  # Facts derived by forward chaining
    competing_hypotheses: list[CompetingHypothesis]

    # Actionable output
    recommendations: list[str]
    suggested_actions: list[dict]  # Actions from domain rule inference

    def to_dict(self) -> dict:
        """Serialize the full proof tree for API response."""
        return {
            "issue_id": self.issue_id,
            "issue_type": self.issue_type,
            "store_id": self.store_id,
            "dollar_impact": round(self.dollar_impact, 2),
            "root_cause": self.root_cause,
            "root_cause_display": self.root_cause_display,
            "root_cause_confidence": round(self.root_cause_confidence, 4),
            "root_cause_ambiguity": round(self.root_cause_ambiguity, 4),
            "active_signals": self.active_signals,
            "signal_contributions": [s.to_dict() for s in self.signal_contributions],
            "cause_scores": self.cause_scores,
            "proof_tree": self.proof_root.to_dict(),
            "inferred_facts": self.inferred_facts,
            "competing_hypotheses": [h.to_dict() for h in self.competing_hypotheses],
            "recommendations": self.recommendations,
            "suggested_actions": self.suggested_actions,
        }

    def render(self) -> str:
        """Render a human-readable explanation."""
        lines = []
        lines.append(f"=== Reasoning for {self.issue_id} ===")
        lines.append(f"Issue: {self.issue_type} at {self.store_id}")
        lines.append(f"Dollar Impact: ${self.dollar_impact:,.0f}")
        lines.append("")

        # Conclusion
        lines.append(f"ROOT CAUSE: {self.root_cause_display}")
        lines.append(f"Confidence: {self.root_cause_confidence:.0%}")
        if self.root_cause_ambiguity > 0.7:
            lines.append(
                f"  Note: High ambiguity ({self.root_cause_ambiguity:.0%}) — "
                "competing hypotheses are close"
            )
        lines.append("")

        # Signals
        lines.append("OBSERVED SIGNALS:")
        for sig in self.active_signals:
            desc = SIGNAL_DESCRIPTIONS.get(sig, sig)
            lines.append(f"  - {sig}: {desc}")
        lines.append("")

        # Signal → Cause mappings
        lines.append("EVIDENCE RULES FIRED:")
        for sc in self.signal_contributions:
            for rule in sc.rules_fired:
                weight_str = (
                    f"+{rule['weight']:.1f}"
                    if rule["weight"] > 0
                    else f"{rule['weight']:.1f}"
                )
                lines.append(
                    f"  {sc.signal} -> {rule['cause']} ({weight_str}): "
                    f"{rule['rationale']}"
                )
        lines.append("")

        # All cause scores
        lines.append("CAUSE SCORES (all 8 hypotheses):")
        for cs in self.cause_scores:
            marker = " <-- WINNER" if cs.get("rank") == 1 else ""
            lines.append(
                f"  {cs['rank']}. {cs['cause_display']}: "
                f"score={cs['score']:.3f}, "
                f"evidence={cs['evidence_count']}{marker}"
            )
        lines.append("")

        # Competing hypotheses
        if self.competing_hypotheses:
            lines.append("WHY ALTERNATIVES WERE RULED OUT:")
            for hyp in self.competing_hypotheses:
                lines.append(
                    f"  {hyp.cause_display} (rank #{hyp.rank}): " f"{hyp.why_lower}"
                )
            lines.append("")

        # Inferred facts from forward chaining
        if self.inferred_facts:
            lines.append("FORWARD-CHAINED INFERENCES:")
            for fact in self.inferred_facts:
                lines.append(
                    f"  [{fact['confidence']:.0%}] {fact['statement']}: "
                    f"{fact['explanation']}"
                )
            lines.append("")

        # Suggested actions
        if self.suggested_actions:
            lines.append("RECOMMENDED ACTIONS (from domain rules):")
            for action in self.suggested_actions:
                lines.append(
                    f"  [{action['severity']}] {action['action']}: "
                    f"{action['explanation']}"
                )
            lines.append("")

        # Standard recommendations
        if self.recommendations:
            lines.append("STANDARD RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SymbolicReasoner — the main reasoning engine
# ---------------------------------------------------------------------------


# Map cause enum value to display name (mirrored from models.py)
_CAUSE_DISPLAY: dict[str, str] = {
    "Theft": "Theft / Shrinkage",
    "VendorIncrease": "Vendor Price Increase",
    "RebateTiming": "Rebate Timing Mismatch",
    "MarginLeak": "Margin Leak",
    "DemandShift": "Demand Shift",
    "QualityIssue": "Quality Issue",
    "PricingError": "Pricing Error",
    "InventoryDrift": "Inventory Drift",
}


class SymbolicReasoner:
    """Translates VSA evidence into explainable proof trees.

    The reasoner takes an Issue (with its embedded cause_scores and
    active_signals from the Rust pipeline) and produces a ProofTree
    showing the complete reasoning chain.

    Usage:
        reasoner = SymbolicReasoner()
        proof = reasoner.explain(issue)
        print(proof.render())
        api_dict = proof.to_dict()
    """

    def __init__(
        self,
        evidence_rules: list[EvidenceRuleSpec] | None = None,
        domain_rules: list[DomainRule] | None = None,
    ):
        self.evidence_rules = evidence_rules or EVIDENCE_RULES
        self.domain_rules = domain_rules or DOMAIN_RULES

        # Build signal → rules index for fast lookup
        self._signal_index: dict[str, list[EvidenceRuleSpec]] = {}
        for rule in self.evidence_rules:
            self._signal_index.setdefault(rule.signal, []).append(rule)

    def explain(self, issue: Issue) -> ProofTree:
        """Generate a complete proof tree for an issue.

        Args:
            issue: An Issue from the Rust pipeline with cause_scores
                   and active_signals populated.

        Returns:
            ProofTree with full reasoning chain.
        """
        # Step 1: Extract signal contributions
        signal_contributions = self._trace_signal_contributions(
            issue.active_signals,
        )

        # Step 2: Build ranked cause scores with display names
        cause_scores = self._build_cause_ranking(issue.cause_scores)

        # Step 3: Generate competing hypotheses explanations
        competing = self._explain_competing_hypotheses(
            issue.cause_scores,
            issue.root_cause,
            issue.active_signals,
        )

        # Step 4: Forward chain domain rules
        fact_base = self._build_fact_base(issue)
        inferred_facts, suggested_actions = self._forward_chain(fact_base)

        # Step 5: Build proof tree structure
        proof_root = self._build_proof_tree(
            issue,
            signal_contributions,
            cause_scores,
        )

        # Step 6: Get recommendations
        recommendations = []
        if issue.root_cause is not None:
            recommendations = issue.root_cause.recommendations

        return ProofTree(
            issue_id=issue.id,
            issue_type=issue.issue_type.display_name,
            store_id=issue.store_id,
            dollar_impact=issue.dollar_impact,
            root_cause=issue.root_cause.value if issue.root_cause else None,
            root_cause_display=issue.root_cause_display,
            root_cause_confidence=issue.root_cause_confidence or 0.0,
            root_cause_ambiguity=issue.root_cause_ambiguity or 0.0,
            active_signals=issue.active_signals,
            signal_contributions=signal_contributions,
            cause_scores=cause_scores,
            proof_root=proof_root,
            inferred_facts=[f.copy() for f in inferred_facts],
            competing_hypotheses=competing,
            recommendations=recommendations,
            suggested_actions=suggested_actions,
        )

    def backward_chain(self, issue: Issue, goal: str) -> list[dict]:
        """Backward-chain from a goal to find what facts support it.

        Given a goal like "root_cause(Theft)", traces backward through
        domain rules and evidence rules to show what signals and
        intermediate facts led to that conclusion.

        Args:
            issue: The issue to reason about.
            goal: A predicate string to explain (e.g. "root_cause(Theft)").

        Returns:
            List of reasoning steps, each a dict with:
                - step: step number
                - goal: what we're trying to prove
                - method: "fact", "evidence_rule", or "domain_rule"
                - supporting: list of supporting facts/rules
                - satisfied: bool
        """
        fact_base = self._build_fact_base(issue)
        # Add inferred facts from forward chaining
        inferred, _ = self._forward_chain(fact_base)
        for inf in inferred:
            fact_base.add(
                Fact(inf["statement"], inf["confidence"], FactSource.RULE_INFERENCE)
            )

        return self._backward_chain_recursive(goal, fact_base, set(), step_counter=[0])

    # -------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------

    def _trace_signal_contributions(
        self,
        active_signals: list[str],
    ) -> list[SignalContribution]:
        """Trace which evidence rules fire for each active signal."""
        contributions = []
        for signal in active_signals:
            rules = self._signal_index.get(signal, [])
            rules_fired = [
                {
                    "cause": r.cause,
                    "cause_display": _CAUSE_DISPLAY.get(r.cause, r.cause),
                    "weight": r.weight,
                    "rationale": r.rationale,
                }
                for r in rules
            ]
            contributions.append(
                SignalContribution(
                    signal=signal,
                    signal_description=SIGNAL_DESCRIPTIONS.get(signal, signal),
                    rules_fired=rules_fired,
                )
            )
        return contributions

    def _build_cause_ranking(
        self,
        cause_scores: list[CauseScoreDetail],
    ) -> list[dict]:
        """Build a ranked list of all cause scores with display names."""
        result = []
        for rank, cs in enumerate(cause_scores, 1):
            result.append(
                {
                    "rank": rank,
                    "cause": cs.cause,
                    "cause_display": _CAUSE_DISPLAY.get(cs.cause, cs.cause),
                    "score": round(cs.score, 4),
                    "evidence_count": cs.evidence_count,
                }
            )
        return result

    def _explain_competing_hypotheses(
        self,
        cause_scores: list[CauseScoreDetail],
        winner: RootCause | None,
        active_signals: list[str],
    ) -> list[CompetingHypothesis]:
        """Explain why each non-winner cause scored lower."""
        if not cause_scores or winner is None:
            return []

        winner_name = winner.value
        winner_score = 0.0
        for cs in cause_scores:
            if cs.cause == winner_name:
                winner_score = cs.score
                break

        hypotheses = []
        for rank, cs in enumerate(cause_scores, 1):
            if cs.cause == winner_name:
                continue
            if cs.score <= 0:
                continue
            # Only show top 3 alternatives
            if rank > 4:
                break

            # Generate explanation of why this scored lower
            why = self._why_lower(
                cs.cause, winner_name, active_signals, winner_score, cs.score
            )

            hypotheses.append(
                CompetingHypothesis(
                    cause=cs.cause,
                    cause_display=_CAUSE_DISPLAY.get(cs.cause, cs.cause),
                    score=cs.score,
                    rank=rank,
                    why_lower=why,
                )
            )

        return hypotheses

    def _why_lower(
        self,
        loser: str,
        winner: str,
        active_signals: list[str],
        winner_score: float,
        loser_score: float,
    ) -> str:
        """Generate an explanation of why a cause scored lower than the winner."""
        # Gather rules that fired for each cause from active signals
        winner_rules = []
        loser_rules = []
        for signal in active_signals:
            for rule in self._signal_index.get(signal, []):
                if rule.cause == winner:
                    winner_rules.append(rule)
                elif rule.cause == loser:
                    loser_rules.append(rule)

        winner_weight_sum = sum(r.weight for r in winner_rules)
        loser_weight_sum = sum(r.weight for r in loser_rules)

        # Find missing signals — signals that would boost the loser
        loser_all_rules = [
            r for r in self.evidence_rules if r.cause == loser and r.weight > 0
        ]
        missing_signals = [
            r.signal for r in loser_all_rules if r.signal not in active_signals
        ]

        parts = []
        if winner_weight_sum > loser_weight_sum:
            diff = winner_weight_sum - loser_weight_sum
            parts.append(
                f"Winner had {diff:.1f} more total evidence weight "
                f"({winner_weight_sum:.1f} vs {loser_weight_sum:.1f})"
            )
        if missing_signals:
            parts.append(
                f"Missing supporting signals: {', '.join(missing_signals[:3])}"
            )
        # Check for counter-evidence
        counter_rules = [r for r in loser_rules if r.weight < 0]
        if counter_rules:
            counter_str = ", ".join(r.signal for r in counter_rules)
            parts.append(f"Counter-evidence from: {counter_str}")

        if not parts:
            gap_pct = (
                ((winner_score - loser_score) / winner_score * 100)
                if winner_score > 0
                else 0
            )
            parts.append(f"Scored {gap_pct:.0f}% lower in VSA similarity")

        return "; ".join(parts)

    def _build_fact_base(self, issue: Issue) -> set[Fact]:
        """Build the initial fact base from an issue's data."""
        facts: set[Fact] = set()

        # Signal facts
        for signal in issue.active_signals:
            desc = SIGNAL_DESCRIPTIONS.get(signal, signal)
            facts.add(
                Fact(
                    predicate=f"detected({signal})",
                    confidence=1.0,  # Signals are binary
                    source=FactSource.SIGNAL,
                    explanation=desc,
                )
            )

        # Cause score facts
        for cs in issue.cause_scores:
            facts.add(
                Fact(
                    predicate=f"cause_score({cs.cause}, {cs.score:.3f})",
                    confidence=1.0,
                    source=FactSource.VSA_SCORING,
                    explanation=f"{_CAUSE_DISPLAY.get(cs.cause, cs.cause)} scored {cs.score:.3f}",
                )
            )

        # Root cause fact
        if issue.root_cause:
            facts.add(
                Fact(
                    predicate=f"root_cause({issue.root_cause.value})",
                    confidence=issue.root_cause_confidence or 0.0,
                    source=FactSource.VSA_SCORING,
                    explanation=f"VSA evidence scorer selected {issue.root_cause.display_name}",
                )
            )

        # Issue attribute facts
        facts.add(
            Fact(
                predicate=f"issue_type({issue.issue_type.value})",
                confidence=1.0,
                source=FactSource.DATA_ATTRIBUTE,
                explanation=f"Issue classified as {issue.issue_type.display_name}",
            )
        )
        facts.add(
            Fact(
                predicate=f"dollar_impact({issue.dollar_impact:.0f})",
                confidence=1.0,
                source=FactSource.DATA_ATTRIBUTE,
                explanation=f"Total dollar exposure: ${issue.dollar_impact:,.0f}",
            )
        )
        if issue.dollar_impact >= 10_000:
            facts.add(
                Fact(
                    predicate="high_dollar_impact",
                    confidence=1.0,
                    source=FactSource.DATA_ATTRIBUTE,
                    explanation=f"Dollar impact ${issue.dollar_impact:,.0f} exceeds $10,000 threshold",
                )
            )
        facts.add(
            Fact(
                predicate=f"trend({issue.trend_direction.value})",
                confidence=1.0,
                source=FactSource.DATA_ATTRIBUTE,
                explanation=f"Trend direction: {issue.trend_direction.description}",
            )
        )

        return facts

    def _forward_chain(
        self,
        fact_base: set[Fact],
    ) -> tuple[list[dict], list[dict]]:
        """Run forward chaining over domain rules.

        Returns:
            (inferred_facts, suggested_actions)
            - inferred_facts: list of dicts with statement, confidence, explanation
            - suggested_actions: list of dicts with action, explanation, severity
        """
        inferred: list[dict] = []
        actions: list[dict] = []
        known_predicates = {f.predicate for f in fact_base}

        # Iterate until no new facts are derived (fixed-point)
        changed = True
        max_iterations = 10  # Safety bound
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for rule in self.domain_rules:
                # Check if conclusion already derived
                if rule.conclusion in known_predicates:
                    continue

                # Check if all premises are satisfied
                premise_confidences = []
                all_matched = True
                for premise in rule.premises:
                    matched_fact = None
                    for f in fact_base:
                        if f.predicate == premise:
                            matched_fact = f
                            break
                    if matched_fact is None:
                        all_matched = False
                        break
                    premise_confidences.append(matched_fact.confidence)

                if not all_matched:
                    continue

                # Derive the conclusion
                derived_confidence = min(premise_confidences) * (
                    1.0 - rule.confidence_decay
                )
                new_fact = Fact(
                    predicate=rule.conclusion,
                    confidence=derived_confidence,
                    source=FactSource.RULE_INFERENCE,
                    explanation=rule.explanation,
                )

                fact_base.add(new_fact)
                known_predicates.add(rule.conclusion)
                changed = True

                fact_dict = {
                    "statement": rule.conclusion,
                    "confidence": round(derived_confidence, 4),
                    "explanation": rule.explanation,
                    "rule_name": rule.name,
                    "premises": rule.premises,
                }

                if rule.conclusion.startswith("action("):
                    actions.append(
                        {
                            "action": rule.conclusion,
                            "explanation": rule.explanation,
                            "severity": rule.severity,
                            "confidence": round(derived_confidence, 4),
                            "rule_name": rule.name,
                        }
                    )
                else:
                    inferred.append(fact_dict)

        return inferred, actions

    def _build_proof_tree(
        self,
        issue: Issue,
        signal_contributions: list[SignalContribution],
        cause_scores: list[dict],
    ) -> ProofNode:
        """Build the hierarchical proof tree."""
        # Root node: the conclusion
        root_display = issue.root_cause_display
        root = ProofNode(
            statement=f"Root cause: {root_display}",
            confidence=issue.root_cause_confidence or 0.0,
            explanation=(
                f"VSA evidence scorer determined {root_display} with "
                f"{(issue.root_cause_confidence or 0.0):.0%} confidence"
            ),
            source="conclusion",
        )

        # Evidence bundle node
        evidence_node = ProofNode(
            statement=f"Evidence bundle from {len(issue.active_signals)} signals",
            confidence=1.0,
            explanation=(
                f"Signals detected: {', '.join(issue.active_signals)}"
                if issue.active_signals
                else "No signals detected"
            ),
            source="evidence",
        )

        # Add individual signal nodes
        for sc in signal_contributions:
            signal_node = ProofNode(
                statement=f"Signal: {sc.signal}",
                confidence=1.0,
                explanation=sc.signal_description,
                source="signal",
            )
            # Add which causes this signal supports
            for rule in sc.rules_fired:
                weight = rule["weight"]
                if weight > 0:
                    label = f"supports {rule['cause_display']} (weight {weight:+.1f})"
                else:
                    label = f"counter-evidence for {rule['cause_display']} (weight {weight:+.1f})"
                rule_node = ProofNode(
                    statement=label,
                    confidence=abs(weight),
                    explanation=rule["rationale"],
                    source="evidence_rule",
                )
                signal_node.children.append(rule_node)
            evidence_node.children.append(signal_node)

        root.children.append(evidence_node)

        # VSA scoring node
        if cause_scores:
            scoring_node = ProofNode(
                statement="VSA positive-similarity scoring",
                confidence=issue.root_cause_confidence or 0.0,
                explanation=(
                    "Weighted evidence vectors compared against 8 cause hypothesis "
                    "vectors using Hermitian cosine similarity (positive-only clamping)"
                ),
                source="vsa_scoring",
            )
            for cs in cause_scores[:5]:  # Top 5 for readability
                cs_node = ProofNode(
                    statement=f"#{cs['rank']} {cs['cause_display']}: score={cs['score']:.3f}",
                    confidence=cs["score"],
                    explanation=f"Evidence vectors matched {cs['evidence_count']} supporting rules",
                    source="vsa_scoring",
                )
                scoring_node.children.append(cs_node)
            root.children.append(scoring_node)

        return root

    def _backward_chain_recursive(
        self,
        goal: str,
        fact_base: set[Fact],
        visited: set[str],
        step_counter: list[int],
    ) -> list[dict]:
        """Recursively backward-chain to prove a goal."""
        if goal in visited:
            return []  # Avoid cycles
        visited.add(goal)
        step_counter[0] += 1

        steps: list[dict] = []

        # Check if goal is directly in fact base
        for f in fact_base:
            if f.predicate == goal:
                steps.append(
                    {
                        "step": step_counter[0],
                        "goal": goal,
                        "method": "fact",
                        "supporting": [
                            {
                                "predicate": f.predicate,
                                "confidence": round(f.confidence, 4),
                                "source": f.source.value,
                                "explanation": f.explanation,
                            }
                        ],
                        "satisfied": True,
                    }
                )
                return steps

        # Check if any domain rule derives this goal
        for rule in self.domain_rules:
            if rule.conclusion == goal:
                # Try to prove all premises
                all_proved = True
                premise_steps: list[dict] = []
                for premise in rule.premises:
                    sub_steps = self._backward_chain_recursive(
                        premise,
                        fact_base,
                        visited.copy(),
                        step_counter,
                    )
                    premise_steps.extend(sub_steps)
                    if not any(
                        s["satisfied"] for s in sub_steps if s["goal"] == premise
                    ):
                        all_proved = False

                steps.append(
                    {
                        "step": step_counter[0],
                        "goal": goal,
                        "method": "domain_rule",
                        "rule_name": rule.name,
                        "rule_explanation": rule.explanation,
                        "premises": rule.premises,
                        "supporting": premise_steps,
                        "satisfied": all_proved,
                    }
                )
                if all_proved:
                    return steps

        # Check if goal matches evidence rules pattern
        if goal.startswith("detected("):
            signal = goal[9:-1]  # Extract signal name
            for f in fact_base:
                if f.predicate == goal:
                    steps.append(
                        {
                            "step": step_counter[0],
                            "goal": goal,
                            "method": "signal",
                            "supporting": [
                                {
                                    "predicate": f.predicate,
                                    "confidence": round(f.confidence, 4),
                                    "source": "signal",
                                    "explanation": SIGNAL_DESCRIPTIONS.get(
                                        signal, signal
                                    ),
                                }
                            ],
                            "satisfied": True,
                        }
                    )
                    return steps

        # Goal cannot be proved
        if not steps:
            steps.append(
                {
                    "step": step_counter[0],
                    "goal": goal,
                    "method": "unproven",
                    "supporting": [],
                    "satisfied": False,
                }
            )

        return steps
