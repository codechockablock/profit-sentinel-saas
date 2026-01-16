"""
reasoning/dsl.py - Domain-Specific Language for Reasoning

Provides a fluent API for defining and executing symbolic reasoning
in a more Pythonic way.

Example:
    from reasoning.dsl import ReasoningDSL

    r = ReasoningDSL()

    # Define predicates
    r.predicate("has_anomaly", 2)
    r.predicate("severity", 2)
    r.predicate("critical_alert", 1)

    # Add facts
    r.fact("has_anomaly", "SKU123", "low_stock")
    r.fact("severity", "low_stock", "medium")

    # Add rules
    r.rule("critical_alert", X) \
        .when("has_anomaly", X, A) \
        .and_("severity", A, "critical") \
        .done()

    # Query
    result = r.query("critical_alert", X)
    for binding in result:
        print(f"Critical alert for: {binding['X']}")
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Iterator
from dataclasses import dataclass

from .terms import Term, Var, Atom, Clause, Predicate, TermLike
from .knowledge_base import KnowledgeBase
from .inference import backward_chain, backward_chain_all, forward_chain, ProofTree
from .unification import Substitution


# Create common variables for DSL
X = Var("X")
Y = Var("Y")
Z = Var("Z")
A = Var("A")
B = Var("B")
C = Var("C")
SKU = Var("SKU")
ANOMALY = Var("ANOMALY")
SEVERITY = Var("SEVERITY")
MARGIN = Var("MARGIN")
QTY = Var("QTY")
VENDOR = Var("VENDOR")
CATEGORY = Var("CATEGORY")


class RuleBuilder:
    """Fluent builder for rules."""

    def __init__(self, dsl: 'ReasoningDSL', head_name: str, *args):
        self.dsl = dsl
        self.head = Term(head_name, *args)
        self.body: List[Term] = []
        self._name: Optional[str] = None
        self._description: Optional[str] = None

    def when(self, predicate: str, *args) -> 'RuleBuilder':
        """Add first condition."""
        self.body.append(Term(predicate, *args))
        return self

    def and_(self, predicate: str, *args) -> 'RuleBuilder':
        """Add another condition (AND)."""
        self.body.append(Term(predicate, *args))
        return self

    def named(self, name: str) -> 'RuleBuilder':
        """Set rule name."""
        self._name = name
        return self

    def described(self, description: str) -> 'RuleBuilder':
        """Set rule description."""
        self._description = description
        return self

    def done(self) -> None:
        """Finalize and add rule to knowledge base."""
        self.dsl.kb.add_rule(
            self.head,
            *self.body,
            name=self._name,
            description=self._description
        )


class ReasoningDSL:
    """Domain-specific language for symbolic reasoning.

    Provides a fluent, Pythonic API for building knowledge bases
    and running inference.
    """

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        """Initialize DSL.

        Args:
            kb: Existing knowledge base (creates new if None)
        """
        self.kb = kb or KnowledgeBase()
        self._predicates: Dict[str, Predicate] = {}

    def predicate(self, name: str, arity: int) -> Predicate:
        """Declare a predicate.

        Args:
            name: Predicate name
            arity: Number of arguments

        Returns:
            Predicate object for creating terms
        """
        pred = Predicate(name, arity)
        self._predicates[name] = pred
        return pred

    def fact(self, predicate: str, *args, **kwargs) -> None:
        """Add a fact.

        Args:
            predicate: Predicate name
            *args: Arguments (converted to Atoms if not Terms)
            **kwargs: Metadata (confidence, source, etc.)
        """
        term = Term(predicate, *args)
        self.kb.add_fact(
            term,
            confidence=kwargs.get("confidence", 1.0),
            source=kwargs.get("source"),
            metadata=kwargs.get("metadata", {})
        )

    def rule(self, head_pred: str, *args) -> RuleBuilder:
        """Start building a rule.

        Args:
            head_pred: Head predicate name
            *args: Head arguments

        Returns:
            RuleBuilder for fluent construction

        Example:
            r.rule("alert", X).when("anomaly", X, A).and_("severe", A).done()
        """
        return RuleBuilder(self, head_pred, *args)

    def query(
        self,
        predicate: str,
        *args,
        all_solutions: bool = False,
        max_results: int = 100
    ) -> Union[ProofTree, List[ProofTree]]:
        """Query the knowledge base.

        Args:
            predicate: Predicate to query
            *args: Arguments (may include variables)
            all_solutions: If True, return all solutions
            max_results: Maximum results for all_solutions

        Returns:
            ProofTree or list of ProofTrees
        """
        goal = Term(predicate, *args)

        if all_solutions:
            return backward_chain_all(self.kb, goal, max_results)
        else:
            return backward_chain(self.kb, goal)

    def prove(self, predicate: str, *args) -> bool:
        """Check if a goal is provable.

        Args:
            predicate: Predicate name
            *args: Arguments

        Returns:
            True if goal can be proved
        """
        proof = self.query(predicate, *args)
        return proof.is_valid

    def solutions(self, predicate: str, *args, max_results: int = 100) -> Iterator[Dict[str, Any]]:
        """Get all solutions as dictionaries.

        Args:
            predicate: Predicate name
            *args: Arguments
            max_results: Maximum results

        Yields:
            Dictionaries mapping variable names to values
        """
        proofs = self.query(predicate, *args, all_solutions=True, max_results=max_results)
        for proof in proofs:
            yield {
                name: val.value if isinstance(val, Atom) else str(val)
                for name, val in proof.bindings.items()
            }

    def derive_all(self, max_iterations: int = 1000) -> List[Term]:
        """Run forward chaining to derive all facts.

        Args:
            max_iterations: Maximum iterations

        Returns:
            List of newly derived facts
        """
        new_facts, _ = forward_chain(self.kb, max_iterations)
        return new_facts

    def explain(self, predicate: str, *args) -> str:
        """Get explanation for a query.

        Args:
            predicate: Predicate name
            *args: Arguments

        Returns:
            Human-readable explanation
        """
        proof = self.query(predicate, *args)
        return proof.explain()

    def clear(self) -> None:
        """Clear all facts and rules."""
        self.kb.clear()

    def save(self, path: str, format: str = "yaml") -> None:
        """Save knowledge base to file.

        Args:
            path: File path
            format: "yaml" or "json"
        """
        if format == "yaml":
            self.kb.to_yaml(path)
        else:
            self.kb.to_json(path)

    def load(self, path: str, format: str = "yaml") -> None:
        """Load knowledge base from file.

        Args:
            path: File path
            format: "yaml" or "json"
        """
        if format == "yaml":
            self.kb = KnowledgeBase.from_yaml(path)
        else:
            self.kb = KnowledgeBase.from_json(path)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_dsl: Optional[ReasoningDSL] = None


def get_dsl() -> ReasoningDSL:
    """Get default DSL instance."""
    global _default_dsl
    if _default_dsl is None:
        _default_dsl = ReasoningDSL()
    return _default_dsl


def define_fact(predicate: str, *args, **kwargs) -> None:
    """Add fact to default DSL."""
    get_dsl().fact(predicate, *args, **kwargs)


def define_rule(head_pred: str, *args) -> RuleBuilder:
    """Start rule definition on default DSL."""
    return get_dsl().rule(head_pred, *args)


def query(predicate: str, *args, **kwargs) -> Union[ProofTree, List[ProofTree]]:
    """Query default DSL."""
    return get_dsl().query(predicate, *args, **kwargs)


# =============================================================================
# PROFIT SENTINEL SPECIFIC DSL EXTENSIONS
# =============================================================================

class ProfitSentinelDSL(ReasoningDSL):
    """Extended DSL with Profit Sentinel-specific predicates.

    Pre-defines common predicates and rules for retail anomaly detection.
    """

    def __init__(self, kb: Optional[KnowledgeBase] = None):
        super().__init__(kb)
        self._setup_predicates()

    def _setup_predicates(self):
        """Define standard Profit Sentinel predicates."""
        # Entity predicates
        self.predicate("sku", 1)  # sku(SKU)
        self.predicate("vendor", 1)  # vendor(Vendor)
        self.predicate("category", 1)  # category(Cat)

        # Relationship predicates
        self.predicate("sku_vendor", 2)  # sku_vendor(SKU, Vendor)
        self.predicate("sku_category", 2)  # sku_category(SKU, Category)

        # Metric predicates
        self.predicate("quantity", 2)  # quantity(SKU, Qty)
        self.predicate("margin", 2)  # margin(SKU, Margin)
        self.predicate("velocity", 2)  # velocity(SKU, Velocity)
        self.predicate("days_of_supply", 2)  # days_of_supply(SKU, Days)

        # Anomaly predicates
        self.predicate("has_anomaly", 2)  # has_anomaly(SKU, AnomalyType)
        self.predicate("anomaly_severity", 2)  # anomaly_severity(Type, Severity)

        # Comparison predicates
        self.predicate("less_than", 2)  # less_than(X, Y)
        self.predicate("greater_than", 2)  # greater_than(X, Y)
        self.predicate("between", 3)  # between(X, Low, High)

        # Alert predicates
        self.predicate("alert", 2)  # alert(SKU, AlertType)
        self.predicate("critical_alert", 1)  # critical_alert(SKU)
        self.predicate("needs_reorder", 1)  # needs_reorder(SKU)

        # Root cause predicates
        self.predicate("root_cause", 2)  # root_cause(Anomaly, Cause)
        self.predicate("recommended_action", 2)  # recommended_action(SKU, Action)

    def add_sku_data(
        self,
        sku: str,
        quantity: float,
        margin: float,
        velocity: float,
        vendor: Optional[str] = None,
        category: Optional[str] = None
    ) -> None:
        """Add SKU data as facts.

        Args:
            sku: SKU identifier
            quantity: Current quantity
            margin: Margin percentage
            velocity: Daily velocity
            vendor: Optional vendor
            category: Optional category
        """
        self.fact("sku", sku)
        self.fact("quantity", sku, quantity)
        self.fact("margin", sku, margin)
        self.fact("velocity", sku, velocity)

        if vendor:
            self.fact("vendor", vendor)
            self.fact("sku_vendor", sku, vendor)

        if category:
            self.fact("category", category)
            self.fact("sku_category", sku, category)

    def define_anomaly_rules(self) -> None:
        """Define standard anomaly detection rules."""
        # Low stock rule
        self.rule("has_anomaly", SKU, "low_stock") \
            .when("quantity", SKU, QTY) \
            .and_("less_than", QTY, 10) \
            .named("low_stock_detection") \
            .done()

        # Margin leak rule
        self.rule("has_anomaly", SKU, "margin_leak") \
            .when("margin", SKU, MARGIN) \
            .and_("less_than", MARGIN, 0.15) \
            .named("margin_leak_detection") \
            .done()

        # Critical alert rule
        self.rule("critical_alert", SKU) \
            .when("has_anomaly", SKU, ANOMALY) \
            .and_("anomaly_severity", ANOMALY, "critical") \
            .named("critical_alert_rule") \
            .done()

        # Reorder rule
        self.rule("needs_reorder", SKU) \
            .when("has_anomaly", SKU, "low_stock") \
            .and_("velocity", SKU, X) \
            .and_("greater_than", X, 1.0) \
            .named("reorder_rule") \
            .done()

        # Define severities
        self.fact("anomaly_severity", "low_stock", "medium")
        self.fact("anomaly_severity", "margin_leak", "high")
        self.fact("anomaly_severity", "negative_inventory", "critical")
        self.fact("anomaly_severity", "dead_item", "medium")

    def detect_anomalies(self, sku: str) -> List[str]:
        """Detect all anomalies for a SKU.

        Args:
            sku: SKU to check

        Returns:
            List of anomaly types detected
        """
        anomalies = []
        for solution in self.solutions("has_anomaly", sku, ANOMALY):
            anomalies.append(solution.get("ANOMALY", "unknown"))
        return anomalies
