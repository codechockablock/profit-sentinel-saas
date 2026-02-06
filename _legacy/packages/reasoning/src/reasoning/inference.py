"""
reasoning/inference.py - Forward and Backward Chaining Inference

Implements two classical inference strategies:

FORWARD CHAINING (Data-Driven):
    Start with known facts and apply rules to derive new facts
    until no more conclusions can be drawn.

    Use when: You have data and want to find all derivable conclusions.

BACKWARD CHAINING (Goal-Driven):
    Start with a goal and work backwards to find supporting facts.
    Returns a proof tree showing how the goal was derived.

    Use when: You have a specific query and want to know if/how it's true.

Both return proof trees for explainability.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .knowledge_base import KnowledgeBase
from .terms import Clause, Term
from .unification import Substitution, compose_substitutions, substitute, unify


class ProofStatus(Enum):
    """Status of a proof attempt."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Some goals proved, others failed
    TIMEOUT = "timeout"


@dataclass
class ProofNode:
    """Node in a proof tree.

    Represents a single step in a derivation.
    """

    goal: Term
    rule_used: Clause | None = None
    substitution: Substitution = field(default_factory=dict)
    children: list[ProofNode] = field(default_factory=list)
    status: ProofStatus = ProofStatus.PARTIAL
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf node (fact or failure)."""
        return len(self.children) == 0

    @property
    def is_success(self) -> bool:
        return self.status == ProofStatus.SUCCESS

    def __repr__(self) -> str:
        status_mark = (
            "✓"
            if self.is_success
            else "✗" if self.status == ProofStatus.FAILURE else "?"
        )
        return f"[{status_mark}] {self.goal}"


@dataclass
class ProofTree:
    """Complete proof tree for a query.

    Represents the full derivation of a goal, including all
    subgoals and their proofs.
    """

    root: ProofNode
    query: Term
    bindings: Substitution = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True if the proof succeeded."""
        return self.root.status == ProofStatus.SUCCESS

    @property
    def status(self) -> ProofStatus:
        return self.root.status

    def get_answer(self) -> Term | None:
        """Get the instantiated query with bindings applied."""
        if not self.is_valid:
            return None
        return substitute(self.query, self.bindings)

    def explain(self, indent: int = 0) -> str:
        """Generate human-readable explanation."""
        return self._explain_node(self.root, indent)

    def _explain_node(self, node: ProofNode, indent: int) -> str:
        lines = []
        prefix = "  " * indent
        status = "✓" if node.is_success else "✗"

        if node.rule_used:
            lines.append(f"{prefix}{status} {node.goal}")
            lines.append(f"{prefix}  by rule: {node.rule_used.head} :- ...")
        else:
            lines.append(f"{prefix}{status} {node.goal} (fact)")

        for child in node.children:
            lines.append(self._explain_node(child, indent + 1))

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export proof tree to dictionary."""
        return {
            "query": str(self.query),
            "valid": self.is_valid,
            "status": self.status.value,
            "bindings": {k: str(v) for k, v in self.bindings.items()},
            "tree": self._node_to_dict(self.root),
        }

    def _node_to_dict(self, node: ProofNode) -> dict[str, Any]:
        return {
            "goal": str(node.goal),
            "status": node.status.value,
            "depth": node.depth,
            "rule": str(node.rule_used) if node.rule_used else None,
            "children": [self._node_to_dict(c) for c in node.children],
        }


# =============================================================================
# FORWARD CHAINING
# =============================================================================


def forward_chain(
    kb: KnowledgeBase, max_iterations: int = 1000
) -> tuple[list[Term], int]:
    """Forward chaining inference.

    Applies all applicable rules to derive new facts until
    no more facts can be derived (fixed point).

    Args:
        kb: Knowledge base with facts and rules
        max_iterations: Maximum iterations to prevent infinite loops

    Returns:
        (new_facts, iterations) - list of derived facts and iteration count
    """
    new_facts: list[Term] = []
    derived: set[str] = set()  # Track derived facts by string representation

    # Initialize with existing facts
    for clause in kb:
        if clause.is_fact:
            derived.add(str(clause.head))

    iterations = 0
    changed = True

    while changed and iterations < max_iterations:
        changed = False
        iterations += 1

        # Try each rule
        for clause in kb:
            if not clause.is_rule:
                continue

            # Find all ways to satisfy the body
            for theta in _satisfy_body(kb, clause.body, derived):
                # Instantiate head with bindings
                new_head = substitute(clause.head, theta)

                # Check if this is a new fact
                key = str(new_head)
                if key not in derived:
                    derived.add(key)
                    new_facts.append(new_head)
                    kb.add_fact(new_head, source="forward_chain")
                    changed = True

    return new_facts, iterations


def _satisfy_body(
    kb: KnowledgeBase, body: list[Term], derived: set[str]
) -> Iterator[Substitution]:
    """Find all substitutions that satisfy a rule body."""
    if not body:
        yield {}
        return

    first, *rest = body

    # Find facts matching first goal
    for theta in kb.query(first):
        # Recursively satisfy rest with current bindings
        rest_subst = [substitute(t, theta) for t in rest]
        for theta2 in _satisfy_body(kb, rest_subst, derived):
            yield compose_substitutions(theta2, theta)


# =============================================================================
# BACKWARD CHAINING
# =============================================================================


def backward_chain(kb: KnowledgeBase, goal: Term, max_depth: int = 100) -> ProofTree:
    """Backward chaining inference with proof tree.

    Works backwards from goal to find supporting facts.
    Returns complete proof tree for explainability.

    Args:
        kb: Knowledge base with facts and rules
        goal: Goal to prove
        max_depth: Maximum proof depth

    Returns:
        ProofTree showing derivation (valid or not)
    """
    root = ProofNode(goal=goal, depth=0)
    bindings: Substitution = {}

    success, bindings = _prove_goal(kb, goal, root, bindings, 0, max_depth)

    root.status = ProofStatus.SUCCESS if success else ProofStatus.FAILURE

    return ProofTree(root=root, query=goal, bindings=bindings)


def _prove_goal(
    kb: KnowledgeBase,
    goal: Term,
    node: ProofNode,
    theta: Substitution,
    depth: int,
    max_depth: int,
) -> tuple[bool, Substitution]:
    """Recursively prove a goal."""
    if depth > max_depth:
        node.status = ProofStatus.TIMEOUT
        return False, theta

    # Apply current substitution
    goal = substitute(goal, theta)
    node.goal = goal

    # Try to unify with facts first
    for fact_theta in kb.query(goal):
        node.status = ProofStatus.SUCCESS
        return True, compose_substitutions(fact_theta, theta)

    # Try rules
    for clause in kb.query_rules(goal):
        # Unify goal with rule head
        head_theta = unify(goal, clause.head)
        if head_theta is None:
            continue

        combined = compose_substitutions(head_theta, theta)

        # Prove all body goals
        body_nodes = []
        all_proved = True
        current_theta = combined

        for body_goal in clause.body:
            child_node = ProofNode(
                goal=substitute(body_goal, current_theta), depth=depth + 1
            )
            body_nodes.append(child_node)

            success, new_theta = _prove_goal(
                kb, body_goal, child_node, current_theta, depth + 1, max_depth
            )

            if not success:
                child_node.status = ProofStatus.FAILURE
                all_proved = False
                break

            current_theta = new_theta
            child_node.status = ProofStatus.SUCCESS

        if all_proved:
            node.rule_used = clause
            node.children = body_nodes
            node.substitution = current_theta
            node.status = ProofStatus.SUCCESS
            return True, current_theta

    node.status = ProofStatus.FAILURE
    return False, theta


def backward_chain_all(
    kb: KnowledgeBase, goal: Term, max_results: int = 100, max_depth: int = 100
) -> list[ProofTree]:
    """Find all proofs for a goal.

    Args:
        kb: Knowledge base
        goal: Goal to prove
        max_results: Maximum number of proofs to return
        max_depth: Maximum proof depth

    Returns:
        List of valid proof trees
    """
    results = []

    for bindings in _prove_all(kb, [goal], {}, 0, max_depth):
        if len(results) >= max_results:
            break

        root = ProofNode(goal=substitute(goal, bindings), status=ProofStatus.SUCCESS)
        tree = ProofTree(root=root, query=goal, bindings=bindings)
        results.append(tree)

    return results


def _prove_all(
    kb: KnowledgeBase,
    goals: list[Term],
    theta: Substitution,
    depth: int,
    max_depth: int,
) -> Iterator[Substitution]:
    """Generate all solutions for a list of goals."""
    if depth > max_depth:
        return

    if not goals:
        yield theta
        return

    goal, *rest = goals
    goal = substitute(goal, theta)

    # Try facts
    for fact_theta in kb.query(goal):
        combined = compose_substitutions(fact_theta, theta)
        yield from _prove_all(kb, rest, combined, depth, max_depth)

    # Try rules
    for clause in kb.query_rules(goal):
        head_theta = unify(goal, clause.head)
        if head_theta is None:
            continue

        combined = compose_substitutions(head_theta, theta)
        new_goals = clause.body + rest
        yield from _prove_all(kb, new_goals, combined, depth + 1, max_depth)


# =============================================================================
# SPECIALIZED INFERENCE
# =============================================================================


def abductive_inference(
    kb: KnowledgeBase, observation: Term, hypotheses: list[Term], max_depth: int = 50
) -> list[tuple[Term, ProofTree]]:
    """Abductive reasoning: find hypotheses that explain observation.

    Given an observation and candidate hypotheses, returns
    hypotheses that, if true, would prove the observation.

    Args:
        kb: Knowledge base
        observation: Observed fact to explain
        hypotheses: Candidate explanations
        max_depth: Maximum proof depth

    Returns:
        List of (hypothesis, proof) pairs that explain observation
    """
    explanations = []

    for hyp in hypotheses:
        # Temporarily add hypothesis
        kb.add_fact(hyp, source="hypothesis")

        # Try to prove observation
        proof = backward_chain(kb, observation, max_depth)

        if proof.is_valid:
            explanations.append((hyp, proof))

        # Remove hypothesis
        kb.retract_fact(hyp)

    return explanations


def counterfactual_query(
    kb: KnowledgeBase, intervention: Term, query: Term, max_depth: int = 50
) -> tuple[ProofTree, ProofTree]:
    """Counterfactual reasoning: what if X were true/false?

    Args:
        kb: Knowledge base
        intervention: Fact to add (or its negation to retract)
        query: Query to evaluate

    Returns:
        (proof_without_intervention, proof_with_intervention)
    """
    # Prove without intervention
    proof_without = backward_chain(kb, query, max_depth)

    # Add intervention and prove
    kb.add_fact(intervention, source="intervention")
    proof_with = backward_chain(kb, query, max_depth)
    kb.retract_fact(intervention)

    return proof_without, proof_with
