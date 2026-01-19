"""
reasoning - Hard Symbolic Reasoning Engine

Prolog-style logic programming for provable conclusions.

This module implements formal logic operations:
- Knowledge base with Horn clauses
- Unification algorithm
- Forward chaining (data-driven)
- Backward chaining (goal-driven)
- Proof tree generation

Example:
    from reasoning import KnowledgeBase, Term, Var, backward_chain

    kb = KnowledgeBase()

    # Define rules
    kb.add_rule(
        head=Term("margin_leak", Var("SKU")),
        body=[
            Term("actual_margin", Var("SKU"), Var("M")),
            Term("less_than", Var("M"), 0.15)
        ]
    )

    # Query
    proof = backward_chain(kb, Term("margin_leak", "SKU123"))
    print(proof.is_valid)
"""

from .dsl import ReasoningDSL, define_fact, define_rule, query
from .inference import ProofNode, ProofTree, backward_chain, forward_chain
from .knowledge_base import Fact, KnowledgeBase, RuleDefinition
from .terms import Atom, Clause, Predicate, Rule, Term, Var
from .unification import occurs_check, substitute, unify

__all__ = [
    # Terms
    "Term",
    "Var",
    "Atom",
    "Predicate",
    "Rule",
    "Clause",
    # Unification
    "unify",
    "substitute",
    "occurs_check",
    # Knowledge Base
    "KnowledgeBase",
    "Fact",
    "RuleDefinition",
    # Inference
    "forward_chain",
    "backward_chain",
    "ProofTree",
    "ProofNode",
    # DSL
    "ReasoningDSL",
    "define_rule",
    "define_fact",
    "query",
]
