"""
reasoning/knowledge_base.py - Knowledge Base for Symbolic Reasoning

The knowledge base stores facts and rules, and provides efficient
indexing for inference operations.

Features:
- Fact storage with indexing by predicate
- Rule storage with head indexing
- Integrity constraints
- Persistence support (JSON/YAML)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Iterator, Any, Callable
from collections import defaultdict
import json
import yaml

from .terms import Term, Var, Atom, Clause, TermLike
from .unification import unify, substitute, Substitution


@dataclass
class Fact:
    """A ground fact in the knowledge base."""
    term: Term
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.term.is_ground():
            raise ValueError(f"Fact must be ground, got: {self.term}")


@dataclass
class RuleDefinition:
    """A rule definition with metadata."""
    clause: Clause
    name: Optional[str] = None
    description: Optional[str] = None
    priority: int = 0
    enabled: bool = True

    @property
    def head(self) -> Term:
        return self.clause.head

    @property
    def body(self) -> List[Term]:
        return self.clause.body


class KnowledgeBase:
    """Knowledge base for facts and rules.

    Provides indexed storage and retrieval for efficient inference.

    Example:
        kb = KnowledgeBase()

        # Add facts
        kb.add_fact(Term("has_anomaly", "SKU123", "low_stock"))
        kb.add_fact(Term("severity", "low_stock", "medium"))

        # Add rules
        kb.add_rule(
            Term("critical_alert", Var("X")),
            Term("has_anomaly", Var("X"), Var("A")),
            Term("severity", Var("A"), "critical")
        )

        # Query
        for match in kb.query(Term("has_anomaly", Var("S"), "low_stock")):
            print(match)  # {"S": Atom("SKU123")}
    """

    def __init__(self):
        # Index facts by predicate name and arity
        self._facts: Dict[str, List[Fact]] = defaultdict(list)

        # Index rules by head predicate
        self._rules: Dict[str, List[RuleDefinition]] = defaultdict(list)

        # All clauses (for iteration)
        self._clauses: List[Clause] = []

        # Integrity constraints
        self._constraints: List[Callable[[Term], bool]] = []

        # Statistics
        self._stats = {
            "facts_added": 0,
            "rules_added": 0,
            "queries": 0,
        }

    def add_fact(
        self,
        term: Term,
        confidence: float = 1.0,
        source: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a fact to the knowledge base.

        Args:
            term: Ground term representing the fact
            confidence: Confidence score (0-1)
            source: Source identifier
            metadata: Additional metadata
        """
        fact = Fact(
            term=term,
            confidence=confidence,
            source=source,
            metadata=metadata or {}
        )

        # Check constraints
        for constraint in self._constraints:
            if not constraint(term):
                raise ValueError(f"Fact violates constraint: {term}")

        key = f"{term.functor}/{term.arity}"
        self._facts[key].append(fact)
        self._clauses.append(Clause(term, []))
        self._stats["facts_added"] += 1

    def add_rule(
        self,
        head: Term,
        *body: Term,
        name: Optional[str] = None,
        description: Optional[str] = None,
        priority: int = 0
    ) -> None:
        """Add a rule to the knowledge base.

        Args:
            head: Rule head (conclusion)
            body: Rule body (conditions)
            name: Optional rule name
            description: Optional description
            priority: Rule priority (higher = evaluated first)
        """
        clause = Clause(head, list(body))
        rule = RuleDefinition(
            clause=clause,
            name=name,
            description=description,
            priority=priority
        )

        key = f"{head.functor}/{head.arity}"
        self._rules[key].append(rule)
        self._clauses.append(clause)
        self._stats["rules_added"] += 1

    def add_clause(self, clause: Clause) -> None:
        """Add a clause (fact or rule)."""
        if clause.is_fact:
            self.add_fact(clause.head)
        else:
            self.add_rule(clause.head, *clause.body)

    def add_constraint(self, constraint: Callable[[Term], bool]) -> None:
        """Add integrity constraint.

        Args:
            constraint: Function that returns True if term is valid
        """
        self._constraints.append(constraint)

    def query(self, pattern: Term) -> Iterator[Substitution]:
        """Query facts matching pattern.

        Yields substitutions that make pattern match facts.

        Args:
            pattern: Query pattern (may contain variables)

        Yields:
            Substitutions for each matching fact
        """
        self._stats["queries"] += 1
        key = f"{pattern.functor}/{pattern.arity}"

        for fact in self._facts.get(key, []):
            theta = unify(pattern, fact.term)
            if theta is not None:
                yield theta

    def query_rules(self, pattern: Term) -> Iterator[Clause]:
        """Get rules whose head matches pattern.

        Args:
            pattern: Pattern to match against rule heads

        Yields:
            Matching rules (with variables renamed)
        """
        key = f"{pattern.functor}/{pattern.arity}"

        for i, rule_def in enumerate(self._rules.get(key, [])):
            if not rule_def.enabled:
                continue

            # Rename variables to avoid capture
            renamed = rule_def.clause.rename_variables(f"_{i}")

            # Check if head unifies with pattern
            if unify(pattern, renamed.head) is not None:
                yield renamed

    def get_facts(self, predicate: str, arity: int) -> List[Fact]:
        """Get all facts for a predicate."""
        key = f"{predicate}/{arity}"
        return list(self._facts.get(key, []))

    def get_rules(self, predicate: str, arity: int) -> List[RuleDefinition]:
        """Get all rules for a predicate."""
        key = f"{predicate}/{arity}"
        return list(self._rules.get(key, []))

    def retract_fact(self, term: Term) -> bool:
        """Remove a fact from the knowledge base.

        Args:
            term: Fact to remove

        Returns:
            True if fact was found and removed
        """
        key = f"{term.functor}/{term.arity}"
        facts = self._facts.get(key, [])

        for i, fact in enumerate(facts):
            if fact.term == term:
                del facts[i]
                return True
        return False

    def clear(self) -> None:
        """Clear all facts and rules."""
        self._facts.clear()
        self._rules.clear()
        self._clauses.clear()

    def __len__(self) -> int:
        """Total number of clauses."""
        return len(self._clauses)

    def __iter__(self) -> Iterator[Clause]:
        """Iterate over all clauses."""
        return iter(self._clauses)

    @property
    def fact_count(self) -> int:
        """Number of facts."""
        return sum(len(facts) for facts in self._facts.values())

    @property
    def rule_count(self) -> int:
        """Number of rules."""
        return sum(len(rules) for rules in self._rules.values())

    @property
    def stats(self) -> Dict[str, int]:
        """Get statistics."""
        return dict(self._stats)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "facts": [
                {
                    "term": _term_to_dict(f.term),
                    "confidence": f.confidence,
                    "source": f.source,
                    "metadata": f.metadata
                }
                for facts in self._facts.values()
                for f in facts
            ],
            "rules": [
                {
                    "name": r.name,
                    "description": r.description,
                    "priority": r.priority,
                    "head": _term_to_dict(r.head),
                    "body": [_term_to_dict(t) for t in r.body]
                }
                for rules in self._rules.values()
                for r in rules
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeBase':
        """Import from dictionary."""
        kb = cls()

        for fact_data in data.get("facts", []):
            term = _dict_to_term(fact_data["term"])
            kb.add_fact(
                term,
                confidence=fact_data.get("confidence", 1.0),
                source=fact_data.get("source"),
                metadata=fact_data.get("metadata", {})
            )

        for rule_data in data.get("rules", []):
            head = _dict_to_term(rule_data["head"])
            body = [_dict_to_term(t) for t in rule_data.get("body", [])]
            kb.add_rule(
                head, *body,
                name=rule_data.get("name"),
                description=rule_data.get("description"),
                priority=rule_data.get("priority", 0)
            )

        return kb

    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'KnowledgeBase':
        """Load from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_yaml(self, path: str) -> None:
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'KnowledgeBase':
        """Load from YAML file."""
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def _term_to_dict(term: TermLike) -> Dict:
    """Convert term to dictionary."""
    if isinstance(term, Var):
        return {"type": "var", "name": term.name}
    elif isinstance(term, Atom):
        return {"type": "atom", "value": term.value}
    elif isinstance(term, Term):
        return {
            "type": "term",
            "functor": term.functor,
            "args": [_term_to_dict(arg) for arg in term.args]
        }
    raise ValueError(f"Unknown term type: {type(term)}")


def _dict_to_term(data: Dict) -> TermLike:
    """Convert dictionary to term."""
    t = data["type"]
    if t == "var":
        return Var(data["name"])
    elif t == "atom":
        return Atom(data["value"])
    elif t == "term":
        args = [_dict_to_term(arg) for arg in data.get("args", [])]
        return Term(data["functor"], *args)
    raise ValueError(f"Unknown term type: {t}")
