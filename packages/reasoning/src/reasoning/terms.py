"""
reasoning/terms.py - Term Algebra for Symbolic Reasoning

Implements the fundamental term structures for logic programming:
- Atom: Ground constants (e.g., "SKU123", 42)
- Var: Logical variables (e.g., X, SKU)
- Term: Compound terms with functor and arguments
- Predicate: Named relation
- Clause: Rule or fact in Horn clause form

This follows classical first-order logic with Horn clause restriction
(at most one positive literal in each clause).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Union


class TermBase(ABC):
    """Base class for all term types."""

    @abstractmethod
    def is_ground(self) -> bool:
        """Return True if term contains no variables."""
        pass

    @abstractmethod
    def variables(self) -> set[str]:
        """Return set of variable names in term."""
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass


@dataclass(frozen=True)
class Var(TermBase):
    """Logical variable.

    Variables are placeholders that can be unified with any term.
    By convention, variable names start with uppercase (X, SKU, Margin).

    Example:
        X = Var("X")
        SKU = Var("SKU")
    """
    name: str

    def is_ground(self) -> bool:
        return False

    def variables(self) -> set[str]:
        return {self.name}

    def __repr__(self) -> str:
        return f"?{self.name}"

    def __hash__(self) -> int:
        return hash(("Var", self.name))

    def __eq__(self, other) -> bool:
        return isinstance(other, Var) and self.name == other.name


@dataclass(frozen=True)
class Atom(TermBase):
    """Ground constant (atom).

    Atoms are constants that represent specific values.
    Can be strings, numbers, or any hashable value.

    Example:
        sku = Atom("SKU123")
        price = Atom(99.99)
    """
    value: Any

    def is_ground(self) -> bool:
        return True

    def variables(self) -> set[str]:
        return set()

    def __repr__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)

    def __hash__(self) -> int:
        return hash(("Atom", self.value))

    def __eq__(self, other) -> bool:
        return isinstance(other, Atom) and self.value == other.value


@dataclass(frozen=True)
class Term(TermBase):
    """Compound term with functor and arguments.

    A term consists of a functor (name) and a list of arguments.
    This is the primary representation for predicates in clauses.

    Example:
        # margin_leak(SKU, Margin)
        term = Term("margin_leak", Var("SKU"), Var("Margin"))

        # has_anomaly("SKU123", "low_stock")
        fact = Term("has_anomaly", Atom("SKU123"), Atom("low_stock"))
    """
    functor: str
    args: tuple[Var | Atom | Term, ...] = field(default_factory=tuple)

    def __init__(self, functor: str, *args: Var | Atom | Term | str | int | float):
        # Convert raw values to Atoms
        processed = []
        for arg in args:
            if isinstance(arg, (Var, Atom, Term)):
                processed.append(arg)
            else:
                processed.append(Atom(arg))

        object.__setattr__(self, 'functor', functor)
        object.__setattr__(self, 'args', tuple(processed))

    @property
    def arity(self) -> int:
        """Number of arguments."""
        return len(self.args)

    def is_ground(self) -> bool:
        return all(arg.is_ground() for arg in self.args)

    def variables(self) -> set[str]:
        result = set()
        for arg in self.args:
            result.update(arg.variables())
        return result

    def __repr__(self) -> str:
        if not self.args:
            return self.functor
        args_str = ", ".join(repr(arg) for arg in self.args)
        return f"{self.functor}({args_str})"

    def __hash__(self) -> int:
        return hash(("Term", self.functor, self.args))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Term):
            return False
        return self.functor == other.functor and self.args == other.args


# Type alias for any term-like value
TermLike = Union[Var, Atom, Term]


@dataclass
class Predicate:
    """Named relation with arity.

    A predicate defines the structure of a relation without
    specific arguments. Used for declaring relation schemas.

    Example:
        margin_leak = Predicate("margin_leak", 2)  # margin_leak/2
    """
    name: str
    arity: int

    def __call__(self, *args: TermLike) -> Term:
        """Create a term from this predicate."""
        if len(args) != self.arity:
            raise ValueError(f"Predicate {self.name}/{self.arity} expects {self.arity} args, got {len(args)}")
        return Term(self.name, *args)

    def __repr__(self) -> str:
        return f"{self.name}/{self.arity}"


@dataclass
class Clause:
    """Horn clause: head :- body.

    A clause represents a logical implication:
    - If body is empty: fact (head is unconditionally true)
    - If body is non-empty: rule (head is true if all body terms are true)

    Example:
        # Fact: has_anomaly("SKU123", "low_stock")
        fact = Clause(Term("has_anomaly", "SKU123", "low_stock"), [])

        # Rule: critical_alert(X) :- has_anomaly(X, A), severity(A, "critical")
        rule = Clause(
            Term("critical_alert", Var("X")),
            [
                Term("has_anomaly", Var("X"), Var("A")),
                Term("severity", Var("A"), "critical")
            ]
        )
    """
    head: Term
    body: list[Term] = field(default_factory=list)

    @property
    def is_fact(self) -> bool:
        """True if this is a fact (no body)."""
        return len(self.body) == 0

    @property
    def is_rule(self) -> bool:
        """True if this is a rule (has body)."""
        return len(self.body) > 0

    def is_ground(self) -> bool:
        """True if clause contains no variables."""
        if not self.head.is_ground():
            return False
        return all(t.is_ground() for t in self.body)

    def variables(self) -> set[str]:
        """All variables in the clause."""
        result = self.head.variables()
        for t in self.body:
            result.update(t.variables())
        return result

    def rename_variables(self, suffix: str) -> Clause:
        """Create copy with renamed variables (for resolution)."""
        var_map = {v: Var(f"{v}{suffix}") for v in self.variables()}
        return Clause(
            _rename_term(self.head, var_map),
            [_rename_term(t, var_map) for t in self.body]
        )

    def __repr__(self) -> str:
        if self.is_fact:
            return f"{self.head}."
        body_str = ", ".join(repr(t) for t in self.body)
        return f"{self.head} :- {body_str}."


def _rename_term(term: TermLike, var_map: dict[str, Var]) -> TermLike:
    """Rename variables in a term."""
    if isinstance(term, Var):
        return var_map.get(term.name, term)
    elif isinstance(term, Atom):
        return term
    elif isinstance(term, Term):
        new_args = tuple(_rename_term(arg, var_map) for arg in term.args)
        return Term(term.functor, *new_args)
    return term


# Convenience function
def Rule(head: Term, *body: Term) -> Clause:
    """Create a rule clause."""
    return Clause(head, list(body))


def Fact(term: Term) -> Clause:
    """Create a fact clause."""
    return Clause(term, [])


# Common variable shortcuts
X = Var("X")
Y = Var("Y")
Z = Var("Z")
SKU = Var("SKU")
ANOMALY = Var("ANOMALY")
SEVERITY = Var("SEVERITY")
MARGIN = Var("MARGIN")
QTY = Var("QTY")
