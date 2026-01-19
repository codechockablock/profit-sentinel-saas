"""
reasoning/unification.py - Unification Algorithm

Implements the Martelli-Montanari unification algorithm for first-order terms.
Unification finds the most general substitution that makes two terms identical.

Key operations:
- unify(t1, t2): Find substitution θ such that t1θ = t2θ
- substitute(t, θ): Apply substitution to term
- occurs_check(var, term): Check for circular references

This is the foundation for pattern matching in inference.
"""
from __future__ import annotations

from .terms import Atom, Term, TermLike, Var

# Type alias for substitution
Substitution = dict[str, TermLike]


def unify(
    t1: TermLike,
    t2: TermLike,
    theta: Substitution | None = None
) -> Substitution | None:
    """Unify two terms and return most general unifier (MGU).

    Finds substitution θ such that t1θ = t2θ, or None if impossible.

    Args:
        t1: First term
        t2: Second term
        theta: Initial substitution (default: empty)

    Returns:
        Most general unifier, or None if unification fails

    Example:
        # Unify margin_leak(?X, ?M) with margin_leak("SKU123", 0.05)
        t1 = Term("margin_leak", Var("X"), Var("M"))
        t2 = Term("margin_leak", Atom("SKU123"), Atom(0.05))
        theta = unify(t1, t2)
        # theta = {"X": Atom("SKU123"), "M": Atom(0.05)}
    """
    if theta is None:
        theta = {}

    # Apply current substitution
    t1 = substitute(t1, theta)
    t2 = substitute(t2, theta)

    # Same term - already unified
    if t1 == t2:
        return theta

    # Variable cases
    if isinstance(t1, Var):
        return _unify_var(t1, t2, theta)
    if isinstance(t2, Var):
        return _unify_var(t2, t1, theta)

    # Atom cases
    if isinstance(t1, Atom) and isinstance(t2, Atom):
        if t1.value == t2.value:
            return theta
        return None

    # Term cases
    if isinstance(t1, Term) and isinstance(t2, Term):
        if t1.functor != t2.functor or t1.arity != t2.arity:
            return None

        # Unify arguments pairwise
        for arg1, arg2 in zip(t1.args, t2.args):
            theta = unify(arg1, arg2, theta)
            if theta is None:
                return None

        return theta

    # Incompatible types
    return None


def _unify_var(var: Var, term: TermLike, theta: Substitution) -> Substitution | None:
    """Unify a variable with a term."""
    # Check if variable is already bound
    if var.name in theta:
        return unify(theta[var.name], term, theta)

    # Occurs check - prevent infinite terms
    if occurs_check(var, term, theta):
        return None

    # Bind variable
    theta = dict(theta)
    theta[var.name] = term
    return theta


def occurs_check(var: Var, term: TermLike, theta: Substitution) -> bool:
    """Check if variable occurs in term (prevents infinite structures).

    Returns True if var appears in term, which would create a circular
    reference like X = f(X).

    Args:
        var: Variable to check for
        term: Term to search in
        theta: Current substitution

    Returns:
        True if var occurs in term
    """
    term = substitute(term, theta)

    if isinstance(term, Var):
        return term.name == var.name
    elif isinstance(term, Atom):
        return False
    elif isinstance(term, Term):
        return any(occurs_check(var, arg, theta) for arg in term.args)
    return False


def substitute(term: TermLike, theta: Substitution) -> TermLike:
    """Apply substitution to term.

    Replaces all variables in term with their bindings in theta.

    Args:
        term: Term to substitute in
        theta: Substitution mapping variable names to terms

    Returns:
        Term with substitutions applied
    """
    if isinstance(term, Var):
        if term.name in theta:
            # Recursively substitute (variable might be bound to another variable)
            return substitute(theta[term.name], theta)
        return term

    elif isinstance(term, Atom):
        return term

    elif isinstance(term, Term):
        new_args = tuple(substitute(arg, theta) for arg in term.args)
        return Term(term.functor, *new_args)

    return term


def compose_substitutions(
    theta1: Substitution,
    theta2: Substitution
) -> Substitution:
    """Compose two substitutions.

    (θ1 ∘ θ2)(t) = θ1(θ2(t))

    Args:
        theta1: First substitution (applied last)
        theta2: Second substitution (applied first)

    Returns:
        Composed substitution
    """
    result = {}

    # Apply theta1 to all values in theta2
    for var, term in theta2.items():
        result[var] = substitute(term, theta1)

    # Add bindings from theta1 not in theta2
    for var, term in theta1.items():
        if var not in result:
            result[var] = term

    return result


def rename_apart(
    terms: list[Term],
    suffix: str
) -> tuple[list[Term], Substitution]:
    """Rename variables in terms to avoid capture.

    Creates fresh variable names by adding suffix.

    Args:
        terms: Terms to rename
        suffix: Suffix to add to variable names

    Returns:
        (renamed_terms, renaming_substitution)
    """
    # Collect all variables
    all_vars = set()
    for t in terms:
        all_vars.update(t.variables())

    # Create renaming
    renaming = {v: Var(f"{v}{suffix}") for v in all_vars}

    # Apply renaming
    renamed = [substitute(t, renaming) for t in terms]

    return renamed, renaming


def is_instance_of(specific: TermLike, general: TermLike) -> bool:
    """Check if specific is an instance of general.

    Returns True if there exists θ such that generalθ = specific.

    Args:
        specific: More specific term
        general: More general term (may have variables)

    Returns:
        True if specific is an instance of general
    """
    theta = unify(general, specific)
    if theta is None:
        return False

    # Check that substitution only binds variables in general
    general_vars = general.variables() if hasattr(general, 'variables') else set()
    for var in theta:
        if var not in general_vars:
            return False

    return True


def most_general_instance(t1: TermLike, t2: TermLike) -> TermLike | None:
    """Find most general instance (anti-unification).

    Finds the most specific term that is more general than both t1 and t2.

    Args:
        t1: First term
        t2: Second term

    Returns:
        Most general instance, or None if terms are incompatible
    """
    counter = [0]

    def fresh_var() -> Var:
        counter[0] += 1
        return Var(f"_G{counter[0]}")

    return _anti_unify(t1, t2, {}, fresh_var)


def _anti_unify(
    t1: TermLike,
    t2: TermLike,
    mapping: dict[tuple, Var],
    fresh_var
) -> TermLike:
    """Anti-unification helper."""
    # Same term
    if t1 == t2:
        return t1

    # Both are terms with same functor
    if isinstance(t1, Term) and isinstance(t2, Term):
        if t1.functor == t2.functor and t1.arity == t2.arity:
            args = tuple(
                _anti_unify(a1, a2, mapping, fresh_var)
                for a1, a2 in zip(t1.args, t2.args)
            )
            return Term(t1.functor, *args)

    # Different structures - need a variable
    key = (id(t1), id(t2))
    if key in mapping:
        return mapping[key]

    var = fresh_var()
    mapping[key] = var
    return var


def ground_term(term: TermLike, theta: Substitution) -> bool:
    """Check if term becomes ground after substitution."""
    result = substitute(term, theta)
    if hasattr(result, 'is_ground'):
        return result.is_ground()
    return True
