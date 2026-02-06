# Reasoning

Hard symbolic reasoning engine for Profit Sentinel - Prolog-style logic programming.

## Features

- Knowledge base with Horn clauses
- Unification algorithm
- Forward chaining (data-driven)
- Backward chaining (goal-driven)
- Proof tree generation

## Installation

```bash
pip install -e .
```

## Usage

```python
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
```
