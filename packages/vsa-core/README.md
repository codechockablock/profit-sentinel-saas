# VSA Core

Hyperdimensional Computing / Vector Symbolic Architecture library for Profit Sentinel.

## Features

- 16,384-dimensional complex vectors on the unit hypersphere
- Deterministic seeding via SHA256
- Algebraic operations: bind, bundle, permute
- Resonator network for query cleanup
- GPU acceleration via PyTorch

## Installation

```bash
pip install -e .
```

## Usage

```python
from vsa_core import seed_hash, bind, bundle, Resonator

# Generate vectors
sku = seed_hash("SKU123")
anomaly = seed_hash("primitive:low_stock")

# Bind fact
fact = bind(sku, anomaly)

# Query with resonator
resonator = Resonator()
result = resonator.resonate(fact)
```
