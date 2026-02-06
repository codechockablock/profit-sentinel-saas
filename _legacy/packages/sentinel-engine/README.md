# Sentinel Engine

Scalable VSA inference engine for Profit Sentinel.

## Features

- Core analysis functions (bundle_pos_facts, query_bundle)
- Tiered pipeline (statistical pre-filter -> VSA deep dive)
- FAISS integration for approximate nearest neighbor
- Persistent codebook with session management
- Batch processing utilities

Designed to handle 150k-1.5M+ entities efficiently.

## Installation

```bash
pip install -e .
```

## Usage

```python
from sentinel_engine import bundle_pos_facts, query_bundle

# Bundle POS data facts
bundle = bundle_pos_facts(rows)

# Query for anomalies
items, scores = query_bundle(bundle, "low_stock")
```
