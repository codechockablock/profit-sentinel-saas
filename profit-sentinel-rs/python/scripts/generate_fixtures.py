#!/usr/bin/env python3
"""Generate test fixtures from real inventory data.

This script reads real inventory data, runs it through the adapter and
bridge, and writes pipeline-ready CSV fixtures for testing.

Output:
    tests/fixtures/inventory_sample.csv  — 1000 diverse rows for unit tests
    tests/fixtures/inventory_full.csv    — All rows for integration tests

Usage:
    cd profit-sentinel-rs/python
    python scripts/generate_fixtures.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel_agent.adapters.bridge import PipelineBridge
from sentinel_agent.adapters.generic_pos import GenericPosAdapter

REAL_DATA = Path("/Users/joseph/Downloads/custom_1.csv")
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main():
    if not REAL_DATA.exists():
        print(f"Real data not found at {REAL_DATA}")
        sys.exit(1)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    # Ingest real data
    print(f"Reading {REAL_DATA}...")
    adapter = GenericPosAdapter()
    result = adapter.ingest(REAL_DATA, store_id="default-store")
    print(f"  Records: {result.total_inventory_records:,}")
    print(f"  Errors: {len(result.errors)}")

    records = result.inventory_records
    bridge = PipelineBridge()

    # --- Full fixture (all records) ---
    full_path = FIXTURE_DIR / "inventory_full.csv"
    print(f"\nGenerating full fixture → {full_path}")
    bridge.to_pipeline_csv(records, full_path)
    print(f"  Written: {len(records):,} records")

    # --- Sample fixture (1000 diverse rows) ---
    # Select a diverse sample: ensure we get negatives, dead stock, low margin, etc.
    negative_qty = [r for r in records if r.qty_on_hand < 0]
    zero_sales = [r for r in records if r.qty_on_hand > 0 and r.sales_ytd == 0]
    low_margin = [r for r in records if r.margin_pct < 0.10 and r.retail_price > 0]
    high_value = sorted(
        [r for r in records if r.qty_on_hand > 0 and r.unit_cost > 0],
        key=lambda r: r.inventory_value_at_cost,
        reverse=True,
    )
    has_on_order = [r for r in records if r.on_order_qty > 0]
    normal = [
        r
        for r in records
        if r.qty_on_hand > 0 and r.unit_cost > 0 and r.retail_price > 0
    ]

    # Use dict keyed by sku_id to avoid unhashable Pydantic models
    sample_by_sku: dict[str, object] = {}

    # Take some from each category
    for pool, take in [
        (negative_qty, 100),  # 100 negative inventory rows
        (zero_sales, 100),  # 100 dead stock candidates
        (low_margin, 100),  # 100 low margin items
        (high_value[:200], 100),  # 100 high value items
        (has_on_order, 50),  # 50 items with on-order
    ]:
        random.seed(42)
        take = min(take, len(pool))
        for r in random.sample(pool, take):
            sample_by_sku[r.sku_id] = r

    # Fill remaining with random normal records
    remaining = 1000 - len(sample_by_sku)
    if remaining > 0:
        available = [r for r in normal if r.sku_id not in sample_by_sku]
        random.seed(42)
        for r in random.sample(available, min(remaining, len(available))):
            sample_by_sku[r.sku_id] = r

    sample_list = sorted(sample_by_sku.values(), key=lambda r: r.sku_id)

    sample_path = FIXTURE_DIR / "inventory_sample.csv"
    print(f"\nGenerating sample fixture → {sample_path}")
    bridge.to_pipeline_csv(sample_list, sample_path)

    # Print sample stats
    neg_count = sum(1 for r in sample_list if r.qty_on_hand < 0)
    dead_count = sum(1 for r in sample_list if r.qty_on_hand > 0 and r.sales_ytd == 0)
    low_margin_count = sum(
        1 for r in sample_list if r.margin_pct < 0.10 and r.retail_price > 0
    )
    print(f"  Written: {len(sample_list)} records")
    print(f"  Negative inventory: {neg_count}")
    print(f"  Dead stock candidates: {dead_count}")
    print(f"  Low margin items: {low_margin_count}")

    print("\nDone!")


if __name__ == "__main__":
    main()
