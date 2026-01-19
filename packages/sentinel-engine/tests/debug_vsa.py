#!/usr/bin/env python3
"""Debug VSA detection to understand what's being returned."""

import sys

sys.path.insert(0, "../src")

from validation_runner import SyntheticDataGenerator

# Generate small dataset
gen = SyntheticDataGenerator(seed=42)
rows, ground_truth = gen.generate(n_total=500, anomaly_rate=0.10)

print(f"Dataset: {len(rows)} rows")
print(f"Ground truth low_stock: {list(ground_truth['low_stock'])[:5]}...")
print()

# Check VSA
try:
    from sentinel_engine import core
    from sentinel_engine.context import create_analysis_context

    print("Creating context...")
    ctx = create_analysis_context(use_gpu=False)
    ctx.iters = 100
    ctx.multi_steps = 2

    print("Bundling...")
    bundle = core.bundle_pos_facts(ctx, rows)

    print(f"Codebook size: {len(ctx.codebook)}")
    print(f"Sample codebook keys: {list(ctx.codebook.keys())[:10]}")
    print()

    # Check what query returns
    print("Querying low_stock primitive...")
    items, scores = core.query_bundle(ctx, bundle, "low_stock", top_k=20)

    print("Top 20 results:")
    for i, (item, score) in enumerate(zip(items, scores)):
        print(f"  {i+1}. {item}: {score:.4f}")

    print()
    print("Ground truth low_stock SKUs (first 10):")
    for sku in list(ground_truth["low_stock"])[:10]:
        print(f"  - {sku}")

    # Check if any ground truth SKUs are in results
    detected_lower = {item.lower() for item in items}
    truth_lower = {s.lower() for s in ground_truth["low_stock"]}
    overlap = detected_lower & truth_lower
    print(f"\nOverlap with ground truth: {len(overlap)}")

    ctx.reset()

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
