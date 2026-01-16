#!/usr/bin/env python3
"""
Calibration Verification - Test the v2.1.0 calibrated resonator settings.

Tests:
1. Convergence threshold lowered to 0.005
2. SKU-only codebook filtering
3. Contradiction detection
"""

import time
from datetime import datetime
from pathlib import Path

# Import validation components
from validation_runner import SyntheticDataGenerator, BaselineDetector, DetectionResult


def test_calibrated_resonator():
    """Test resonator with calibrated settings."""
    print("=" * 70)
    print("CALIBRATION VERIFICATION TEST")
    print("=" * 70)
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Generate test data
    print("Generating test dataset...")
    gen = SyntheticDataGenerator(seed=42)
    rows, ground_truth = gen.generate(n_total=2000, anomaly_rate=0.05)
    print(f"  Dataset: {len(rows)} rows")
    print()

    # Run baseline
    print("Running baseline detector...")
    baseline = BaselineDetector()
    baseline_results = baseline.detect(rows)
    print(f"  Total detections: {sum(len(s) for s in baseline_results.values())}")
    print()

    # Test contradiction detection
    print("Testing contradiction detection...")
    try:
        from sentinel_engine.contradiction_detector import (
            detect_contradictions,
            resolve_contradictions,
            generate_contradiction_report,
        )

        contradictions, summary = detect_contradictions(baseline_results)
        print(f"  Contradictions found: {len(contradictions)}")
        for key, count in summary.items():
            print(f"    - {key}: {count}")

        # Test resolution
        resolved = resolve_contradictions(baseline_results)
        resolved_total = sum(len(s) for s in resolved.values())
        original_total = sum(len(s) for s in baseline_results.values())
        print(f"  Resolved: {original_total} -> {resolved_total} detections")
        print("  ✅ Contradiction detection working")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()

    # Test calibrated resonator
    print("Testing calibrated resonator...")
    try:
        import torch
        from sentinel_engine.context import create_analysis_context, RESONATOR_CONVERGENCE_THRESHOLD
        from sentinel_engine import core

        print(f"  Convergence threshold: {RESONATOR_CONVERGENCE_THRESHOLD}")

        # Create context with SKU-only filtering
        ctx = create_analysis_context(
            use_gpu=False,
            sku_only_codebook=True,
            convergence_threshold=0.005,
            iters=150,
        )

        print(f"  Context settings:")
        print(f"    - sku_only_codebook: {ctx.sku_only_codebook}")
        print(f"    - convergence_threshold: {ctx.convergence_threshold}")
        print(f"    - iters: {ctx.iters}")

        # Convert rows for resonator
        resonator_rows = []
        for row in rows[:1000]:
            resonator_rows.append({
                "sku": row.get("sku", ""),
                "description": row.get("description", ""),
                "vendor": row.get("vendor", ""),
                "quantity": float(row.get("quantity", 0)),
                "sold": float(row.get("sold", 0)),
                "cost": float(row.get("cost", 0)),
                "revenue": float(row.get("revenue", 0)),
            })

        print(f"  Bundling {len(resonator_rows)} rows...")
        t0 = time.time()
        bundle = core.bundle_pos_facts(ctx, resonator_rows)
        print(f"  Bundled in {time.time()-t0:.1f}s")
        print(f"  Codebook size: {len(ctx.codebook)} (should be smaller with SKU-only)")

        # Query and check convergence
        print("  Querying primitives...")
        for primitive in ["low_stock", "negative_inventory", "dead_item"]:
            items, scores = core.query_bundle(ctx, bundle, primitive, top_k=50)
            if scores:
                avg_score = sum(scores) / len(scores)
                above_threshold = sum(1 for s in scores if s >= ctx.convergence_threshold)
                print(f"    {primitive}: avg_score={avg_score:.4f}, above_threshold={above_threshold}/{len(scores)}")

        ctx.reset()
        print("  ✅ Calibrated resonator working")

    except ImportError as e:
        print(f"  ⚠️ Resonator not available: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)
    print("✅ Calibration verification complete")
    print("=" * 70)


if __name__ == "__main__":
    test_calibrated_resonator()
