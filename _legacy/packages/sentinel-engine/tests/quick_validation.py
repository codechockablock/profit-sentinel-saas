#!/usr/bin/env python3
"""
Quick VSA Validation - Faster test with reduced iterations.

Tests VSA functionality without full resonator convergence.
"""

import json
import time
from datetime import datetime
from pathlib import Path

# Import validation_runner components
from validation_runner import BaselineDetector, DetectionResult, SyntheticDataGenerator


class QuickVSADetector:
    """
    Quick VSA detector with reduced iterations for faster testing.
    """

    def __init__(self, iters: int = 50):
        self.available = False
        self.iters = iters
        try:
            import torch
            from sentinel_engine import core
            from sentinel_engine.context import create_analysis_context

            self._torch = torch
            self._create_ctx = create_analysis_context
            self._core = core
            self.available = True
            print(f"  VSA engine loaded successfully (torch {torch.__version__})")
        except ImportError as e:
            print(f"  WARNING: VSA not available: {e}")

    def detect(
        self, rows: list[dict], score_threshold: float = 0.05
    ) -> dict[str, set[str]]:
        """Run VSA detection with reduced iterations."""
        if not self.available:
            return {
                p: set()
                for p in [
                    "low_stock",
                    "high_margin_leak",
                    "dead_item",
                    "negative_inventory",
                    "overstock",
                    "price_discrepancy",
                    "shrinkage_pattern",
                    "margin_erosion",
                ]
            }

        # Create context with reduced iterations
        ctx = self._create_ctx(use_gpu=False)
        ctx.iters = self.iters  # Reduce iterations for speed
        ctx.multi_steps = 1

        results = {}

        try:
            # Bundle facts
            print(f"    Bundling {len(rows)} rows...")
            t0 = time.time()
            bundle = self._core.bundle_pos_facts(ctx, rows)
            print(f"    Bundled in {time.time()-t0:.1f}s")

            # Query each primitive
            for primitive in [
                "low_stock",
                "high_margin_leak",
                "dead_item",
                "negative_inventory",
                "overstock",
                "price_discrepancy",
                "shrinkage_pattern",
                "margin_erosion",
            ]:
                t0 = time.time()
                items, scores = self._core.query_bundle(
                    ctx, bundle, primitive, top_k=100
                )

                # Filter by score threshold and collect SKUs
                detected = set()
                for item, score in zip(items, scores):
                    item_lower = item.lower()
                    # Only count actual SKU patterns (not descriptions/vendors)
                    if score > score_threshold and (
                        item_lower.startswith("low_stock_")
                        or item_lower.startswith("high_margin_leak_")
                        or item_lower.startswith("dead_item_")
                        or item_lower.startswith("negative_inventory_")
                        or item_lower.startswith("overstock_")
                        or item_lower.startswith("price_discrepancy_")
                        or item_lower.startswith("shrinkage_pattern_")
                        or item_lower.startswith("margin_erosion_")
                        or item_lower.startswith("normal_")
                    ):
                        detected.add(item_lower)

                results[primitive] = detected
                print(
                    f"    {primitive}: {len(detected)} detections ({time.time()-t0:.1f}s)"
                )

        finally:
            ctx.reset()

        return results


def run_quick_validation():
    """Run quick validation with reduced VSA iterations."""
    print("=" * 70)
    print("QUICK VSA VALIDATION (Reduced Iterations)")
    print("=" * 70)
    print(f"Run time: {datetime.now().isoformat()}")
    print()

    # Generate smaller dataset for speed
    print("Generating synthetic dataset...")
    gen = SyntheticDataGenerator(seed=42)
    rows, ground_truth = gen.generate(n_total=2000, anomaly_rate=0.05)

    print(f"  Total rows: {len(rows)}")
    for primitive, skus in ground_truth.items():
        print(f"  {primitive}: {len(skus)} anomalies")
    print()

    # Run baseline detector
    print("Running BASELINE detector...")
    baseline = BaselineDetector()
    t0 = time.time()
    baseline_results = baseline.detect(rows)
    baseline_time = time.time() - t0
    print(f"  Completed in {baseline_time:.2f}s")
    print()

    # Run VSA detector
    print("Running VSA detector (quick mode)...")
    vsa = QuickVSADetector(iters=50)
    if vsa.available:
        t0 = time.time()
        vsa_results = vsa.detect(rows)
        vsa_time = time.time() - t0
        print(f"  Completed in {vsa_time:.2f}s")
    else:
        print("  SKIPPED - sentinel_engine not available")
        vsa_results = None
        vsa_time = 0
    print()

    # Calculate metrics
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    primitives = list(ground_truth.keys())

    print(
        f"{'Primitive':<25} {'Method':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}"
    )
    print("-" * 85)

    baseline_metrics = {}
    vsa_metrics = {}

    for primitive in primitives:
        truth = ground_truth[primitive]

        # Baseline
        br = DetectionResult(primitive, baseline_results[primitive])
        br.calculate(truth)
        baseline_metrics[primitive] = br
        print(
            f"{primitive:<25} {'BASELINE':<10} {br.precision:>7.1%} {br.recall:>7.1%} {br.f1:>7.1%} {br.true_positives:>6} {br.false_positives:>6} {br.false_negatives:>6}"
        )

        # VSA
        if vsa_results:
            vr = DetectionResult(primitive, vsa_results[primitive])
            vr.calculate(truth)
            vsa_metrics[primitive] = vr
            print(
                f"{'':<25} {'VSA':<10} {vr.precision:>7.1%} {vr.recall:>7.1%} {vr.f1:>7.1%} {vr.true_positives:>6} {vr.false_positives:>6} {vr.false_negatives:>6}"
            )

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_avg_f1 = sum(m.f1 for m in baseline_metrics.values()) / len(
        baseline_metrics
    )
    baseline_avg_prec = sum(m.precision for m in baseline_metrics.values()) / len(
        baseline_metrics
    )
    baseline_avg_recall = sum(m.recall for m in baseline_metrics.values()) / len(
        baseline_metrics
    )

    print("\nBASELINE Averages:")
    print(f"  Precision: {baseline_avg_prec:.1%}")
    print(f"  Recall:    {baseline_avg_recall:.1%}")
    print(f"  F1:        {baseline_avg_f1:.1%}")
    print(f"  Time:      {baseline_time:.2f}s")

    if vsa_metrics:
        vsa_avg_f1 = sum(m.f1 for m in vsa_metrics.values()) / len(vsa_metrics)
        vsa_avg_prec = sum(m.precision for m in vsa_metrics.values()) / len(vsa_metrics)
        vsa_avg_recall = sum(m.recall for m in vsa_metrics.values()) / len(vsa_metrics)

        print("\nVSA Averages:")
        print(f"  Precision: {vsa_avg_prec:.1%}")
        print(f"  Recall:    {vsa_avg_recall:.1%}")
        print(f"  F1:        {vsa_avg_f1:.1%}")
        print(f"  Time:      {vsa_time:.2f}s")

        # Comparison
        print("\nVSA vs BASELINE:")
        f1_diff = vsa_avg_f1 - baseline_avg_f1
        print(
            f"  F1 Difference: {f1_diff:+.1%} ({'VSA better' if f1_diff > 0 else 'Baseline better'})"
        )

        # Critical primitives check
        critical = ["negative_inventory", "high_margin_leak"]
        critical_pass = all(vsa_metrics[p].recall >= 0.30 for p in critical)

        # Decision
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)

        if not critical_pass:
            decision = "KILL"
            print("\n  KILL - Critical primitives failing (recall < 30%)")
            for p in critical:
                if vsa_metrics[p].recall < 0.30:
                    print(f"    - {p}: {vsa_metrics[p].recall:.1%} recall")
        elif vsa_avg_f1 < baseline_avg_f1 - 0.05:
            decision = "KILL"
            print("\n  KILL - VSA significantly underperforms baseline")
        elif vsa_avg_f1 < baseline_avg_f1:
            decision = "CALIBRATE"
            print(
                "\n  CALIBRATE - VSA slightly underperforms baseline, tune thresholds"
            )
        elif vsa_avg_prec < 0.50:
            decision = "CALIBRATE"
            print("\n  CALIBRATE - VSA has too many false positives")
        else:
            decision = "KEEP"
            print("\n  KEEP - VSA performs adequately")

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(rows),
            "mode": "quick_validation",
            "vsa_iterations": 50,
            "baseline": {
                "avg_precision": baseline_avg_prec,
                "avg_recall": baseline_avg_recall,
                "avg_f1": baseline_avg_f1,
                "time_seconds": baseline_time,
                "per_primitive": {
                    p: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
                    for p, m in baseline_metrics.items()
                },
            },
            "vsa": {
                "avg_precision": vsa_avg_prec,
                "avg_recall": vsa_avg_recall,
                "avg_f1": vsa_avg_f1,
                "time_seconds": vsa_time,
                "per_primitive": {
                    p: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
                    for p, m in vsa_metrics.items()
                },
            },
            "decision": decision,
        }

        results_file = Path(__file__).parent / "quick_validation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_file}")

        return results

    else:
        print("\n  Cannot make recommendation - VSA not available")
        return None


if __name__ == "__main__":
    run_quick_validation()
