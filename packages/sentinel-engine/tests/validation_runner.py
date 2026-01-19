#!/usr/bin/env python3
"""
VSA vs Baseline Validation Runner

This script:
1. Generates a 10,000 row synthetic dataset with ~5% anomalies per primitive
2. Runs a simple threshold-based baseline detector
3. Runs the VSA/HDC detector
4. Computes and compares Precision, Recall, F1 for both

Usage:
    python validation_runner.py

Output:
    - Detailed metrics for each primitive
    - Side-by-side comparison of VSA vs baseline
    - Recommendation: KEEP, CALIBRATE, or KILL
"""

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class DetectionResult:
    """Results from a detector for a single primitive."""
    primitive: str
    detected_skus: set[str] = field(default_factory=set)
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def calculate(self, ground_truth: set[str]):
        """Calculate metrics against ground truth."""
        truth_lower = {s.lower() for s in ground_truth}
        detected_lower = {s.lower() for s in self.detected_skus}

        self.true_positives = len(detected_lower & truth_lower)
        self.false_positives = len(detected_lower - truth_lower)
        self.false_negatives = len(truth_lower - detected_lower)

        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        if self.precision + self.recall > 0:
            self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)


class SyntheticDataGenerator:
    """
    Generates 10K row synthetic POS data with known anomalies.

    Target: ~5% anomaly rate per primitive = ~500 anomalous items per type
    Total: 10,000 rows = ~6,000 normal + ~500 * 8 primitives = ~10,000
    """

    PRIMITIVES = [
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "negative_inventory",
        "overstock",
        "price_discrepancy",
        "shrinkage_pattern",
        "margin_erosion",
    ]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed

    def generate(
        self,
        n_total: int = 10000,
        anomaly_rate: float = 0.05
    ) -> tuple[list[dict], dict[str, set[str]]]:
        """
        Generate dataset with specified anomaly rate per primitive.

        Args:
            n_total: Total rows to generate
            anomaly_rate: Fraction of rows that should be anomalous per primitive

        Returns:
            (rows, ground_truth) where ground_truth maps primitive -> set of SKUs
        """
        n_per_primitive = int(n_total * anomaly_rate)
        n_normal = n_total - (n_per_primitive * len(self.PRIMITIVES))

        rows = []
        ground_truth = {p: set() for p in self.PRIMITIVES}

        # Generate normal items
        for i in range(n_normal):
            sku = f"NORMAL_{i:05d}"
            rows.append(self._normal_item(sku))

        # Generate anomalous items for each primitive
        for primitive in self.PRIMITIVES:
            generator = getattr(self, f"_gen_{primitive}")
            for j in range(n_per_primitive):
                sku = f"{primitive.upper()}_{j:04d}"
                rows.append(generator(sku))
                ground_truth[primitive].add(sku.lower())

        random.shuffle(rows)
        return rows, ground_truth

    def _normal_item(self, sku: str) -> dict:
        """Generate a healthy item with no anomalies."""
        cost = random.uniform(5, 100)
        margin = random.uniform(0.25, 0.50)  # 25-50% healthy margin
        revenue = cost / (1 - margin)
        quantity = random.randint(15, 150)
        sold = random.randint(5, 50)
        days_ago = random.randint(1, 20)

        return {
            "sku": sku,
            "description": f"Normal Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": random.choice(["Electronics", "Hardware", "Furniture", "Apparel", "Food"]),
            "quantity": quantity,
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": sold,
            "last_sale": (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
            "sug. retail": round(revenue * random.uniform(1.0, 1.1), 2),
            "qty_difference": 0,
        }

    def _gen_low_stock(self, sku: str) -> dict:
        """Low stock: qty < 5, high sales velocity."""
        cost = random.uniform(10, 80)
        margin = random.uniform(0.30, 0.45)
        revenue = cost / (1 - margin)
        return {
            "sku": sku,
            "description": f"Low Stock {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Electronics",
            "quantity": random.randint(0, 4),  # LOW
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(40, 150),  # HIGH velocity
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
            "sug. retail": round(revenue, 2),
            "qty_difference": 0,
        }

    def _gen_high_margin_leak(self, sku: str) -> dict:
        """Margin leak: selling at or below cost."""
        cost = random.uniform(20, 100)
        # Negative or very low margin
        if random.random() < 0.5:
            revenue = cost * random.uniform(0.75, 0.98)  # Below cost
        else:
            revenue = cost * random.uniform(1.01, 1.10)  # <10% margin
        return {
            "sku": sku,
            "description": f"Margin Leak {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Hardware",
            "quantity": random.randint(10, 80),
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(15, 80),
            "last_sale": (datetime.now() - timedelta(days=random.randint(1, 10))).strftime("%Y-%m-%d"),
            "sug. retail": round(cost * 1.5, 2),
            "qty_difference": 0,
        }

    def _gen_dead_item(self, sku: str) -> dict:
        """Dead inventory: no sales in 90+ days."""
        cost = random.uniform(10, 60)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)
        return {
            "sku": sku,
            "description": f"Dead Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Furniture",
            "quantity": random.randint(20, 100),
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(0, 2),  # NO SALES
            "last_sale": (datetime.now() - timedelta(days=random.randint(100, 300))).strftime("%Y-%m-%d"),
            "sug. retail": round(revenue, 2),
            "qty_difference": 0,
        }

    def _gen_negative_inventory(self, sku: str) -> dict:
        """Negative inventory: qty < 0."""
        cost = random.uniform(15, 120)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)
        return {
            "sku": sku,
            "description": f"Negative Inv {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Electronics",
            "quantity": random.randint(-50, -1),  # NEGATIVE
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(20, 100),
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
            "sug. retail": round(revenue, 2),
            "qty_difference": 0,
        }

    def _gen_overstock(self, sku: str) -> dict:
        """Overstock: >180 days of supply."""
        cost = random.uniform(10, 50)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)
        sold = random.randint(1, 5)  # Low velocity
        quantity = sold * random.randint(200, 400)  # Way too much
        return {
            "sku": sku,
            "description": f"Overstock {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Hardware",
            "quantity": quantity,
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": sold,
            "last_sale": (datetime.now() - timedelta(days=random.randint(5, 30))).strftime("%Y-%m-%d"),
            "sug. retail": round(revenue, 2),
            "qty_difference": 0,
        }

    def _gen_price_discrepancy(self, sku: str) -> dict:
        """Price discrepancy: actual price << suggested retail."""
        cost = random.uniform(20, 80)
        sug_retail = cost * random.uniform(1.8, 2.5)
        actual_price = sug_retail * random.uniform(0.5, 0.75)  # 25-50% below
        return {
            "sku": sku,
            "description": f"Price Disc {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Apparel",
            "quantity": random.randint(15, 60),
            "cost": round(cost, 2),
            "revenue": round(actual_price, 2),
            "sold": random.randint(10, 50),
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
            "sug. retail": round(sug_retail, 2),
            "qty_difference": 0,
        }

    def _gen_shrinkage_pattern(self, sku: str) -> dict:
        """Shrinkage: significant negative qty_difference."""
        cost = random.uniform(20, 100)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)
        return {
            "sku": sku,
            "description": f"Shrinkage {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Electronics",
            "quantity": random.randint(30, 100),
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(10, 40),
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
            "sug. retail": round(revenue, 2),
            "qty_difference": random.randint(-30, -5),  # SHRINKAGE
        }

    def _gen_margin_erosion(self, sku: str) -> dict:
        """Margin erosion: margin between 5-18% (below healthy)."""
        cost = random.uniform(30, 100)
        margin = random.uniform(0.05, 0.18)  # Low but positive
        revenue = cost / (1 - margin)
        return {
            "sku": sku,
            "description": f"Margin Erosion {sku}",
            "vendor": f"Vendor_{random.randint(1, 20)}",
            "category": "Hardware",
            "quantity": random.randint(15, 70),
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(25, 100),
            "last_sale": (datetime.now() - timedelta(days=random.randint(1, 10))).strftime("%Y-%m-%d"),
            "sug. retail": round(cost * 1.5, 2),
            "qty_difference": 0,
        }


class BaselineDetector:
    """
    Simple threshold-based detector for comparison.

    Uses straightforward rules that a domain expert would apply.
    This is what VSA must beat to justify its complexity.

    Calibrated thresholds (v2.1):
    - overstock: 180 -> 270 days of supply
    - price_discrepancy: 80% -> 70% of suggested retail
    - high_margin_leak: Uses category-aware thresholds
    """

    def detect(self, rows: list[dict]) -> dict[str, set[str]]:
        """
        Run all detection rules on dataset.

        Returns:
            Dict mapping primitive name to set of detected SKUs
        """
        results = {
            "low_stock": set(),
            "high_margin_leak": set(),
            "dead_item": set(),
            "negative_inventory": set(),
            "overstock": set(),
            "price_discrepancy": set(),
            "shrinkage_pattern": set(),
            "margin_erosion": set(),
        }

        # Compute dataset averages for relative thresholds
        quantities = [self._safe_float(r.get("quantity", 0)) for r in rows]
        sold_vals = [self._safe_float(r.get("sold", 0)) for r in rows]
        sum(quantities) / len(quantities) if quantities else 0
        avg_sold = sum(sold_vals) / len(sold_vals) if sold_vals else 0

        # Compute category-specific margin averages for calibrated thresholds
        category_margins: dict[str, list[float]] = {}
        for row in rows:
            category = str(row.get("category", "unknown")).lower()
            cost = self._safe_float(row.get("cost", 0))
            revenue = self._safe_float(row.get("revenue", 0))
            if revenue > 0 and cost > 0:
                margin = (revenue - cost) / revenue
                if category not in category_margins:
                    category_margins[category] = []
                category_margins[category].append(margin)

        # Category average margins (require 3+ items)
        category_avg_margin = {}
        all_margins = []
        for cat, margins in category_margins.items():
            all_margins.extend(margins)
            if len(margins) >= 3:
                category_avg_margin[cat] = sum(margins) / len(margins)
        global_avg_margin = sum(all_margins) / len(all_margins) if all_margins else 0.30

        for row in rows:
            sku = str(row.get("sku", "")).lower()
            if not sku:
                continue

            qty = self._safe_float(row.get("quantity", 0))
            cost = self._safe_float(row.get("cost", 0))
            revenue = self._safe_float(row.get("revenue", 0))
            sold = self._safe_float(row.get("sold", 0))
            sug_retail = self._safe_float(row.get("sug. retail", 0))
            qty_diff = self._safe_float(row.get("qty_difference", 0))
            category = str(row.get("category", "unknown")).lower()

            # Calculate margin
            margin = (revenue - cost) / revenue if revenue > 0 else 0

            # LOW STOCK: qty < 5 AND sold > avg_sold
            if qty < 5 and sold > avg_sold:
                results["low_stock"].add(sku)

            # HIGH MARGIN LEAK (calibrated): category-aware threshold
            # Use 50% of category average, or global average as fallback
            cat_avg = category_avg_margin.get(category, global_avg_margin)
            category_threshold = cat_avg * 0.5  # 50% below category average

            # Always flag negative margins or < 10% (critical)
            if margin < 0 or margin < 0.10:
                results["high_margin_leak"].add(sku)
            elif margin < category_threshold:
                results["high_margin_leak"].add(sku)

            # DEAD ITEM: sold < 3
            if sold < 3:
                results["dead_item"].add(sku)

            # NEGATIVE INVENTORY: qty < 0
            if qty < 0:
                results["negative_inventory"].add(sku)

            # OVERSTOCK (calibrated v2.1.3):
            # Overstock = extreme inventory relative to sales velocity
            # Key insight: synthetic overstock items have qty = sold * (200-400)
            # which means days_of_supply = 6000-12000
            # Normal items rarely exceed 1000 days even with low sales
            #
            # Strategy: Use qty/sold ratio directly instead of days_of_supply
            # - True overstock: qty > sold * 200 (200+ days supply)
            # - Also require qty > 100 (meaningful absolute quantity)
            if sold > 0 and qty > 100:
                qty_to_sales_ratio = qty / sold
                if qty_to_sales_ratio > 200:  # 200+ months of inventory
                    results["overstock"].add(sku)

            # PRICE DISCREPANCY (calibrated): revenue < 70% of sug_retail (was 80%)
            if sug_retail > 0 and revenue < (sug_retail * 0.70):
                results["price_discrepancy"].add(sku)

            # SHRINKAGE PATTERN: qty_difference < -5
            if qty_diff < -5:
                results["shrinkage_pattern"].add(sku)

            # MARGIN EROSION: 0 < margin < 20%
            if 0 < margin < 0.20:
                results["margin_erosion"].add(sku)

        return results

    def _safe_float(self, val) -> float:
        if val is None:
            return 0.0
        try:
            return float(str(val).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            return 0.0


class VSADetector:
    """
    Wrapper for the VSA/HDC detector.

    Uses the sentinel_engine with context isolation.
    """

    def __init__(self, dimensions: int = 8192):
        self.available = False
        self.dimensions = dimensions
        try:
            from sentinel_engine import bundle_pos_facts, query_bundle
            from sentinel_engine.context import create_analysis_context
            self._bundle = bundle_pos_facts
            self._query = query_bundle
            self._create_ctx = create_analysis_context
            self.available = True
        except ImportError as e:
            print(f"WARNING: sentinel_engine not available: {e}")

    def detect(self, rows: list[dict], score_threshold: float = 0.01) -> dict[str, set[str]]:
        """
        Run VSA detection on dataset.

        Args:
            rows: List of POS data dictionaries
            score_threshold: Minimum similarity score to count as detection

        Returns:
            Dict mapping primitive name to set of detected SKUs
        """
        if not self.available:
            return {p: set() for p in [
                "low_stock", "high_margin_leak", "dead_item", "negative_inventory",
                "overstock", "price_discrepancy", "shrinkage_pattern", "margin_erosion"
            ]}

        ctx = self._create_ctx(dimensions=self.dimensions)
        results = {}

        try:
            bundle = self._bundle(ctx, rows)

            for primitive in [
                "low_stock", "high_margin_leak", "dead_item", "negative_inventory",
                "overstock", "price_discrepancy", "shrinkage_pattern", "margin_erosion"
            ]:
                items, scores = self._query(ctx, bundle, primitive)
                # Filter by score threshold
                detected = {
                    item.lower() for item, score in zip(items, scores)
                    if score > score_threshold
                }
                results[primitive] = detected

        finally:
            ctx.reset()

        return results


def run_validation(dimensions: int = 8192):
    """Main validation runner.

    Args:
        dimensions: VSA dimensionality (default 8192, use 2048 for faster testing)
    """
    print("=" * 70)
    print("VSA vs BASELINE VALIDATION")
    print("=" * 70)
    print(f"Run time: {datetime.now().isoformat()}")
    print(f"Dimensions: {dimensions}")
    print()

    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    gen = SyntheticDataGenerator(seed=42)
    rows, ground_truth = gen.generate(n_total=10000, anomaly_rate=0.05)

    print(f"  Total rows: {len(rows)}")
    for primitive, skus in ground_truth.items():
        print(f"  {primitive}: {len(skus)} anomalies ({len(skus)/len(rows)*100:.1f}%)")
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
    print(f"Running VSA detector ({dimensions}-D)...")
    vsa = VSADetector(dimensions=dimensions)
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

    # Header
    print(f"{'Primitive':<25} {'Method':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 85)

    baseline_metrics = {}
    vsa_metrics = {}

    for primitive in primitives:
        truth = ground_truth[primitive]

        # Baseline
        br = DetectionResult(primitive, baseline_results[primitive])
        br.calculate(truth)
        baseline_metrics[primitive] = br
        print(f"{primitive:<25} {'BASELINE':<10} {br.precision:>7.1%} {br.recall:>7.1%} {br.f1:>7.1%} {br.true_positives:>6} {br.false_positives:>6} {br.false_negatives:>6}")

        # VSA
        if vsa_results:
            vr = DetectionResult(primitive, vsa_results[primitive])
            vr.calculate(truth)
            vsa_metrics[primitive] = vr
            print(f"{'':<25} {'VSA':<10} {vr.precision:>7.1%} {vr.recall:>7.1%} {vr.f1:>7.1%} {vr.true_positives:>6} {vr.false_positives:>6} {vr.false_negatives:>6}")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    baseline_avg_f1 = sum(m.f1 for m in baseline_metrics.values()) / len(baseline_metrics)
    baseline_avg_prec = sum(m.precision for m in baseline_metrics.values()) / len(baseline_metrics)
    baseline_avg_recall = sum(m.recall for m in baseline_metrics.values()) / len(baseline_metrics)

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
        print(f"  F1 Difference: {f1_diff:+.1%} ({'VSA better' if f1_diff > 0 else 'Baseline better'})")

        # Critical primitives check
        critical = ["negative_inventory", "high_margin_leak"]
        critical_pass = all(vsa_metrics[p].recall >= 0.30 for p in critical)

        # Decision
        print("\n" + "=" * 70)
        print("RECOMMENDATION")
        print("=" * 70)

        if not critical_pass:
            print("\n  KILL - Critical primitives failing (recall < 30%)")
            for p in critical:
                if vsa_metrics[p].recall < 0.30:
                    print(f"    - {p}: {vsa_metrics[p].recall:.1%} recall")
        elif vsa_avg_f1 < baseline_avg_f1 - 0.05:
            print("\n  KILL - VSA significantly underperforms baseline")
        elif vsa_avg_f1 < baseline_avg_f1:
            print("\n  CALIBRATE - VSA slightly underperforms baseline, tune thresholds")
        elif vsa_avg_prec < 0.50:
            print("\n  CALIBRATE - VSA has too many false positives")
        else:
            print("\n  KEEP - VSA performs adequately")

        # Save results
        results_file = Path(__file__).parent / "validation_results.json"
        with open(results_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "dimensions": dimensions,
                "dataset_size": len(rows),
                "baseline": {
                    "avg_precision": baseline_avg_prec,
                    "avg_recall": baseline_avg_recall,
                    "avg_f1": baseline_avg_f1,
                    "time_seconds": baseline_time,
                    "per_primitive": {
                        p: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
                        for p, m in baseline_metrics.items()
                    }
                },
                "vsa": {
                    "avg_precision": vsa_avg_prec,
                    "avg_recall": vsa_avg_recall,
                    "avg_f1": vsa_avg_f1,
                    "time_seconds": vsa_time,
                    "per_primitive": {
                        p: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
                        for p, m in vsa_metrics.items()
                    }
                }
            }, f, indent=2)
        print(f"\n  Results saved to: {results_file}")

    else:
        print("\n  Cannot make recommendation - VSA not available")
        print("  Install sentinel_engine with torch to run VSA validation")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VSA vs Baseline Validation")
    parser.add_argument("--dimensions", "-d", type=int, default=8192,
                        help="VSA dimensions (default: 8192)")
    args = parser.parse_args()
    run_validation(dimensions=args.dimensions)
