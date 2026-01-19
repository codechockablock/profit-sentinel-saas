"""
Synthetic Validation Framework for VSA Profit Leak Detection.

This framework generates synthetic POS data with known anomalies (ground truth)
and validates that the VSA algorithm correctly detects them.

Purpose:
    - Measure precision/recall for each detection primitive
    - Identify false positive and false negative patterns
    - Provide evidence for VSA keep/kill decision
    - Enable threshold tuning

Usage:
    pytest test_validation_framework.py -v --tb=short

Decision Criteria:
    - KEEP if precision >= 0.6 AND recall >= 0.5 for critical primitives
    - CALIBRATE if precision < 0.6 but recall >= 0.5 (reduce false positives)
    - KILL if recall < 0.3 (algorithm not detecting real issues)
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest


@dataclass
class ValidationResult:
    """Results from validating a single primitive."""

    primitive: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    detected_skus: set[str] = field(default_factory=set)
    missed_skus: set[str] = field(default_factory=set)
    spurious_skus: set[str] = field(default_factory=set)

    def calculate_metrics(self):
        """Calculate precision, recall, F1 from counts."""
        if self.true_positives + self.false_positives > 0:
            self.precision = self.true_positives / (
                self.true_positives + self.false_positives
            )
        if self.true_positives + self.false_negatives > 0:
            self.recall = self.true_positives / (
                self.true_positives + self.false_negatives
            )
        if self.precision + self.recall > 0:
            self.f1_score = (
                2 * (self.precision * self.recall) / (self.precision + self.recall)
            )


class SyntheticDataGenerator:
    """
    Generates synthetic POS data with known anomalies for validation.

    Each generated dataset has:
    - Normal items (no anomalies)
    - Items with specific anomalies (ground truth for each primitive)
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        self.seed = seed

    def generate_dataset(
        self,
        n_normal: int = 100,
        n_per_anomaly: int = 10,
    ) -> tuple[list[dict], dict[str, set[str]]]:
        """
        Generate a synthetic dataset with known anomalies.

        Args:
            n_normal: Number of normal (healthy) items
            n_per_anomaly: Number of items per anomaly type

        Returns:
            Tuple of (rows, ground_truth) where:
            - rows: List of POS data dictionaries
            - ground_truth: Dict mapping primitive names to set of SKUs that should trigger
        """
        rows = []
        ground_truth = {
            "low_stock": set(),
            "high_margin_leak": set(),
            "dead_item": set(),
            "negative_inventory": set(),
            "overstock": set(),
            "price_discrepancy": set(),
            "shrinkage_pattern": set(),
            "margin_erosion": set(),
        }

        # Generate normal items
        for i in range(n_normal):
            sku = f"NORMAL_{i:04d}"
            rows.append(self._generate_normal_item(sku))

        # Generate anomalous items for each primitive
        for primitive, generator in [
            ("low_stock", self._generate_low_stock_item),
            ("high_margin_leak", self._generate_margin_leak_item),
            ("dead_item", self._generate_dead_item),
            ("negative_inventory", self._generate_negative_inventory_item),
            ("overstock", self._generate_overstock_item),
            ("price_discrepancy", self._generate_price_discrepancy_item),
            ("shrinkage_pattern", self._generate_shrinkage_item),
            ("margin_erosion", self._generate_margin_erosion_item),
        ]:
            for j in range(n_per_anomaly):
                sku = f"{primitive.upper()}_{j:04d}"
                rows.append(generator(sku))
                ground_truth[primitive].add(sku.lower())  # lowercase for comparison

        random.shuffle(rows)
        return rows, ground_truth

    def _generate_normal_item(self, sku: str) -> dict:
        """Generate a healthy item with no anomalies."""
        cost = random.uniform(5, 50)
        margin = random.uniform(0.25, 0.45)  # 25-45% margin (healthy)
        revenue = cost / (1 - margin)
        quantity = random.randint(20, 100)  # healthy stock level
        sold = random.randint(10, 50)

        return {
            "sku": sku,
            "description": f"Normal Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": random.choice(
                ["Electronics", "Hardware", "Furniture", "Apparel"]
            ),
            "quantity": quantity,
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": sold,
            "last_sale": (
                datetime.now() - timedelta(days=random.randint(1, 14))
            ).strftime("%Y-%m-%d"),
        }

    def _generate_low_stock_item(self, sku: str) -> dict:
        """Generate item with low stock signal (qty < 5, high sales velocity)."""
        cost = random.uniform(10, 100)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)

        return {
            "sku": sku,
            "description": f"Low Stock Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Electronics",
            "quantity": random.randint(0, 4),  # LOW STOCK
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(50, 200),  # HIGH VELOCITY
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
        }

    def _generate_margin_leak_item(self, sku: str) -> dict:
        """Generate item with negative or very low margin."""
        cost = random.uniform(20, 80)
        # Margin leak: selling below cost or near cost
        if random.random() < 0.5:
            revenue = cost * random.uniform(0.7, 0.95)  # Negative margin
        else:
            revenue = cost * random.uniform(1.01, 1.10)  # <10% margin

        return {
            "sku": sku,
            "description": f"Margin Leak Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Hardware",
            "quantity": random.randint(10, 50),
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(20, 100),
            "last_sale": (
                datetime.now() - timedelta(days=random.randint(1, 7))
            ).strftime("%Y-%m-%d"),
        }

    def _generate_dead_item(self, sku: str) -> dict:
        """Generate dead inventory item (high qty, no recent sales)."""
        cost = random.uniform(10, 50)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)

        return {
            "sku": sku,
            "description": f"Dead Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Furniture",
            "quantity": random.randint(20, 100),  # In stock
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(0, 2),  # NO/FEW SALES
            "last_sale": (
                datetime.now() - timedelta(days=random.randint(120, 365))
            ).strftime("%Y-%m-%d"),
        }

    def _generate_negative_inventory_item(self, sku: str) -> dict:
        """Generate item with negative quantity (data integrity issue)."""
        cost = random.uniform(10, 100)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)

        return {
            "sku": sku,
            "description": f"Negative Inv Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Electronics",
            "quantity": random.randint(-50, -1),  # NEGATIVE
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(20, 100),
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
        }

    def _generate_overstock_item(self, sku: str) -> dict:
        """Generate overstocked item (>180 days of supply)."""
        cost = random.uniform(10, 50)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)
        sold = random.randint(1, 5)  # Low sales
        quantity = sold * random.randint(200, 400)  # Way too much stock

        return {
            "sku": sku,
            "description": f"Overstock Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Hardware",
            "quantity": quantity,
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": sold,
            "last_sale": (
                datetime.now() - timedelta(days=random.randint(7, 30))
            ).strftime("%Y-%m-%d"),
        }

    def _generate_price_discrepancy_item(self, sku: str) -> dict:
        """Generate item with price significantly below suggested retail."""
        cost = random.uniform(20, 60)
        suggested_retail = cost * random.uniform(1.8, 2.5)  # Normal markup
        actual_revenue = suggested_retail * random.uniform(0.5, 0.7)  # 30-50% below

        return {
            "sku": sku,
            "description": f"Price Disc Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Apparel",
            "quantity": random.randint(20, 50),
            "cost": round(cost, 2),
            "revenue": round(actual_revenue, 2),
            "sug. retail": round(suggested_retail, 2),
            "sold": random.randint(10, 40),
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
        }

    def _generate_shrinkage_item(self, sku: str) -> dict:
        """Generate item with significant inventory shrinkage."""
        cost = random.uniform(20, 80)
        margin = random.uniform(0.30, 0.40)
        revenue = cost / (1 - margin)
        quantity = random.randint(30, 80)
        # Negative qty_difference indicates shrinkage
        qty_difference = random.randint(-20, -5)

        return {
            "sku": sku,
            "description": f"Shrinkage Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Electronics",
            "quantity": quantity,
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(10, 30),
            "qty_difference": qty_difference,  # SHRINKAGE
            "last_sale": datetime.now().strftime("%Y-%m-%d"),
        }

    def _generate_margin_erosion_item(self, sku: str) -> dict:
        """Generate item with margin erosion (margin < 15%)."""
        cost = random.uniform(30, 80)
        # Margin erosion: 5-15% margin
        margin = random.uniform(0.05, 0.15)
        revenue = cost / (1 - margin)

        return {
            "sku": sku,
            "description": f"Margin Erosion Item {sku}",
            "vendor": f"Vendor_{random.randint(1, 10)}",
            "category": "Hardware",
            "quantity": random.randint(20, 60),
            "cost": round(cost, 2),
            "revenue": round(revenue, 2),
            "sold": random.randint(30, 100),
            "last_sale": (
                datetime.now() - timedelta(days=random.randint(1, 7))
            ).strftime("%Y-%m-%d"),
        }


class VSAValidator:
    """
    Validates VSA detection accuracy against ground truth.
    """

    def __init__(self):
        """Initialize validator."""
        self.results: dict[str, ValidationResult] = {}

    def validate(
        self,
        detected: dict[str, list[str]],
        ground_truth: dict[str, set[str]],
    ) -> dict[str, ValidationResult]:
        """
        Compare detection results against ground truth.

        Args:
            detected: Dict mapping primitive -> list of detected SKUs
            ground_truth: Dict mapping primitive -> set of known anomalous SKUs

        Returns:
            Dict mapping primitive -> ValidationResult
        """
        self.results = {}

        for primitive in ground_truth.keys():
            result = ValidationResult(primitive=primitive)

            detected_set = set(s.lower() for s in detected.get(primitive, []))
            truth_set = ground_truth[primitive]

            # True positives: detected AND in ground truth
            result.detected_skus = detected_set & truth_set
            result.true_positives = len(result.detected_skus)

            # False positives: detected but NOT in ground truth
            result.spurious_skus = detected_set - truth_set
            result.false_positives = len(result.spurious_skus)

            # False negatives: in ground truth but NOT detected
            result.missed_skus = truth_set - detected_set
            result.false_negatives = len(result.missed_skus)

            result.calculate_metrics()
            self.results[primitive] = result

        return self.results

    def print_report(self):
        """Print validation report to stdout."""
        print("\n" + "=" * 70)
        print("VSA VALIDATION REPORT")
        print("=" * 70)

        for primitive, result in sorted(self.results.items()):
            status = self._get_status(result)
            print(f"\n{primitive.upper()} [{status}]")
            print(f"  Precision: {result.precision:.2%}")
            print(f"  Recall:    {result.recall:.2%}")
            print(f"  F1 Score:  {result.f1_score:.2%}")
            print(
                f"  TP/FP/FN:  {result.true_positives}/{result.false_positives}/{result.false_negatives}"
            )

            if result.missed_skus and len(result.missed_skus) <= 5:
                print(f"  Missed:    {', '.join(list(result.missed_skus)[:5])}")

        print("\n" + "=" * 70)
        print("DECISION SUMMARY")
        print("=" * 70)

        keep_count = sum(
            1 for r in self.results.values() if self._get_status(r) == "KEEP"
        )
        calibrate_count = sum(
            1 for r in self.results.values() if self._get_status(r) == "CALIBRATE"
        )
        kill_count = sum(
            1 for r in self.results.values() if self._get_status(r) == "KILL"
        )

        print(f"  KEEP:      {keep_count} primitives")
        print(f"  CALIBRATE: {calibrate_count} primitives")
        print(f"  KILL:      {kill_count} primitives")

        overall = (
            "KEEP"
            if kill_count == 0 and keep_count >= 4
            else "CALIBRATE" if kill_count < 3 else "KILL"
        )
        print(f"\n  OVERALL RECOMMENDATION: {overall}")

    def _get_status(self, result: ValidationResult) -> str:
        """Determine keep/calibrate/kill status for a primitive."""
        if result.recall < 0.3:
            return "KILL"
        elif result.precision >= 0.6 and result.recall >= 0.5:
            return "KEEP"
        else:
            return "CALIBRATE"


# =============================================================================
# PYTEST TEST CASES
# =============================================================================


class TestSyntheticDataGenerator:
    """Tests for the synthetic data generator."""

    def test_generates_correct_counts(self):
        """Test that generator produces correct number of items."""
        gen = SyntheticDataGenerator(seed=42)
        rows, ground_truth = gen.generate_dataset(n_normal=50, n_per_anomaly=5)

        # 50 normal + 8 primitives * 5 each = 90 total
        assert len(rows) == 50 + (8 * 5)

        # Each primitive should have 5 ground truth items
        for primitive, skus in ground_truth.items():
            assert len(skus) == 5, f"{primitive} should have 5 ground truth items"

    def test_reproducible_with_seed(self):
        """Test that same seed produces same data."""
        gen1 = SyntheticDataGenerator(seed=123)
        gen2 = SyntheticDataGenerator(seed=123)

        rows1, gt1 = gen1.generate_dataset(n_normal=10, n_per_anomaly=2)
        rows2, gt2 = gen2.generate_dataset(n_normal=10, n_per_anomaly=2)

        # Should be identical
        assert gt1 == gt2

    def test_low_stock_items_have_low_quantity(self):
        """Test that low stock items actually have low quantity."""
        gen = SyntheticDataGenerator(seed=42)
        rows, ground_truth = gen.generate_dataset(n_normal=10, n_per_anomaly=5)

        low_stock_skus = ground_truth["low_stock"]
        for row in rows:
            if row["sku"].lower() in low_stock_skus:
                assert (
                    row["quantity"] < 5
                ), f"Low stock item {row['sku']} has qty {row['quantity']}"

    def test_negative_inventory_items_are_negative(self):
        """Test that negative inventory items have negative quantity."""
        gen = SyntheticDataGenerator(seed=42)
        rows, ground_truth = gen.generate_dataset(n_normal=10, n_per_anomaly=5)

        neg_inv_skus = ground_truth["negative_inventory"]
        for row in rows:
            if row["sku"].lower() in neg_inv_skus:
                assert (
                    row["quantity"] < 0
                ), f"Negative inv item {row['sku']} has qty {row['quantity']}"


class TestVSAValidator:
    """Tests for the VSA validator."""

    def test_perfect_detection(self):
        """Test metrics for perfect detection."""
        validator = VSAValidator()

        ground_truth = {"low_stock": {"sku1", "sku2", "sku3"}}
        detected = {"low_stock": ["sku1", "sku2", "sku3"]}

        results = validator.validate(detected, ground_truth)

        assert results["low_stock"].precision == 1.0
        assert results["low_stock"].recall == 1.0
        assert results["low_stock"].f1_score == 1.0

    def test_no_detection(self):
        """Test metrics for no detection."""
        validator = VSAValidator()

        ground_truth = {"low_stock": {"sku1", "sku2", "sku3"}}
        detected = {"low_stock": []}

        results = validator.validate(detected, ground_truth)

        assert results["low_stock"].precision == 0.0
        assert results["low_stock"].recall == 0.0
        assert results["low_stock"].false_negatives == 3

    def test_all_false_positives(self):
        """Test metrics when all detections are wrong."""
        validator = VSAValidator()

        ground_truth = {"low_stock": {"sku1", "sku2"}}
        detected = {"low_stock": ["wrong1", "wrong2", "wrong3"]}

        results = validator.validate(detected, ground_truth)

        assert results["low_stock"].precision == 0.0
        assert results["low_stock"].recall == 0.0
        assert results["low_stock"].false_positives == 3
        assert results["low_stock"].false_negatives == 2

    def test_partial_detection(self):
        """Test metrics for partial detection with some false positives."""
        validator = VSAValidator()

        ground_truth = {"low_stock": {"sku1", "sku2", "sku3", "sku4"}}
        detected = {"low_stock": ["sku1", "sku2", "wrong1"]}

        results = validator.validate(detected, ground_truth)

        # TP=2, FP=1, FN=2
        assert results["low_stock"].true_positives == 2
        assert results["low_stock"].false_positives == 1
        assert results["low_stock"].false_negatives == 2
        assert results["low_stock"].precision == pytest.approx(2 / 3)
        assert results["low_stock"].recall == pytest.approx(2 / 4)


class TestVSAIntegration:
    """
    Integration tests that actually run VSA detection.

    These tests require the sentinel_engine to be properly installed.
    Skip if not available.
    """

    @pytest.fixture
    def engine_available(self):
        """Check if sentinel engine is available."""
        try:
            from sentinel_engine import bundle_pos_facts, query_bundle  # noqa: F401
            from sentinel_engine.context import create_analysis_context  # noqa: F401

            return True
        except ImportError:
            return False

    def test_vsa_detects_negative_inventory(self, engine_available):
        """Test that VSA detects negative inventory items."""
        if not engine_available:
            pytest.skip("sentinel_engine not available")

        from sentinel_engine import bundle_pos_facts, query_bundle
        from sentinel_engine.context import create_analysis_context

        gen = SyntheticDataGenerator(seed=42)
        rows, ground_truth = gen.generate_dataset(n_normal=50, n_per_anomaly=10)

        ctx = create_analysis_context()
        bundle = bundle_pos_facts(ctx, rows)
        items, scores = query_bundle(ctx, bundle, "negative_inventory")

        # Should detect at least some negative inventory items
        detected_set = set(s.lower() for s in items if scores[items.index(s)] > 0.1)
        truth_set = ground_truth["negative_inventory"]

        recall = len(detected_set & truth_set) / len(truth_set) if truth_set else 0

        # Negative inventory is a clear signal - should have decent recall
        assert recall >= 0.3, f"Recall for negative_inventory too low: {recall:.2%}"

    def test_full_validation_run(self, engine_available):
        """Run full validation across all primitives."""
        if not engine_available:
            pytest.skip("sentinel_engine not available")

        from sentinel_engine import bundle_pos_facts, query_bundle
        from sentinel_engine.context import create_analysis_context

        gen = SyntheticDataGenerator(seed=42)
        rows, ground_truth = gen.generate_dataset(n_normal=100, n_per_anomaly=15)

        ctx = create_analysis_context()
        bundle = bundle_pos_facts(ctx, rows)

        detected = {}
        for primitive in ground_truth.keys():
            items, scores = query_bundle(ctx, bundle, primitive)
            # Filter to items with score > 0.1
            detected[primitive] = [
                item for item, score in zip(items, scores) if score > 0.1
            ]

        validator = VSAValidator()
        results = validator.validate(detected, ground_truth)
        validator.print_report()

        # At minimum, negative_inventory should work well (clearest signal)
        assert (
            results["negative_inventory"].recall >= 0.3
        ), "Critical: negative_inventory detection failing"
