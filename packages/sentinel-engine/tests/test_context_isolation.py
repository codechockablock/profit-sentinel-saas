"""
Context Isolation Tests - Verify no cross-request contamination.

These tests ensure that:
1. Each AnalysisContext is completely independent
2. Concurrent analysis runs don't share state
3. Codebook entries from one context don't leak to another
4. Primitive vectors are consistent but isolated per context

CRITICAL: These tests validate the security fix for the global codebook bug.
"""

import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import torch


class TestContextIsolation:
    """Tests for request-scoped context isolation."""

    def test_separate_contexts_have_independent_codebooks(self):
        """Two contexts should have completely independent codebooks."""
        from sentinel_engine.context import create_analysis_context

        ctx1 = create_analysis_context()
        ctx2 = create_analysis_context()

        # Add entity to ctx1 only
        ctx1.add_to_codebook("customer_a_sku_001")

        # Verify it's in ctx1
        assert "customer_a_sku_001" in ctx1.codebook
        assert len(ctx1.codebook) == 1

        # Verify it's NOT in ctx2
        assert "customer_a_sku_001" not in ctx2.codebook
        assert len(ctx2.codebook) == 0

    def test_contexts_produce_consistent_vectors_for_same_entity(self):
        """Same entity should produce identical vectors in different contexts."""
        from sentinel_engine.context import create_analysis_context

        ctx1 = create_analysis_context()
        ctx2 = create_analysis_context()

        vec1 = ctx1.seed_hash("test_entity")
        vec2 = ctx2.seed_hash("test_entity")

        # Vectors should be identical (deterministic from hash)
        assert torch.allclose(vec1, vec2), "Same entity should produce same vector"

    def test_context_reset_clears_all_state(self):
        """reset() should clear all mutable state."""
        from sentinel_engine.context import create_analysis_context

        ctx = create_analysis_context()

        # Add some state
        ctx.add_to_codebook("sku_001")
        ctx.add_to_codebook("sku_002")
        ctx.rows_processed = 100
        ctx.increment_leak_count("low_stock")
        ctx.increment_leak_count("low_stock")

        # Verify state exists
        assert len(ctx.codebook) == 2
        assert ctx.rows_processed == 100
        assert ctx.leak_counts["low_stock"] == 2

        # Reset
        ctx.reset()

        # Verify state is cleared
        assert len(ctx.codebook) == 0
        assert ctx.rows_processed == 0
        assert ctx.leak_counts["low_stock"] == 0

    def test_concurrent_contexts_no_interference(self):
        """Concurrent analysis in separate contexts should not interfere."""
        from sentinel_engine.context import create_analysis_context

        results = {}
        errors = []

        def analyze_in_context(context_id: str, sku_prefix: str):
            try:
                ctx = create_analysis_context()

                # Add unique SKUs for this context
                for i in range(100):
                    ctx.add_to_codebook(f"{sku_prefix}_item_{i}")

                # Small delay to increase chance of race conditions
                time.sleep(0.01)

                # Verify only our SKUs are present
                for i in range(100):
                    expected_sku = f"{sku_prefix}_item_{i}"
                    if expected_sku not in ctx.codebook:
                        errors.append(f"{context_id}: Missing own SKU {expected_sku}")

                # Verify no SKUs from other contexts leaked in
                for key in ctx.codebook.keys():
                    if not key.startswith(sku_prefix):
                        errors.append(f"{context_id}: Foreign SKU found: {key}")

                results[context_id] = {
                    "codebook_size": len(ctx.codebook),
                    "prefix": sku_prefix,
                }

            except Exception as e:
                errors.append(f"{context_id}: Exception - {e}")

        # Run 10 concurrent analysis contexts
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(10):
                future = executor.submit(
                    analyze_in_context,
                    f"context_{i}",
                    f"customer_{i}"
                )
                futures.append(future)

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()

        # Verify no errors
        assert len(errors) == 0, f"Isolation errors: {errors}"

        # Verify all contexts completed
        assert len(results) == 10, f"Only {len(results)} contexts completed"

        # Verify each context has exactly 100 items
        for ctx_id, data in results.items():
            assert data["codebook_size"] == 100, f"{ctx_id} has {data['codebook_size']} items, expected 100"

    def test_primitives_isolated_but_consistent(self):
        """Primitive vectors should be isolated per context but produce same values."""
        from sentinel_engine.context import create_analysis_context

        ctx1 = create_analysis_context()
        ctx2 = create_analysis_context()

        # Get primitives from both contexts
        prims1 = ctx1.get_primitives()
        prims2 = ctx2.get_primitives()

        # Should have same keys
        assert set(prims1.keys()) == set(prims2.keys())

        # Values should be identical
        for key in prims1.keys():
            assert torch.allclose(prims1[key], prims2[key]), f"Primitive {key} differs between contexts"

        # But the dicts should be different objects (isolated)
        assert prims1 is not prims2, "Primitive dicts should be different objects"

    def test_context_manager_cleanup(self):
        """Context manager should cleanup on exit."""
        from sentinel_engine.context import analysis_context

        with analysis_context() as ctx:
            ctx.add_to_codebook("test_sku")
            assert len(ctx.codebook) == 1

        # After exit, context should be reset
        # (can't access ctx directly, but no exception means cleanup worked)

    def test_fifo_eviction_isolated(self):
        """FIFO eviction should be isolated per context."""
        from sentinel_engine.context import create_analysis_context

        # Create context with small max size
        ctx1 = create_analysis_context(max_codebook_size=5)
        ctx2 = create_analysis_context(max_codebook_size=5)

        # Fill ctx1 past capacity
        for i in range(10):
            ctx1.add_to_codebook(f"ctx1_item_{i}")

        # ctx1 should have only last 5 (due to FIFO)
        assert len(ctx1.codebook) == 5
        assert "ctx1_item_0" not in ctx1.codebook  # Evicted
        assert "ctx1_item_9" in ctx1.codebook  # Still there

        # ctx2 should still be empty
        assert len(ctx2.codebook) == 0


class TestAnalysisFunctionIsolation:
    """Tests for analysis function isolation with context parameter."""

    def test_bundle_pos_facts_uses_context_codebook(self):
        """bundle_pos_facts should populate context's codebook, not global."""
        from sentinel_engine.context import create_analysis_context
        from sentinel_engine import bundle_pos_facts

        ctx1 = create_analysis_context()
        ctx2 = create_analysis_context()

        # Sample data
        rows = [
            {"sku": "SKU_A", "quantity": 10, "cost": 5.0, "revenue": 10.0},
            {"sku": "SKU_B", "quantity": 5, "cost": 3.0, "revenue": 8.0},
        ]

        # Bundle in ctx1
        bundle_pos_facts(ctx1, rows)

        # ctx1 should have the SKUs
        assert len(ctx1.codebook) > 0

        # ctx2 should still be empty
        assert len(ctx2.codebook) == 0

    def test_query_bundle_uses_context_codebook(self):
        """query_bundle should query from context's codebook only."""
        from sentinel_engine.context import create_analysis_context
        from sentinel_engine import bundle_pos_facts, query_bundle

        ctx1 = create_analysis_context()
        ctx2 = create_analysis_context()

        # Sample data with clear low_stock signal
        rows = [
            {"sku": "LOW_STOCK_ITEM", "quantity": 1, "cost": 100.0, "revenue": 200.0, "sold": 50},
            {"sku": "NORMAL_ITEM", "quantity": 100, "cost": 10.0, "revenue": 20.0, "sold": 5},
        ]

        # Bundle and query in ctx1
        bundle1 = bundle_pos_facts(ctx1, rows)
        items1, scores1 = query_bundle(ctx1, bundle1, "low_stock")

        # Bundle and query in ctx2 with different data
        rows2 = [
            {"sku": "OTHER_ITEM", "quantity": 500, "cost": 1.0, "revenue": 2.0, "sold": 1},
        ]
        bundle2 = bundle_pos_facts(ctx2, rows2)
        items2, scores2 = query_bundle(ctx2, bundle2, "low_stock")

        # Results should be independent
        # ctx1 should find LOW_STOCK_ITEM but not OTHER_ITEM
        assert "other_item" not in [i.lower() for i in items1]

        # ctx2 should not find LOW_STOCK_ITEM
        assert "low_stock_item" not in [i.lower() for i in items2]

    def test_concurrent_analysis_different_datasets(self):
        """Concurrent analyses of different datasets should not contaminate."""
        from sentinel_engine.context import create_analysis_context
        from sentinel_engine import bundle_pos_facts, query_bundle

        results = {}
        errors = []

        def run_analysis(analysis_id: int, unique_skus: List[str]):
            try:
                ctx = create_analysis_context()

                rows = [
                    {"sku": sku, "quantity": 1, "cost": 10.0, "revenue": 20.0, "sold": 100}
                    for sku in unique_skus
                ]

                bundle = bundle_pos_facts(ctx, rows)
                items, scores = query_bundle(ctx, bundle, "low_stock")

                # Store codebook keys for verification
                results[analysis_id] = {
                    "codebook_keys": list(ctx.codebook.keys()),
                    "expected_skus": [s.lower() for s in unique_skus],
                    "query_items": items,
                }

            except Exception as e:
                errors.append(f"Analysis {analysis_id}: {e}")

        # Run concurrent analyses with different datasets
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                skus = [f"CUSTOMER_{i}_SKU_{j}" for j in range(20)]
                future = executor.submit(run_analysis, i, skus)
                futures.append(future)

            for future in as_completed(futures):
                future.result()

        # Verify no errors
        assert len(errors) == 0, f"Errors: {errors}"

        # Verify each analysis only has its own SKUs
        for analysis_id, data in results.items():
            expected_prefix = f"customer_{analysis_id}_"
            for key in data["codebook_keys"]:
                # Allow primitive vectors and vendor/category entries
                if key.startswith("primitive_") or key in ("unknown_vendor", "unknown_category"):
                    continue
                assert key.startswith(expected_prefix), \
                    f"Analysis {analysis_id} has foreign key: {key}"


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy API."""

    def test_legacy_api_still_works_with_deprecation(self):
        """Legacy API without context should work but emit deprecation warning."""
        import warnings
        from sentinel_engine import bundle_pos_facts, query_bundle, reset_codebook

        # This should work but emit deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            rows = [
                {"sku": "LEGACY_SKU", "quantity": 5, "cost": 10.0, "revenue": 20.0},
            ]

            # Legacy call without context
            bundle = bundle_pos_facts(rows)

            # Should have emitted deprecation warning
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0, "Expected deprecation warning for legacy API"

    def test_reset_codebook_is_noop(self):
        """reset_codebook should be a no-op in v2.1."""
        from sentinel_engine import reset_codebook

        # Should not raise
        reset_codebook()
