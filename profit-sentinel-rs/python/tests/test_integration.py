"""End-to-end integration tests: CSV → Rust pipeline → Python digest → formatted output.

These tests call the actual Rust binary with real CSV data and validate
the full round-trip through the Python layer.

Requires the sentinel-server binary to be built:
    cargo build --release -p sentinel-server
"""

import os
import subprocess
from pathlib import Path

import pytest
from sentinel_agent.delegation import DelegationManager
from sentinel_agent.digest import MorningDigestGenerator
from sentinel_agent.engine import PipelineError, SentinelEngine
from sentinel_agent.llm_layer import render_digest
from sentinel_agent.models import Digest, IssueType
from sentinel_agent.vendor_assist import VendorCallAssistant

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = WORKSPACE_ROOT / "fixtures"
SAMPLE_CSV = FIXTURES_DIR / "sample_inventory.csv"


def _find_binary() -> Path | None:
    """Find the sentinel-server binary if it exists."""
    for profile in ("release", "debug"):
        candidate = WORKSPACE_ROOT / "target" / profile / "sentinel-server"
        if candidate.is_file():
            return candidate
    return None


BINARY = _find_binary()
SKIP_REASON = (
    "sentinel-server binary not found. Run: cargo build --release -p sentinel-server"
)


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(BINARY is None, reason=SKIP_REASON)
class TestSentinelEngine:
    def setup_method(self):
        self.engine = SentinelEngine(binary_path=BINARY)

    def test_health_check(self):
        health = self.engine.health_check()
        assert health["exists"] is True
        assert health["executable"] is True

    def test_run_full_csv(self):
        digest = self.engine.run(SAMPLE_CSV)
        assert len(digest.issues) > 0
        assert digest.summary.records_processed == 20
        assert digest.summary.stores_affected >= 1

    def test_run_with_store_filter(self):
        digest = self.engine.run(SAMPLE_CSV, stores=["store-7"])
        for issue in digest.issues:
            assert issue.store_id == "store-7"

    def test_run_with_top_k(self):
        digest = self.engine.run(SAMPLE_CSV, top_k=2)
        assert len(digest.issues) <= 2

    def test_run_missing_csv(self):
        with pytest.raises(FileNotFoundError):
            self.engine.run("/nonexistent/file.csv")

    def test_dollar_impact_correctness(self):
        """Validate that the key dollar impact flows through correctly."""
        digest = self.engine.run(
            SAMPLE_CSV,
            stores=["store-7", "store-12"],
            top_k=10,
        )
        # Find the negative inventory issue
        neg_inv = next(
            (
                i
                for i in digest.issues
                if i.issue_type == IssueType.NEGATIVE_INVENTORY
                and i.store_id == "store-7"
            ),
            None,
        )
        assert neg_inv is not None, "Pipeline should produce a NegativeInventory issue for store-7"
        # -47 * $23.50 = $1,104.50
        assert abs(neg_inv.dollar_impact - 1104.5) < 0.01
        assert neg_inv.skus[0].sku_id == "ELC-4401"


# ---------------------------------------------------------------------------
# Full round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(BINARY is None, reason=SKIP_REASON)
class TestFullRoundTrip:
    """CSV → Rust pipeline → Python digest → natural language output."""

    def setup_method(self):
        self.gen = MorningDigestGenerator(engine=SentinelEngine(binary_path=BINARY))

    def test_generate_and_render(self):
        text = self.gen.generate_and_render(SAMPLE_CSV, top_k=5)
        assert "Good morning" in text
        assert "need" in text and "attention" in text
        assert "$" in text  # dollar amounts present

    def test_generate_returns_typed_digest(self):
        digest = self.gen.generate(SAMPLE_CSV)
        assert isinstance(digest, Digest)
        assert len(digest.issues) > 0

    def test_render_all_issue_types_mentioned(self):
        digest = self.gen.generate(SAMPLE_CSV, top_k=10)
        text = self.gen.render(digest)
        # At least Dead Stock should appear (it's the biggest issue)
        assert "Dead Stock" in text

    def test_delegation_from_pipeline_output(self):
        digest = self.gen.generate(SAMPLE_CSV, top_k=5)
        dm = DelegationManager()

        # Delegate every issue
        tasks = []
        for issue in digest.issues:
            task = dm.create_task(issue, assignee="Store Manager")
            tasks.append(task)

        assert len(tasks) == len(digest.issues)
        for task in tasks:
            text = dm.format_for_store_manager(task)
            assert "TASK:" in text
            assert "Action Items:" in text
            assert len(task.action_items) >= 3

    def test_vendor_assist_from_pipeline_output(self):
        digest = self.gen.generate(SAMPLE_CSV, top_k=10)
        assistant = VendorCallAssistant()

        # Find a vendor-related issue (VendorShortShip or MarginErosion)
        vendor_issue = next(
            (
                i
                for i in digest.issues
                if i.issue_type
                in (
                    IssueType.VENDOR_SHORT_SHIP,
                    IssueType.MARGIN_EROSION,
                )
            ),
            None,
        )
        assert vendor_issue is not None, "Pipeline should produce a vendor issue (short ship or margin erosion)"
        prep = assistant.prepare_call(vendor_issue)
        text = assistant.render(prep)
        assert "VENDOR CALL BRIEF" in text
        assert "Talking Points" in text
        assert prep.total_dollar_impact > 0

    def test_pipeline_timing(self):
        """Pipeline should complete in under 1 second for 20 rows."""
        digest = self.gen.generate(SAMPLE_CSV)
        assert digest.pipeline_ms < 1000

    def test_full_output_looks_reasonable(self):
        """Smoke test: the full output should be readable and complete."""
        text = self.gen.generate_and_render(SAMPLE_CSV, top_k=3)

        # Should have greeting
        assert text.startswith("Good morning")

        # Should mention stores
        assert "store" in text.lower() or "Store" in text

        # Should have dollar amounts
        dollar_count = text.count("$")
        assert dollar_count >= 3  # at least in heading + each issue

        # Should mention pipeline metrics
        assert "ms" in text

        # Should not be absurdly short or long
        assert 200 < len(text) < 5000
