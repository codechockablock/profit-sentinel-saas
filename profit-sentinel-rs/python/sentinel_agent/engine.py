"""Rust pipeline bridge via subprocess + JSON.

Calls the sentinel-server binary with --json flag and parses the output
into typed Pydantic models. This is Option B from the roadmap — simpler
than PyO3, the binary already works, and subprocess overhead is negligible
compared to the pipeline's own execution time.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

from .models import Digest


class PipelineError(Exception):
    """Raised when the Rust pipeline fails."""

    def __init__(self, message: str, stderr: str = "", returncode: int = -1):
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(message)


def _find_binary() -> Path:
    """Locate the sentinel-server binary.

    Search order:
    1. SENTINEL_BIN environment variable
    2. target/release/sentinel-server (relative to workspace root)
    3. target/debug/sentinel-server (relative to workspace root)
    """
    import os

    env_bin = os.environ.get("SENTINEL_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.is_file():
            return p
        raise PipelineError(f"SENTINEL_BIN set but not found: {env_bin}")

    # Walk up from this file to find the workspace root
    # python/sentinel_agent/engine.py -> profit-sentinel-rs/
    workspace = Path(__file__).resolve().parent.parent.parent

    for profile in ("release", "debug"):
        candidate = workspace / "target" / profile / "sentinel-server"
        if candidate.is_file():
            return candidate

    raise PipelineError(
        "sentinel-server binary not found. "
        "Run 'cargo build --release -p sentinel-server' first, "
        "or set SENTINEL_BIN environment variable."
    )


class SentinelEngine:
    """Bridge to the Rust sentinel-server binary.

    Usage:
        engine = SentinelEngine()
        digest = engine.run("fixtures/sample_inventory.csv")
        digest = engine.run("data.csv", stores=["store-7"], top_k=3)
    """

    def __init__(self, binary_path: Path | str | None = None):
        if binary_path:
            self.binary = Path(binary_path)
            if not self.binary.is_file():
                raise PipelineError(f"Binary not found: {self.binary}")
        else:
            self.binary = _find_binary()

    def run(
        self,
        csv_path: str | Path,
        stores: list[str] | None = None,
        top_k: int = 5,
        timeout_seconds: int = 30,
    ) -> Digest:
        """Run the Rust pipeline and return a typed Digest.

        Args:
            csv_path: Path to inventory CSV file.
            stores: Optional list of store IDs to filter.
            top_k: Number of top issues to return.
            timeout_seconds: Max time to wait for the pipeline.

        Returns:
            Digest model with all issues, SKU details, and summary.

        Raises:
            PipelineError: If the binary fails or output is invalid.
            FileNotFoundError: If the CSV file doesn't exist.
        """
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        cmd = [str(self.binary), str(csv_path), "--json", "--top", str(top_k)]
        if stores:
            cmd.extend(["--stores", ",".join(stores)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            raise PipelineError(
                f"Pipeline timed out after {timeout_seconds}s",
                stderr=str(e),
            )

        if result.returncode != 0:
            raise PipelineError(
                f"Pipeline failed (exit code {result.returncode})",
                stderr=result.stderr.strip(),
                returncode=result.returncode,
            )

        if not result.stdout.strip():
            raise PipelineError("Pipeline produced empty output")

        try:
            return Digest.model_validate_json(result.stdout)
        except Exception as e:
            raise PipelineError(
                f"Failed to parse pipeline JSON: {e}",
                stderr=result.stdout[:500],
            )

    def run_from_adapter(
        self,
        records: list,
        stores: list[str] | None = None,
        top_k: int = 5,
        timeout_seconds: int = 60,
        reference_date: date | None = None,
    ) -> AnalysisResult:
        """Run the pipeline directly from adapter output.

        This is the primary entry point for the Phase 8 bridge. It:
        1. Converts NormalizedInventory records to pipeline CSV via PipelineBridge
        2. Runs the Rust pipeline
        3. Returns an AnalysisResult with both the Digest and enrichment data

        Args:
            records: List of NormalizedInventory records from any adapter.
            stores: Optional list of store IDs to filter.
            top_k: Number of top issues to return.
            timeout_seconds: Max time to wait for the pipeline.
            reference_date: Date for computing days_since_receipt.

        Returns:
            AnalysisResult with digest, enrichment index, and bridge metadata.

        Raises:
            PipelineError: If the pipeline fails.
            ValueError: If no records are provided.
        """
        if not records:
            raise ValueError("No inventory records provided")

        from .adapters.bridge import PipelineBridge

        bridge = PipelineBridge(reference_date=reference_date)

        # Convert to pipeline CSV (temp file, auto-cleaned)
        import os
        import tempfile

        fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="sentinel_bridge_")
        os.close(fd)

        try:
            csv_path = bridge.to_pipeline_csv(records, tmp_path)

            # Run the Rust pipeline
            digest = self.run(
                csv_path,
                stores=stores,
                top_k=top_k,
                timeout_seconds=timeout_seconds,
            )

            # Build enrichment index for bidirectional mapping
            enrichment = bridge.build_enrichment_index(records)

            return AnalysisResult(
                digest=digest,
                enrichment_index=enrichment,
                records_converted=len(records),
                bridge_reference_date=bridge.reference_date,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def health_check(self) -> dict:
        """Verify the binary exists and is executable."""
        return {
            "binary": str(self.binary),
            "exists": self.binary.is_file(),
            "executable": (
                self.binary.stat().st_mode & 0o111 != 0
                if self.binary.is_file()
                else False
            ),
        }


class AnalysisResult:
    """Result of running the pipeline from adapter output.

    Combines the Rust Digest with the original adapter data for
    bidirectional enrichment.
    """

    def __init__(
        self,
        digest: Digest,
        enrichment_index: dict,
        records_converted: int,
        bridge_reference_date: date,
    ):
        self.digest = digest
        self.enrichment_index = enrichment_index
        self.records_converted = records_converted
        self.bridge_reference_date = bridge_reference_date

    def enrich_sku(self, sku_id: str, store_id: str | None = None):
        """Look up original adapter data for a SKU.

        Args:
            sku_id: The SKU identifier from the pipeline output.
            store_id: Optional store ID for multi-store disambiguation.

        Returns:
            NormalizedInventory record or None if not found.
        """
        if store_id:
            key = f"{store_id}::{sku_id}"
            if key in self.enrichment_index:
                return self.enrichment_index[key]
        return self.enrichment_index.get(sku_id)

    @property
    def total_issues(self) -> int:
        return self.digest.summary.total_issues

    @property
    def total_dollar_impact(self) -> float:
        return self.digest.summary.total_dollar_impact

    @property
    def summary(self) -> str:
        """Human-readable summary combining bridge and pipeline data."""
        parts = [
            f"Records converted: {self.records_converted:,}",
            f"Reference date: {self.bridge_reference_date}",
            f"Issues found: {self.total_issues}",
            f"Total dollar impact: ${self.total_dollar_impact:,.0f}",
            f"Records processed by pipeline: {self.digest.summary.records_processed:,}",
            f"Pipeline time: {self.digest.pipeline_ms}ms",
        ]
        return "\n".join(parts)
