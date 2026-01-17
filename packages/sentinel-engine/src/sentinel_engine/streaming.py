"""
Streaming Module for Large-Scale POS Data Processing

Handles files with 150k+ rows using memory-efficient chunked processing.
Uses a two-pass approach:
1. First pass: Compute streaming statistics (Welford's algorithm)
2. Second pass: Chunked bundling with pre-computed stats

Memory target: <2GB for 150k rows at 8192-D complex64

Usage:
    from sentinel_engine.streaming import process_large_file

    results = process_large_file("inventory_150k.csv")
    print(results.total_rows)
    print(results.leak_counts)
    print(results.top_leaks_by_primitive)
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Any, Tuple, Union, IO
import io

import torch

from .context import AnalysisContext, create_analysis_context


def _unbind(bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Unbind using complex conjugate: unbind(a*b, a) ≈ b."""
    return bound * torch.conj(key)


from .core import (
    bundle_pos_facts,
    query_bundle,
    QUANTITY_ALIASES,
    COST_ALIASES,
    REVENUE_ALIASES,
    SOLD_ALIASES,
    MARGIN_ALIASES,
    CATEGORY_ALIASES,
    _get_field,
    _safe_float,
)

logger = logging.getLogger(__name__)

# Chunk size calibrated for 8192-D complex64: ~1.7GB RAM for 150k rows
# v3.2: Back to 15k for faster codebook saturation
DEFAULT_CHUNK_SIZE = 15000

# Large file threshold for auto-optimizations
LARGE_FILE_THRESHOLD = 30000  # Force SKU-only codebook above this


@dataclass
class StreamingStats:
    """Streaming statistics using Welford's online algorithm.

    Memory-efficient: O(1) space regardless of data size.
    Numerically stable for large datasets.
    """
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences

    def update(self, value: float) -> None:
        """Add a single value to running statistics."""
        if value is None or value != value:  # NaN check
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Population variance."""
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return self.variance ** 0.5

    def to_dict(self) -> Dict[str, float]:
        """Export statistics."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "variance": self.variance,
        }


@dataclass
class CategoryStats:
    """Per-category margin statistics."""
    margins: Dict[str, StreamingStats] = field(default_factory=dict)

    def update(self, category: str, margin: float) -> None:
        """Update margin stat for category."""
        if category not in self.margins:
            self.margins[category] = StreamingStats()
        self.margins[category].update(margin)

    def get_avg_margin(self, category: str, fallback: float = 0.3) -> float:
        """Get average margin for category, with fallback."""
        if category in self.margins and self.margins[category].count >= 3:
            return self.margins[category].mean
        return fallback


@dataclass
class StreamingResult:
    """Results from streaming analysis."""
    total_rows: int
    processed_rows: int
    chunks_processed: int
    leak_counts: Dict[str, int]
    top_leaks_by_primitive: Dict[str, List[Tuple[str, float]]]
    dataset_stats: Dict[str, Any]
    elapsed_seconds: float
    peak_memory_mb: float

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Streaming Analysis Complete",
            f"  Rows: {self.total_rows:,} total, {self.processed_rows:,} processed",
            f"  Chunks: {self.chunks_processed}",
            f"  Time: {self.elapsed_seconds:.1f}s",
            f"  Peak Memory: {self.peak_memory_mb:.0f}MB",
            f"  Detections:",
        ]
        for prim, count in sorted(self.leak_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"    {prim}: {count:,}")
        return "\n".join(lines)


def read_file_chunked(
    filepath: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding: str = "utf-8",
) -> Iterator[List[Dict]]:
    """
    Read CSV/TSV file in chunks, yielding lists of row dicts.

    Handles:
    - CSV (comma) and TSV (tab) formats
    - Various encodings (utf-8, latin-1)
    - Files with or without headers

    Args:
        filepath: Path to file
        chunk_size: Rows per chunk
        encoding: File encoding

    Yields:
        List of row dictionaries per chunk
    """
    filepath = Path(filepath)

    # Detect delimiter
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(8192)
        if "\t" in sample and "," not in sample:
            delimiter = "\t"
        else:
            delimiter = ","

    with open(filepath, "r", encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:  # Don't forget the last partial chunk
            yield chunk


SKU_ALIASES = ['sku', 'item', 'product_id', 'productid', 'item_id', 'itemid', 'upc', 'barcode']


def compute_streaming_stats(
    filepath: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    collect_skus: bool = False,
    max_skus: int = 10000,
) -> Tuple[Dict[str, StreamingStats], CategoryStats, Optional[List[str]]]:
    """
    First pass: Compute streaming statistics for adaptive thresholds.

    v3.2: Optionally collects unique SKUs for batch codebook pre-population.

    Returns:
        (field_stats, category_stats, unique_skus) tuple
        unique_skus is None if collect_skus=False
    """
    logger.info(f"Pass 1: Computing streaming statistics for {filepath}")

    field_stats = {
        "quantity": StreamingStats(),
        "margin": StreamingStats(),
        "sold": StreamingStats(),
        "rows": StreamingStats(),  # Track total rows
    }
    category_stats = CategoryStats()

    # v3.2: Collect unique SKUs for batch pre-population
    unique_skus: set = set() if collect_skus else None

    rows_scanned = 0
    for chunk in read_file_chunked(filepath, chunk_size):
        for row in chunk:
            rows_scanned += 1
            field_stats["rows"].update(1.0)  # Count every row

            # v3.2: Collect SKU if enabled and under limit
            if collect_skus and len(unique_skus) < max_skus:
                sku = str(_get_field(row, SKU_ALIASES, '')).strip().lower()
                if sku and sku not in ('unknown', 'unknown_sku', ''):
                    unique_skus.add(sku)

            # Quantity
            qty = _safe_float(_get_field(row, QUANTITY_ALIASES, None))
            if qty is not None and qty > 0:
                field_stats["quantity"].update(qty)

            # Margin
            cost = _safe_float(_get_field(row, COST_ALIASES, 0))
            revenue = _safe_float(_get_field(row, REVENUE_ALIASES, 0))
            if revenue > 0 and cost > 0:
                margin = (revenue - cost) / revenue
                field_stats["margin"].update(margin)

                # Category margin
                category = str(_get_field(row, CATEGORY_ALIASES, "unknown")).strip().lower()
                category_stats.update(category, margin)

            # Sold
            sold = _safe_float(_get_field(row, SOLD_ALIASES, None))
            if sold is not None and sold >= 0:
                field_stats["sold"].update(sold)

    sku_count = len(unique_skus) if unique_skus else 0
    logger.info(
        f"Stats computed: {rows_scanned:,} rows, "
        f"avg_qty={field_stats['quantity'].mean:.1f}, "
        f"avg_margin={field_stats['margin'].mean:.2%}, "
        f"avg_sold={field_stats['sold'].mean:.1f}"
        + (f", unique_skus={sku_count:,}" if collect_skus else "")
    )

    sku_list = list(unique_skus) if unique_skus else None
    return field_stats, category_stats, sku_list


def bundle_pos_facts_streaming(
    ctx: AnalysisContext,
    filepath: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    field_stats: Optional[Dict[str, StreamingStats]] = None,
    category_stats: Optional[CategoryStats] = None,
) -> torch.Tensor:
    """
    Second pass: Bundle POS facts in chunks with memory-efficient accumulation.

    This function processes the file in chunks, bundling each chunk and
    accumulating into a single bundle vector. This keeps memory bounded
    regardless of file size.

    Args:
        ctx: Analysis context
        filepath: Path to CSV/TSV file
        chunk_size: Rows per chunk
        field_stats: Pre-computed field statistics (optional, computes if None)
        category_stats: Pre-computed category stats (optional)

    Returns:
        Final bundled hypervector
    """
    logger.info(f"Pass 2: Chunked bundling for {filepath}")

    # Compute stats if not provided
    if field_stats is None:
        field_stats, category_stats, _ = compute_streaming_stats(filepath, chunk_size)

    # Update context with pre-computed stats
    ctx.update_stats(
        avg_quantity=field_stats["quantity"].mean,
        avg_margin=field_stats["margin"].mean,
        avg_sold=field_stats["sold"].mean,
    )

    # Accumulator bundle
    final_bundle = ctx.zeros()
    chunks_processed = 0

    for chunk in read_file_chunked(filepath, chunk_size):
        # Process chunk
        chunk_bundle = bundle_pos_facts(ctx, chunk)

        # Accumulate into final bundle
        final_bundle = final_bundle + chunk_bundle
        chunks_processed += 1

        logger.info(
            f"Chunk {chunks_processed}: {len(chunk)} rows, "
            f"codebook={len(ctx.codebook)}, "
            f"leaks={sum(ctx.leak_counts.values())}"
        )

    # Normalize final bundle
    final_bundle = ctx.normalize(final_bundle)

    logger.info(
        f"Bundling complete: {chunks_processed} chunks, "
        f"{ctx.rows_processed} rows processed"
    )

    return final_bundle


def process_large_file(
    filepath: Union[str, Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    dimensions: int = 8192,
    top_k_per_primitive: int = 20,
    primitives: Optional[List[str]] = None,
    force_sku_only: Optional[bool] = None,
) -> StreamingResult:
    """
    High-level API for processing large POS files.

    Performs two-pass analysis:
    1. Statistics computation (also counts rows for auto-optimization)
    2. Chunked bundling and detection

    v3.1 Auto-optimizations:
    - Files >30k rows: Force SKU-only codebook (excludes descriptions/vendors)
    - Codebook >18k entries: Use HierarchicalResonator for O(sqrt(n)) queries

    Args:
        filepath: Path to CSV/TSV file
        chunk_size: Rows per chunk (default 20k)
        dimensions: VSA dimensionality (default 8192)
        top_k_per_primitive: Top leaks to return per type
        primitives: List of primitives to query (default: all 8)
        force_sku_only: Override SKU-only mode (None=auto based on file size)

    Returns:
        StreamingResult with detections and statistics
    """
    import psutil
    import math

    start_time = time.time()
    process = psutil.Process()

    # Default primitives
    if primitives is None:
        primitives = [
            "low_stock",
            "high_margin_leak",
            "dead_item",
            "negative_inventory",
            "overstock",
            "price_discrepancy",
            "shrinkage_pattern",
            "margin_erosion",
        ]

    filepath = Path(filepath)
    logger.info(f"Processing large file: {filepath}")

    # Pass 1: Statistics (also counts rows and collects SKUs for batch pre-pop)
    from .context import DEFAULT_MAX_CODEBOOK_SIZE, HIERARCHICAL_CODEBOOK_THRESHOLD
    field_stats, category_stats, unique_skus = compute_streaming_stats(
        filepath, chunk_size, collect_skus=True, max_skus=DEFAULT_MAX_CODEBOOK_SIZE
    )
    total_rows = field_stats["rows"].count  # Use dedicated row counter

    # v3.1: Auto-enable SKU-only codebook for large files
    if force_sku_only is None:
        sku_only = total_rows > LARGE_FILE_THRESHOLD
    else:
        sku_only = force_sku_only

    if sku_only:
        logger.info(f"Large file detected ({total_rows:,} rows) - enabling SKU-only codebook")

    # Create context
    ctx = create_analysis_context(dimensions=dimensions, sku_only_codebook=sku_only)

    # v3.2: Batch pre-populate codebook with collected SKUs (major speedup)
    if unique_skus:
        logger.info(f"Pre-populating codebook with {len(unique_skus):,} unique SKUs")
        ctx.batch_add_to_codebook(unique_skus, is_sku=True)
        logger.info(f"Codebook pre-populated: {len(ctx.codebook):,} entries")

    # Pass 2: Bundling with optimized context
    bundle = bundle_pos_facts_streaming(
        ctx, filepath, chunk_size, field_stats, category_stats
    )

    # v3.1: Track codebook size for diagnostics
    codebook_size = len(ctx.codebook)

    # v3.2: HierarchicalResonator disabled - query phase is already fast (<1s)
    # and the hierarchy building adds ~2GB memory overhead (triples codebook memory)
    # Main bottleneck is bundling phase, not query phase
    use_hierarchical = False
    hierarchical_resonator = None

    if codebook_size > HIERARCHICAL_CODEBOOK_THRESHOLD:
        logger.info(
            f"Large codebook ({codebook_size:,} entries) - "
            f"using direct query (hierarchical disabled for memory savings)"
        )

    # Query for top leaks per primitive
    # v3.2: Use HierarchicalResonator for faster codebook lookup if available
    top_leaks: Dict[str, List[Tuple[str, float]]] = {}

    if hierarchical_resonator is not None:
        # Use hierarchical query for each primitive
        logger.info("Using HierarchicalResonator for fast primitive queries")
        for prim in primitives:
            # Get primitive vector from context
            prim_vec = ctx.get_primitive(prim)
            if prim_vec is None:
                top_leaks[prim] = []
                continue

            # Unbind bundle with primitive to get query vector
            query_vec = _unbind(bundle, prim_vec)

            # Use hierarchical resonator for fast top-k
            result = hierarchical_resonator.resonate(query_vec)
            top_leaks[prim] = result.top_matches[:top_k_per_primitive]
    else:
        # Standard query
        for prim in primitives:
            items, scores = query_bundle(ctx, bundle, prim, top_k=top_k_per_primitive)
            top_leaks[prim] = list(zip(items, scores))

    elapsed = time.time() - start_time
    peak_memory = process.memory_info().rss / (1024 * 1024)  # MB

    return StreamingResult(
        total_rows=total_rows,
        processed_rows=ctx.rows_processed,
        chunks_processed=ctx.rows_processed // chunk_size + 1,
        leak_counts=ctx.leak_counts.copy(),
        top_leaks_by_primitive=top_leaks,
        dataset_stats={
            "avg_quantity": field_stats["quantity"].mean,
            "avg_margin": field_stats["margin"].mean,
            "avg_sold": field_stats["sold"].mean,
            "categories": len(category_stats.margins),
            "sku_only_codebook": sku_only,
            "hierarchical_resonator": use_hierarchical,
            "codebook_size": codebook_size,
        },
        elapsed_seconds=elapsed,
        peak_memory_mb=peak_memory,
    )


def process_dataframe(
    df: "pd.DataFrame",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    dimensions: int = 8192,
) -> StreamingResult:
    """
    Process a pandas DataFrame in chunks.

    Converts DataFrame to row dicts and processes with streaming.

    Args:
        df: pandas DataFrame with POS data
        chunk_size: Rows per chunk
        dimensions: VSA dimensionality

    Returns:
        StreamingResult
    """
    import psutil

    start_time = time.time()
    process = psutil.Process()

    logger.info(f"Processing DataFrame: {len(df)} rows")

    # Convert to records
    rows = df.to_dict("records")

    # Create context
    ctx = create_analysis_context(dimensions=dimensions)

    # Process in chunks
    final_bundle = ctx.zeros()

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        chunk_bundle = bundle_pos_facts(ctx, chunk)
        final_bundle = final_bundle + chunk_bundle

        logger.info(
            f"Chunk {i // chunk_size + 1}: {len(chunk)} rows, "
            f"leaks={sum(ctx.leak_counts.values())}"
        )

    final_bundle = ctx.normalize(final_bundle)

    elapsed = time.time() - start_time
    peak_memory = process.memory_info().rss / (1024 * 1024)

    return StreamingResult(
        total_rows=len(df),
        processed_rows=ctx.rows_processed,
        chunks_processed=len(rows) // chunk_size + 1,
        leak_counts=ctx.leak_counts.copy(),
        top_leaks_by_primitive={},  # Skip query for DataFrame processing
        dataset_stats=ctx.dataset_stats.copy(),
        elapsed_seconds=elapsed,
        peak_memory_mb=peak_memory,
    )
