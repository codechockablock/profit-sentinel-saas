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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .context import AnalysisContext, create_analysis_context


def _unbind(bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Unbind using complex conjugate: unbind(a*b, a) ≈ b."""
    return bound * torch.conj(key)


from .core import (
    CATEGORY_ALIASES,
    COST_ALIASES,
    QUANTITY_ALIASES,
    REVENUE_ALIASES,
    SOLD_ALIASES,
    _get_field,
    _safe_float,
    bundle_pos_facts,
    query_bundle,
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
        return self.variance**0.5

    def to_dict(self) -> dict[str, float]:
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

    margins: dict[str, StreamingStats] = field(default_factory=dict)

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
    """Results from streaming analysis.

    v3.3: Enhanced with audit data for reproducibility.
    """

    total_rows: int
    processed_rows: int
    chunks_processed: int
    leak_counts: dict[str, int]
    top_leaks_by_primitive: dict[str, list[tuple[str, float]]]
    dataset_stats: dict[str, Any]
    elapsed_seconds: float
    peak_memory_mb: float

    # v3.3: Audit data for reproducibility
    file_hash: str | None = None  # SHA256 of input file
    seeding_summary: dict[str, Any] | None = None  # Deterministic seeding info
    audit_trail: dict[str, Any] | None = None  # Full evidence chain per leak

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "Streaming Analysis Complete",
            f"  Rows: {self.total_rows:,} total, {self.processed_rows:,} processed",
            f"  Chunks: {self.chunks_processed}",
            f"  Time: {self.elapsed_seconds:.1f}s",
            f"  Peak Memory: {self.peak_memory_mb:.0f}MB",
        ]
        if self.seeding_summary:
            lines.append(
                f"  Seed Hash: {self.seeding_summary.get('master_seed', 'N/A')}"
            )
        lines.append("  Detections:")
        for prim, count in sorted(self.leak_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                lines.append(f"    {prim}: {count:,}")
        return "\n".join(lines)

    def to_synopsis(self) -> dict[str, Any]:
        """
        v3.3: Generate analysis synopsis for Supabase storage.

        Returns dict matching analysis_synopses table schema:
        - file_hash, file_row_count
        - detection_counts
        - top_leaks_by_primitive (top 10 SKU + score per primitive)
        - seeding_summary (master_seed, entity counts)
        - dataset_stats, performance_metrics
        - engine_version, codebook_size
        """
        top_10_leaks = {}
        for prim, leaks in self.top_leaks_by_primitive.items():
            top_10_leaks[prim] = [
                {"sku": sku, "score": round(score, 4)} for sku, score in leaks[:10]
            ]

        return {
            # Core identity
            "file_hash": self.file_hash,
            "file_row_count": self.total_rows,
            "file_column_count": None,  # Not tracked in streaming
            # Detections
            "detection_counts": self.leak_counts,
            "top_leaks_by_primitive": top_10_leaks,
            # Audit trail
            "seeding_summary": self.seeding_summary,
            # Statistics
            "dataset_stats": self.dataset_stats,
            # Performance
            "processing_time_seconds": round(self.elapsed_seconds, 2),
            "peak_memory_mb": int(self.peak_memory_mb),
            "dimensions_used": self.dataset_stats.get("dimensions", 8192),
            # Version tracking
            "engine_version": "3.3",
            "codebook_size": self.dataset_stats.get("codebook_size"),
        }

    def to_audit_json(self, include_all_leaks: bool = False) -> dict[str, Any]:
        """
        v3.3: Generate full audit export JSON for evidence chain.

        This is the complete audit trail that can be:
        - Exported for regulatory compliance
        - Used to reproduce exact findings
        - Shared with auditors for verification

        Args:
            include_all_leaks: Include all leaks (not just top 10) per primitive

        Returns:
            Complete audit JSON with evidence chain
        """
        import datetime

        # Build full leaks structure
        leaks_by_primitive = {}
        for prim, leaks in self.top_leaks_by_primitive.items():
            leak_list = leaks if include_all_leaks else leaks[:10]
            leaks_by_primitive[prim] = [
                {"sku": sku, "similarity_score": round(score, 6)}
                for sku, score in leak_list
            ]

        return {
            "audit_version": "1.0",
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "engine_version": "3.3",
            # File identity
            "file_identity": {
                "sha256_hash": self.file_hash,
                "row_count": self.total_rows,
                "processed_rows": self.processed_rows,
            },
            # Reproducibility
            "reproducibility": {
                "seeding_summary": self.seeding_summary,
                "dimensions": self.dataset_stats.get("dimensions", 8192),
                "codebook_size": self.dataset_stats.get("codebook_size"),
                "sku_only_codebook": self.dataset_stats.get("sku_only_codebook", False),
            },
            # Detection results
            "detection_summary": {
                "total_detections": sum(self.leak_counts.values()),
                "counts_by_primitive": self.leak_counts,
            },
            "evidence_chain": leaks_by_primitive,
            # Dataset statistics (for context)
            "dataset_context": {
                "avg_quantity": self.dataset_stats.get("avg_quantity"),
                "avg_margin": self.dataset_stats.get("avg_margin"),
                "avg_sold": self.dataset_stats.get("avg_sold"),
                "categories_analyzed": self.dataset_stats.get("categories"),
            },
            # Performance (for SLA verification)
            "performance": {
                "elapsed_seconds": round(self.elapsed_seconds, 2),
                "peak_memory_mb": int(self.peak_memory_mb),
                "chunks_processed": self.chunks_processed,
            },
        }


def read_file_chunked(
    filepath: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    encoding: str = "utf-8",
) -> Iterator[list[dict]]:
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
    with open(filepath, encoding=encoding, errors="replace") as f:
        sample = f.read(8192)
        if "\t" in sample and "," not in sample:
            delimiter = "\t"
        else:
            delimiter = ","

    with open(filepath, encoding=encoding, errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:  # Don't forget the last partial chunk
            yield chunk


SKU_ALIASES = [
    "sku",
    "item",
    "product_id",
    "productid",
    "item_id",
    "itemid",
    "upc",
    "barcode",
]
VENDOR_ALIASES = ["vendor", "supplier", "manufacturer", "brand", "vendor_name"]
DEPARTMENT_ALIASES = ["department", "dept", "division", "dept_name"]


@dataclass
class EntityRanking:
    """
    v3.3: Ranked entities for deterministic seeding priority.

    Entities are ranked by significance (qty × revenue or sold count).
    Most significant entities get seeded first for reproducibility.
    """

    skus: list[tuple[str, float]] = field(default_factory=list)  # (sku, score)
    categories: list[str] = field(default_factory=list)
    vendors: list[str] = field(default_factory=list)
    departments: list[str] = field(default_factory=list)

    def get_top_skus(self, n: int = 10000) -> list[str]:
        """Get top N SKUs by significance score."""
        # Sort by score descending, return just SKU names
        sorted_skus = sorted(self.skus, key=lambda x: -x[1])
        return [sku for sku, _ in sorted_skus[:n]]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"EntityRanking: {len(self.skus)} SKUs, "
            f"{len(self.categories)} categories, "
            f"{len(self.vendors)} vendors, "
            f"{len(self.departments)} departments"
        )


def compute_streaming_stats(
    filepath: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    collect_skus: bool = False,
    max_skus: int = 10000,
    collect_entities: bool = False,
) -> tuple[
    dict[str, StreamingStats], CategoryStats, list[str] | None, EntityRanking | None
]:
    """
    First pass: Compute streaming statistics for adaptive thresholds.

    v3.2: Optionally collects unique SKUs for batch codebook pre-population.
    v3.3: Optionally collects ranked entities for deterministic seeding.

    Returns:
        (field_stats, category_stats, unique_skus, entity_ranking) tuple
        unique_skus is None if collect_skus=False
        entity_ranking is None if collect_entities=False
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

    # v3.3: Collect ranked entities for deterministic seeding
    if collect_entities:
        sku_scores: dict[str, float] = {}  # sku -> significance score
        categories_seen: set = set()
        vendors_seen: set = set()
        departments_seen: set = set()
    else:
        sku_scores = None
        categories_seen = None
        vendors_seen = None
        departments_seen = None

    rows_scanned = 0
    for chunk in read_file_chunked(filepath, chunk_size):
        for row in chunk:
            rows_scanned += 1
            field_stats["rows"].update(1.0)  # Count every row

            # Extract key fields
            sku = str(_get_field(row, SKU_ALIASES, "")).strip().lower()
            qty = _safe_float(_get_field(row, QUANTITY_ALIASES, None)) or 0
            sold = _safe_float(_get_field(row, SOLD_ALIASES, None)) or 0
            revenue = _safe_float(_get_field(row, REVENUE_ALIASES, 0)) or 0
            cost = _safe_float(_get_field(row, COST_ALIASES, 0)) or 0
            category = str(_get_field(row, CATEGORY_ALIASES, "unknown")).strip().lower()
            vendor = str(_get_field(row, VENDOR_ALIASES, "unknown")).strip().lower()
            department = (
                str(_get_field(row, DEPARTMENT_ALIASES, "unknown")).strip().lower()
            )

            # v3.2: Collect SKU if enabled and under limit
            if collect_skus and len(unique_skus) < max_skus:
                if sku and sku not in ("unknown", "unknown_sku", ""):
                    unique_skus.add(sku)

            # v3.3: Collect ranked entities
            if collect_entities:
                # Score = qty * max(revenue, 1) + sold * 100 (prioritize items with activity)
                if sku and sku not in ("unknown", "unknown_sku", ""):
                    score = qty * max(revenue, 1) + sold * 100
                    sku_scores[sku] = max(sku_scores.get(sku, 0), score)

                if category and category not in ("unknown", "unknown_category", ""):
                    categories_seen.add(category)

                if vendor and vendor not in ("unknown", "unknown_vendor", ""):
                    vendors_seen.add(vendor)

                if department and department not in ("unknown", ""):
                    departments_seen.add(department)

            # Quantity stats
            if qty > 0:
                field_stats["quantity"].update(qty)

            # Margin stats
            if revenue > 0 and cost > 0:
                margin = (revenue - cost) / revenue
                field_stats["margin"].update(margin)
                category_stats.update(category, margin)

            # Sold stats
            if sold >= 0:
                field_stats["sold"].update(sold)

    # Build results
    sku_count = len(unique_skus) if unique_skus else 0
    entity_ranking = None

    if collect_entities:
        # Convert to ranked lists
        ranked_skus = [(sku, score) for sku, score in sku_scores.items()]
        entity_ranking = EntityRanking(
            skus=ranked_skus,
            categories=list(categories_seen),
            vendors=list(vendors_seen),
            departments=list(departments_seen),
        )
        logger.info(f"Entity ranking: {entity_ranking.summary()}")

    logger.info(
        f"Stats computed: {rows_scanned:,} rows, "
        f"avg_qty={field_stats['quantity'].mean:.1f}, "
        f"avg_margin={field_stats['margin'].mean:.2%}, "
        f"avg_sold={field_stats['sold'].mean:.1f}"
        + (f", unique_skus={sku_count:,}" if collect_skus else "")
    )

    sku_list = list(unique_skus) if unique_skus else None
    return field_stats, category_stats, sku_list, entity_ranking


def bundle_pos_facts_streaming(
    ctx: AnalysisContext,
    filepath: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    field_stats: dict[str, StreamingStats] | None = None,
    category_stats: CategoryStats | None = None,
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
        field_stats, category_stats, _, _ = compute_streaming_stats(
            filepath, chunk_size
        )

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


@dataclass
class SeedingSummary:
    """v3.3: Summary of deterministic seeding for audit trail."""

    master_seed: str  # Hash of all seeded entity names
    primitives_seeded: int
    categories_seeded: int
    vendors_seeded: int
    departments_seeded: int
    skus_seeded: int
    total_entities: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "master_seed": self.master_seed,
            "primitives": self.primitives_seeded,
            "categories": self.categories_seeded,
            "vendors": self.vendors_seeded,
            "departments": self.departments_seeded,
            "skus": self.skus_seeded,
            "total": self.total_entities,
        }


def _deterministic_seed_entities(
    ctx: AnalysisContext,
    entity_ranking: EntityRanking | None,
    max_codebook_size: int,
) -> SeedingSummary:
    """
    v3.3: Deterministically seed all significant entities in priority order.

    This ensures reproducibility: same input file = same findings every time.
    Entities are seeded in deterministic order:
    1. Primitives (always first, via ctx.get_primitives())
    2. Categories/departments (alphabetically sorted)
    3. Vendors (alphabetically sorted)
    4. Top SKUs by significance score

    Args:
        ctx: Analysis context
        entity_ranking: Ranked entities from stats pass
        max_codebook_size: Maximum codebook size

    Returns:
        SeedingSummary with audit information
    """
    import hashlib

    all_seeded_names = []

    # 1. Seed primitives first (triggers lazy initialization)
    primitives = ctx.get_primitives()
    primitive_names = sorted(primitives.keys())
    all_seeded_names.extend([f"primitive:{p}" for p in primitive_names])
    primitives_count = len(primitives)

    categories_count = 0
    vendors_count = 0
    departments_count = 0
    skus_count = 0

    if entity_ranking:
        # 2. Seed categories (sorted for determinism)
        sorted_categories = sorted(entity_ranking.categories)
        for cat in sorted_categories:
            if len(ctx.codebook) >= max_codebook_size:
                break
            ctx.add_to_codebook(f"category:{cat}", is_sku=False)
            all_seeded_names.append(f"category:{cat}")
            categories_count += 1

        # 3. Seed departments (sorted for determinism)
        sorted_departments = sorted(entity_ranking.departments)
        for dept in sorted_departments:
            if len(ctx.codebook) >= max_codebook_size:
                break
            ctx.add_to_codebook(f"department:{dept}", is_sku=False)
            all_seeded_names.append(f"department:{dept}")
            departments_count += 1

        # 4. Seed vendors (sorted for determinism)
        sorted_vendors = sorted(entity_ranking.vendors)
        for vendor in sorted_vendors:
            if len(ctx.codebook) >= max_codebook_size:
                break
            ctx.add_to_codebook(f"vendor:{vendor}", is_sku=False)
            all_seeded_names.append(f"vendor:{vendor}")
            vendors_count += 1

        # 5. Seed top SKUs by significance (deterministic order)
        top_skus = entity_ranking.get_top_skus(max_codebook_size - len(ctx.codebook))
        if top_skus:
            ctx.batch_add_to_codebook(top_skus, is_sku=True)
            all_seeded_names.extend(top_skus)
            skus_count = len(top_skus)

    # Compute master seed hash for audit
    combined = "|".join(all_seeded_names)
    master_seed = hashlib.sha256(combined.encode()).hexdigest()[:16]

    total = (
        primitives_count
        + categories_count
        + vendors_count
        + departments_count
        + skus_count
    )

    logger.info(
        f"Deterministic seeding complete - hash seed {master_seed} for {total:,} entities "
        f"({primitives_count} primitives, {categories_count} categories, "
        f"{vendors_count} vendors, {departments_count} depts, {skus_count:,} SKUs)"
    )

    return SeedingSummary(
        master_seed=master_seed,
        primitives_seeded=primitives_count,
        categories_seeded=categories_count,
        vendors_seeded=vendors_count,
        departments_seeded=departments_count,
        skus_seeded=skus_count,
        total_entities=total,
    )


def process_large_file(
    filepath: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    dimensions: int = 8192,
    top_k_per_primitive: int = 20,
    primitives: list[str] | None = None,
    force_sku_only: bool | None = None,
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

    start_time = time.time()

    # Try psutil for memory tracking, fallback to resource module
    try:
        import psutil

        process = psutil.Process()
        use_psutil = True
    except ImportError:
        import resource

        use_psutil = False

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

    # v3.3: Compute file hash for audit trail
    import hashlib

    file_hash = None
    try:
        with open(filepath, "rb") as f:
            # Read in chunks to handle large files
            sha256 = hashlib.sha256()
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
            file_hash = sha256.hexdigest()
        logger.info(f"File hash: {file_hash[:16]}...")
    except Exception as e:
        logger.warning(f"Could not compute file hash: {e}")

    # Pass 1: Statistics with entity ranking for deterministic seeding
    from .context import DEFAULT_MAX_CODEBOOK_SIZE, HIERARCHICAL_CODEBOOK_THRESHOLD

    field_stats, category_stats, _, entity_ranking = compute_streaming_stats(
        filepath, chunk_size, collect_skus=False, collect_entities=True
    )
    total_rows = field_stats["rows"].count  # Use dedicated row counter

    # v3.1: Auto-enable SKU-only codebook for large files
    if force_sku_only is None:
        sku_only = total_rows > LARGE_FILE_THRESHOLD
    else:
        sku_only = force_sku_only

    if sku_only:
        logger.info(
            f"Large file detected ({total_rows:,} rows) - enabling SKU-only codebook"
        )

    # Create context
    ctx = create_analysis_context(dimensions=dimensions, sku_only_codebook=sku_only)

    # v3.3: Deterministic seeding in priority order
    # 1. Primitives are auto-seeded on first access (already deterministic)
    # 2. Pre-seed categories, vendors, departments
    # 3. Pre-seed top SKUs by significance score
    seeding_summary = _deterministic_seed_entities(
        ctx, entity_ranking, DEFAULT_MAX_CODEBOOK_SIZE
    )

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
    top_leaks: dict[str, list[tuple[str, float]]] = {}

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

    # Get peak memory
    if use_psutil:
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
    else:
        # Fallback to resource module (macOS/Linux)
        peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

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
            "dimensions": dimensions,
        },
        elapsed_seconds=elapsed,
        peak_memory_mb=peak_memory,
        # v3.3: Audit data
        file_hash=file_hash,
        seeding_summary=seeding_summary.to_dict(),
    )


def process_dataframe(
    df: pd.DataFrame,
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
        chunk = rows[i : i + chunk_size]
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
