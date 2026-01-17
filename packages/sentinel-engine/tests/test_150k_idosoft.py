#!/usr/bin/env python3
"""
Test script for 150k+ row IdoSoft file using streaming module.

This script:
1. Processes the IdoSoft inventory file with 150k+ rows
2. Uses streaming module to handle memory efficiently
3. Reports detection statistics and timing

Usage:
    python test_150k_idosoft.py /path/to/idosoft_inventory.csv

Expected output:
    - Processing time < 60s at 8192-D
    - Peak memory < 2GB
    - Detection counts for all 8 primitives
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sentinel_engine.streaming import process_large_file, DEFAULT_CHUNK_SIZE
from sentinel_engine.context import DEFAULT_DIMENSIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(
        description="Test streaming processing on large IdoSoft file"
    )
    parser.add_argument(
        "filepath",
        help="Path to IdoSoft CSV/TSV file (150k+ rows)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per processing chunk (default: {DEFAULT_CHUNK_SIZE})"
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"VSA dimensionality (default: {DEFAULT_DIMENSIONS})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top leaks to show per primitive (default: 20)"
    )
    args = parser.parse_args()

    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print("=" * 70)
    print("IDOSOFT 150K FILE STREAMING TEST")
    print("=" * 70)
    print(f"File: {filepath}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Chunk size: {args.chunk_size}")
    print()

    # Process file
    result = process_large_file(
        filepath,
        chunk_size=args.chunk_size,
        dimensions=args.dimensions,
        top_k_per_primitive=args.top_k,
    )

    # Report results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(result.summary())
    print()

    # Performance metrics
    print("=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    rows_per_sec = result.processed_rows / result.elapsed_seconds
    print(f"  Processing rate: {rows_per_sec:,.0f} rows/sec")
    print(f"  Time target: {'PASS' if result.elapsed_seconds < 60 else 'FAIL'} (< 60s)")
    print(f"  Memory target: {'PASS' if result.peak_memory_mb < 2048 else 'FAIL'} (< 2GB)")
    print()

    # Top leaks by primitive
    print("=" * 70)
    print("TOP LEAKS BY PRIMITIVE")
    print("=" * 70)
    for prim, leaks in result.top_leaks_by_primitive.items():
        if result.leak_counts.get(prim, 0) > 0:
            print(f"\n{prim.upper()} ({result.leak_counts[prim]:,} detected):")
            for sku, score in leaks[:5]:
                print(f"  {sku}: {score:.4f}")

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
