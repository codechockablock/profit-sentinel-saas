"""CLI entry point for Profit Sentinel.

Usage:
    # Start the sidecar API server
    python -m sentinel_agent serve
    SIDECAR_DEV_MODE=true python -m sentinel_agent serve

    # Ingest data from a source
    python -m sentinel_agent ingest --source /path/to/data
    python -m sentinel_agent ingest --source /path/to/data --store-id my-store
    python -m sentinel_agent ingest --source /path/to/data --adapter orgill

    # Ingest AND run analysis pipeline
    python -m sentinel_agent ingest --source /path/to/data --analyze
    python -m sentinel_agent ingest --source /path/to/data --analyze --top 10

    # List available adapters
    python -m sentinel_agent adapters
"""

from __future__ import annotations

import argparse
import sys
import time


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the sidecar API server."""
    import uvicorn

    from .sidecar import create_app
    from .sidecar_config import get_settings

    settings = get_settings()
    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.sidecar_host,
        port=settings.sidecar_port,
        log_level="info",
    )


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Run data ingestion from a source."""
    from pathlib import Path

    from .adapters.detection import detect_adapter, detect_and_ingest

    source = Path(args.source)
    store_id = args.store_id

    if not source.exists():
        print(f"Error: Path not found: {source}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect or use specified adapter
    if args.adapter:
        from .adapters.orgill import OrgillPOAdapter

        adapter_map = {
            "orgill": OrgillPOAdapter(),
        }
        adapter = adapter_map.get(args.adapter.lower())
        if not adapter:
            print(f"Error: Unknown adapter '{args.adapter}'", file=sys.stderr)
            print(f"Available: {', '.join(adapter_map.keys())}", file=sys.stderr)
            sys.exit(1)
        result = adapter.ingest(source, store_id=store_id)
    else:
        result = detect_and_ingest(source, store_id=store_id)

    # Print results
    print(result.summary)
    print()

    if result.errors:
        print(f"--- Errors ({len(result.errors)}) ---")
        for err in result.errors[:20]:
            print(f"  ! {err}")
        if len(result.errors) > 20:
            print(f"  ... and {len(result.errors) - 20} more")
        print()

    if result.warnings:
        print(f"--- Warnings ({len(result.warnings)}) ---")
        for warn in result.warnings[:10]:
            print(f"  * {warn}")
        print()

    # Short-ship detail report for PO data
    if result.purchase_orders:
        short_shipped = []
        for po in result.purchase_orders:
            for item in po.product_line_items:
                if item.is_short_ship and item.short_ship_value > 0:
                    short_shipped.append((po.po_number, item))

        if short_shipped:
            short_shipped.sort(key=lambda x: x[1].short_ship_value, reverse=True)
            print(f"--- Top Short-Ships ({len(short_shipped)} items) ---")
            for po_num, item in short_shipped[:25]:
                print(
                    f"  PO {po_num}: {item.description[:35]:<35} "
                    f"Ordered: {item.qty_ordered:>3}  Filled: {item.qty_filled:>3}  "
                    f"Lost: ${item.short_ship_value:>8,.2f}"
                )
            if len(short_shipped) > 25:
                total_lost = sum(item.short_ship_value for _, item in short_shipped)
                print(f"  ... {len(short_shipped) - 25} more items")
                print(f"  Total short-ship value: ${total_lost:,.2f}")
            print()

    # Output file option
    if args.output:
        _write_output(result, args.output)

    # Run Rust pipeline analysis if requested
    if getattr(args, "analyze", False) and result.inventory_records:
        _run_analysis(result, store_id, args.top)


def _write_output(result, output_path: str) -> None:
    """Write normalized records to CSV for pipeline consumption."""
    import csv
    from pathlib import Path

    path = Path(output_path)

    if result.inventory_records:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "sku_id",
                    "description",
                    "vendor",
                    "vendor_sku",
                    "qty_on_hand",
                    "unit_cost",
                    "retail_price",
                    "last_sale_date",
                    "last_receipt_date",
                    "bin_location",
                    "store_id",
                    "category",
                    "department",
                    "barcode",
                    "on_order_qty",
                    "sales_ytd",
                ]
            )
            for rec in result.inventory_records:
                writer.writerow(
                    [
                        rec.sku_id,
                        rec.description or "",
                        rec.vendor or "",
                        rec.vendor_sku or "",
                        rec.qty_on_hand,
                        rec.unit_cost,
                        rec.retail_price,
                        rec.last_sale_date or "",
                        rec.last_receipt_date or "",
                        rec.bin_location or "",
                        rec.store_id,
                        rec.category or "",
                        rec.department or "",
                        rec.barcode or "",
                        rec.on_order_qty,
                        rec.sales_ytd,
                    ]
                )
        print(f"Wrote {len(result.inventory_records):,} records to {path}")

    elif result.purchase_orders:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "po_number",
                    "order_date",
                    "status",
                    "line_number",
                    "sku_id",
                    "description",
                    "qty_ordered",
                    "qty_filled",
                    "unit_cost",
                    "ext_cost",
                    "retail_price",
                    "department",
                    "is_short_ship",
                    "short_ship_value",
                ]
            )
            for po in result.purchase_orders:
                for item in po.line_items:
                    writer.writerow(
                        [
                            po.po_number,
                            po.order_date or "",
                            po.status.value,
                            item.line_number,
                            item.sku_id,
                            item.description,
                            item.qty_ordered,
                            item.qty_filled,
                            item.unit_cost,
                            item.ext_cost,
                            item.retail_price,
                            item.department or "",
                            item.is_short_ship,
                            item.short_ship_value,
                        ]
                    )
        total_lines = sum(po.total_line_items for po in result.purchase_orders)
        print(f"Wrote {total_lines:,} PO line items to {path}")


def _run_analysis(result, store_id: str, top_k: int) -> None:
    """Run Rust pipeline analysis on ingested inventory records."""
    from .engine import AnalysisResult, PipelineError, SentinelEngine
    from .llm_layer import render_digest

    print("=" * 64)
    print("  RUNNING RUST PIPELINE ANALYSIS")
    print("=" * 64)
    print()

    start = time.time()

    try:
        engine = SentinelEngine()

        # Scale timeout based on record count (156K rows takes ~4 min)
        record_count = len(result.inventory_records)
        timeout = max(60, int(record_count / 500))  # ~1s per 500 records

        analysis = engine.run_from_adapter(
            result.inventory_records,
            stores=[store_id],
            top_k=top_k,
            timeout_seconds=timeout,
        )
    except PipelineError as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        if e.stderr:
            print(f"  stderr: {e.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start

    # Print bridge summary
    print(analysis.summary)
    print(f"Total wall time: {elapsed:.1f}s")
    print()

    # Print rendered morning digest
    digest = analysis.digest
    if digest.issues:
        print("=" * 64)
        print("  MORNING DIGEST — DETECTED ISSUES")
        print("=" * 64)
        print()
        print(render_digest(digest))
        print()

        # Enriched issue detail (bidirectional bridge)
        print("=" * 64)
        print("  ENRICHED ISSUE DETAIL (adapter + pipeline)")
        print("=" * 64)
        print()
        for issue in digest.issues:
            print(f"  [{issue.issue_type.display_name}] {issue.dollar_display}")
            print(
                f"    Store: {issue.store_id}  |  "
                f"Confidence: {issue.confidence:.0%}  |  "
                f"Priority: {issue.priority_score:.1f}"
            )
            for sku in issue.skus[:5]:
                original = analysis.enrich_sku(sku.sku_id, issue.store_id)
                if original:
                    desc = (original.description or "—")[:40]
                    vendor = original.vendor or "—"
                    bin_loc = original.bin_location or "—"
                    print(f"      SKU {sku.sku_id}: {desc}")
                    print(
                        f"        Vendor: {vendor}  |  Bin: {bin_loc}  |  "
                        f"Qty: {sku.qty_on_hand:.0f}  |  "
                        f"Cost: ${sku.unit_cost:.2f}  |  "
                        f"Margin: {sku.margin_display}"
                    )
                else:
                    print(
                        f"      SKU {sku.sku_id}: qty={sku.qty_on_hand:.0f} "
                        f"cost=${sku.unit_cost:.2f} margin={sku.margin_display}"
                    )
            if len(issue.skus) > 5:
                print(f"      ... and {len(issue.skus) - 5} more SKUs")
            print()
    else:
        print("No actionable issues detected. All clear!")


def _cmd_adapters(args: argparse.Namespace) -> None:
    """List available adapters."""
    from .adapters.detection import list_adapters

    adapters = list_adapters()
    print("Available data adapters:")
    for a in adapters:
        print(f"  {a['name']:<25} ({a['class']})")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel_agent",
        description="Profit Sentinel — inventory intelligence for hardware retail",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # serve
    subparsers.add_parser("serve", help="Start the sidecar API server")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data from a source")
    ingest_parser.add_argument(
        "--source",
        required=True,
        help="Path to file or directory to ingest",
    )
    ingest_parser.add_argument(
        "--store-id",
        default="default-store",
        help="Store ID for normalized records (default: default-store)",
    )
    ingest_parser.add_argument(
        "--adapter",
        help="Force a specific adapter (orgill)",
    )
    ingest_parser.add_argument(
        "--output",
        "-o",
        help="Write normalized CSV output to this path",
    )
    ingest_parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run Rust pipeline analysis after ingestion",
    )
    ingest_parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top issues to return (default: 10, used with --analyze)",
    )

    # adapters
    subparsers.add_parser("adapters", help="List available adapters")

    args = parser.parse_args()

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "ingest":
        _cmd_ingest(args)
    elif args.command == "adapters":
        _cmd_adapters(args)
    else:
        # Default: start serve (backward compatible)
        if len(sys.argv) == 1:
            args.command = "serve"
            _cmd_serve(args)
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
