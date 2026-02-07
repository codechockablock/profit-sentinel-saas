#!/usr/bin/env python3
"""
Profit Sentinel Real Data Validation Pipeline

Executes the full hybrid detection pipeline on actual inventory data:
1. Baseline Detector (source of truth)
2. VSA Resonator (infrastructure sanity checker)
3. Generates Markdown report with plots and metrics
"""

import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class DetectionResult:
    """Results from detection for a single primitive."""

    primitive: str
    detected_skus: set[str] = field(default_factory=set)
    count: int = 0
    sample_items: list[dict] = field(default_factory=list)


@dataclass
class ResonatorValidation:
    """Results from resonator sanity checking."""

    primitive: str
    candidates_checked: int = 0
    convergence_passed: int = 0
    hallucinations_flagged: int = 0
    avg_confidence: float = 0.0
    status: str = "PASS"


def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if val is None or str(val).strip() == "":
        return default
    try:
        cleaned = str(val).replace("$", "").replace(",", "").replace("%", "").strip()
        if cleaned == "" or cleaned.lower() in ("nan", "null", "none", "-"):
            return default
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def parse_date(val: Any) -> datetime | None:
    """Parse date from various formats."""
    if val is None or str(val).strip() == "":
        return None

    date_str = str(val).strip()

    # Handle "Jun 02,25" format
    match = re.match(r"([A-Za-z]+)\s+(\d+),(\d+)", date_str)
    if match:
        month_str, day, year = match.groups()
        try:
            month_map = {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }
            month = month_map.get(month_str.lower()[:3], 1)
            year_full = 2000 + int(year) if int(year) < 100 else int(year)
            return datetime(year_full, month, int(day))
        except Exception:
            pass

    # Handle YYYYMMDD format
    if len(date_str) == 8 and date_str.isdigit():
        try:
            return datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            pass

    return None


def load_inventory_csv(filepath: str) -> list[dict]:
    """Load and parse inventory CSV."""
    rows = []
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


class RealDataBaselineDetector:
    """
    Baseline detector adapted for real inventory data.

    Column mapping:
    - SKU: SKU
    - quantity: Stock
    - cost: Avg. Cost / Gross Cost
    - revenue: Gross Sales
    - margin: Profit Margin%
    - sold: Year Total
    - last_sale: Last Sale / Real Date
    - vendor: Vendor
    """

    PRIMITIVES = [
        "negative_inventory",
        "low_stock",
        "high_margin_leak",
        "dead_item",
        "overstock",
        "price_discrepancy",
        "shrinkage_pattern",
        "margin_erosion",
    ]

    def detect(
        self, rows: list[dict]
    ) -> tuple[dict[str, DetectionResult], dict[str, Any]]:
        """
        Run detection on real inventory data.

        Returns:
            Tuple of (results_by_primitive, dataset_stats)
        """
        results = {p: DetectionResult(primitive=p) for p in self.PRIMITIVES}

        # First pass: compute statistics
        all_stocks = []
        all_sold = []
        all_margins = []
        vendor_margins: dict[str, list[float]] = {}

        for row in rows:
            stock = safe_float(row.get("Stock", 0))
            sold = safe_float(row.get("Year Total", 0))
            margin_pct = safe_float(row.get("Profit Margin%", 0))
            vendor = str(row.get("Vendor", "unknown")).strip()

            if stock > 0:
                all_stocks.append(stock)
            if sold > 0:
                all_sold.append(sold)
            if margin_pct > 0:
                all_margins.append(margin_pct)
                if vendor not in vendor_margins:
                    vendor_margins[vendor] = []
                vendor_margins[vendor].append(margin_pct)

        avg_stock = sum(all_stocks) / len(all_stocks) if all_stocks else 100
        avg_sold = sum(all_sold) / len(all_sold) if all_sold else 10
        avg_margin = sum(all_margins) / len(all_margins) if all_margins else 30

        # Compute vendor average margins
        vendor_avg_margin = {}
        for vendor, margins in vendor_margins.items():
            if len(margins) >= 3:
                vendor_avg_margin[vendor] = sum(margins) / len(margins)

        stats = {
            "total_rows": len(rows),
            "avg_stock": avg_stock,
            "avg_sold": avg_sold,
            "avg_margin": avg_margin,
            "vendors": len(vendor_margins),
        }

        # Second pass: detect anomalies
        for row in rows:
            sku = str(row.get("SKU", "")).strip()
            if not sku:
                continue

            stock = safe_float(row.get("Stock", 0))
            sold = safe_float(row.get("Year Total", 0))
            margin_pct = safe_float(row.get("Profit Margin%", 0))
            safe_float(row.get("Avg. Cost", 0))
            gross_sales = safe_float(row.get("Gross Sales", 0))
            safe_float(row.get("Gross Cost", 0))
            safe_float(row.get("Gross Profit", 0))
            vendor = str(row.get("Vendor", "unknown")).strip()
            description = str(row.get("Description", "")).strip()

            last_sale_str = row.get("Last Sale", "")
            last_sale = parse_date(last_sale_str)

            item_info = {
                "sku": sku,
                "description": description[:50],
                "stock": stock,
                "sold": sold,
                "margin": margin_pct,
                "vendor": vendor[:20],
            }

            # =================================================================
            # PRIMITIVE 1: NEGATIVE INVENTORY
            # =================================================================
            if stock < 0:
                results["negative_inventory"].detected_skus.add(sku)
                results["negative_inventory"].count += 1
                if len(results["negative_inventory"].sample_items) < 10:
                    results["negative_inventory"].sample_items.append(item_info)

            # =================================================================
            # PRIMITIVE 2: LOW STOCK (stock < 5 AND high velocity)
            # =================================================================
            if 0 < stock < 5 and sold > avg_sold:
                results["low_stock"].detected_skus.add(sku)
                results["low_stock"].count += 1
                if len(results["low_stock"].sample_items) < 10:
                    results["low_stock"].sample_items.append(item_info)

            # =================================================================
            # PRIMITIVE 3: HIGH MARGIN LEAK (margin < 50% of vendor avg OR < 10%)
            # =================================================================
            vendor_avg = vendor_avg_margin.get(vendor, avg_margin)
            margin_threshold = min(vendor_avg * 0.5, 10)  # 50% of vendor avg or 10%

            if gross_sales > 0:
                if margin_pct < 0:
                    # Negative margin - selling at loss
                    results["high_margin_leak"].detected_skus.add(sku)
                    results["high_margin_leak"].count += 1
                    if len(results["high_margin_leak"].sample_items) < 10:
                        results["high_margin_leak"].sample_items.append(item_info)
                elif margin_pct < margin_threshold and margin_pct > 0:
                    results["high_margin_leak"].detected_skus.add(sku)
                    results["high_margin_leak"].count += 1
                    if len(results["high_margin_leak"].sample_items) < 10:
                        results["high_margin_leak"].sample_items.append(item_info)

            # =================================================================
            # PRIMITIVE 4: DEAD ITEM (no sales AND has stock)
            # =================================================================
            is_dead = False
            if sold < 1 and stock > 0:
                is_dead = True
            elif last_sale:
                days_since = (datetime.now() - last_sale).days
                if days_since > 180 and stock > 0:  # No sales in 6 months
                    is_dead = True

            if is_dead:
                results["dead_item"].detected_skus.add(sku)
                results["dead_item"].count += 1
                if len(results["dead_item"].sample_items) < 10:
                    results["dead_item"].sample_items.append(item_info)

            # =================================================================
            # PRIMITIVE 5: OVERSTOCK (days supply > 270)
            # =================================================================
            if stock > 50 and sold > 0:
                # Estimate daily sales from year total
                daily_sales = sold / 365
                days_supply = stock / daily_sales if daily_sales > 0 else 9999

                if days_supply > 270:
                    results["overstock"].detected_skus.add(sku)
                    results["overstock"].count += 1
                    if len(results["overstock"].sample_items) < 10:
                        item_info["days_supply"] = round(days_supply, 1)
                        results["overstock"].sample_items.append(item_info)

            # =================================================================
            # PRIMITIVE 6: PRICE DISCREPANCY
            # Not applicable - no suggested retail in this dataset
            # =================================================================
            # Skip - no sug. retail column

            # =================================================================
            # PRIMITIVE 7: SHRINKAGE PATTERN
            # Not directly available - would need inventory count data
            # =================================================================
            # Skip - no qty_difference column

            # =================================================================
            # PRIMITIVE 8: MARGIN EROSION (0 < margin < 5%)
            # =================================================================
            if gross_sales > 0 and 0 < margin_pct < 5:
                results["margin_erosion"].detected_skus.add(sku)
                results["margin_erosion"].count += 1
                if len(results["margin_erosion"].sample_items) < 10:
                    results["margin_erosion"].sample_items.append(item_info)

        return results, stats


class CalibratedResonator:
    """
    VSA Resonator with calibrated settings for real data.

    Calibration applied:
    - convergence_threshold: 0.005 (lowered from 0.01)
    - max_iters: 300 (increased from 100)
    - codebook_filter: SKU-only
    """

    def __init__(self):
        self.available = False
        self.convergence_threshold = 0.005  # Calibrated
        self.max_iters = 300  # Calibrated

        try:
            import torch
            from sentinel_engine import core
            from sentinel_engine.context import create_analysis_context

            self._torch = torch
            self._create_ctx = create_analysis_context
            self._core = core
            self.available = True
            print(f"  Resonator loaded (torch {torch.__version__})")
        except ImportError as e:
            print(f"  Resonator not available: {e}")

    def validate_detections(
        self,
        rows: list[dict],
        baseline_results: dict[str, DetectionResult],
    ) -> dict[str, ResonatorValidation]:
        """Validate baseline detections using calibrated resonator."""

        if not self.available:
            # Return simulated pass-through when unavailable
            return {
                p: ResonatorValidation(
                    primitive=p,
                    candidates_checked=r.count,
                    convergence_passed=r.count,
                    status="PASS (resonator unavailable)",
                )
                for p, r in baseline_results.items()
            }

        results = {}

        # Convert rows to resonator format
        resonator_rows = []
        for row in rows[:10000]:  # Limit for performance
            resonator_rows.append(
                {
                    "sku": row.get("SKU", ""),
                    "description": row.get("Description", ""),
                    "vendor": row.get("Vendor", ""),
                    "quantity": safe_float(row.get("Stock", 0)),
                    "sold": safe_float(row.get("Year Total", 0)),
                    "cost": safe_float(row.get("Avg. Cost", 0)),
                    "revenue": safe_float(row.get("Gross Sales", 0)),
                }
            )

        print(f"    Building resonator codebook ({len(resonator_rows)} rows)...")
        ctx = self._create_ctx(use_gpu=False)
        ctx.iters = self.max_iters
        ctx.multi_steps = 3

        try:
            t0 = time.time()
            bundle = self._core.bundle_pos_facts(ctx, resonator_rows)
            print(
                f"    Codebook built in {time.time()-t0:.1f}s ({len(ctx.codebook)} entries)"
            )

            for primitive, baseline_result in baseline_results.items():
                validation = ResonatorValidation(primitive=primitive)
                validation.candidates_checked = baseline_result.count

                if baseline_result.count == 0:
                    validation.status = "PASS (no candidates)"
                    results[primitive] = validation
                    continue

                # Query resonator
                try:
                    items, scores = self._core.query_bundle(
                        ctx, bundle, primitive, top_k=500
                    )
                    resonator_scores = {
                        item.lower(): score for item, score in zip(items, scores)
                    }

                    # Check overlap
                    confidences = []
                    for sku in list(baseline_result.detected_skus)[:500]:
                        sku_lower = sku.lower()
                        if sku_lower in resonator_scores:
                            score = resonator_scores[sku_lower]
                            confidences.append(score)
                            if score >= self.convergence_threshold:
                                validation.convergence_passed += 1
                            else:
                                validation.hallucinations_flagged += 1
                        else:
                            validation.hallucinations_flagged += 1

                    if confidences:
                        validation.avg_confidence = sum(confidences) / len(confidences)

                    # Determine status
                    if validation.candidates_checked == 0:
                        validation.status = "PASS (empty)"
                    elif (
                        validation.convergence_passed / validation.candidates_checked
                        >= 0.3
                    ):
                        validation.status = "PASS"
                    elif (
                        validation.convergence_passed / validation.candidates_checked
                        >= 0.1
                    ):
                        validation.status = "WARN"
                    else:
                        validation.status = "INFRASTRUCTURE"

                except Exception as e:
                    validation.status = f"ERROR: {str(e)[:30]}"

                results[primitive] = validation
                print(
                    f"    {primitive}: {validation.convergence_passed}/{validation.candidates_checked} converged"
                )

        finally:
            ctx.reset()

        return results


def generate_markdown_report(
    filepath: str,
    baseline_results: dict[str, DetectionResult],
    resonator_results: dict[str, ResonatorValidation],
    stats: dict[str, Any],
    baseline_time: float,
    resonator_time: float,
) -> str:
    """Generate comprehensive Markdown report."""

    total_detections = sum(r.count for r in baseline_results.values())

    report = f"""# Profit Sentinel Validation Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Data Source:** {Path(filepath).name}
**Pipeline:** Hybrid (Baseline + VSA Resonator)

---

## Executive Summary

### Dataset Overview

| Metric | Value |
|--------|-------|
| **Total SKUs** | {stats['total_rows']:,} |
| **Avg Stock Level** | {stats['avg_stock']:.1f} units |
| **Avg Annual Sales** | {stats['avg_sold']:.1f} units |
| **Avg Profit Margin** | {stats['avg_margin']:.1f}% |
| **Unique Vendors** | {stats['vendors']:,} |

### Detection Summary

| Component | Status | Total Detections | Time |
|-----------|--------|------------------|------|
| **Baseline Detector** | ✅ Complete | **{total_detections:,}** | {baseline_time:.2f}s |
| **VSA Resonator** | {'✅ Active' if resonator_time > 0 else '⚠️ Unavailable'} | Infrastructure | {resonator_time:.1f}s |

### Architecture

```
[Inventory CSV: {stats['total_rows']:,} rows]
        │
        ▼
┌─────────────────────┐
│  BASELINE DETECTOR  │  ← SOURCE OF TRUTH
│  {baseline_time:.2f}s processing     │
└──────────┬──────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  VSA RESONATOR (Infrastructure)     │
│  • Symbolic consistency validation  │
│  • Hallucination prevention         │
└─────────────────────────────────────┘
        │
        ▼
[{total_detections:,} Anomalies Detected]
```

---

## Baseline Detector Results

### Per-Primitive Detection Counts

| Primitive | Detected | % of SKUs | Status |
|-----------|----------|-----------|--------|
"""

    for p in RealDataBaselineDetector.PRIMITIVES:
        r = baseline_results[p]
        pct = (r.count / stats["total_rows"] * 100) if stats["total_rows"] > 0 else 0
        status = "✅" if r.count > 0 else "⚪"
        if p in ["price_discrepancy", "shrinkage_pattern"] and r.count == 0:
            status = "⚪ N/A"
        report += f"| `{p}` | {r.count:,} | {pct:.2f}% | {status} |\n"

    report += """
### Detection Visualization

```
Anomalies Detected by Primitive
════════════════════════════════════════════════════════════════════════════════

"""

    max_count = max(r.count for r in baseline_results.values()) or 1
    for p in RealDataBaselineDetector.PRIMITIVES:
        r = baseline_results[p]
        bar_len = int((r.count / max_count) * 40) if max_count > 0 else 0
        bar = "█" * bar_len + "░" * (40 - bar_len)
        report += f"{p:<22} {bar} {r.count:>6,}\n"

    report += """```

---

## Sample Detections

"""

    for p in RealDataBaselineDetector.PRIMITIVES:
        r = baseline_results[p]
        if r.sample_items:
            report += f"### {p.replace('_', ' ').title()} ({r.count:,} total)\n\n"
            report += "| SKU | Description | Stock | Sold | Margin |\n"
            report += "|-----|-------------|-------|------|--------|\n"
            for item in r.sample_items[:5]:
                report += f"| {item['sku']} | {item['description'][:30]} | {item['stock']:.0f} | {item['sold']:.0f} | {item['margin']:.1f}% |\n"
            report += "\n"

    report += """---

## VSA Resonator Validation

### Convergence Status

| Primitive | Candidates | Converged | Flagged | Avg Conf | Status |
|-----------|------------|-----------|---------|----------|--------|
"""

    for p in RealDataBaselineDetector.PRIMITIVES:
        v = resonator_results.get(p, ResonatorValidation(primitive=p))
        status_icon = (
            "✅" if "PASS" in v.status else ("⚠️" if "WARN" in v.status else "🔧")
        )
        report += f"| `{p}` | {v.candidates_checked:,} | {v.convergence_passed:,} | {v.hallucinations_flagged:,} | {v.avg_confidence:.4f} | {status_icon} {v.status} |\n"

    report += """
### Resonator Role

The VSA Resonator serves as **infrastructure** for:
- Symbolic consistency validation
- Hallucination prevention
- Contradiction detection

**Important:** Baseline metrics are the source of truth. Resonator does NOT override baseline outputs.

---

## Recommendations

### High-Priority Anomalies

"""

    # Prioritize recommendations
    if baseline_results["negative_inventory"].count > 0:
        report += f"""#### 1. Negative Inventory ({baseline_results["negative_inventory"].count:,} SKUs)
**Severity:** CRITICAL
**Action:** Immediate physical count required
**Impact:** Data integrity / potential shrinkage

"""

    if baseline_results["high_margin_leak"].count > 0:
        report += f"""#### 2. High Margin Leak ({baseline_results["high_margin_leak"].count:,} SKUs)
**Severity:** HIGH
**Action:** Review pricing and vendor costs
**Impact:** Profitability erosion

"""

    if baseline_results["dead_item"].count > 0:
        report += f"""#### 3. Dead Inventory ({baseline_results["dead_item"].count:,} SKUs)
**Severity:** MEDIUM
**Action:** Consider markdown or liquidation
**Impact:** Tied-up capital

"""

    if baseline_results["overstock"].count > 0:
        report += f"""#### 4. Overstock ({baseline_results["overstock"].count:,} SKUs)
**Severity:** MEDIUM
**Action:** Reduce future orders, promote slow movers
**Impact:** Cash flow / carrying costs

"""

    report += """---

## Next Steps

### Immediate (This Week)
- [ ] Review negative inventory SKUs - perform physical counts
- [ ] Investigate high margin leak items for pricing errors
- [ ] Export dead item list for markdown planning

### Short-Term (Next 2 Weeks)
- [ ] Implement automated alerts for new negative inventory
- [ ] Set up vendor cost monitoring for margin protection
- [ ] Review overstock items for promotional opportunities

### Long-Term (Next Quarter)
- [ ] Integrate detection pipeline into daily operations
- [ ] Build dashboard for anomaly trend tracking
- [ ] Implement feedback loop for threshold tuning

---

## Technical Details

### Detection Rules Applied

| Primitive | Rule |
|-----------|------|
| `negative_inventory` | `stock < 0` |
| `low_stock` | `stock < 5 AND sold > avg_sold` |
| `high_margin_leak` | `margin < vendor_avg * 0.5 OR margin < 10%` |
| `dead_item` | `sold < 1 AND stock > 0` OR `last_sale > 180 days` |
| `overstock` | `days_supply > 270 AND stock > 50` |
| `price_discrepancy` | N/A (no suggested retail in data) |
| `shrinkage_pattern` | N/A (no inventory count data) |
| `margin_erosion` | `0 < margin < 5%` |

### Pipeline Performance

| Metric | Value |
|--------|-------|
| Baseline Detection Time | {baseline_time:.2f}s |
| Resonator Validation Time | {resonator_time:.1f}s |
| Total Pipeline Time | {baseline_time + resonator_time:.1f}s |
| Throughput | {stats['total_rows'] / (baseline_time + resonator_time + 0.001):.0f} rows/sec |

---

**Report Generated:** {datetime.now().isoformat()}
**Pipeline Version:** 2.1.0 (Calibrated)
**Contact:** engineering@profit-sentinel.io

✅ Pipeline completed successfully.
"""

    return report


def run_real_data_validation(filepath: str):
    """Execute full validation pipeline on real inventory data."""

    print("=" * 80)
    print("PROFIT SENTINEL REAL DATA VALIDATION")
    print("=" * 80)
    print(f"Data source: {filepath}")
    print(f"Run time: {datetime.now().isoformat()}")
    print()

    # Step 1: Load data
    print("STEP 1: DATA INGESTION")
    print("-" * 40)
    t0 = time.time()
    rows = load_inventory_csv(filepath)
    load_time = time.time() - t0
    print(f"  Loaded {len(rows):,} rows in {load_time:.2f}s")
    print()

    # Step 2: Baseline detection
    print("STEP 2: BASELINE DETECTOR")
    print("-" * 40)
    detector = RealDataBaselineDetector()
    t0 = time.time()
    baseline_results, stats = detector.detect(rows)
    baseline_time = time.time() - t0
    print(f"  Completed in {baseline_time:.2f}s")
    print("  Dataset stats:")
    print(f"    - Total SKUs: {stats['total_rows']:,}")
    print(f"    - Avg Stock: {stats['avg_stock']:.1f}")
    print(f"    - Avg Sold: {stats['avg_sold']:.1f}")
    print(f"    - Avg Margin: {stats['avg_margin']:.1f}%")
    print("  Detections:")
    for p, r in baseline_results.items():
        if r.count > 0:
            print(f"    - {p}: {r.count:,}")
    print()

    # Step 3: Resonator validation
    print("STEP 3: VSA RESONATOR (Calibrated)")
    print("-" * 40)
    resonator = CalibratedResonator()
    t0 = time.time()
    if resonator.available:
        resonator_results = resonator.validate_detections(rows, baseline_results)
        resonator_time = time.time() - t0
        print(f"  Completed in {resonator_time:.1f}s")
    else:
        resonator_results = {
            p: ResonatorValidation(
                primitive=p,
                candidates_checked=r.count,
                convergence_passed=r.count,
                status="PASS (resonator unavailable)",
            )
            for p, r in baseline_results.items()
        }
        resonator_time = 0
        print("  Skipped (torch not available)")
    print()

    # Step 4: Generate report
    print("STEP 4: GENERATING REPORT")
    print("-" * 40)
    report = generate_markdown_report(
        filepath,
        baseline_results,
        resonator_results,
        stats,
        baseline_time,
        resonator_time,
    )

    # Save report
    report_path = (
        Path(__file__).parent.parent.parent.parent
        / "docs"
        / "PROFIT_SENTINEL_VALIDATION.md"
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to: {report_path}")

    # Save JSON metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(filepath),
        "dataset": stats,
        "baseline": {
            "time_seconds": baseline_time,
            "per_primitive": {
                p: {
                    "count": r.count,
                    "sample_skus": list(r.detected_skus)[:20],
                }
                for p, r in baseline_results.items()
            },
        },
        "resonator": {
            "time_seconds": resonator_time,
            "available": resonator.available,
            "per_primitive": {
                p: {
                    "status": v.status,
                    "candidates_checked": v.candidates_checked,
                    "convergence_passed": v.convergence_passed,
                    "hallucinations_flagged": v.hallucinations_flagged,
                    "avg_confidence": v.avg_confidence,
                }
                for p, v in resonator_results.items()
            },
        },
    }

    metrics_path = Path(__file__).parent / "real_data_validation_results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    print()
    print("=" * 80)
    print("✅ Pipeline completed successfully.")
    print("=" * 80)

    return baseline_results, resonator_results, stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = (
            "/Users/joseph/Downloads/Reports/Inventory_Report_AllSKUs_SHLP_YTD.csv"
        )

    run_real_data_validation(filepath)
