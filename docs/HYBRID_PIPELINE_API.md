# Hybrid Pipeline API Integration Guide

**Version:** 2.1.0
**Architecture:** Baseline Detector (Source of Truth) + VSA Resonator (Infrastructure Mode)

---

## Overview

The Profit Sentinel Hybrid Pipeline combines:

1. **Baseline Detector** - Fast, rule-based anomaly detection (SOURCE OF TRUTH)
2. **VSA Resonator** - Symbolic validation infrastructure (validates, does not override)

```
[POS Data] → [Baseline Detector] → [Anomaly Candidates] → [VSA Resonator] → [Report]
                 (34ms/10K)             (5,656 items)         (20s/10K)
```

---

## Quick Start

### Python API

```python
from sentinel_engine.context import create_analysis_context
from sentinel_engine.core import bundle_pos_facts, query_bundle, get_all_primitives
from sentinel_engine.contradiction_detector import detect_contradictions, resolve_contradictions

# 1. Create analysis context
ctx = create_analysis_context(
    dimensions=16384,
    use_gpu=True,  # Falls back to CPU if unavailable
)

# 2. Load your POS data
rows = [
    {"sku": "SKU001", "quantity": 5, "cost": 10.0, "revenue": 15.0, "sold": 50},
    {"sku": "SKU002", "quantity": -3, "cost": 20.0, "revenue": 25.0, "sold": 30},
    # ... more rows
]

# 3. Run baseline detection (SOURCE OF TRUTH)
bundle = bundle_pos_facts(ctx, rows)

# 4. Query for each primitive
detections = {}
for primitive in get_all_primitives():
    results, scores = query_bundle(ctx, bundle, primitive, top_k=50)
    detections[primitive] = set(results)

# 5. Detect contradictions
contradictions, summary = detect_contradictions(detections)

# 6. (Optional) Resolve contradictions by priority
resolved = resolve_contradictions(detections)
```

### REST API

```bash
# Upload and analyze POS data
curl -X POST https://api.profitsentinel.ai/v1/analyze \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@inventory.csv" \
  -F "config={\"resonator_mode\": \"infrastructure\"}"
```

**Response:**
```json
{
  "analysis_id": "abc123",
  "status": "completed",
  "baseline": {
    "time_ms": 34,
    "detections": {
      "low_stock": ["SKU001", "SKU002"],
      "negative_inventory": ["SKU003"],
      "high_margin_leak": ["SKU004", "SKU005"]
    }
  },
  "resonator": {
    "mode": "infrastructure",
    "validation": {
      "low_stock": {"status": "PASS", "convergence_rate": 1.0},
      "negative_inventory": {"status": "PASS", "convergence_rate": 1.0}
    },
    "contradictions": [
      {"sku": "SKU003", "primitives": ["negative_inventory", "low_stock"]}
    ]
  }
}
```

---

## Configuration

### Baseline Detector Thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `low_stock_qty` | 10 | Flag if qty below this AND high velocity |
| `low_stock_critical` | 3 | Critical alert threshold |
| `dead_item_days` | 60 | Days since last sale |
| `dead_item_sold_threshold` | 2 | Items sold in period |
| `margin_leak_threshold` | 0.25 | 25% margin minimum |
| `margin_critical_threshold` | 0.10 | 10% triggers critical |
| `overstock_qty_threshold` | 100 | Min qty to consider |
| `overstock_qty_to_sold_ratio` | 200 | Months of inventory |
| `shrinkage_threshold` | -1 | Any negative variance |
| `price_discrepancy_threshold` | 0.30 | 30% variance from MSRP |

### VSA Resonator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `convergence_threshold` | 0.005 | Convergence check |
| `max_iterations` | 100 (CPU) / 300 (GPU) | Max resonator iterations |
| `top_k` | 16 | Sparse selection |
| `dimensions` | 16384 | Hypervector dimensions |

---

## Detection Primitives

### 1. `negative_inventory` (Critical)
- **Condition:** `quantity < 0`
- **F1 Score:** 100%
- **Impact:** Data integrity issue, potential shrinkage

### 2. `overstock` (Medium)
- **Condition:** `qty > 100 AND qty/sold > 200`
- **F1 Score:** 99.9%
- **Impact:** Tied capital, carrying costs

### 3. `shrinkage_pattern` (High)
- **Condition:** `qty_difference < -1 OR shrinkage_rate > 5%`
- **F1 Score:** 98.1%
- **Impact:** Inventory loss, theft prevention

### 4. `dead_item` (Medium)
- **Condition:** `sold < 2 in 60 days`
- **F1 Score:** 84.6%
- **Impact:** Dead capital

### 5. `high_margin_leak` (Critical)
- **Condition:** `margin < 10% OR margin < 50% of category avg`
- **F1 Score:** 71.5%
- **Impact:** Profitability erosion

### 6. `margin_erosion` (High)
- **Condition:** `margin < 20% of dataset avg`
- **F1 Score:** 71.2%
- **Impact:** Profitability trend

### 7. `low_stock` (High)
- **Condition:** `qty < 10 AND sold > avg_sold`
- **F1 Score:** 70.9%
- **Impact:** Lost sales

### 8. `price_discrepancy` (Warning)
- **Condition:** `|price - msrp| / msrp > 30%`
- **F1 Score:** 63.1%
- **Impact:** Pricing integrity

---

## Resonator Modes

### Infrastructure Mode (Default)
```python
resonator_mode = "infrastructure"
```
- Validates baseline detections symbolically
- Flags contradictions for review
- **Does NOT override baseline results**
- A "FAIL" means needs review, not false anomaly

### Active Filter Mode (Experimental)
```python
resonator_mode = "active_filter"
```
- Resonator can suppress low-confidence detections
- Use with caution - may reduce recall

### Disabled
```python
resonator_mode = "disabled"
```
- Baseline only, no symbolic validation

---

## Contradiction Handling

### Detected Contradictory Pairs

| Pair | Resolution |
|------|------------|
| `low_stock` + `overstock` | Keep `low_stock` (lost sales worse) |
| `dead_item` + `high_velocity` | Keep `high_velocity` (recent activity) |
| `negative_inventory` + `overstock` | Keep `negative_inventory` (data issue) |
| `negative_inventory` + `low_stock` | Keep `negative_inventory` (more severe) |

### Manual Review
Contradictions default to manual review:
```python
contradictions, summary = detect_contradictions(detections)
for c in contradictions:
    print(f"REVIEW: {c.sku} - {c.primitive_a} vs {c.primitive_b}")
```

### Auto-Resolution
Enable if needed:
```python
resolved = resolve_contradictions(detections, auto_resolve=True)
```

---

## Performance Benchmarks

| Dataset Size | Baseline Time | Resonator Time | Total |
|--------------|---------------|----------------|-------|
| 1,000 rows | 3ms | 2s | ~2s |
| 10,000 rows | 34ms | 20s | ~20s |
| 100,000 rows | 340ms | 200s* | ~3min |
| 1,000,000 rows | 3.4s | 2000s* | ~33min |

*GPU reduces resonator time by 5-10x

### AWS Recommendations

| Dataset Size | Instance | GPU | Cost/hr |
|--------------|----------|-----|---------|
| < 50K | g4dn.xlarge | T4 | $0.526 |
| 50K - 200K | g4dn.2xlarge | T4 | $0.752 |
| 200K - 1M | g4dn.4xlarge | T4 | $1.204 |
| > 1M | g4dn.8xlarge | T4 | $2.176 |

---

## Error Handling

### Resonator Unavailable
```python
if not resonator.available:
    # Fall back to baseline-only mode
    results = baseline.detect(rows)
```

### GPU Fallback
```python
ctx = create_analysis_context(use_gpu=True)
# Automatically falls back to CPU if CUDA unavailable
print(f"Using device: {ctx.device}")  # 'cuda' or 'cpu'
```

### Large Dataset Batching
```python
# For datasets > 100K rows, use batch processing
from sentinel_engine.batch import BatchProcessor

processor = BatchProcessor(batch_size=10000)
results = processor.process(rows, ctx)
```

---

## Monitoring & Metrics

### Key Metrics to Track

```python
metrics = {
    "baseline_time_ms": 34,
    "resonator_time_ms": 20000,
    "total_candidates": 5656,
    "convergence_rate": 1.0,
    "contradiction_count": 410,
    "avg_f1": 0.824,
}
```

### CloudWatch Integration
```python
import boto3

cloudwatch = boto3.client('cloudwatch')
cloudwatch.put_metric_data(
    Namespace='ProfitSentinel',
    MetricData=[
        {'MetricName': 'BaselineF1', 'Value': 0.824, 'Unit': 'None'},
        {'MetricName': 'ContradictionCount', 'Value': 410, 'Unit': 'Count'},
    ]
)
```

---

## Changelog

### v2.1.0 (2026-01-16)
- Added hybrid baseline + VSA architecture
- Implemented batch processing for resonator
- Added contradiction detection and resolution
- Validated on 10K synthetic dataset (82.4% avg F1)
- GPU acceleration support with CPU fallback

---

*Generated by Profit Sentinel Hybrid Validation Pipeline v2.1.0*
