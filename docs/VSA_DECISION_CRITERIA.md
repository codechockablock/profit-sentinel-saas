# VSA Algorithm Keep/Kill Decision Criteria

## Overview

This document defines the criteria for deciding whether to keep, calibrate, or kill the VSA (Vector Symbolic Architecture) profit leak detection algorithm.

## Decision Framework

### Metrics Definitions

- **Precision**: Of items flagged as anomalies, what % are actual anomalies?
  - Formula: `TP / (TP + FP)`
  - High precision = few false alarms

- **Recall**: Of actual anomalies, what % did we detect?
  - Formula: `TP / (TP + FN)`
  - High recall = few missed issues

- **F1 Score**: Harmonic mean of precision and recall
  - Formula: `2 * (P * R) / (P + R)`
  - Balanced measure of overall accuracy

### Decision Thresholds

| Decision | Precision | Recall | Interpretation |
|----------|-----------|--------|----------------|
| **KEEP** | >= 60% | >= 50% | Algorithm works well |
| **CALIBRATE** | < 60% | >= 50% | Too many false positives, tune thresholds |
| **KILL** | any | < 30% | Algorithm not detecting real issues |

### Per-Primitive Requirements

#### Critical Primitives (Must Pass)
These detect high-impact issues. Failure here = serious concern.

1. **negative_inventory** (Data Integrity)
   - Minimum Recall: 70% (clearest signal - should almost always detect)
   - Acceptable Precision: 50%+

2. **high_margin_leak** (Direct Profit Loss)
   - Minimum Recall: 50%
   - Acceptable Precision: 60%+

#### High Priority Primitives
3. **low_stock** (Lost Sales)
   - Minimum Recall: 50%
   - Acceptable Precision: 50%+

4. **shrinkage_pattern** (Inventory Loss)
   - Minimum Recall: 40%
   - Acceptable Precision: 50%+

#### Medium Priority Primitives
5. **dead_item** (Dead Capital)
   - Minimum Recall: 40%
   - Acceptable Precision: 40%+

6. **margin_erosion** (Profitability Trend)
   - Minimum Recall: 40%
   - Acceptable Precision: 40%+

7. **overstock** (Cash Flow)
   - Minimum Recall: 30%
   - Acceptable Precision: 40%+

8. **price_discrepancy** (Pricing Integrity)
   - Minimum Recall: 30%
   - Acceptable Precision: 40%+

## Overall Algorithm Decision

### KEEP the Algorithm If:
- All critical primitives (negative_inventory, high_margin_leak) pass
- At least 5/8 primitives have recall >= 40%
- No primitive has recall < 20%
- Average F1 across all primitives >= 0.45

### CALIBRATE the Algorithm If:
- Critical primitives pass but non-critical primitives struggle
- Precision is low but recall is acceptable (tune thresholds down)
- Recall is inconsistent across primitives (investigate data quality)

### KILL the Algorithm If:
- Any critical primitive has recall < 30%
- More than 3 primitives have recall < 20%
- Average precision < 30% (too noisy to be useful)
- Algorithm takes >30s for 10K rows (performance concern)

## Validation Process

### Step 1: Generate Synthetic Data
```bash
cd packages/sentinel-engine
pytest tests/test_validation_framework.py -v -k "test_full_validation"
```

### Step 2: Review Report
Check the validation report output for:
- Per-primitive precision/recall
- Missed SKUs (false negatives)
- Spurious SKUs (false positives)

### Step 3: Make Decision
Based on the metrics against the thresholds above.

### Step 4: If CALIBRATE
1. Identify primitives with low precision
2. Review threshold values in `core.py`
3. Adjust and re-validate

## Threshold Calibration Guide

### Current Thresholds (core.py)

| Primitive | Key Thresholds |
|-----------|---------------|
| low_stock | qty < 5 AND sold > avg_sold |
| high_margin_leak | margin < 15% (warning), < 5% (critical), < 0% (negative) |
| dead_item | sold < 3 OR last_sale > 90 days |
| negative_inventory | qty < 0 |
| overstock | days_of_supply > 180 |
| price_discrepancy | price < 80% of suggested retail |
| shrinkage_pattern | shrinkage > 5% of expected |
| margin_erosion | margin < 20% |

### Calibration Strategies

**To reduce false positives (improve precision):**
- Tighten thresholds (e.g., qty < 3 instead of < 5)
- Add secondary conditions (e.g., low_stock requires high velocity)
- Increase minimum score threshold for query results

**To reduce false negatives (improve recall):**
- Loosen thresholds (e.g., margin < 20% instead of < 15%)
- Remove secondary conditions
- Lower minimum score threshold

## Real-World Validation

After synthetic validation passes, validate on real customer data:

1. Export 1000 rows from a real POS system
2. Have domain expert manually tag 50 known issues
3. Run algorithm and compare
4. Acceptable if recall > 40% on real data

## Decision Log

| Date | Version | Decision | Rationale |
|------|---------|----------|-----------|
| TBD | v2.1.0 | TBD | Awaiting validation run |

---

## Running Validation

```bash
# Full validation with report
cd /Users/joseph/profit-sentinel-saas
PYTHONPATH=packages/sentinel-engine/src pytest packages/sentinel-engine/tests/test_validation_framework.py::TestVSAIntegration::test_full_validation_run -v -s

# Quick unit tests only (no engine required)
pytest packages/sentinel-engine/tests/test_validation_framework.py -v -k "not Integration"
```
