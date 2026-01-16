# Profit Sentinel Validation Report

**Date:** 2026-01-16 13:55
**Data Source:** Inventory_Report_AllSKUs_SHLP_YTD.csv
**Pipeline:** Hybrid (Baseline + VSA Resonator)

---

## Executive Summary

### Dataset Overview

| Metric | Value |
|--------|-------|
| **Total SKUs** | 156,139 |
| **Avg Stock Level** | 9.9 units |
| **Avg Annual Sales** | 17.4 units |
| **Avg Profit Margin** | 49.3% |
| **Unique Vendors** | 38 |

### Detection Summary

| Component | Status | Detections | Time |
|-----------|--------|------------|------|
| **Baseline Detector** | ✅ Complete | **27,892** anomalies | 0.27s |
| **VSA Resonator** | ✅ Active | Infrastructure | 311.5s |

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               PROFIT SENTINEL DETECTION PIPELINE                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Inventory CSV: 156,139 SKUs]                              │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────┐                                        │
│  │  BASELINE DETECTOR  │  ◄── SOURCE OF TRUTH                  │
│  │  Time: 0.27s           │                                        │
│  └──────────┬──────────┘                                        │
│             │                                                   │
│             ▼                                                   │
│  ┌─────────────────────────────────┐                            │
│  │  VSA RESONATOR (Infrastructure) │                            │
│  │  • Sanity check / Hallucination │                            │
│  │  • Does NOT override baseline   │                            │
│  └──────────┬──────────────────────┘                            │
│             │                                                   │
│             ▼                                                   │
│  [27,892 Anomalies Detected]                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Baseline Detector Results

### Anomaly Counts by Primitive

| Primitive | Count | % of SKUs | Status |
|-----------|-------|-----------|--------|
| `negative_inventory` | 3,996 | 2.56% | ✅ |
| `low_stock` | 968 | 0.62% | ✅ |
| `high_margin_leak` | 82 | 0.05% | ✅ |
| `dead_item` | 22,629 | 14.49% | ✅ |
| `overstock` | 189 | 0.12% | ✅ |
| `price_discrepancy` | 0 | 0.00% | ⚪ N/A |
| `shrinkage_pattern` | 0 | 0.00% | ⚪ N/A |
| `margin_erosion` | 28 | 0.02% | ✅ |

### Detection Visualization

```
Anomalies Detected by Primitive
════════════════════════════════════════════════════════════════════

negative_inventory     ███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   3,996
low_stock              █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     968
high_margin_leak       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      82
dead_item              ████████████████████████████████████████  22,629
overstock              ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░     189
price_discrepancy      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       0
shrinkage_pattern      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       0
margin_erosion         ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░      28
```

---

## Sample Detections

### Negative Inventory (3,996 total)

| SKU | Description | Stock | Sold | Margin |
|-----|-------------|-------|------|--------|
| `G8917221` | TIE-DOWN RATCHET ORNG 1IN | -1 | 19 | 44.0% |
| `"LOVE" DECOR` | "LOVE" DECOR KEY HOOK | -1 | 1 | 35.8% |
| `YORK PEP PATTI` | YORK PEPPERMENT PATTIE | -1 | 5 | 30.2% |
| `0001-5448` | DAIWA NM-4.5-956 YAMAMOTO | -1 | 2 | 40.1% |
| `0001-6091` | DAIWA DTU80F602ML D-TURBO | -1 | 0 | 0.0% |

### Low Stock (968 total)

| SKU | Description | Stock | Sold | Margin |
|-----|-------------|-------|------|--------|
| `G8408049` | SHEARS PRUNING BYPASS 8 I | 4 | 22 | 59.1% |
| `G4882932` | ICE MELT ORGANIC PET BAG | 4 | 389 | 36.5% |
| `G4904744` | TAPE DUCT 1.88INX10YD 691 | 4 | 21 | 52.1% |
| `G1978147` | LINT TRAP ALUM W/TWIST PP | 4 | 25 | 70.4% |
| `G4862777` | 32OZ MULTI PURPOSE CLEANE | 4 | 64 | 31.2% |

### High Margin Leak (82 total)

| SKU | Description | Stock | Sold | Margin |
|-----|-------------|-------|------|--------|
| `G6113534` | SAND PLAY PREMIUM 50LB 11 | 58 | 15 | 9.6% |
| `6064804` | JARRITOS PINEAPPLE SODA 1 | 23 | 1 | 5.0% |
| `G7241425` | PAINT SPRY GLS RICH PLUM | 15 | 1 | 3.2% |
| `1/2 BIRCH PLY` | 1/2 BIRCH/ MAPLE VC PLYWO | 12 | 32 | 1.8% |
| `G7241649` | PAINT SPRY STN BURGUNDY 1 | 12 | 1 | -0.1% |

### Dead Item (22,629 total)

| SKU | Description | Stock | Sold | Margin |
|-----|-------------|-------|------|--------|
| `G1956663` | COLORED KEY CAPS BULK/200 | 40124 | 28 | 0.0% |
| `G6134043` | SCANHOOK FASTWIST 10IN R3 | 10400 | 0 | 0.0% |
| `G8261489` | SPLINE 11/64" 250 FT BLAC | 6500 | 0 | 0.0% |
| `G6861215` | PAINT STICK HDWD 14X1/16I | 5500 | 0 | 0.0% |
| `G6328082` | HOLDER LABEL WIRE QUAD 1- | 5000 | 0 | 0.0% |

### Overstock (189 total)

| SKU | Description | Stock | Sold | Margin |
|-----|-------------|-------|------|--------|
| `G1956663` | COLORED KEY CAPS BULK/200 | 40124 | 28 | 0.0% |
| `G8221681` | PELLET WOOD FUEL 40LB FG1 | 2187 | 11 | 43.3% |
| `G6784961` | RIBBON FLY TERRO CD4 T510 | 1101 | 70 | 98.8% |
| `G6710776` | TUBING VINYL CLEAR 1/2IDX | 789 | 6 | 99.7% |
| `G6251185` | M1      KEYBLANK MASTER L | 592 | 38 | 90.7% |

### Margin Erosion (28 total)

| SKU | Description | Stock | Sold | Margin |
|-----|-------------|-------|------|--------|
| `G7241425` | PAINT SPRY GLS RICH PLUM | 15 | 1 | 3.2% |
| `1/2 BIRCH PLY` | 1/2 BIRCH/ MAPLE VC PLYWO | 12 | 32 | 1.8% |
| `G2809176` | PAINT SPRY GLOSS IVORY 12 | 8 | 1 | 3.3% |
| `G2809663` | PAINT SPRY TXTRD DESRT SD | 7 | 2 | 4.6% |
| `G7382484` | PAINT INTR FLAT PASTEL 1G | 7 | 1 | 1.4% |

---

## VSA Resonator Status

### Convergence Summary

| Primitive | Checked | Converged | Flagged | Confidence | Status |
|-----------|---------|-----------|---------|------------|--------|
| `negative_inventory` | 200 | 0 | 200 | 0.0000 | 🔧 INFRASTRUCTURE |
| `low_stock` | 200 | 0 | 200 | 0.0000 | 🔧 INFRASTRUCTURE |
| `high_margin_leak` | 82 | 0 | 82 | 0.0000 | 🔧 INFRASTRUCTURE |
| `dead_item` | 200 | 1 | 199 | 0.0115 | 🔧 INFRASTRUCTURE |
| `overstock` | 189 | 6 | 183 | 0.0139 | 🔧 INFRASTRUCTURE |
| `price_discrepancy` | 0 | 0 | 0 | 0.0000 | ✅ PASS (no candidates) |
| `shrinkage_pattern` | 0 | 0 | 0 | 0.0000 | ✅ PASS (no candidates) |
| `margin_erosion` | 28 | 0 | 28 | 0.0000 | 🔧 INFRASTRUCTURE |

### Resonator Role

The VSA Resonator operates in **infrastructure mode**:
- Validates symbolic consistency of baseline detections
- Flags potential hallucinations for review
- Does **NOT** override baseline outputs

Baseline metrics remain the **source of truth**.

---

## Recommendations

### High Priority Actions

#### 1. Negative Inventory (3,996 SKUs)
- **Severity:** CRITICAL
- **Action:** Immediate physical count
- **Impact:** Data integrity issue

#### 2. Margin Leak (82 SKUs)
- **Severity:** HIGH
- **Action:** Review pricing & vendor costs
- **Impact:** Profitability at risk

#### 3. Dead Inventory (22,629 SKUs)
- **Severity:** MEDIUM
- **Action:** Markdown or liquidate
- **Impact:** Capital tied up

#### 4. Overstock (189 SKUs)
- **Severity:** MEDIUM
- **Action:** Reduce orders, promote
- **Impact:** Cash flow

---

## Next Steps

### Immediate
- [ ] Review 3,996 negative inventory SKUs
- [ ] Investigate 82 margin leak items
- [ ] Export dead inventory list for review

### Short-Term
- [ ] Set up automated anomaly alerts
- [ ] Build vendor cost monitoring
- [ ] Plan promotional strategy for overstock

### Long-Term
- [ ] Integrate into daily operations
- [ ] Build trend tracking dashboard
- [ ] Implement feedback loop

---

## Technical Summary

| Metric | Value |
|--------|-------|
| Dataset Size | 156,139 SKUs |
| Baseline Time | 0.27s |
| Resonator Time | 311.5s |
| Total Time | 311.8s |
| Throughput | 581030 rows/sec |

---

**Generated:** 2026-01-16T13:55:22.671883
**Pipeline Version:** 2.1.0 (Calibrated)

✅ Pipeline completed successfully.
