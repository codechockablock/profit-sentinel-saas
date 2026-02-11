# Test Status

> Current test coverage as of production readiness audit.

## Summary

| Suite | Framework | Count | Status | Command |
|-------|-----------|-------|--------|---------|
| Rust Pipeline | `cargo test` | 132 | ✅ All passing | `cd profit-sentinel-rs && cargo test --workspace` |
| Python World Model | `pytest` | 9 | ✅ All passing | `cd profit-sentinel-rs && python -m pytest python/sentinel_agent/world_model/tests/ -v` |
| Python Standalone | manual | 4 | ✅ All passing | See standalone commands below |
| Frontend | — | 0 | ⚠️ No tests | — |
| **Total** | | **145** | **145 passing** | |

## Rust Tests (132)

Run: `cd profit-sentinel-rs && cargo test --workspace`

### By Crate

| Crate | Tests | Description |
|-------|-------|-------------|
| `sentinel-vsa` | ~40 | VSA encoding, binding, similarity, Complex64 operations |
| `sentinel-pipeline` | ~50 | Issue classification, scoring, thresholds, CSV parsing |
| `sentinel-server` | ~15 | CLI argument parsing, JSON output, E2E pipeline |
| `sentinel-bridge` | ~27 | Response validator (25 banned terms, 8 rules), ops parsing |

### Key Test Areas
- **Response Validator**: Tests all 25 banned terms are rejected, dollar amounts required, confidence validation
- **VSA Operations**: All 16 operation variants parse correctly from JSON, read-only vs mutation classification
- **Pipeline**: Issue type detection (NegativeInventory, MarginLeak, DeadStock, Overstock, etc.)

## Python World Model Tests (9)

Run: `cd profit-sentinel-rs && python -m pytest python/sentinel_agent/world_model/tests/ -v`

| Test | File | Description |
|------|------|-------------|
| `test_config_presets` | `test_dead_stock_config.py` | All 4 industry presets validate cleanly |
| `test_tier_classification` | `test_dead_stock_config.py` | Items classified into correct tiers at threshold boundaries |
| `test_category_overrides` | `test_dead_stock_config.py` | Per-category thresholds override globals |
| `test_capital_threshold` | `test_dead_stock_config.py` | Items below capital threshold don't trigger alerts |
| `test_serialization_roundtrip` | `test_dead_stock_config.py` | Config survives JSON serialization/deserialization |
| `test_validation_catches_bad_config` | `test_dead_stock_config.py` | Invalid threshold ordering rejected |
| `test_lifecycle_tracker` | `test_dead_stock_config.py` | Tier transitions tracked (escalations + recoveries) |
| `test_full_integration` | `test_dead_stock_config.py` | End-to-end config + tracker integration |
| `test_world_model_battery` | `test_world_model.py` | VSA world model battery: 4/4 health checks pass |

## Python Standalone Tests (4)

These run as scripts (not pytest) and verify integration between modules:

| Test | Command | Description |
|------|---------|-------------|
| Transfer Matching | `python -m sentinel_agent.world_model.transfer_matching` | Multi-store transfer recommendation engine |
| Pipeline | `python -m sentinel_agent.world_model.pipeline` | Full sentinel pipeline with all components |
| Dead Stock Config | `python -m sentinel_agent.world_model.config` | Config presets, classification, lifecycle tracking |
| Battery-driven | (via test_world_model.py) | PhasorAlgebra + WorldModelBattery integration |

## Coverage Gaps

### High Priority (needs tests)

| Area | What's Missing | Priority |
|------|---------------|----------|
| **Sidecar API routes** | No integration tests for FastAPI endpoints | P0 |
| **Column mapping** | No tests for MappingService (Anthropic Claude integration) | P0 |
| **S3 service** | No tests for presign, upload, delete, load_dataframe | P1 |
| **Result adapter** | No tests for Rust→frontend result transformation | P1 |
| **Frontend** | Zero test files exist | P2 |

### Medium Priority

| Area | What's Missing | Priority |
|------|---------------|----------|
| **Dual auth** | No tests for anonymous vs authenticated rate limiting | P1 |
| **Email service** | No tests for report PDF generation + email sending | P2 |
| **Anonymizer** | No tests for PII stripping pipeline | P2 |
| **Diagnostic engine** | No tests for diagnostic flow | P2 |

### Low Priority

| Area | What's Missing | Priority |
|------|---------------|----------|
| **Vendor assist** | No tests for vendor call assistant | P3 |
| **Delegation** | No tests for delegation manager | P3 |
| **Category mix** | No tests for category mix optimizer | P3 |

## Previous Test Claims (README)

The README previously claimed:
- "85 Rust tests" → **Actual: 132** (updated)
- "243 Python tests" → **Actual: 9 world model + 4 standalone = 13** (sidecar tests don't exist)
- "27 Frontend tests" → **Actual: 0** (referenced `_legacy/` which is removed)
- "355 total" → **Actual: 145**

These numbers have been corrected in the README.
