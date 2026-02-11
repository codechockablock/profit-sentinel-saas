# Profit Sentinel — Architecture Analysis

> **Date:** 2025-02-05
> **Scope:** Full inventory of original system (`apps/`, `packages/`), new system (`profit-sentinel-rs/`), gap analysis, integration opportunities, and recommended next steps.
> **Constraint:** All future work must be Rust or Python only.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Original System Inventory](#2-original-system-inventory)
3. [New System Inventory](#3-new-system-inventory)
4. [Gap Analysis](#4-gap-analysis)
5. [Integration Opportunities](#5-integration-opportunities)
6. [Real Data Requirements](#6-real-data-requirements)
7. [Recommended Next Steps](#7-recommended-next-steps)

---

## 1. Executive Summary

The original Profit Sentinel is a production-deployed MLP (Machine Learning Pipeline) built entirely in Python across three packages (`sentinel-engine`, `vsa-core`, `reasoning`) and a FastAPI application (`apps/api`). It totals approximately **17,500 LOC** with sophisticated VSA-based anomaly detection, conversational diagnostics, evidence-grounded cause scoring, and a gamified repair engine.

The new system (`profit-sentinel-rs/`) is a Rust+Python hybrid totaling approximately **15,400 LOC** (4,941 Rust + 10,412 Python). The Rust layer handles the hot path — VSA bundling and issue classification — achieving **45x speedup** over the Python equivalent (~2s vs ~90s for 156K rows). The Python layer provides domain intelligence: morning digest generation, co-op patronage analysis, vendor call preparation, task delegation, and a FastAPI sidecar for mobile access.

### The Critical Gap

The **single most important missing piece** is the data bridge: Python adapters produce `NormalizedInventory` records (18 fields from real POS data), but the Rust pipeline expects `InventoryRecord` structs (11 fields in a specific CSV format). There is currently no converter between them. Closing this gap is the prerequisite for running real analysis on store data.

### What's Complete, What's Not

| Capability | Original | New (Rust+Python) | Status |
|---|---|---|---|
| VSA bundling (hot path) | Python (~90s) | Rust (~2s) | ✅ Done, 45x faster |
| Issue classification | 11 primitives | 7 types w/ dollar impact | ✅ Done, typed |
| Pipeline architecture | Ad-hoc service | Async trait-based 9-stage | ✅ Done, extensible |
| POS data adapters | AI mapping service | Sample store + Orgill parsers | ✅ Done for sample store |
| NormalizedInventory→Rust bridge | N/A | **Missing** | ❌ Critical gap |
| Morning digest | Email-based | Text rendering + mobile API | ✅ Done |
| Co-op intelligence | Not present | Full suite (patronage, GMROI, rebates) | ✅ New capability |
| Vendor call assistant | Not present | Call prep with talking points | ✅ New capability |
| Task delegation | Not present | Manager with auto-deadlines | ✅ New capability |
| Mobile sidecar API | Not present | FastAPI + static UI | ✅ Done |
| Conversational diagnostics | 26 patterns, interactive | Not ported | ❌ Gap |
| Evidence-based cause scoring | Vectorized scorer | Not ported | ❌ Gap |
| VSA→Symbolic bridge | Full proof trees | Not ported | ❌ Gap |
| Schema evolution | Auto-detect changes | Not ported | ⚠️ Low priority |
| Repair engine (gamified) | Badges, streaks | Not ported | ⚠️ Low priority |
| Column mapping AI | 3-tier (AI/heuristic/sample) | Per-adapter hardcoded | ⚠️ Different approach |

---

## 2. Original System Inventory

### 2.1 Package Structure

```
apps/api/src/                    # FastAPI application (38 files)
├── config.py                    # Thresholds & settings (249 LOC)
├── dependencies.py              # Supabase JWT auth
├── main.py                      # App entry point
├── middleware/auth.py            # Auth middleware
├── routes/
│   ├── analysis.py              # POST /analyze — main entry (777 LOC)
│   ├── diagnostic.py            # Conversational diagnostic routes
│   ├── repair.py                # Gamified repair routes
│   ├── reports.py               # Report generation
│   ├── uploads.py               # S3 file handling
│   ├── billing.py               # Stripe integration
│   ├── employee.py              # Employee management
│   ├── premium.py               # Premium features
│   ├── metrics.py               # Metrics endpoint
│   └── health.py                # Health check
├── services/
│   ├── analysis.py              # Core analysis service (1,631 LOC) ★
│   ├── mapping.py               # Column mapping AI (394 LOC) ★
│   ├── anonymization.py         # PII scrubbing
│   ├── billing.py               # Billing service
│   ├── email.py                 # Email delivery
│   ├── file_validator.py        # Upload validation
│   ├── grok_vision.py           # Grok-3 vision for PDFs
│   ├── rate_limiting.py         # Rate limiter
│   └── s3.py                    # S3 operations
└── utils/column_mappings.py     # Alias dictionaries

packages/sentinel-engine/        # VSA engine (40+ files)
├── core.py                      # VSA resonator + bundling (2,600+ LOC) ★
├── context.py                   # Request-scoped state (3,000+ LOC) ★
├── batch.py                     # Batch/stream processing (368 LOC)
├── bridge.py                    # VSA→Symbolic bridge (430 LOC) ★
├── codebook.py                  # VSA codebook management
├── contradiction_detector.py    # Inconsistency detection
├── flagging.py                  # Issue flagging logic
├── pipeline.py                  # Processing pipeline
├── repair_engine.py             # Repair orchestrator
├── repair_models.py             # Gamified models (473 LOC)
├── streaming.py                 # Stream processing
├── diagnostic/
│   ├── engine.py                # Conversational diagnostics (907 LOC) ★
│   ├── agent.py                 # Diagnostic agent
│   ├── email.py                 # Diagnostic email reports
│   ├── integrated.py            # Integrated diagnostic
│   ├── multi_file.py            # Multi-file analysis
│   └── report.py                # Diagnostic reports
├── domains/                     # Multi-domain reasoning (8 files)
│   ├── biology.py, chemistry.py, cs.py, economics.py
│   ├── math.py, philosophy.py, physics.py, web.py
├── dorian/                      # Knowledge persistence (5 files)
│   ├── agent.py, core.py, ontology.py, persistence.py, pipeline.py
├── routing/smart_router.py      # Hot/cold path routing
├── vsa_evidence/                # Evidence scoring (6 files)
│   ├── scorer.py                # Cause scorer (661 LOC) ★
│   ├── encoder.py               # Evidence encoder
│   ├── knowledge.py             # Domain knowledge
│   ├── rules.py                 # Evidence rules
│   └── causes.py                # Cause definitions
└── loaders/                     # External data loaders
    ├── arxiv.py, conceptnet.py, wikidata.py

packages/reasoning/              # Symbolic inference (6 files)
├── inference.py                 # Forward/backward chaining (432 LOC) ★
├── knowledge_base.py            # Rule + fact storage
├── dsl.py                       # Rule definition DSL
├── terms.py                     # Term types
└── unification.py               # Unification algorithm

packages/vsa-core/               # VSA primitives (8 files)
├── resonator.py                 # Convergence-lock resonator (374 LOC)
├── schema.py                    # Schema evolution (608 LOC) ★
├── vectors.py                   # Complex phasor vectors
├── operators.py                 # VSA algebra
├── types.py                     # Type definitions
├── probabilistic.py             # Probabilistic extensions
└── loader.py                    # Data loading utilities
```

### 2.2 Core Capabilities (★ = High Value for Port)

#### A. Analysis Service (1,631 LOC) ★★★

The crown jewel. Implements 11 detection primitives with production-tuned thresholds:

| Primitive | Threshold | Impact Cap |
|---|---|---|
| `high_margin_leak` | margin < 0.25 | $50K |
| `negative_inventory` | qty < 0 | $10K |
| `negative_profit` | (retail - cost) < 0 | $25K |
| `severe_inventory_deficit` | qty < -100 | $100K |
| `low_stock` | qty < 10 | — |
| `shrinkage_pattern` | complex multi-signal | $75K |
| `margin_erosion` | margin < avg × 0.7 | $50K |
| `zero_cost_anomaly` | cost = 0 & retail > 0 | — |
| `dead_item` | no sales > 90 days | — |
| `overstock` | days supply > 180 | — |
| `price_discrepancy` | |price - MSRP| > 15% | — |

Key architectural decision: **hybrid execution** — heuristic fast path runs all 11 primitives instantly; VSA slow path runs in parallel for cause diagnosis. Results merge with heuristics as the floor.

Production safeguards:
- Impact sanity caps prevent $10M hallucinations
- Paladin margin explicitly flagged as unreliable (hardcoded note at lines 1029, 1045)
- Auto-delete uploaded files after 60s (privacy)

#### B. Column Mapping Service (394 LOC) ★★

Three-tier column mapping for unknown POS formats:
1. **AI tier** — Grok-3 with temperature=0.1 analyzes column names + sample values
2. **Heuristic tier** — Normalized substring matching against 400+ aliases
3. **Sample value tier** — Detects numeric ranges, date patterns, UPC formats

This is the "magic" that lets the original system accept any POS CSV without manual configuration.

#### C. Conversational Diagnostics (907 LOC) ★★

26 hardcoded patterns for hardware retail negative inventory:

| Pattern | Example |
|---|---|
| RECEIVING_GAP | Items received but not scanned in |
| NON_TRACKED | Consumables/samples not tracked |
| VENDOR_MANAGED | Vendor fills without PO |
| EXPIRATION | Expired items removed but not written off |
| THEFT | Shrinkage from theft |
| INVESTIGATE | Unknown cause requiring physical audit |

Interactive flow: Present patterns → User confirms → System learns → Reduces negative inventory exposure ($726K → $178K demonstrated).

#### D. Evidence-Based Cause Scoring (661 LOC) ★★

0% quantitative hallucination via positive-similarity summing:
- Encodes known causes as VSA vectors
- Scores each candidate cause by dot-product against evidence
- Only positive similarities contribute (prevents negative cancellation)
- Hot/cold path routing: high-confidence → fast path; ambiguous → deep analysis

Validated metrics: +159% actionability improvement, 100% multi-hop accuracy.

#### E. VSA→Symbolic Bridge (430 LOC) ★

Translates VSA resonator anomalies into symbolic facts, runs forward/backward chaining, produces proof trees. The `BridgeResult` provides: `vsa_anomalies`, `symbolic_conclusions`, `root_causes`, `recommended_actions`, `proof_tree`, `combined_confidence`.

#### F. Schema Evolution (608 LOC)

Handles POS system upgrades that add/rename/remove columns between exports. Useful but lower priority — the adapter-per-vendor approach in the new system sidesteps this.

#### G. Gamified Repair Engine (473+ LOC)

Badges, streaks, levels for technicians resolving issues. Nice-to-have, not core analytics.

### 2.3 Configuration & Thresholds

From `config.py`:
```
margin_leak_threshold:        0.25
margin_below_average_factor:  0.70
dead_stock_days:              90
low_stock_threshold:          10
overstock_days_supply:        180
price_discrepancy_threshold:  0.15
```

These thresholds are battle-tested from production. They should be carried forward into the new system.

---

## 3. New System Inventory

### 3.1 Rust Layer (4,941 LOC)

```
profit-sentinel-rs/
├── sentinel-vsa/                # VSA hot path (Rust)
│   ├── primitives.rs            # 11 phasor vectors (D=1024)
│   ├── codebook.rs              # Lock-free SKU→vector map
│   ├── bundling.rs              # Parallel bundle encoding ★
│   ├── similarity.rs            # Cosine similarity
│   ├── py_bindings.rs           # PyO3 exports (unused currently)
│   ├── correctness.rs           # 13 property tests
│   └── benchmark.rs             # Criterion benchmarks
│
├── sentinel-pipeline/           # Async pipeline framework (Rust)
│   ├── candidate_pipeline.rs    # Trait-based pipeline ★
│   ├── issue_classifier.rs      # 7 issue types ★
│   ├── inventory_loader.rs      # CSV → InventoryRecord
│   ├── types.rs                 # Shared types
│   └── pipelines/
│       └── executive_digest.rs  # Concrete wiring (9 stages)
│   └── components/              # Pipeline stages
│       ├── top_k_selector.rs
│       ├── low_impact_filter.rs ($100 threshold)
│       ├── dollar_impact_scorer.rs
│       ├── store_diversity_scorer.rs
│       └── ...
│
├── sentinel-server/             # CLI binary (Rust)
│   └── main.rs                  # Load CSV, run pipeline, output JSON
│
└── fixtures/
    └── sample_inventory.csv     # 21 synthetic rows, 11 columns
```

#### A. VSA Bundling (Rust) ★★★

The `bundling.rs` hot path:

1. **Warmup phase**: Parallel codebook population via Rayon
2. **Bundle phase**: Lock-free parallel encoding of inventory rows

`InventoryRow` struct (10 fields):
```
sku, qty_on_hand, unit_cost, margin_pct, sales_last_30d,
days_since_receipt, retail_price, is_damaged, on_order_qty, is_seasonal
```

11 signal thresholds encoded as phasor primitives:
- `negative_qty`, `high_cost`, `low_margin`, `zero_sales`, `high_qty`
- `recent_receipt`, `old_receipt`, `negative_retail`, `damaged`, `on_order`, `seasonal`

Performance: 156K rows bundled in ~2s (vs ~90s in Python).

#### B. Issue Classifier (477 LOC) ★★★

7 typed issue categories with dollar-impact formulas:

| Issue Type | Condition | Dollar Impact Formula |
|---|---|---|
| `ReceivingGap` | qty < 0 | `|qty| × unit_cost` |
| `DeadStock` | no sales > 90 days, qty > 0 | `qty × unit_cost` |
| `MarginErosion` | margin < 15% & retail > 0 | `(expected_margin - actual_margin) × retail × qty` |
| `NegativeInventory` | qty < -100 | `|qty| × unit_cost` |
| `VendorShortShip` | on_order > 0 & qty < 0 | `on_order × unit_cost` |
| `PurchasingLeakage` | margin < 5% & cost > 10 | `potential_savings × qty` |
| `PatronageMiss` | (reserved for co-op) | — |

Each issue carries: `confidence` (0.0–1.0), `dollar_impact`, `store_id`, `sku`, `issue_type`.

8 unit tests, all passing.

#### C. Pipeline Architecture (Async Traits) ★★★

```
Source → Hydrator → Filter → Scorer → Selector → SideEffect
```

Concrete `ExecutiveDigestPipeline` wiring:
```
TimeRangeQueryHydrator
  → GlmAnalyticsSource
  → StoreContextHydrator
  → LowImpactFilter ($100 threshold)
  → DollarImpactScorer
  → StoreDiversityScorer
  → TopKSelector
  → DigestCacheSideEffect
```

All stages implement the `PipelineComponent` async trait — fully extensible.

#### D. Inventory Loader (165 LOC)

Reads CSV with 11 columns into `InventoryRecord`:
```
store_id, sku, qty_on_hand, unit_cost, margin_pct, sales_last_30d,
days_since_receipt, retail_price, is_damaged, on_order_qty, is_seasonal
```

Flexible bool parsing: accepts "true"/"false", "1"/"0", "yes"/"no".

**This is the format that the Python adapters must produce.**

#### E. CLI Binary (487 LOC)

`sentinel-server` accepts CSV path, runs `ExecutiveDigestPipeline`, outputs:
- `--json`: Structured `DigestJson` with per-issue `SkuJson` details
- Default: Human-readable context with issue summaries

### 3.2 Python Layer (10,412 LOC)

```
python/sentinel_agent/
├── engine.py                    # Subprocess bridge to Rust binary (147 LOC)
├── models.py                    # Pydantic models for Digest (226 LOC)
├── digest.py                    # MorningDigestGenerator ★
├── llm_layer.py                 # Template rendering (all NL text)
├── delegation.py                # Task delegation manager
├── vendor_assist.py             # Vendor call preparation
├── inventory_health.py          # Health scoring
├── vendor_rebates.py            # Rebate tracking
├── category_mix.py              # Category optimization
├── coop_intelligence.py         # Co-op patronage analysis ★
├── coop_models.py               # Co-op data models
├── sidecar.py                   # FastAPI mobile API (10 endpoints)
├── sidecar_config.py            # Pydantic-settings config
├── api_models.py                # API request/response models
├── __main__.py                  # CLI entry point
│
├── adapters/
│   ├── base.py                  # BaseAdapter ABC + canonical models ★
│   ├── detection.py             # Auto-detection logic
│   ├── sample_store/
│   │   ├── inventory.py         # Sample store/IdoSoft adapter ★
│   │   └── __init__.py
│   ├── orgill/
│   │   ├── po_parser.py         # Orgill PO parser ★
│   │   └── __init__.py
│   ├── do_it_best/
│   │   └── __init__.py          # Stub
│   └── ace/
│       └── __init__.py          # Stub
│
├── static/                      # Mobile UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
└── tests/                       # 14 test modules, 4.5K+ LOC
    ├── test_models.py
    ├── test_engine.py
    ├── test_adapters.py
    ├── test_digest.py
    ├── test_delegation.py
    ├── test_llm_layer.py
    ├── test_vendor_assist.py
    ├── test_coop_intelligence.py
    ├── test_coop_models.py
    ├── test_inventory_health.py
    ├── test_vendor_rebates.py
    ├── test_category_mix.py
    ├── test_api_models.py
    └── test_sidecar.py
```

#### A. Adapter Layer ★★★

`BaseAdapter` ABC with `can_handle()` + `ingest()` → `AdapterResult`:

**`NormalizedInventory`** (18 fields):
```
sku_id, description, vendor, vendor_sku, qty_on_hand, unit_cost,
retail_price, last_receipt_date, last_sale_date, bin_location,
store_id, category, department, barcode, on_order_qty,
min_qty, max_qty, sales_ytd, cost_ytd
```

**Sample store adapter** — Handles two formats:
1. `custom_1.csv` (53 columns) → Full inventory with 50+ fields
2. `Inventory_Report_AllSKUs_SHLP_YTD.csv` (36 columns) → Monthly sales rollup

**Orgill PO parser** — Handles 148 PO files with 14-row header blocks and 28+ columns per line item. Produces `PurchaseOrder` with `POLineItem` records including short-ship detection.

**Auto-detection** (`detection.py`) — Determines which adapter handles a given file based on column fingerprints.

#### B. Morning Digest Generator ★★

Orchestrates the full morning briefing:
1. Calls `SentinelEngine.run()` → gets `Digest` from Rust pipeline
2. Runs `CoopIntelligence` analysis → patronage leakage, consolidation
3. Runs `InventoryHealthScorer` → store health grade
4. Runs `VendorRebateTracker` → rebate tier progress
5. Runs `CategoryMixOptimizer` → category rebalancing
6. Renders all via `llm_layer` templates

#### C. Co-op Intelligence Suite ★★ (NEW — not in original)

- **Patronage tracking**: Calculates earned patronage by category, identifies leakage to non-co-op vendors
- **GMROI optimization**: Gross Margin Return on Investment analysis
- **Vendor rebates**: Tier tracking (Bronze/Silver/Gold/Platinum), progress to next tier
- **Category mix**: Optimal allocation across departments

Co-op prefixes: DIB, WHS, HRD, FLR, TLS, PLB, ELC, PNT, SSN (warehouse) vs DMG, EXT, LOC, OEM (outside buys).

#### D. Vendor Call Assistant ★★ (NEW — not in original)

Prepares call briefs for VendorShortShip and PurchasingLeakage issues:
- Talking points with dollar amounts
- Historical vendor performance (stub catalog, production would use DB)
- Questions to ask the vendor
- Rendered via templates for mobile consumption

#### E. Task Delegation Manager ★ (NEW — not in original)

Creates task packages from pipeline issues:
- Auto-calculates deadline based on issue urgency
- Formats for store manager audience
- In-memory task store (MVP)

#### F. Sidecar API ★

10 endpoints:
- `GET /api/v1/digest` — Morning briefing
- `GET /api/v1/digest/{store_id}` — Per-store view
- `POST /api/v1/delegate` — Create task from issue
- `GET /api/v1/tasks` — List delegated tasks
- `GET /api/v1/tasks/{task_id}` — Task detail
- `GET /api/v1/vendor-call/{issue_id}` — Vendor call prep
- `GET /api/v1/coop/{store_id}` — Co-op report
- `GET /health` — Health check
- Static files at `/` (mobile UI)

Auth: Supabase JWT with dev-mode bypass.

### 3.3 Test Coverage

| Suite | Tests | Status |
|---|---|---|
| Rust (`cargo test --workspace`) | 58 | ✅ All passing |
| Python (`python -m pytest`) | 322 | ✅ All passing |
| **Total** | **380** | ✅ All passing |

---

## 4. Gap Analysis

### 4.1 Critical Gaps (Block Real Data Ingestion)

#### Gap 1: NormalizedInventory → InventoryRecord Bridge ❌❌❌

**The single most important gap in the entire system.**

The Python adapters output `NormalizedInventory` (18 fields from real POS data). The Rust pipeline expects `InventoryRecord` (11 fields in CSV format). There is **no code** that converts between them.

Field mapping required:

| NormalizedInventory (Python) | InventoryRecord (Rust) | Conversion |
|---|---|---|
| `store_id` | `store_id` | Direct |
| `sku_id` | `sku` | Rename |
| `qty_on_hand` | `qty_on_hand` | int → f64 |
| `unit_cost` | `unit_cost` | Direct |
| (computed) `margin_pct` | `margin_pct` | `(retail - cost) / retail` |
| (not present) | `sales_last_30d` | Must derive from `sales_ytd` or default 0.0 |
| `last_receipt_date` | `days_since_receipt` | `(today - date).days` or default 365.0 |
| `retail_price` | `retail_price` | Direct |
| (not present) | `is_damaged` | Default false |
| `on_order_qty` | `on_order_qty` | int → f64 |
| (not present) | `is_seasonal` | Default false |

Missing data problems:
- **`sales_last_30d`**: Not in any sample store export. `sales_ytd` exists but is annual. Could approximate as `sales_ytd / months_elapsed` or require a separate sales report.
- **`is_damaged`**: Not in POS data. Default to `false`.
- **`is_seasonal`**: Not in POS data. Could derive from category/department (e.g., "Christmas", "Seasonal").
- **`days_since_receipt`**: Calculable from `last_receipt_date` when present.

#### Gap 2: No Real Fixtures ❌❌

The only fixture is `sample_inventory.csv` with 21 synthetic rows. There are no test fixtures generated from real store data. The entire pipeline has never been tested against real-world data.

### 4.2 Significant Gaps (Missing Production Capabilities)

#### Gap 3: No Conversational Diagnostics ❌

The original system's 26-pattern diagnostic engine reduces negative inventory exposure by 75% ($726K → $178K) through interactive user confirmation. This is a significant value driver that has no equivalent in the new system.

**Port priority: HIGH** — But requires a different interface in the new architecture (mobile UI flow rather than the original web chat).

#### Gap 4: No Evidence-Based Cause Scoring ❌

The original system's evidence scorer provides 0% quantitative hallucination via positive-similarity summing. The new system classifies issues but doesn't explain **why** they occurred.

Currently the Rust `issue_classifier.rs` determines *what* the issue is (e.g., MarginErosion) but not the root cause (e.g., vendor price increase, incorrect retail setup, competitive pressure).

**Port priority: HIGH** — The evidence scoring algorithm (vectorized matrix multiply) is a natural fit for the Rust VSA layer.

#### Gap 5: No VSA→Symbolic Bridge ❌

The bridge translates anomalies into proof trees. Without it, the system can't explain its reasoning chain. The new system has the Rust VSA layer (bundling + similarity) but no symbolic reasoning to produce proof chains.

**Port priority: MEDIUM** — Useful for trust-building with store owners but not required for basic functionality.

#### Gap 6: No Column Mapping AI ❌

The original system's 3-tier mapping (AI → heuristic → sample value) lets it accept any unknown POS format. The new system uses hardcoded adapters per vendor.

**Port priority: LOW for now** — The adapter-per-vendor approach works fine for the sample store + known vendors. The AI mapping becomes important only when onboarding many unknown POS systems rapidly.

### 4.3 Minor Gaps (Nice to Have)

| Gap | Original System | Priority |
|---|---|---|
| Schema evolution | Auto-detect column changes between exports | LOW |
| Repair engine | Gamified badges/streaks for technicians | LOW |
| Multi-domain reasoning | Biology, chemistry, etc. domain codebooks | NOT NEEDED |
| Dorian knowledge persistence | Ontology + persistence layer | LOW |
| ConceptNet/Wikidata loaders | External knowledge graphs | NOT NEEDED |

### 4.4 Reverse Gaps (New System Has, Original Doesn't)

| Capability | New System Only |
|---|---|
| Co-op patronage analysis | Patronage leakage, consolidation opportunities |
| Vendor call preparation | Talking points, questions, history |
| Task delegation | Auto-deadline assignment |
| Mobile sidecar API | 10-endpoint FastAPI + static UI |
| Typed issue model | 7 categories with dollar impact formulas |
| Rust hot path | 45x faster VSA bundling |
| PO short-ship analysis | Automated short-ship detection + dollar impact |
| Category mix optimization | GMROI-based rebalancing |

---

## 5. Integration Opportunities

### 5.1 Direct Port to Rust (Performance-Critical)

#### Evidence Cause Scoring → `sentinel-vsa/src/evidence.rs`

The Python evidence scorer (`scorer.py`, 661 LOC) uses vectorized NumPy operations:
```
scores = max(0, similarity_matrix @ evidence_vector)
```

This is ideal for Rust — the VSA codebook is already in Rust, and the scoring can use the same phasor primitives. Expected: ~200-300 LOC Rust, 10-50x speedup over NumPy for large codebooks.

#### Additional Issue Types → `sentinel-pipeline/src/issue_classifier.rs`

Port the 4 missing detection primitives from the original's 11:
- `shrinkage_pattern` — Multi-signal shrinkage detection
- `zero_cost_anomaly` — Items with $0 cost but positive retail
- `price_discrepancy` — Price deviation from expected retail
- `overstock` — Days-of-supply > 180 days

These fit naturally into the existing `classify_row()` match pattern.

### 5.2 Direct Port to Python (Domain Logic)

#### Conversational Diagnostics → `python/sentinel_agent/diagnostics.py`

Port the 26-pattern diagnostic engine. The patterns themselves are domain knowledge (hardware retail) and don't benefit from Rust performance. The interactive flow works well as a Python module called from the sidecar API.

Estimated: ~400-500 LOC Python (simplified from 907 LOC — remove legacy abstractions).

#### Production Thresholds → Configuration

Port the battle-tested thresholds from `config.py`:
```python
margin_leak_threshold = 0.25
margin_below_average_factor = 0.70
dead_stock_days = 90
low_stock_threshold = 10
overstock_days_supply = 180
price_discrepancy_threshold = 0.15
```

These should be configurable (not hardcoded) in the new system. Add to `sidecar_config.py`.

### 5.3 Hybrid (Rust Compute + Python Orchestration)

#### VSA→Symbolic Bridge

The bridge has two parts:
1. **VSA→Facts translation** (Python, ~150 LOC) — Interprets resonator results as symbolic facts
2. **Forward/backward chaining** (could be Rust, ~300 LOC) — Inference engine with unification

The inference engine (`reasoning/inference.py`, 432 LOC) is a good Rust candidate if proof-tree generation becomes a hot path. For MVP, keep in Python.

### 5.4 Reference Data to Carry Forward

| Asset | Location | Value |
|---|---|---|
| Column alias dictionary | `config/pos_mappings/standard_fields.yaml` | 400+ vendor-specific aliases |
| POS mapping reference | `docs/POS_COLUMN_MAPPING_REFERENCE.md` | 20+ POS system column maps |
| Paladin margin note | `analysis.py:1029,1045` | "Paladin margin is unreliable" |
| Impact sanity caps | `analysis.py` | Prevents $10M hallucinated impacts |

---

## 6. Real Data Requirements

### 6.1 Available Data

| Source | Path | Files | Rows | Format |
|---|---|---|---|---|
| Sample store inventory | `Reports/` | 17 CSVs | 861K+ | SHLP (36 col) + sales detail |
| Sample store inventory | `custom_1.csv` | 1 CSV | 156K | IdoSoft custom (53 col) |
| Orgill POs | `/Users/joseph/Downloads/OrgilPO/` | 148 files | ~15K items | PO format (14-row header + items) |

**Total**: 166 files, ~176 MB, ~1M rows.

### 6.2 Data Flow (Target State)

```
                    ┌─────────────┐
                    │  Raw Files  │
                    │ (Store Data │
                    │   + Orgill) │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Python    │
                    │  Adapters   │
                    │(sample_store│
                    │  /orgill/)  │
                    └──────┬──────┘
                           │
                    NormalizedInventory (18 fields)
                           │
                    ┌──────▼──────┐
                    │   Bridge    │  ← THIS IS THE CRITICAL GAP
                    │  Converter  │
                    │ (Python)    │
                    └──────┬──────┘
                           │
                    Intermediate CSV (11 fields)
                    or direct InventoryRecord JSON
                           │
                    ┌──────▼──────┐
                    │    Rust     │
                    │  Pipeline   │
                    │ (sentinel-  │
                    │  server)    │
                    └──────┬──────┘
                           │
                    Digest JSON
                           │
                    ┌──────▼──────┐
                    │   Python    │
                    │  Agent Layer│
                    │ (digest,    │
                    │  coop, etc.)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Sidecar    │
                    │  API +      │
                    │  Mobile UI  │
                    └─────────────┘
```

### 6.3 Missing Data Fields

The Rust pipeline needs fields that aren't directly in sample store exports:

| Required Field | Source | Derivation Strategy |
|---|---|---|
| `sales_last_30d` | Not in single export | `sales_ytd / months_elapsed` (rough), or join with `Sales_DetailYTD.csv` |
| `days_since_receipt` | `last_receipt_date` | `(today - date).days`; default 365 if null |
| `is_damaged` | Not in POS data | Default `false` |
| `is_seasonal` | Category/department | Heuristic from department name (e.g., "SEASONAL", "CHRISTMAS") |
| `margin_pct` | Computed | `(retail - cost) / retail` if retail > 0 |

---

## 7. Recommended Next Steps

### Phase 8: Data Bridge & Real Pipeline (NEXT — Highest Priority)

**Goal:** Run the full pipeline on real store data and produce actual insights.

#### Step 1: NormalizedInventory → Pipeline CSV Converter (~150 LOC Python)

Create `python/sentinel_agent/adapters/pipeline_bridge.py`:
- Accepts `list[NormalizedInventory]`
- Derives computed fields (`margin_pct`, `days_since_receipt`, `sales_last_30d`)
- Writes intermediate CSV in `InventoryRecord` format
- Returns path for `SentinelEngine.run()`

#### Step 2: Real Data Fixture Generation (~100 LOC Python)

Create `python/scripts/generate_real_fixtures.py`:
- Reads `custom_1.csv` through the sample store inventory adapter
- Converts via bridge to pipeline CSV format
- Writes to `fixtures/sample_store_real.csv` (sanitized subset: first 1000 rows)
- Creates `fixtures/orgill_po_sample.csv` from one PO file

#### Step 3: Integration Test (~100 LOC Python)

Create `python/tests/test_real_pipeline.py`:
- Loads real fixture through adapter → bridge → engine → digest
- Asserts: issues found, dollar impacts > 0, no crashes
- Validates that NormalizedInventory round-trips correctly

#### Step 4: Enhance `SentinelEngine` for Adapter Input (~50 LOC Python)

Add `SentinelEngine.run_from_adapter()` method:
- Accepts `AdapterResult` directly
- Internally converts to CSV via bridge
- Runs pipeline
- Returns `Digest`

This eliminates the need for callers to know about the CSV intermediate format.

### Phase 9: Port Missing Detection Primitives (~200 LOC Rust)

Add to `issue_classifier.rs`:
- `ShrinkagePattern` — Multi-signal shrinkage detection
- `ZeroCostAnomaly` — $0 cost with positive retail
- `PriceDiscrepancy` — Deviation from expected retail
- `Overstock` — Days-of-supply exceeding threshold

Port production thresholds from original `config.py`.

### Phase 10: Evidence Cause Scoring in Rust (~300 LOC Rust)

Create `sentinel-vsa/src/evidence.rs`:
- Define cause vectors (receiving_gap, vendor_price_increase, etc.)
- Implement positive-similarity scoring
- Integrate with `issue_classifier.rs` — each issue gets a `root_cause` field
- Expose via pipeline JSON output

### Phase 11: Conversational Diagnostics in Python (~500 LOC Python)

Create `python/sentinel_agent/diagnostics.py`:
- Port 26 patterns from original `diagnostic/engine.py`
- Expose via sidecar API: `POST /api/v1/diagnose/{issue_id}`
- Mobile UI flow: present pattern → user taps confirm/deny → system learns

### Phase 12: Sales Data Integration (~200 LOC Python + ~50 LOC Rust)

Currently `sales_last_30d` is approximated. To get accurate values:
- Parse `Sales_DetailYTD.csv` (116K rows) in a new adapter
- Join sales data with inventory by SKU
- Feed accurate `sales_last_30d` into pipeline

This unlocks accurate `DeadStock` and `Overstock` detection.

### Future Phases (Lower Priority)

| Phase | Work | LOC Estimate |
|---|---|---|
| 13 | VSA→Symbolic bridge in Python | ~400 LOC |
| 14 | Schema evolution handling | ~200 LOC |
| 15 | Additional POS adapters (Paladin, Epicor) | ~300 LOC each |
| 16 | Database-backed task store (replace in-memory) | ~200 LOC |
| 17 | Production deployment (Docker, CI/CD) | Config only |

---

## Appendix A: File Count Summary

| Component | Files | LOC | Language |
|---|---|---|---|
| Original `apps/api/` | 38 | ~5,000 | Python |
| Original `packages/sentinel-engine/` | 40+ | ~9,000 | Python |
| Original `packages/vsa-core/` | 8 | ~2,500 | Python |
| Original `packages/reasoning/` | 6 | ~1,000 | Python |
| **Original Total** | **~92** | **~17,500** | **Python** |
| New `sentinel-vsa/` | 7 | ~1,500 | Rust |
| New `sentinel-pipeline/` | 10 | ~2,400 | Rust |
| New `sentinel-server/` | 1 | ~500 | Rust |
| New Python agent layer | 25 | ~6,000 | Python |
| New Python tests | 14 | ~4,500 | Python |
| **New Total** | **~57** | **~15,400** | **Rust + Python** |

## Appendix B: Test Coverage

| Suite | Tests | Passing |
|---|---|---|
| Original Python (apps + packages) | ~120 | ✅ |
| New Rust (`cargo test`) | 58 | ✅ |
| New Python (`pytest`) | 322 | ✅ |
| **Total verified** | **380** (new system) | ✅ |

## Appendix C: Key Decision Record

| Decision | Rationale |
|---|---|
| Subprocess bridge over PyO3 | Simpler, binary already works, subprocess overhead negligible vs pipeline time |
| Adapter-per-vendor over AI mapping | Deterministic, testable, handles real edge cases (e.g., "Qty." vs "Qty On Hand") |
| Rust for hot path only | VSA bundling + classification benefit from parallelism; domain logic stays in Python |
| FastAPI sidecar over Axum | All business logic in Python; avoids FFI for every endpoint |
| In-memory task store for MVP | Simple, sufficient for single-store; database in Phase 16 |
| Dark-theme mobile-first UI | Store managers check at 6 AM in dim warehouse; thumb-friendly tap targets |
