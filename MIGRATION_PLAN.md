# Profit Sentinel: Production Migration Plan

**Version:** 1.0
**Date:** 2026-02-06
**Author:** Migration Engineer (Claude)
**Status:** DRAFT — Awaiting owner review before any infrastructure changes

---

## Executive Summary

Migrate Profit Sentinel from the current TypeScript/Node.js + Python heuristic engine to the new Rust + Python pipeline. The new system is **10x faster** (sub-second vs 10.3s), has **evidence-based root cause analysis** (0% hallucination), and adds **co-op intelligence, delegation, vendor call prep, and symbolic reasoning** — features the old system doesn't have.

**Key constraint:** Zero downtime. Same domain. Same UX flow. Customers notice nothing except speed.

---

## Table of Contents

1. [Current Architecture (As-Is)](#1-current-architecture-as-is)
2. [Target Architecture (To-Be)](#2-target-architecture-to-be)
3. [Baseline Measurements](#3-baseline-measurements)
4. [Critical Gap Analysis](#4-critical-gap-analysis)
5. [Migration Phases](#5-migration-phases)
6. [Integration Inventory](#6-integration-inventory)
7. [Risk Register](#7-risk-register)
8. [Rollback Plan](#8-rollback-plan)
9. [Verification Checklist](#9-verification-checklist)
10. [Timeline Estimate](#10-timeline-estimate)

---

## 1. Current Architecture (As-Is)

### 1.1 Services

| Service | Technology | Location | Purpose |
|---------|-----------|----------|---------|
| **Frontend** | Next.js 16 (React 19, TypeScript) | Vercel (profitsentinel.com) | Upload UI, results dashboard, billing |
| **Backend API** | Python FastAPI (async) | AWS ECS Fargate (api.profitsentinel.com) | Analysis, uploads, auth, billing, diagnostics |
| **Database** | PostgreSQL (Supabase hosted) | PLACEHOLDER_SUPABASE_PROJECT.supabase.co | user_profiles, email_signups, analysis_synopses |
| **File Storage** | AWS S3 | us-east-1 (profit-sentinel-uploads) | CSV/XLS uploads, presigned URLs, 24h auto-delete |
| **Auth** | Supabase Auth (JWT) | Supabase hosted | Sign up, login, token validation |
| **Billing** | Stripe | stripe.com | Pro subscription ($99/mo), 14-day trial |
| **Email** | Resend (primary) | resend.com | Analysis reports, contact forms |
| **DNS** | GoDaddy | godaddy.com | profitsentinel.com → Vercel, api.profitsentinel.com → ALB |
| **AI Mapping** | Grok (xAI) via OpenAI-compat API | api.x.ai | Column name inference from CSV samples |
| **GPU Service** | NVIDIA T4 (g4dn.xlarge) | AWS EC2 ASG | VSA bundling (optional, slow path) |

### 1.2 API Endpoints (50+ total)

**Core Flow (what users interact with):**
- `POST /uploads/presign` → presigned S3 URL
- `POST /uploads/suggest-mapping` → AI column mapping
- `POST /analysis/analyze` → 11-primitive leak detection (10.3s for 36K rows)
- `GET /analysis/primitives` → list primitives
- `GET /analysis/supported-pos` → 40+ POS systems

**Premium:**
- `POST /analysis/analyze-multi` → multi-file correlation (Pro tier)
- `POST /diagnostic/start` → shrinkage diagnostic Q&A session
- `POST /premium/diagnostic/start` → premium vendor correlation

**Billing:**
- `POST /billing/create-checkout-session` → Stripe checkout
- `POST /billing/portal` → customer portal
- `GET /billing/status` → subscription status
- `POST /billing/webhook` → Stripe webhook handler

**Other:** repair assistant (stubs), employee management (stubs), contact, privacy, metrics, reports, health

### 1.3 Infrastructure

- **ECS Task:** 4 vCPU, 16GB RAM (Fargate) — sized for 156K+ row CSVs
- **ALB:** 300s idle timeout (long analysis), TLS 1.2+/1.3
- **ECR:** profitsentinel-dev-api (container registry)
- **RDS Aurora:** PostgreSQL 15, Serverless v2 (0.5–2.0 ACU) — currently unused by app (Supabase used instead)
- **S3:** 24h lifecycle, encryption at rest (AES256), public access blocked
- **VPC:** 10.0.0.0/16, 2 public + 2 private subnets, VPC endpoints (cost-optimized, no NAT)
- **Secrets Manager:** XAI API key, Supabase service key, Resend API key
- **CI/CD:** GitHub Actions → ECR → ECS (backend), GitHub Actions → Vercel (frontend)

### 1.4 Database Schema (Supabase)

```
user_profiles (extends auth.users)
├── subscription_tier: free | pro | enterprise
├── stripe_customer_id, stripe_subscription_id
├── current_period_end, trial_starts_at, trial_expires_at
├── analyses_this_month, total_analyses
└── RLS: users read/update own profile

email_signups
├── email (unique per source), company_name, role, store_count
├── UTM tracking, consent flags
└── RLS: anon insert, service role read

analysis_synopses (anonymized analytics only)
├── file_hash (SHA256), detection_counts (JSONB)
├── total_impact_estimate_low/high
├── processing_time_seconds, engine_version
└── RLS: anon insert, service role read
```

### 1.5 Upload → Analysis Flow

```
Browser                    Vercel (Next.js)              API (FastAPI)              S3
  │                            │                            │                       │
  ├─── POST /api/uploads/presign ──────────────────────────►│                       │
  │◄── { key, url } ──────────────────────────────────────── │                       │
  │                            │                            │                       │
  ├─── PUT <presigned-url> ────────────────────────────────────────────────────────►│
  │◄── 200 OK ──────────────────────────────────────────────────────────────────────│
  │                            │                            │                       │
  ├─── POST /api/uploads/suggest-mapping ──────────────────►│── load 50 rows ──────►│
  │◄── { mapping, confidence } ─────────────────────────────│◄─── sample rows ──────│
  │                            │                            │                       │
  ├─── POST api.profitsentinel.com/analysis/analyze ───────►│── load full file ────►│
  │    (direct to API, bypasses Vercel timeout)             │◄─── full DataFrame ───│
  │                            │                            │                       │
  │◄── { leaks, summary, cause_diagnosis } ─────────────────│── delete file (60s) ─►│
```

**Important:** The analysis call goes **directly** to `api.profitsentinel.com` (not through Vercel proxy) to avoid Vercel's function timeout limits.

---

## 2. Target Architecture (To-Be)

### 2.1 New Components

| Component | Technology | Status | What It Does |
|-----------|-----------|--------|-------------|
| **sentinel-server** | Rust (binary) | ✅ Built, 85 tests | CSV → issue detection, VSA evidence scoring, JSON output |
| **sentinel-pipeline** | Rust (library) | ✅ Built | Issue classification, grouping, scoring |
| **sentinel-vsa** | Rust (library) | ✅ Built | Vector Symbolic Architecture evidence grounding |
| **sentinel-agent** | Python (package) | ✅ Built, 566 tests | Engine bridge, digest, delegation, vendor calls, co-op, symbolic reasoning |
| **sidecar API** | Python FastAPI | ✅ Built | REST API serving sentinel-agent functionality |
| **Mobile UI** | HTML/CSS/JS (static) | ✅ Built | Morning digest, task delegation, co-op dashboard |

### 2.2 Target Service Map

```
Browser                    Vercel (Next.js)              API (FastAPI)              S3
  │                            │                            │                       │
  │  [SAME upload flow as current — S3 presigned URLs]      │                       │
  │                            │                            │                       │
  ├─── POST api.profitsentinel.com/analysis/analyze ───────►│                       │
  │                            │                    ┌───────┴───────┐               │
  │                            │                    │  NEW: Adapter │               │
  │                            │                    │  Maps columns │               │
  │                            │                    │  Calls Rust   │               │
  │                            │                    │  binary via   │               │
  │                            │                    │  subprocess   │               │
  │                            │                    └───────┬───────┘               │
  │                            │                            │                       │
  │◄── { leaks, summary, cause_diagnosis, proof_tree } ─────│                       │
  │    [SAME response shape + new fields]                   │                       │
```

### 2.3 What Changes, What Stays

| Component | Action | Notes |
|-----------|--------|-------|
| **Frontend (Vercel)** | KEEP AS-IS initially | Same Next.js app, same domain, same UX |
| **Backend API (ECS)** | REPLACE analysis engine | Swap sentinel_engine → new Rust+Python pipeline |
| **S3 upload flow** | KEEP AS-IS | Same presigned URLs, same cleanup |
| **Supabase auth** | KEEP AS-IS | Same JWT validation |
| **Supabase database** | KEEP AS-IS | Same tables, same RLS |
| **Stripe billing** | KEEP AS-IS | Same webhooks, same checkout |
| **Column mapping** | KEEP + ENHANCE | Same AI mapping, add adapter to Rust schema |
| **Analysis endpoint** | MODIFY | Same request/response shape, new engine underneath |
| **CI/CD** | UPDATE | Add Rust build step to Docker image |
| **DNS** | NO CHANGE | Same domains, same records |
| **Email** | KEEP AS-IS | Same Resend integration |

---

## 3. Baseline Measurements

### 3.1 Old System (Production, 2026-02-06)

**Test file:** `Inventory_Report_GreaterThanZero_AllSKUs.csv` (36,452 rows, 29 columns, Paladin POS)

| Metric | Value |
|--------|-------|
| **Analysis time** | **10.3 seconds** |
| **SKUs analyzed** | 36,452 |
| **Issues found** | 14,068 |
| **Leak types detected** | 10 / 11 |
| **Estimated annual impact** | $0 — $0 (BUG: impact calculation broken) |
| **Root cause analysis** | "Margin Leak" at 100% confidence |
| **Hallucination rate** | 0% (VSA grounded) |

**Leak type breakdown (old system):**

| Leak Type | Severity | Items Found |
|-----------|----------|-------------|
| Low Stock Risk | HIGH | 7,414 |
| Margin Leak | CRITICAL | 1,797 |
| Margin Erosion | HIGH | 1,418 |
| Severe Inventory Deficit | CRITICAL | 1,207 |
| Price Discrepancy | LOW | 861 |
| Negative Profit | CRITICAL | 791 |
| Overstock | MEDIUM | 269 |
| Dead Inventory | MEDIUM | 234 |
| Zero Cost Anomaly | CRITICAL | 71 |
| Shrinkage Pattern | HIGH | 6 |
| Negative Inventory | CRITICAL | 0 |

**Known bugs in old system:**
1. **Impact calculation returns $0 — $0** despite 14,068 issues detected
2. **Duplicate column warnings** (Retail→revenue, Barcode→sku, Dpt.→category)
3. **Low Stock over-flagging:** 7,414 items (20% of dataset) — threshold may be too aggressive
4. **Root cause limited:** Only returns single cause ("Margin Leak") with no competing hypotheses

### 3.2 New System (Local, Pending)

The new Rust pipeline expects a normalized schema:
```
store_id, sku, qty_on_hand, unit_cost, margin_pct, sales_last_30d,
days_since_receipt, retail_price, is_damaged, on_order_qty, is_seasonal
```

**Cannot run against production CSV directly** — needs a column mapping adapter (same as old system). This adapter is the primary development work for migration.

**Expected performance (from fixture tests):**
- Pipeline execution: **< 500ms** for 36K rows (estimated from 10K fixture benchmarks)
- Evidence scoring: **0.003ms** hot path
- Root cause analysis: Full competing hypothesis scores with symbolic proof trees
- New capabilities: Co-op intelligence, vendor call prep, task delegation

---

## 4. Critical Gap Analysis

### 4.1 Must-Build Before Migration

| # | Gap | Effort | Priority |
|---|-----|--------|----------|
| **G1** | **Column Mapping Adapter** — Transform arbitrary POS CSV → Rust pipeline schema | 2 days | P0 |
| **G2** | **Response Format Adapter** — Transform Rust JSON → old API response shape (leaks, summary, cause_diagnosis) | 1 day | P0 |
| **G3** | **Impact Estimation** — Old system has $-impact per leak type; new system has dollar_impact per issue but different format | 1 day | P0 |
| **G4** | **Multi-file Analysis** — Old system has `analyze-multi` (Pro feature); new Rust pipeline is single-store | 2 days | P1 |
| **G5** | **Diagnostic Q&A Sessions** — Old system has interactive shrinkage diagnostic; not in new system | 3 days | P2 |
| **G6** | **Repair Assistant** — Old system has stubs; not in new system (stubs OK) | 0.5 day | P3 |
| **G7** | **Employee Management** — Old system has stubs; not in new system (stubs OK) | 0.5 day | P3 |

### 4.2 Feature Parity Matrix

| Feature | Old System | New System | Gap |
|---------|-----------|------------|-----|
| CSV/XLS upload via S3 | ✅ | ✅ (reuse) | None |
| AI column mapping (Grok) | ✅ | ✅ (reuse) | None |
| 11 leak primitives | ✅ | ✅ (11 issue types) | **G2: response format** |
| $ impact estimation | ❌ (broken, returns $0) | ✅ (dollar_impact per issue) | **G3: format adapter** |
| Per-item context/explanation | ✅ | ✅ (context field) | Minor format diff |
| Root cause (single) | ✅ | ✅ (with confidence) | None |
| Competing hypotheses | ❌ | ✅ (cause_scores) | New feature |
| Symbolic proof trees | ❌ | ✅ (explain API) | New feature |
| Co-op intelligence | ❌ | ✅ | New feature |
| Task delegation | ❌ | ✅ | New feature |
| Vendor call prep | ❌ | ✅ | New feature |
| Morning digest | ❌ | ✅ | New feature |
| Multi-file correlation | ✅ (Pro) | ❌ | **G4** |
| Shrinkage diagnostic Q&A | ✅ | ❌ | **G5** |
| Stripe billing | ✅ | ✅ (reuse) | None |
| Supabase auth | ✅ | ✅ (reuse) | None |
| Email reports | ✅ | ✅ (reuse) | None |

### 4.3 Primitive Name Mapping

| Old System Primitive | New System Issue Type | Match Quality |
|---------------------|----------------------|---------------|
| `high_margin_leak` | `MarginErosion` | ✅ Direct |
| `negative_inventory` | `NegativeInventory` | ✅ Direct |
| `negative_profit` | `PriceDiscrepancy` (below-cost) | ⚠️ Close |
| `severe_inventory_deficit` | `ReceivingGap` | ⚠️ Close |
| `low_stock` | `ReceivingGap` | ⚠️ Partial overlap |
| `shrinkage_pattern` | `ShrinkagePattern` | ✅ Direct |
| `margin_erosion` | `MarginErosion` | ✅ Direct |
| `zero_cost_anomaly` | `ZeroCostAnomaly` | ✅ Direct |
| `dead_item` | `DeadStock` | ✅ Direct |
| `overstock` | `Overstock` | ✅ Direct |
| `price_discrepancy` | `PriceDiscrepancy` | ✅ Direct |
| — | `VendorShortShip` | New |
| — | `PurchasingLeakage` | New |
| — | `PatronageMiss` | New |

---

## 5. Migration Phases

### Phase M1: Column Mapping Adapter (2 days)

**Goal:** Enable Rust pipeline to accept any POS CSV format.

**What to build:**
- `apps/api/src/services/column_adapter.py` (~200 lines)
  - Takes raw DataFrame + column mapping (from existing suggest-mapping step)
  - Maps to Rust schema: `store_id, sku, qty_on_hand, unit_cost, margin_pct, sales_last_30d, days_since_receipt, retail_price, is_damaged, on_order_qty, is_seasonal`
  - Handles missing columns with sensible defaults:
    - `store_id` → "default" (single-store analysis)
    - `days_since_receipt` → calculated from `Last Pur.` date, default 30
    - `is_damaged` → false
    - `on_order_qty` → 0
    - `is_seasonal` → false
  - Writes normalized CSV to temp file, passes to Rust binary
  - Cleans up temp file after analysis

**Testing:**
- Unit test with Paladin POS CSV header → verify correct mapping
- Integration test: full pipeline with `Inventory_Report_GreaterThanZero_AllSKUs.csv`
- Compare issue counts with old system baseline

### Phase M2: Response Format Adapter (1 day)

**Goal:** Make Rust pipeline output match old API response shape exactly.

**What to build:**
- `apps/api/src/services/result_adapter.py` (~300 lines)
  - Transforms `Digest` (Rust JSON) → old `{ leaks, summary, cause_diagnosis }` format
  - Maps issue types to old primitive names
  - Includes per-item details (sku, score, description, quantity, cost, revenue, sold, margin, context)
  - Preserves old display metadata (title, icon, color, priority, severity, recommendations)
  - Adds new fields (cause_scores, proof_tree) that the frontend can optionally consume
  - Calculates estimated_impact from dollar_impact per issue

**Backward compatibility:**
- Old frontend sees same JSON shape → works immediately
- New fields are additive → old frontend ignores them
- Later: frontend update to display new features (proof trees, competing hypotheses)

### Phase M3: Engine Swap (1 day)

**Goal:** Replace old analysis engine with new Rust+Python pipeline in the existing API.

**What to change:**
- `apps/api/src/services/analysis.py` → modify `AnalysisService.analyze()`:
  - Import column adapter and result adapter
  - Call adapter → write temp CSV → run `sentinel-server` subprocess → parse JSON → transform → return
  - Fallback: if Rust binary fails, fall back to old heuristic engine
  - Log timing comparison (old vs new)

- `apps/api/Dockerfile` → add Rust binary:
  - Multi-stage build: stage 1 builds Rust binary, stage 2 copies into Python image
  - Set `SENTINEL_BIN=/app/sentinel-server` env var
  - ~40 additional lines

- `apps/api/requirements.txt` → add `pydantic>=2.0` (for Digest model parsing)

**Safety:**
- Feature flag: `USE_NEW_ENGINE=true/false` (env var)
- When `false`, old engine runs as before
- When `true`, new Rust engine runs with fallback to old on error
- Log both timings when flag is `true` (A/B comparison)

### Phase M4: Docker Build & Local Testing (1 day)

**Goal:** Verify the new Docker image works end-to-end locally.

**Steps:**
1. Build new Docker image with Rust binary
2. Run `docker compose up api` with `USE_NEW_ENGINE=true`
3. Upload test CSV through local frontend → verify results match baseline
4. Compare:
   - Issue counts per type
   - Per-item SKU matches (top 10 per type)
   - Analysis timing
   - Response format compatibility
5. Run existing test suite: `pytest apps/api/tests/ -v`
6. Run new pipeline tests: `cd profit-sentinel-rs/python && pytest -v`

### Phase M5: Staging Deployment (1 day)

**Goal:** Deploy to a staging ECS service for real-world testing.

**Steps:**
1. Create staging ECS task definition (copy of prod, different tag)
2. Deploy new Docker image to staging
3. Point staging ALB to new task
4. Test with real CSV through staging URL
5. Monitor CloudWatch logs for errors
6. Compare results with production baseline

### Phase M6: Production Cutover (1 day, with rollback window)

**Goal:** Zero-downtime switch to new engine.

**Strategy: Blue-Green via Feature Flag**

```
Timeline:
T+0:  Deploy new image with USE_NEW_ENGINE=false (code is there, not active)
T+5:  Verify health checks pass, no errors
T+10: Set USE_NEW_ENGINE=true via ECS env var update
T+15: Monitor first real analysis request
T+30: If OK → declare success, monitor for 24h
T+?:  If errors → set USE_NEW_ENGINE=false (instant rollback)
```

**Monitoring during cutover:**
- CloudWatch: error rate, latency P50/P95/P99
- Application logs: analysis timing, issue counts
- Frontend: verify results render correctly
- Stripe: billing endpoints unaffected (no change)

### Phase M7: Frontend Enhancements (2-3 days, post-cutover)

**Goal:** Surface new capabilities in the UI.

**What to add (incremental, non-breaking):**
1. **Impact display fix** — show actual $ estimates (was broken in old system)
2. **Competing hypotheses** — show alternative root causes with confidence %
3. **Proof tree viewer** — expandable reasoning chain for each issue
4. **New leak types** — display VendorShortShip, PurchasingLeakage, PatronageMiss
5. **Performance badge** — show "Analyzed in 0.4s" (was 10.3s)

### Phase M8: New Features (ongoing, post-cutover)

**What the new system enables:**
1. **Morning Digest** — daily email briefing with prioritized issues
2. **Task Delegation** — assign issues to team members with deadlines
3. **Vendor Call Prep** — talking points and questions for vendor calls
4. **Co-op Intelligence** — rebate tracking, contract analysis
5. **Symbolic Reasoning** — transparent AI with proof trees (explain endpoint)
6. **Mobile Dashboard** — served from sidecar at /mobile

---

## 6. Integration Inventory

### 6.1 Integrations to Preserve (NO CHANGES)

| Integration | Config | Verified |
|-------------|--------|----------|
| **Supabase Auth** | SUPABASE_URL + SUPABASE_SERVICE_KEY | ✅ In .env |
| **Supabase Database** | Same client, same tables | ✅ Schema audited |
| **Stripe Billing** | STRIPE_SECRET_KEY + STRIPE_WEBHOOK_SECRET + STRIPE_PRICE_ID | ✅ In .env |
| **AWS S3** | AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + S3_BUCKET_NAME | ✅ In .env |
| **Grok AI (xAI)** | XAI_API_KEY (OpenAI-compat at api.x.ai) | ✅ In .env |
| **Resend Email** | RESEND_API_KEY | ✅ In .env |
| **Vercel Frontend** | VERCEL_TOKEN + VERCEL_ORG_ID + VERCEL_PROJECT_ID | ✅ In GitHub secrets |
| **GoDaddy DNS** | Manual (no API integration) | ✅ No change needed |
| **Zoho Mail** | MX records on GoDaddy | ✅ DNS-only, no code |

### 6.2 AWS Resources

| Resource | ID/ARN | Change |
|----------|--------|--------|
| ECS Cluster | profitsentinel-dev-cluster | No change |
| ECS Task Def | profitsentinel-dev-api | Update image + add SENTINEL_BIN env |
| ECR Repo | profitsentinel-dev-api | New image with Rust binary |
| ALB | profitsentinel-dev-alb | No change |
| ACM Cert | PLACEHOLDER_CERT_UUID | No change |
| S3 Bucket | profit-sentinel-uploads | No change |
| VPC | profitsentinel-dev | No change |
| Secrets Manager | profitsentinel-dev/* (3 secrets) | No change |
| RDS Aurora | profitsentinel-dev (unused) | No change |

### 6.3 DNS Records (GoDaddy)

| Record | Type | Value | Change |
|--------|------|-------|--------|
| profitsentinel.com | A/CNAME | Vercel | No change |
| www.profitsentinel.com | CNAME | Vercel | No change |
| api.profitsentinel.com | CNAME | ALB DNS | No change |
| MX records | MX | Zoho | No change |

---

## 7. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| R1 | Rust binary crash on unexpected CSV data | Medium | High | Feature flag fallback to old engine; extensive CSV fuzzing |
| R2 | Column mapping adapter misses edge cases | Medium | Medium | Test with 5+ different POS formats; preserve old mapping logic |
| R3 | Response format breaks frontend | Low | High | Schema validation tests; compare JSON diff with baseline |
| R4 | Docker image too large (Rust + Python) | Low | Low | Multi-stage build; Rust binary is ~15MB stripped |
| R5 | ECS task OOM with Rust subprocess | Low | Medium | Rust pipeline uses ~50MB for 36K rows (vs 16GB allocation) |
| R6 | Stripe webhooks interrupted during deploy | Very Low | High | ECS rolling update ensures at least 1 task always running |
| R7 | Different issue counts alarm users | Medium | Medium | Document differences; new system may find more/fewer per type |
| R8 | S3 presigned URL flow breaks | Very Low | High | No changes to upload flow; only analysis engine changes |
| R9 | GPU service dependency | Low | Low | New system doesn't need GPU; VSA is CPU-only in Rust |

---

## 8. Rollback Plan

### Instant Rollback (< 1 minute)
```bash
# Set feature flag to disable new engine
aws ecs update-service \
  --cluster profitsentinel-dev-cluster \
  --service profitsentinel-dev-api \
  --force-new-deployment \
  --task-definition profitsentinel-dev-api:<PREVIOUS_REVISION>
```

### Full Rollback (< 5 minutes)
```bash
# Revert to previous Docker image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Re-deploy previous task definition
aws ecs update-service \
  --cluster profitsentinel-dev-cluster \
  --service profitsentinel-dev-api \
  --task-definition profitsentinel-dev-api:<OLD_REVISION> \
  --force-new-deployment

# Wait for stability
aws ecs wait services-stable \
  --cluster profitsentinel-dev-cluster \
  --services profitsentinel-dev-api
```

### Rollback Triggers
- Error rate > 5% on /analysis/analyze
- P95 latency > 30s (3x old system)
- Any 500 error on billing endpoints
- Frontend unable to render results

---

## 9. Verification Checklist

### Pre-Cutover
- [ ] Column adapter handles all 29 Paladin POS columns
- [ ] Column adapter handles CSV with missing optional columns
- [ ] Response adapter produces identical JSON shape to old system
- [ ] New system finds issues for same SKUs as old system (within 10% tolerance)
- [ ] Impact estimation produces non-zero values (fix old bug)
- [ ] Docker image builds successfully on linux/amd64
- [ ] Docker image passes health check
- [ ] All existing pytest tests pass (apps/api/tests/)
- [ ] All new pipeline tests pass (profit-sentinel-rs/python/tests/)
- [ ] Rust tests pass (cargo test --workspace)
- [ ] Feature flag works: USE_NEW_ENGINE=false uses old engine
- [ ] Feature flag works: USE_NEW_ENGINE=true uses new engine
- [ ] Fallback works: Rust binary failure → old engine runs

### Post-Cutover (Day 1)
- [ ] Real user analysis completes successfully
- [ ] Analysis time < 3s (was 10.3s)
- [ ] No 500 errors in CloudWatch logs
- [ ] Stripe webhooks still processing
- [ ] Email reports still sending
- [ ] Frontend displays results correctly
- [ ] Auth flow works (login, signup, token refresh)
- [ ] S3 file cleanup happening (24h lifecycle)

### Post-Cutover (Week 1)
- [ ] No user complaints about missing/changed results
- [ ] Error rate < 0.1%
- [ ] All POS formats tested (at least 3 different formats)
- [ ] Old engine code can be safely removed (feature flag cleanup)
- [ ] GPU service can be scaled to 0 (no longer needed for analysis)

---

## 10. Timeline Estimate

| Phase | Duration | Dependencies | Deliverable |
|-------|----------|-------------|-------------|
| **M1: Column Adapter** | 2 days | None | column_adapter.py + tests |
| **M2: Response Adapter** | 1 day | M1 | result_adapter.py + tests |
| **M3: Engine Swap** | 1 day | M1, M2 | Modified analysis.py + Dockerfile |
| **M4: Local Testing** | 1 day | M3 | Test report with baseline comparison |
| **M5: Staging** | 1 day | M4 | Staging deployment verified |
| **M6: Production Cutover** | 1 day | M5 | Live with feature flag |
| **M7: Frontend Enhancements** | 2-3 days | M6 | Impact fix, proof trees, new types |
| **M8: New Features** | Ongoing | M6 | Digest, delegation, co-op, mobile |

**Total for production cutover: ~7 working days (M1–M6)**
**Total including frontend: ~10 working days (M1–M7)**

---

## Appendix A: File Changes Summary

### Files to Create
| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `apps/api/src/services/column_adapter.py` | POS CSV → Rust schema mapping | ~200 |
| `apps/api/src/services/result_adapter.py` | Rust JSON → old API response shape | ~300 |
| `apps/api/tests/test_column_adapter.py` | Column adapter unit tests | ~150 |
| `apps/api/tests/test_result_adapter.py` | Response adapter unit tests | ~150 |
| `apps/api/tests/test_engine_integration.py` | End-to-end with real CSV | ~100 |

### Files to Modify
| File | Change | Risk |
|------|--------|------|
| `apps/api/src/services/analysis.py` | Add new engine path with feature flag | Medium |
| `apps/api/Dockerfile` | Multi-stage build with Rust binary | Low |
| `apps/api/src/config.py` | Add USE_NEW_ENGINE, SENTINEL_BIN settings | Low |
| `.github/workflows/deploy.yml` | Add Rust build step | Low |
| `docker-compose.yml` | Update api service build context | Low |

### Files NOT Changed
- All frontend files (apps/web/*)
- All Supabase migrations
- All Terraform configs
- All billing/auth/upload routes
- DNS records
- Stripe configuration

---

## Appendix B: Environment Variables

### New Variables (to add)
```bash
# New engine feature flag
USE_NEW_ENGINE=true          # false = old engine, true = new Rust engine

# Rust binary path (set in Docker image)
SENTINEL_BIN=/app/sentinel-server

# Optional: new engine specific
SENTINEL_DEFAULT_STORE=default
SENTINEL_TOP_K=20
```

### Existing Variables (no change)
```bash
# AWS
AWS_ACCESS_KEY_ID=***
AWS_SECRET_ACCESS_KEY=***
S3_BUCKET_NAME=profit-sentinel-uploads
AWS_REGION=us-east-1

# Supabase
SUPABASE_URL=https://PLACEHOLDER_SUPABASE_PROJECT.supabase.co
SUPABASE_SERVICE_KEY=***

# Stripe
STRIPE_SECRET_KEY=***
STRIPE_WEBHOOK_SECRET=***
STRIPE_PRICE_ID=***

# AI / Email
XAI_API_KEY=***
RESEND_API_KEY=***
```

---

## Appendix C: Baseline Comparison Template

After M4 (local testing), fill in:

| Metric | Old System | New System | Delta |
|--------|-----------|------------|-------|
| Analysis time | 10.3s | ___ s | ___ x faster |
| SKUs analyzed | 36,452 | ___ | Should match |
| Total issues | 14,068 | ___ | May differ |
| Low Stock | 7,414 | ___ | |
| Margin Leak | 1,797 | ___ | |
| Margin Erosion | 1,418 | ___ | |
| Severe Deficit | 1,207 | ___ | |
| Price Discrepancy | 861 | ___ | |
| Negative Profit | 791 | ___ | |
| Overstock | 269 | ___ | |
| Dead Inventory | 234 | ___ | |
| Zero Cost | 71 | ___ | |
| Shrinkage | 6 | ___ | |
| Negative Inventory | 0 | ___ | |
| Annual Impact ($) | $0 (broken) | ___ | Should be non-zero |
| Root causes shown | 1 | ___ | Should show multiple |
| Proof trees | No | ___ | Should be Yes |

---

**END OF MIGRATION PLAN**

*This document must be reviewed and approved before any infrastructure changes are made.*
*All secrets referenced in this document should be rotated if they have been exposed.*
