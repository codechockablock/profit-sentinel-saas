# Fixes Applied During Production Audit

**Date:** 2026-02-11 (session 1) | 2026-02-12 (session 2)

---

## Session 1 Fixes (2026-02-11) — Marketing & Terminology

### 1. Removed Unverified "71% Shrinkage Reduction" Claim

**File:** `web/src/app/page.tsx`, `web/src/app/layout.tsx`

**Before:** "Our proprietary AI engine analyzes 156,000+ SKUs in seconds, detecting 11 types of profit leaks that humans miss. Retailers see 71% average shrinkage reduction."

**After:** "Our deterministic analysis engine detects 11 types of profit leaks that humans miss. Built in Rust for speed — 36,000 SKUs analyzed in under 3 seconds."

**Reason:** The "71% average shrinkage reduction" figure has no supporting data. The "156,000+ SKUs" claim was an unvalidated extrapolation. Replaced with verified 36K/3s benchmark.

---

### 2. Removed "VSA" Internal Terminology from About Page

**File:** `web/src/app/about/page.tsx`

**Before:** "...built in Rust that uses Vector Symbolic Architecture (VSA) and rule-based pattern recognition to find profit leaks."

**After:** "...built in Rust that uses advanced pattern recognition to find profit leaks."

**Reason:** "Vector Symbolic Architecture (VSA)" is internal terminology that should never appear in customer-facing copy.

---

### 3. Replaced "11-Primitive" with Customer-Friendly Term

**File:** `web/src/components/diagnostic/AnalysisDashboard.tsx`

**Before:** "Full 11-Primitive Analysis"
**After:** "Full 11-Type Profit Leak Analysis"

**Reason:** "Primitive" is internal VSA terminology.

---

### 4. Fixed POS Integrations Roadmap Status (shipped -> in-progress)

**File:** `web/src/app/roadmap/page.tsx`

**Before:** Status: `shipped` with "OAuth2 sync and connection lifecycle management"
**After:** Status: `in-progress`, eta: "Coming Soon"

**Reason:** POS integrations are stubbed — sync returns 0 rows, no OAuth2 implementation, no API clients.

---

### 5. Fixed Multi-File Vendor Correlation Roadmap Status (shipped -> in-progress)

**File:** `web/src/app/roadmap/page.tsx`

**Before:** Status: `shipped` with "Upload up to 200 vendor invoices"
**After:** Status: `in-progress`, eta: "Coming Soon"

**Reason:** No cross-file correlation logic exists. The "200 vendor invoices" claim was fabricated.

---

### 6. Softened "100+ Pages" PDF Report Claim

**File:** `web/src/app/roadmap/page.tsx`

**Before:** "CFO-ready PDF reports with 100+ pages of detailed analysis."
**After:** "CFO-ready PDF reports with detailed analysis, financial impact, and prioritized action items."

---

### 7. Fixed "156,000+ SKUs" Claim on Roadmap

**File:** `web/src/app/roadmap/page.tsx`

**Before:** "Process 156,000+ SKUs in seconds with sub-second analysis."
**After:** "Process large inventory files with the Rust-powered analysis engine. 36K SKUs in under 3 seconds."

---

### 8. Renamed "Symbolic Reasoning & Proof Trees" on Roadmap

**File:** `web/src/app/roadmap/page.tsx`

**Before:** "Symbolic Reasoning & Proof Trees"
**After:** "Transparent Explanations"

---

### 9. Updated Meta Description

**File:** `web/src/app/layout.tsx`

Removed unverified claims from HTML meta description. Aligned with fix #1.

---

## Session 2 Fixes (2026-02-12) — Infrastructure & Production

### 10. Configured Supabase Custom SMTP with Resend

**Method:** Supabase Management API (`/v1/projects/{ref}/config/auth`)

**Problem:** Supabase was using default SMTP with no custom domain, so confirmation and password reset emails were not being delivered reliably.

**Fix:** Configured custom SMTP:
- Host: `smtp.resend.com`
- Port: 465
- Username: `resend`
- Sender: `Profit Sentinel <noreply@profitsentinel.com>`
- Also reduced `smtp_max_frequency` from 60s to 5s

**Result:** Emails now delivered successfully via Resend.

---

### 11. Fixed AWS Secrets Manager Placeholder Key

**Problem:** `SUPABASE_SERVICE_KEY` in AWS Secrets Manager was set to `placeholder-supabase-key`, causing all authenticated API calls to fail with 401.

**Fix:** Updated secret via `aws secretsmanager update-secret` to the real service role key.

**Result:** Backend auth validation now works — all API endpoints return data for authenticated users.

---

### 12. Fixed ECS Environment Variable Prefix Mismatch

**Problem:** Initially added `SIDECAR_SUPABASE_URL` and `SIDECAR_SUPABASE_SERVICE_KEY` to the ECS task definition, but the Python `SidecarSettings` class has no `env_prefix` — it reads `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` directly.

**Fix:** Removed the incorrect `SIDECAR_`-prefixed variables. The correct vars were already available via Secrets Manager and plain environment variables.

---

### 13. Built Digest Cache Bridge (Analysis -> Dashboard)

**Files:** `upload_routes.py`, `routes/state.py`

**Problem:** The `/analysis/analyze` endpoint saved results to `analysis_store` (Supabase) but the dashboard endpoints (`/api/v1/digest`, `/api/v1/dashboard`, `/api/v1/findings`) read from `state.digest_cache` (in-memory). No bridge existed between the two. Dashboard always showed "CSV file not found" because `digest_cache` was empty and the fallback tried to read a local CSV that doesn't exist in production.

**Fix (upload_routes.py):** After analysis completes, cache the `Digest` object in `app_state.digest_cache` with a 1-hour TTL:
```python
from .routes.state import DigestCacheEntry
cache_key = f":{settings.sentinel_top_k}"
app_state.digest_cache[cache_key] = DigestCacheEntry(digest, ttl_seconds=3600)
```

**Fix (routes/state.py):** Modified `get_or_run_digest()` to fall back to any non-expired cache entry when the exact key doesn't match:
```python
for key, entry in self.digest_cache.items():
    if not entry.is_expired:
        return entry.digest
```

Also wrapped the `FileNotFoundError` to return a clean 404 instead of crashing.

**Result:** After running `/analysis/analyze`, all dashboard endpoints immediately display the analysis results.

---

### 14. Fixed Issue Model AttributeError in Dashboard & Findings Routes

**Files:** `routes/dashboard.py`, `routes/findings.py`

**Problem:** Both routes referenced `issue.title`, `issue.severity`, `issue.description`, `issue.sku`, and `issue.recommendation` — none of which exist on the `Issue` Pydantic model. This caused `AttributeError: 'Issue' object has no attribute 'title'` (HTTP 500) on every request.

**Fix:** Mapped to actual Issue model fields:

| Route Referenced | Actual Issue Field |
|---|---|
| `issue.title` | `issue.issue_type.display_name` |
| `issue.severity` | Derived from `issue.priority_score` (>= 8.0 = critical, >= 5.0 = high, >= 3.0 = medium, else low) |
| `issue.description` | `issue.context` |
| `issue.sku` | `issue.skus[0].sku_id if issue.skus else None` |
| `issue.recommendation` | `issue.root_cause_display` |

**Result:** Dashboard and Findings pages render correctly with real data.

---

### 15. Fixed Predictions Page Frontend Crash

**Files:** `web/src/app/dashboard/predictions/page.tsx`, `web/src/lib/sentinel-api.ts`

**Problem:** The Predictions page crashed with `TypeError: Cannot read properties of undefined (reading 'toFixed')` because the `InventoryPrediction` TypeScript interface didn't match the API response field names:

| Frontend Expected | API Returns |
|---|---|
| `current_stock` | `current_qty` |
| `current_velocity` | `daily_velocity` |
| `recommended_action` | `recommendation` |

**Fix:**
1. Updated `InventoryPrediction` interface to match API: `current_qty`, `daily_velocity`, `recommendation`
2. Updated component to use correct field names
3. Added null-coalescing guards: `(value ?? 0).toFixed(1)` and `(value ?? 0).toLocaleString()`

**Result:** Predictions page renders 1,796 predictions with all data fields displayed correctly.

---

### 16. Deployed Docker Image with All Fixes to ECS

**Problem:** Previous Docker builds used `git rev-parse --short HEAD` as the image tag, but the code changes were uncommitted. The deployed container ran old code without the digest cache bridge or Issue model fixes.

**Fix:** Built Docker image from working tree with `--no-cache`, tagged as `profitsentinel-prod-api:issue-fix`, pushed to ECR, registered as task definition revision 13, and deployed to ECS.

**Result:** Production container runs code with all fixes. Health check passes. All API endpoints return correct data.

---

## Summary

| # | Fix | Type | Severity | Session |
|---|-----|------|----------|---------|
| 1 | Removed "71% shrinkage reduction" claim | Unverified claim | HIGH | 1 |
| 2 | Removed "VSA" from About page | Internal terminology | HIGH | 1 |
| 3 | Replaced "11-Primitive" | Internal terminology | HIGH | 1 |
| 4 | POS integrations roadmap status | False "shipped" status | HIGH | 1 |
| 5 | Vendor correlation roadmap status | False "shipped" status | HIGH | 1 |
| 6 | Softened "100+ pages" PDF claim | Unvalidated claim | MEDIUM | 1 |
| 7 | Fixed "156K+ SKUs" claim | Unvalidated claim | MEDIUM | 1 |
| 8 | Renamed "Symbolic Reasoning" | Internal terminology | MEDIUM | 1 |
| 9 | Updated meta description | Unverified claims | MEDIUM | 1 |
| 10 | Configured Supabase custom SMTP | Infrastructure | HIGH | 2 |
| 11 | Fixed Secrets Manager placeholder key | Infrastructure | **CRITICAL** | 2 |
| 12 | Fixed ECS env var prefix mismatch | Infrastructure | HIGH | 2 |
| 13 | Built digest cache bridge | Architecture | **CRITICAL** | 2 |
| 14 | Fixed Issue model AttributeError | Bug | **CRITICAL** | 2 |
| 15 | Fixed Predictions page crash | Bug | HIGH | 2 |
| 16 | Deployed Docker image with fixes | Deployment | HIGH | 2 |
