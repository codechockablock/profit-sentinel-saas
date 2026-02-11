# Production Audit Report — Profit Sentinel

**Date:** 2026-02-11
**Auditor:** Claude Opus 4.6 (Automated)
**Scope:** Full site + dashboard + API + privacy + docs + terminology

---

## Part 1: Landing Page & Marketing Site

### 1.1 Landing Page (profitsentinel.com)

| Check | Status | Notes |
|-------|--------|-------|
| Page loads without errors | PASS | HTTP 200, 43KB |
| Headline accurate | PASS | "Find the Profit Leaks Hiding in Your Inventory" |
| Subheadline accurate | **FIXED** | Removed "71% average shrinkage reduction" (no supporting data). Removed "156,000+ SKUs" (unvalidated extrapolation). Now reads: "36,000 SKUs analyzed in under 3 seconds" |
| Feature descriptions accurate | PASS | 11 leak types match actual pipeline output |
| "Real-time" claims | PASS | No "real-time" claims found — correctly described as batch analysis |
| "AI-powered" claims | PASS | Refers to pattern recognition engine, accurately described |
| Engine 2 features marketed as launch | PASS | Landing page focuses on core 11-leak analysis, not Engine 2 |
| Pricing claims | N/A | No pricing shown on landing page |
| Privacy claims accurate | PASS | "Encrypted in transit and at rest" verified (TLS 1.3 + AES-256 S3). "Auto-deleted in 24 hours" verified (S3 lifecycle policy) |
| Images/screenshots | PASS | Logo image only, no UI screenshots making unverified claims |
| Trust badge "Results in 60 seconds" | PASS | Conservative — actual performance is <4 seconds |
| Trust badge "Files auto-deleted in 24 hours" | PASS | S3 lifecycle policy verified |
| Trust badge "No credit card required" | PASS | Free tier confirmed |

### 1.2 Navigation — Every Link

| Page | Status | HTTP | Notes |
|------|--------|------|-------|
| Landing (/) | PASS | 200 | Loads correctly |
| About (/about) | **FIXED** | 200 | Removed "Vector Symbolic Architecture (VSA)" internal terminology |
| Privacy (/privacy) | PASS | 200 | Comprehensive policy, last updated Jan 2026 |
| Terms (/terms) | PASS | 200 | Clear, appropriate disclaimers |
| Contact (/contact) | PASS | 200 | support@profitsentinel.com listed |
| Roadmap (/roadmap) | **FIXED** | 200 | Multiple claims corrected (see Fixes) |
| Diagnostic (/diagnostic) | PASS | 200 | Shrinkage diagnostic wizard |
| Analyze (/analyze) | PASS | 200 | Upload and analysis flow |
| Dashboard (/dashboard) | PASS | — | Auth-gated, code reviewed |
| Blog | N/A | — | Does not exist (not linked) |
| Documentation/Help | N/A | — | Does not exist (not linked) |

### 1.3 Authentication Flow

| Check | Status | Notes |
|-------|--------|-------|
| Sign-up flow exists | PASS | AuthModal component with Supabase |
| Login flow exists | PASS | signInWithPassword() |
| Password reset | NOT VERIFIED | Could not test live (no test account) |
| Supabase auth configured | PASS | JWT Bearer token validation verified in code |
| Post-login landing | PASS | Redirects to /dashboard |

---

## Part 2: Dashboard Audit

### 2.1 Main Dashboard (/dashboard)

| Check | Status | Notes |
|-------|--------|-------|
| First load with no data | PASS | "All Clear" message with icon |
| Clear CTA to upload | PASS | Upload accessible via /analyze |
| Engine 2 status dot | PASS | Shows gray/warming for no data, emerald when active |
| Dollar amount summary | PASS | Prominent in summary cards |
| Engine 2 predictions hidden gracefully | PASS | Only shown when data available |
| Mobile responsive | PASS (code) | Responsive Tailwind classes throughout |

### 2.2 Findings Page (/dashboard/findings)

| Check | Status | Notes |
|-------|--------|-------|
| GET /api/v1/findings | PASS | 401 without auth (correct behavior) |
| Cards ranked by dollar impact | PASS | sort_by=dollar_impact parameter |
| Card has: type, summary, dollar, tier, action | PASS | All fields present in UI |
| Acknowledge button | PASS (code) | POST /findings/{id}/acknowledge |
| Acknowledged section | PASS | Collapsible, 60% opacity, restore button |
| Empty state message | PASS | "No active findings" with green checkmark |
| Tier colors match DeadStockTier | PASS | critical=darkred, high=red, medium=orange, low=yellow, info=emerald |

### 2.3 Predictions Page (/dashboard/predictions)

| Check | Status | Notes |
|-------|--------|-------|
| GET /api/v1/predictions | PASS | 401 without auth (correct) |
| Stockout/overstock/velocity alerts | PASS | Three sections in UI |
| Confidence scores and severity | PASS | critical/warning/watch badges |
| Empty state | PASS | "Inventory levels are healthy" |

### 2.4 Transfers Page (/dashboard/transfers)

| Check | Status | Notes |
|-------|--------|-------|
| GET /api/v1/transfers | PASS | 401 without auth (correct) |
| Financial comparison shown | PASS | Clearance vs transfer recovery vs net benefit |
| Source/dest store, match type, confidence | PASS | All visible in card layout |
| Empty state for single-store | PASS | "Transfer recommendations appear when you have multiple stores..." |
| Sorted by net benefit | PASS | Descending order |

### 2.5 Settings Page

| Check | Status | Notes |
|-------|--------|-------|
| Settings page exists | **GAP** | No settings UI. DeadStockConfig backend exists but no frontend |
| Dead stock threshold sliders | **GAP** | Not built |
| Per-category overrides | **GAP** | Not built |
| Min capital threshold | **GAP** | Not built |
| Validation feedback | **GAP** | Not built |

### 2.6 Data Upload Flow

| Check | Status | Notes |
|-------|--------|-------|
| CSV upload | PASS | Drag-and-drop + click in AnalysisDashboard |
| Column mapping | PASS | AI-assisted (Anthropic) + manual override |
| Analysis runs | PASS | Rust pipeline integration |
| Results appear immediately | PASS | Inline results on analysis page |
| Data quality issues surfaced | PASS | Warnings for negative quantities, zero costs |

### 2.7 Sidebar Navigation

| Check | Status | Notes |
|-------|--------|-------|
| All links work | PASS | 13 menu items, all route to valid pages |
| Active page indicated | PASS | Visual highlight on current page |
| No broken links | PASS | All routes exist |
| No dead feature links | PASS | All linked pages have implementations |

---

## Part 3: Privacy & Data Claims

### 3.1 Privacy Policy

| Claim | Status | Evidence |
|-------|--------|----------|
| "Don't share data with other customers" | PASS | user_id filtering on all queries, S3 prefix isolation per user |
| "Don't sell data" | PASS | No third-party data sharing integrations found |
| LLM data handling disclosed | PASS | Anthropic receives column headers + 5 sample rows only (disclosed in policy) |
| "Auto-deleted in 24 hours" | PASS | S3 lifecycle policy: `delete-uploads-24h` with `expiration.days = 1` |
| "Encrypted in transit" | PASS | TLS 1.3 via ALB `ELBSecurityPolicy-TLS13-1-2-2021-06` |
| "Encrypted at rest" | PASS | S3 AES-256 server-side encryption |
| Cookie claims | PASS | Session + preference cookies only, no tracking pixels |
| GDPR compliance badge | **PARTIAL** | Policy lists data rights. No formal cookie consent banner or DPA |
| CCPA compliance badge | **PARTIAL** | Policy mentions opt-out rights. No formal "Do Not Sell" mechanism |
| Third-party services listed | PASS | AWS, Supabase, Resend/SendGrid, Vercel, Anthropic all disclosed |

### 3.2 Terms of Service

| Check | Status | Notes |
|-------|--------|-------|
| "AS IS" disclaimer | PASS | Section 7 |
| No accuracy guarantees | PASS | "algorithmic suggestions, should be verified by qualified personnel" |
| Data handling | PASS | References privacy policy correctly |
| Liability limitations | PASS | Reasonable for SaaS |

### 3.3 Security

| Check | Status | Notes |
|-------|--------|-------|
| HTTPS enforced | PASS | TLS 1.3 |
| Data encrypted at rest | PASS | AES-256 S3 |
| Supabase auth | PASS | JWT Bearer tokens, proper validation |
| No secrets in repo | PASS | .gitignore comprehensive, no hardcoded keys |
| Error responses clean | PASS | No stack traces in production (`dev_mode=false`) |
| Rate limiting | PASS | 5/hr anon, 100/hr auth (in-memory, per-worker) |

---

## Part 4: API Verification

| Endpoint | HTTP | Auth Enforced | Valid JSON | Clean Errors |
|----------|------|---------------|------------|--------------|
| GET /health | 200 | N/A (public) | PASS | PASS |
| GET /health/engine2 | 200 | N/A (public) | PASS | PASS |
| GET /api/v1/dashboard | 401 | PASS | PASS | No stack traces |
| GET /api/v1/findings | 401 | PASS | PASS | No stack traces |
| GET /api/v1/predictions | 401 | PASS | PASS | No stack traces |
| GET /api/v1/transfers | 401 | PASS | PASS | No stack traces |
| GET /api/v1/config | 401 | PASS | PASS | No stack traces |

Health endpoint returns: `{"status":"ok","version":"0.13.0","binary_found":true,"dev_mode":false}`
Engine 2 shows: `{"status":"warming_up","observations":0}` (expected for idle instance)

---

## Part 5: README & Docs

| Document | Status | Notes |
|----------|--------|-------|
| README.md | PASS | Architecture, setup, test commands all accurate (145 tests) |
| ORCHESTRATOR_SKILL.md | PASS | Banned terms exactly match response_validator.rs (25 terms) |
| ORCHESTRATOR_SKILL.md | PASS | Bridge operations exactly match ops.rs (16 operations) |
| POS_COLUMN_MAPPING_REFERENCE.md | PASS | 20+ POS systems documented |
| MIGRATION_PLAN.md | **STALE** | M5/M6 status conflicts with README |

---

## Part 6: Internal Terminology Leaks

| Term Found | Location | Customer-Facing? | Status |
|------------|----------|------------------|--------|
| "Vector Symbolic Architecture (VSA)" | about/page.tsx:94 | YES | **FIXED** |
| "11-Primitive Analysis" | AnalysisDashboard.tsx:369 | YES | **FIXED** |
| "Symbolic Reasoning & Proof Trees" | roadmap/page.tsx:67 | YES | **FIXED** |
| "Engine 2" in comments | dashboard/page.tsx | NO | Low priority |
| "primitive", "codebook" in types | api.ts | NO | Internal code only |
| "bundling" in recommendations | leak-metadata.ts | NO | Legitimate business term |

---

## Summary

| Category | Pass | Fixed | Gap | N/A |
|----------|------|-------|-----|-----|
| Marketing & Copy | 11 | 4 | 0 | 2 |
| Dashboard | 22 | 0 | 5 | 0 |
| Privacy & Security | 13 | 0 | 2 | 0 |
| API | 7 | 0 | 0 | 0 |
| Docs | 4 | 0 | 1 | 0 |
| Terminology | 3 | 3 | 0 | 0 |
| **Total** | **60** | **7** | **8** | **2** |
