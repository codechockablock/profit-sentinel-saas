# Feature Gaps — Marketing Claims vs Reality

**Date:** 2026-02-11

Features referenced in marketing/roadmap that are not fully built, with recommendations.

---

## HIGH PRIORITY — Remove or Reword Immediately

### 1. POS System Integrations (Square, Lightspeed, Clover, Shopify)

**Claim:** "Direct connections to Square, Lightspeed, Clover, and Shopify POS with OAuth2 sync and connection lifecycle management." (Roadmap: was "Shipped")

**Reality:** The UI for managing POS connections exists (`/dashboard/pos`) and the backend routes work for creating/listing connections. However:
- **No OAuth2 implementation** for any POS system
- **Sync always returns 0 rows** (simulated, comment says "production would call the actual POS API")
- **No API client libraries** for Square/Lightspeed/Clover/Shopify
- Connections stored in-memory only (lost on restart)

**Status after fix:** Changed to "In Progress" with "Coming Soon" label.

**Recommendation:** Either build real OAuth2 integrations or remove the POS Connect page from the dashboard sidebar until real connections are available. Current stub could mislead users who try to sync.

---

### 2. Multi-File Vendor Correlation

**Claim:** "Upload up to 200 vendor invoices. Cross-reference to find short ships & cost variances." (Roadmap: was "Shipped")

**Reality:**
- **No cross-file correlation logic exists** in the codebase
- The system processes individual CSV files independently
- No vendor invoice parsing, cost variance detection, or short-ship identification
- The "200" file count was fabricated

**Status after fix:** Changed to "In Progress" with "Coming Soon" label.

**Recommendation:** Remove until built. No partial capability exists.

---

## MEDIUM PRIORITY — Build or Document as Planned

### 3. Settings / Configuration UI

**Claim:** DeadStockConfig exists in the backend with configurable thresholds (GET/PUT /api/v1/config). Referenced in ONBOARDING_SPEC.md.

**Reality:**
- Backend API exists and works (3 presets: hardware_store, garden_center, specialty_retail)
- **No frontend UI exists** — no settings page, no threshold sliders, no per-category overrides
- Config is stored in-memory with hardcoded "default" user_id (TODO in code)
- Users cannot configure their dead stock thresholds

**Recommendation:** Build a `/dashboard/settings` page with:
- 4 tier sliders (Watchlist, Attention, Action Required, Write-off)
- Per-category overrides
- Minimum capital threshold
- Store type preset selector
- This is referenced in ONBOARDING_SPEC.md and should be part of the onboarding flow.

---

### 4. GDPR / CCPA Compliance Badges

**Claim:** Footer shows "GDPR Compliant" and "CCPA Compliant" badges.

**Reality:**
- Privacy policy lists user rights (access, deletion, portability, correction)
- Contact email provided: privacy@profitsentinel.com
- 48-hour response time commitment

**Missing for full GDPR compliance:**
- No cookie consent banner
- No data processing agreement (DPA) available
- No formal data export mechanism
- No automated deletion workflow (manual email process)

**Missing for full CCPA compliance:**
- No "Do Not Sell My Personal Information" link
- No formal opt-out mechanism beyond email
- No privacy notice at point of collection

**Recommendation:** Either:
- (a) Remove the "GDPR Compliant" / "CCPA Compliant" badges from the footer and replace with "Privacy First" or similar non-certification language, OR
- (b) Implement a cookie consent banner, add a DPA template, and add a "Do Not Sell" link for CCPA

---

## LOW PRIORITY — Future Features Accurately Labeled

### 5. Multi-Location Support

**Status:** Correctly labeled "Exploring" on roadmap. No action needed.

### 6. Scheduled Analysis

**Status:** Correctly labeled "Exploring" on roadmap. No action needed.

### 7. Mobile App

**Status:** Correctly labeled "Exploring" on roadmap. No action needed.

---

## INFORMATIONAL — Production Readiness Items

These are not marketing gaps but are noted for production scale:

### 8. In-Memory Storage (Tasks, Sessions, Config, API Keys, Acknowledged Findings)

Multiple stores use in-memory dictionaries that are lost on container restart:
- Task delegation store
- Diagnostic session store
- User config store
- API key store
- Acknowledged findings set

**Recommendation:** Migrate to Supabase (TODO comments exist in code).

### 9. Rate Limiter is Per-Worker

In-memory rate limiter works per-worker, not cluster-wide. With N workers, effective limit is N x configured limit.

**Recommendation:** Migrate to Redis for production at scale.

---

## Summary

| Gap | Severity | Action |
|-----|----------|--------|
| POS Integrations (stubbed) | HIGH | Roadmap fixed. Build real integrations or hide POS page |
| Vendor Correlation (missing) | HIGH | Roadmap fixed. Build or remove |
| Settings UI (missing) | MEDIUM | Build frontend for existing backend |
| GDPR/CCPA badges (unsubstantiated) | MEDIUM | Remove badges or implement compliance |
| In-memory stores | LOW | Migrate to Supabase |
| Rate limiter scope | LOW | Migrate to Redis |
