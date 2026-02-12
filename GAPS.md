# Remaining Gaps & Recommended Next Steps

**Date:** 2026-02-11 (initial) | 2026-02-12 (updated)

Issues identified during audit that are not yet resolved, ordered by priority.

---

## CRITICAL — Fix Before Accepting Customers

### 1. analysis_synopses Table Not Created in Supabase

**Impact:** Analysis results are not persisted to Supabase. The `SupabaseAnalysisStore.save()` call fails with PGRST205 ("Could not find the table 'public.analysis_synopses'"). This is non-blocking (analysis still returns results to the user), but means:
- No analysis history
- No daily_analysis_stats materialized view
- Migration 002_create_analysis_synopses.sql has not been run

**Fix:** Run the migration SQL in Supabase SQL Editor:
```
supabase/migrations/002_create_analysis_synopses.sql
```

---

### 2. DMARC Email Deliverability

**Impact:** Emails from `profitsentinel.com` are delivered but Gmail filters them to "All Mail" (not inbox) due to `p=quarantine` DMARC policy. This affects:
- Email confirmation after signup
- Password reset emails
- Digest subscription emails

**Fix:** Align SPF and DKIM records for Resend with the `profitsentinel.com` domain. Options:
1. Update DNS to add Resend's SPF include and DKIM CNAME records
2. Or relax DMARC to `p=none` while setting up proper alignment
3. Or send from a subdomain like `mail.profitsentinel.com` with its own SPF/DKIM

---

## HIGH PRIORITY — Remove or Reword Immediately

### 3. POS System Integrations (Square, Lightspeed, Clover, Shopify)

**Claim:** Roadmap shows "In Progress" (was "Shipped", fixed during audit).

**Reality:** The UI for managing POS connections exists (`/dashboard/pos`) and backend routes work for creating/listing connections. However:
- No OAuth2 implementation for any POS system
- Sync always returns 0 rows (simulated)
- No API client libraries for any POS system
- Connections stored in-memory only

**Recommendation:** Hide the POS Connect page from the sidebar until real integrations exist, or add a clear "Coming Soon" banner on the page itself.

---

### 4. Multi-File Vendor Correlation

**Claim:** Roadmap shows "In Progress" (was "Shipped", fixed during audit).

**Reality:** No cross-file correlation logic exists. The system processes individual CSV files independently.

**Recommendation:** Remove from sidebar or add "Coming Soon" banner.

---

## MEDIUM PRIORITY — Build or Document as Planned

### 5. Settings / Configuration UI

**Claim:** DeadStockConfig exists in the backend (GET/PUT /api/v1/config) with 3 presets.

**Reality:** Backend API works, but no frontend UI exists. Users cannot configure thresholds.

**Recommendation:** Build `/dashboard/settings` with tier sliders, per-category overrides, and store type preset selector.

---

### 6. GDPR / CCPA Compliance Badges

**Claim:** Footer shows "GDPR Compliant" and "CCPA Compliant" badges.

**Missing for GDPR:**
- No cookie consent banner
- No data processing agreement (DPA) available
- No formal data export mechanism

**Missing for CCPA:**
- No "Do Not Sell My Personal Information" link
- No formal opt-out mechanism beyond email

**Recommendation:** Either remove compliance badges and use "Privacy First" language, or implement proper consent mechanisms.

---

### 7. Digest Cache Expiration (1-hour TTL)

**Impact:** The digest cache bridge (fix #13) uses a 1-hour TTL. After 1 hour, dashboard endpoints will return "CSV file not found" again until the user re-runs analysis. This is a UX issue — returning users who signed in 2 hours after their last analysis will see an error.

**Recommendation:**
- Increase TTL to 24 hours, or
- Persist the last digest to Supabase and load it on startup, or
- Show a "Run analysis to see results" CTA instead of an error when cache is empty

---

### 8. Claude API Key Invalid in Production

**Impact:** The `/uploads/suggest-mapping` endpoint (AI-assisted column mapping) falls back to defaults because the Anthropic API key returns 401. Mapping suggestions still work but are less accurate.

**Fix:** Update the `ANTHROPIC_API_KEY` secret in AWS Secrets Manager with a valid key:
```bash
aws secretsmanager update-secret \
  --secret-id "profitsentinel/anthropic-api-key" \
  --secret-string "sk-ant-..."
```

---

## LOW PRIORITY — Future Improvements

### 9. In-Memory Storage (Tasks, Sessions, Config, API Keys, Acknowledged Findings)

Multiple stores use in-memory dictionaries that are lost on container restart:
- Task delegation store
- Diagnostic session store
- User config store
- API key store
- Acknowledged findings set

**Recommendation:** Migrate to Supabase (TODO comments exist in code).

---

### 10. Rate Limiter is Per-Worker

In-memory rate limiter works per-worker, not cluster-wide. With N workers, effective limit is N x configured limit.

**Recommendation:** Migrate to Redis for production at scale.

---

### 11. MIGRATION_PLAN.md is Stale

M5/M6 status in MIGRATION_PLAN.md conflicts with README. Should be updated or removed.

---

### 12. Vulnerability Scanning Traffic

The production ALB receives steady automated vulnerability scanning attempts (phpunit path traversal, ThinkPHP RCE, PEAR cmd injection, Docker socket probing). All return 404 correctly, but:

**Recommendation:** Consider adding AWS WAF rules to block common scanner patterns and reduce log noise.

---

## Summary

| Gap | Severity | Effort | Action |
|-----|----------|--------|--------|
| analysis_synopses table missing | CRITICAL | 5 min | Run migration SQL |
| DMARC email deliverability | CRITICAL | 30 min | DNS records update |
| POS integrations (stubbed) | HIGH | 5 min | Hide page or add banner |
| Vendor correlation (missing) | HIGH | 5 min | Hide page or add banner |
| Settings UI (missing) | MEDIUM | 2-3 days | Build frontend |
| GDPR/CCPA badges | MEDIUM | 1-2 days | Remove or implement |
| Digest cache 1hr TTL | MEDIUM | 2 hours | Increase TTL or persist |
| Claude API key invalid | MEDIUM | 5 min | Update secret |
| In-memory stores | LOW | 1-2 weeks | Migrate to Supabase |
| Rate limiter scope | LOW | 1 day | Migrate to Redis |
| MIGRATION_PLAN.md stale | LOW | 15 min | Update or remove |
| Vulnerability scanning | LOW | 1 hour | Add WAF rules |
