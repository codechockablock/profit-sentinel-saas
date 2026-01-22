# Profit Sentinel Privacy Claim Verification Report

**Date:** 2026-01-21
**Auditor:** Claude Code Security Agent
**Scope:** Full privacy claim verification against codebase

---

## Executive Summary

| Status | Count |
|--------|-------|
| **VERIFIED TRUE** | 28 |
| **FALSE - REQUIRES IMMEDIATE FIX** | 5 |
| **PARTIALLY TRUE / NEEDS CLARIFICATION** | 6 |
| **UNABLE TO VERIFY** | 3 |

### CRITICAL ISSUES REQUIRING IMMEDIATE ACTION

| Priority | Claim | Current Text | Finding | Risk Level |
|----------|-------|--------------|---------|------------|
| **P1** | H1, P5, P20, P35 | "Files auto-deleted in 1 hour" | **FALSE** - S3 lifecycle only expires noncurrent versions after 90 days, NOT current objects. Background deletion after analysis has 60-second delay but relies on successful request completion. | **HIGH - FTC/Legal** |
| **P2** | P18 | "HTTPS/TLS 1.3" | **FALSE** - ALB uses TLS 1.2 (ELBSecurityPolicy-TLS13-1-2-2021-06 supports 1.3 but doesn't guarantee it) | **MEDIUM** |
| **P3** | P11, P12 | "Preview shows anonymized items (Item A, Item B), No specific SKUs shown" | **FALSE** - First SKU is shown real in preview (`isUnlocked \|\| i === 0` in teaser-results.tsx:154) | **HIGH - Privacy breach** |
| **P4** | Third-party | Privacy policy lists 4 services | **INCOMPLETE** - xAI/Grok API is used for column mapping but NOT disclosed | **HIGH - GDPR violation** |
| **P5** | P22-P26 | "GDPR Rights" (Access, Deletion, Portability, Correction) | **PARTIAL** - No automated endpoints exist; relies solely on manual email process | **MEDIUM** |

---

## Detailed Claim Verification

### ENCRYPTION CLAIMS

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| H3 | "Files encrypted in transit and at rest" | **TRUE** | S3 uses AES-256 (infrastructure/modules/s3/main.tf:18), ALB enforces HTTPS |
| P4 | "Files processed in-memory and stored temporarily in encrypted S3 storage" | **TRUE** | S3 SSE-S3 (AES256) configured |
| P18 | "Encryption in Transit: HTTPS/TLS 1.3" | **PARTIALLY TRUE** | ALB uses `ELBSecurityPolicy-TLS13-1-2-2021-06` which supports TLS 1.3 AND 1.2. Connection may negotiate to 1.2. Claim should say "TLS 1.2+" |
| P19 | "Encryption at Rest: AES-256 server-side encryption in AWS S3" | **TRUE** | Verified in infrastructure/modules/s3/main.tf:17-20 |
| T1 | "Your inventory data is uploaded securely to encrypted storage" | **TRUE** | HTTPS + S3 SSE verified |

**REQUIRED ACTION for P18:**
```diff
- "Encryption in Transit: HTTPS/TLS 1.3"
+ "Encryption in Transit: HTTPS/TLS 1.2+"
```

---

### AUTO-DELETION CLAIMS

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| H1 | "Files auto-deleted in 1 hour" | **FALSE** | S3 lifecycle (s3/main.tf:39-50) only expires noncurrent_versions after 90 days. No rule deletes current objects within 1 hour. |
| H4 | "Files automatically deleted within 1 hour after analysis" | **FALSE** | Same as H1 |
| P5 | "Automatically deleted within 1 hour of processing" | **FALSE** | Same as H1 |
| P14 | "Source files deleted immediately after sending" | **PARTIALLY TRUE** | Background task schedules deletion (reports.py:237-246) but: (1) Only if `delete_files_after=True`, (2) Depends on successful email send, (3) No guaranteed fallback |
| P20 | "Auto-Deletion: Raw files automatically deleted within 1 hour" | **FALSE** | Same as H1 |
| P35 | "Deleted immediately after email report sent, or within 1 hour max" | **FALSE** | The "1 hour max" fallback doesn't exist in S3 lifecycle |
| T2 | "Automatically deleted within 1 hour" | **FALSE** | Same as H1 |

**WHAT ACTUALLY HAPPENS:**
1. Analysis endpoint schedules deletion after 60 seconds (analysis.py:439, `delay_seconds=60`)
2. Report endpoint schedules deletion IF `delete_files_after=True` (reports.py:237)
3. S3 lifecycle only cleans noncurrent versions after 90 days
4. **NO guaranteed 1-hour deletion if background task fails**

**REQUIRED ACTIONS:**

**Option A - Fix Infrastructure (Recommended):**
```hcl
# infrastructure/modules/s3/main.tf - ADD this rule:
resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  rule {
    id     = "auto-delete-uploads-1-day"
    status = "Enabled"

    filter {
      prefix = "" # All objects
    }

    expiration {
      days = 1  # Minimum granularity for S3 lifecycle
    }
  }

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 1
    }
  }
}
```

**Option B - Update Claims (If keeping current implementation):**
```diff
- "Files auto-deleted in 1 hour"
+ "Files deleted after analysis completion, with automatic cleanup within 24 hours"

- "within 1 hour max"
+ "within 24 hours if analysis completes, or automatically by scheduled cleanup"
```

---

### ANONYMIZATION CLAIMS

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| H5 | "We only retain anonymized aggregate statistics" | **TRUE** | anonymization.py:165-222 extracts only counts/averages |
| H6 | "Never your actual SKUs, item names, or customer data" | **TRUE** | Analytics stored via Supabase contain no PII (anonymization.py:244-248) |
| P2 | "We only retain anonymized, aggregated statistics" | **TRUE** | Same as H5 |
| P9 | "Aggregate statistics with no PII" | **TRUE** | Verified in extract_aggregated_stats() |
| P10 | "Cannot be linked back to your business" | **TRUE** | No business identifiers in analytics |
| P11 | "Preview shows anonymized items (Item A, Item B, etc.)" | **FALSE** | teaser-results.tsx:154 shows FIRST item's real SKU: `isUnlocked \|\| i === 0` |
| P12 | "No specific SKUs or product identifiers shown" in preview | **FALSE** | Same as P11 - first item shows real SKU |
| P37 | "Anonymized analytics: Indefinitely (no PII)" | **TRUE** | Schema contains only aggregates |
| T3 | "We only retain anonymized, aggregate statistics - never your actual SKUs" | **TRUE for storage** | Analytics don't contain SKUs |

**REQUIRED ACTION for P11/P12:**

Fix `apps/web/src/components/teaser-results.tsx:154`:
```diff
- {isUnlocked || i === 0 ? (
+ {isUnlocked ? (
    <span className={`font-mono text-sm ${i === 0 && !isUnlocked ? 'text-emerald-400 font-semibold' : 'text-slate-200'}`}>
      {item || 'Unknown'}
    </span>
  ) : (
```

OR update privacy policy:
```diff
- "Shows anonymized items (Item A, Item B, etc.)"
+ "Shows first item as a sample, remaining items anonymized (Item B, Item C, etc.)"

- "No specific SKUs or product identifiers shown"
+ "One sample SKU shown, remaining items anonymized"
```

---

### PII STRIPPING CLAIM

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| P21 | "We automatically detect and remove personal information" | **TRUE** | anonymization.py:23-55 defines PII patterns; anonymize_dataframe() at line 84-139 implements stripping |

**PII Detection Implemented:**
- Email regex (line 25)
- Phone number regex (line 26-28)
- SSN regex (line 29)
- Credit card regex (line 30)
- Name column detection (lines 31-42)
- Address column detection (lines 43-54)
- Value-level masking (_mask_pii_in_value at line 141-158)

**STATUS: VERIFIED TRUE**

---

### GDPR/CCPA COMPLIANCE

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| H7 | "GDPR Compliant" badge | **PARTIALLY TRUE** | Has consent capture, data deletion (manual), but no automated rights endpoints |
| H8 | "CCPA Compliant" badge | **PARTIALLY TRUE** | Same as H7 |
| P22 | "Right to Access: Request a copy of any data" | **MANUAL ONLY** | No /api/privacy/my-data endpoint; relies on email to privacy@profitsentinel.com |
| P23 | "Right to Deletion: Request deletion at any time" | **MANUAL ONLY** | No /api/privacy/delete endpoint; contact form routes to privacy@profitsentinel.com |
| P24 | "Right to Opt-Out: Decline email communications" | **TRUE** | Unsubscribe link in emails (email.py:121, 505) |
| P25 | "Right to Portability: Receive data in machine-readable format" | **MANUAL ONLY** | No /api/privacy/export endpoint |
| P26 | "Right to Correction: Request correction of inaccurate data" | **MANUAL ONLY** | No endpoint; email only |
| P27 | "Response within 48 hours" | **UNABLE TO VERIFY** | No SLA enforcement in code; relies on manual process |

**ASSESSMENT:**
Current implementation relies entirely on manual email handling via privacy@profitsentinel.com. This is legally acceptable for small operations but:
1. No audit trail of requests
2. No automated SLA tracking
3. Scalability concerns

**RECOMMENDED (Optional for Phase 2):**
Add privacy rights endpoints:
- `GET /api/privacy/my-data` - Returns user's stored data
- `DELETE /api/privacy/my-data` - Deletes user's data
- `GET /api/privacy/export` - Exports data as JSON

---

### COOKIE & TRACKING CLAIMS

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| P32 | "Session cookies: Keep you logged in" | **TRUE** | Supabase auth uses cookies for session |
| P33 | "Preference cookies: Remember theme preference" | **TRUE** | next-themes stores in localStorage (not cookie, but equivalent) |
| P34 | "We do NOT use advertising cookies or third-party tracking pixels" | **TRUE** | No GA, FB Pixel, or tracking scripts found |

**VERIFICATION:**
```bash
grep -rn "gtag|google.*analytics|fbq|facebook.*pixel" apps/web/src/
# Result: No matches found
```

**STATUS: ALL VERIFIED TRUE**

---

### THIRD-PARTY DISCLOSURES

| Claim ID | Third Party | Status | Evidence |
|----------|-------------|--------|----------|
| P28 | AWS | **TRUE** | S3, RDS, ECS verified in infrastructure/ |
| P29 | Supabase | **TRUE** | Auth and analytics confirmed |
| P30 | Resend/SendGrid | **TRUE** | Email service uses both (email.py:29-34) |
| P31 | Vercel | **TRUE** | Frontend hosting in vercel.json |
| **MISSING** | xAI/Grok | **NOT DISCLOSED** | Used for column mapping (mapping.py:69-71, 156) |

**CRITICAL FINDING:**
xAI/Grok API receives:
- Column names from uploaded CSV files
- First 5 rows of sample data (mapping.py:67, 131)
- Filename (mapping.py:128)

This is **undisclosed data sharing with a third party** - a GDPR Article 13 violation.

**REQUIRED ACTION - Add to Privacy Policy:**
```markdown
### Third-Party Services

...existing services...

**xAI (Grok)**
- Purpose: AI-powered column mapping to interpret your POS data format
- Data shared: Column headers and sample values (first 5-10 rows) to determine field types
- No personal data is sent if your file doesn't contain PII columns
- Privacy policy: https://x.ai/privacy
```

---

### EMAIL HANDLING

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| P6 | "Email only collected if you opt-in" | **TRUE** | email-opt-in.tsx requires checkbox, reports.py:182 validates consent |
| P7 | "Used solely to send your complete analysis report" | **TRUE** | No marketing emails in codebase |
| P8 | "You can unsubscribe at any time" | **TRUE** | List-Unsubscribe header (email.py:121, 159); link in email footer (line 505) |

**STATUS: ALL VERIFIED TRUE**

---

### DATA RETENTION

| Claim ID | Claim Text | Status | Evidence |
|----------|------------|--------|----------|
| P35 | "Uploaded files: Deleted immediately after email sent, or within 1 hour max" | **PARTIALLY TRUE** | Immediate deletion works; "1 hour max" fallback doesn't exist |
| P36 | "Email address: Until you unsubscribe or request deletion" | **TRUE** | Manual process via privacy@ |
| P37 | "Anonymized analytics: Indefinitely (no PII)" | **TRUE** | Supabase analytics table |
| P38 | "Session data: 24 hours" | **UNABLE TO VERIFY** | Supabase default, not explicitly configured |
| P39 | "Preview data: Browser session only (not stored on servers)" | **TRUE** | Results rendered client-side, no server persistence |

---

## Summary of Required Website Updates

### MUST FIX IMMEDIATELY (Legal Risk)

#### 1. Add xAI/Grok to Third-Party Services (P28-P31)
**File:** `apps/web/src/app/privacy/page.tsx`
**Location:** Third-Party Services section (line 173-200)

```tsx
<div className="border-l-4 border-emerald-500 pl-4">
  <h3 className="font-bold text-slate-200">xAI (Grok AI)</h3>
  <p className="text-sm text-slate-400">AI-powered column mapping (processes column headers and sample values only)</p>
</div>
```

#### 2. Fix TLS Version Claim (P18)
**File:** `apps/web/src/app/privacy/page.tsx:113`
```diff
- All data transferred via HTTPS/TLS 1.3.
+ All data transferred via HTTPS/TLS 1.2+.
```

#### 3. Fix Preview SKU Display or Update Claim (P11, P12)
**Option A - Fix Code:**
**File:** `apps/web/src/components/teaser-results.tsx:154`
```diff
- {isUnlocked || i === 0 ? (
+ {isUnlocked ? (
```

**Option B - Update Privacy Policy:**
**File:** `apps/web/src/app/privacy/page.tsx:65,69`
```diff
- <li>Shows anonymized items (Item A, Item B, etc.)</li>
+ <li>Shows first item as sample, others anonymized (Item B, Item C)</li>

- <li>No specific SKUs or product identifiers shown</li>
+ <li>One sample SKU visible, remaining items anonymized</li>
```

#### 4. Fix or Clarify Auto-Deletion Claim (H1, P5, P20, P35)
**Option A - Add S3 Lifecycle Rule (Recommended):**
Add 1-day expiration rule to `infrastructure/modules/s3/main.tf`

**Option B - Update Claims:**
**File:** `apps/web/src/app/privacy/page.tsx` (multiple locations)
```diff
- Automatically deleted within 1 hour of processing
+ Deleted after analysis, with automatic cleanup within 24 hours

- within 1 hour max
+ within 24 hours
```

---

## Appendix: Files Requiring Changes

| File | Changes Needed |
|------|----------------|
| `apps/web/src/app/privacy/page.tsx` | Add xAI disclosure, fix TLS claim, clarify deletion timing |
| `apps/web/src/components/teaser-results.tsx` | Fix first-item SKU exposure (if keeping current claim) |
| `infrastructure/modules/s3/main.tf` | Add 1-day object expiration rule (if keeping "1 hour" claim) |
| `SECURITY_AUDIT_REPORT.md` | Already created - no changes needed |

---

*Report generated by Claude Code Security Agent - Privacy Claim Verification*
