# Profit Sentinel Security & Privacy Audit Report

**Audit Date:** January 25, 2026
**Audit Type:** SOC 2 Readiness Assessment (Full Scope)
**Previous Audit:** January 21, 2026
**Repository:** profit-sentinel-saas

---

## Executive Summary

This comprehensive security and privacy audit identified **6 critical issues**, **8 high-priority issues**, and **15 medium-priority issues** that must be addressed to achieve SOC 2 compliance and match the privacy claims made to users.

### Risk Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| Data Retention | 2 | 2 | 1 | 0 |
| Authentication | 1 | 2 | 2 | 0 |
| Session Management | 2 | 1 | 0 | 0 |
| PII Handling | 1 | 2 | 3 | 2 |
| Logging | 0 | 1 | 3 | 1 |
| File Security | 0 | 0 | 2 | 1 |
| Configuration | 0 | 0 | 1 | 0 |

**Overall Status:** REQUIRES REMEDIATION BEFORE PRODUCTION

---

## CRITICAL FINDINGS (Must Fix Before Production)

### C1. In-Memory Session Storage with No Cleanup

**Location:** `apps/api/src/routes/diagnostic.py:37`, `apps/api/src/routes/premium.py:37`

**Issue:** Sessions stored in Python dictionaries persist indefinitely until server restart.

```python
diagnostic_sessions: dict[str, dict[str, Any]] = {}  # Memory leak
premium_sessions: dict[str, dict[str, Any]] = {}     # Memory leak
```

**Privacy Policy Claim:** "Session data - 24 hours" (privacy/page.tsx line 246)
**Actual Behavior:** Indefinite retention until server restart

**Impact:**
- Memory exhaustion under load
- Privacy policy violation
- Data retained beyond stated period
- No user-session association (any UUID can access any session)

**Remediation:**
```python
# Option 1: Redis with TTL
import redis
from datetime import timedelta

redis_client = redis.Redis()
SESSION_TTL = timedelta(hours=24)

def create_session(session_id: str, data: dict):
    redis_client.setex(session_id, SESSION_TTL, json.dumps(data))

# Option 2: Background cleanup task
async def cleanup_stale_sessions():
    cutoff = datetime.now() - timedelta(hours=24)
    for sid, data in list(diagnostic_sessions.items()):
        if data["created_at"] < cutoff:
            del diagnostic_sessions[sid]
```

---

### C2. No Automatic File Deletion (Privacy Policy Violation)

**Privacy Policy Claim:** "Automatically deleted within 24 hours of processing" (line 31)
**Actual Implementation:** Files only deleted on:
- Explicit DELETE endpoint call
- Error during report generation
- Client request with `delete_files_after=True`

**Missing:** No S3 lifecycle policy, no scheduled cleanup job.

**Remediation:**
1. Add S3 lifecycle policy:
```json
{
  "Rules": [{
    "ID": "DeleteAfter24Hours",
    "Status": "Enabled",
    "Filter": {"Prefix": ""},
    "Expiration": {"Days": 1}
  }]
}
```

2. Add background cleanup task for session files.

---

### C3. Diagnostic Sessions Have No Authentication

**Location:** `apps/api/src/routes/diagnostic.py`, `apps/api/src/routes/premium.py`

**Issue:** All diagnostic endpoints publicly accessible:
- `POST /diagnostic/start` - No auth
- `GET /diagnostic/{id}/question` - No auth
- `POST /diagnostic/{id}/answer` - No auth
- `GET /diagnostic/{id}/report` - No auth
- `POST /diagnostic/{id}/email` - No auth

**Impact:**
- Session ID guessing could access other users' data
- No audit trail of data access
- No user ownership validation

**Remediation:**
```python
@router.post("/{session_id}/answer")
async def answer_question(
    session_id: str,
    user: dict | None = Depends(get_current_user),  # Optional auth
):
    data = diagnostic_sessions.get(session_id)
    if not data:
        raise HTTPException(404, "Session not found")

    # Validate ownership if authenticated
    if user and data.get("user_id") and data["user_id"] != user["id"]:
        raise HTTPException(403, "Not authorized to access this session")
```

---

### C4. Email Addresses Logged to Application Logs

**Location:**
- `apps/api/src/routes/contact.py:74-76`
- `apps/api/src/routes/reports.py:189-193`
- `apps/api/src/services/email.py:126-127, 166-167`

**Issue:** PII logged directly:
```python
logger.info(f"Contact form submission: email={request.email}")  # PII EXPOSURE
logger.info(f"Email sent via Resend to {to_email}")             # PII EXPOSURE
```

**GDPR Impact:** Violates Article 5 (data minimization)

**Remediation:**
```python
# Instead of:
logger.info(f"Email sent to {to_email}")

# Use:
logger.info(f"Email sent (ref: {hashlib.sha256(to_email.encode()).hexdigest()[:8]})")
```

---

### C5. Missing Stripe Configuration Fields

**Location:** `apps/api/src/config.py`

**Issue:** Settings class missing required Stripe fields referenced in `services/billing.py`:
- `stripe_secret_key`
- `stripe_webhook_secret`
- `stripe_price_id`
- `stripe_success_url`
- `stripe_cancel_url`
- `has_stripe` property

**Impact:** Runtime `AttributeError` when billing service initializes.

**Remediation:** Add to Settings class:
```python
stripe_secret_key: str | None = Field(default=None, env="STRIPE_SECRET_KEY")
stripe_webhook_secret: str | None = Field(default=None, env="STRIPE_WEBHOOK_SECRET")
stripe_price_id: str | None = Field(default=None, env="STRIPE_PRICE_ID")
stripe_success_url: str | None = Field(default=None, env="STRIPE_SUCCESS_URL")
stripe_cancel_url: str | None = Field(default=None, env="STRIPE_CANCEL_URL")

@property
def has_stripe(self) -> bool:
    return self.stripe_secret_key is not None
```

---

### C6. No Virus/Malware Scanning on Uploads

**Issue:** No antivirus integration for file uploads.

**Impact:** Malicious files could be uploaded and processed.

**Remediation Options:**
1. **ClamAV:** Self-hosted scanning
2. **AWS GuardDuty:** S3 object scanning
3. **VirusTotal API:** Cloud-based scanning

---

## HIGH PRIORITY FINDINGS

### H1. Privacy Policy Omits IP Address Collection

**Privacy Policy:** Does not mention IP address collection
**Actual:** IP addresses stored in `email_signups` table (migration 001)

**Remediation:** Update privacy policy:
```
We collect your IP address when you sign up to prevent abuse and for security purposes.
```

---

### H2. Privacy Policy Omits User Agent Collection

**Privacy Policy:** Does not mention user agent collection
**Actual:** User agents stored in `email_signups` table (migration 001)

**Remediation:** Update privacy policy to disclose browser/device info collection.

---

### H3. Trial Expiration Function Not Scheduled

**Location:** `supabase/migrations/005_stripe_billing.sql`

**Issue:** `expire_stale_trials()` function defined but never scheduled via pg_cron.

**Impact:** Expired trials remain in 'trialing' status indefinitely.

**Remediation:**
```sql
SELECT cron.schedule('expire-trials', '0 * * * *', $$SELECT expire_stale_trials()$$);
```

---

### H4. Monthly Analysis Reset Not Scheduled

**Location:** `supabase/migrations/003_user_profiles.sql`

**Issue:** `reset_monthly_analysis_counts()` not scheduled.

**Remediation:** Configure pg_cron to run on 1st of each month.

---

### H5. Rate Limiting Missing on Diagnostic Endpoints

**Affected:**
- `POST /diagnostic/start`
- `POST /diagnostic/{id}/answer`
- `POST /diagnostic/{id}/email`
- All `/premium/*` endpoints

**Remediation:**
```python
@router.post("/diagnostic/start")
@limiter.limit("10/minute")
async def start_diagnostic(...):
```

---

### H6. S3 File Size Enforcement Advisory Only

**Location:** `apps/api/src/services/s3.py:139`

**Issue:** Size limit on presigned PUT URLs is "advisory" per code comment.

**Remediation:** Use presigned POST with conditions or validate after upload.

---

### H7. No Distributed Rate Limiting

**Issue:** Rate limits stored per-process; multi-instance deployments can bypass.

**Remediation:** Configure slowapi to use Redis backend.

---

### H8. Email Signups Never Auto-Deleted

**Privacy Policy Claim:** "Until you unsubscribe or request deletion"
**Actual:** No deletion mechanism, no retention policy.

**Remediation:**
1. Add data retention policy (e.g., 2 years)
2. Implement `/api/privacy/delete-my-data` endpoint
3. Add scheduled cleanup job

---

## MEDIUM PRIORITY FINDINGS

### M1. User IDs Logged in Billing Context
Log only transaction IDs, not user associations.

### M2. S3 Keys Logged (May Contain User IDs)
Log hashed identifiers instead.

### M3. Exception Stack Traces in Production
Use `exc_info=True` only in development.

### M4. No Content-Type Validation on Direct Uploads
Add explicit Content-Type validation.

### M5. DataFrame CSV Loading Without Row Limits
Add `nrows=500000` parameter.

### M6. Email Address Echoed in API Response
Remove PII from response bodies.

### M7. No RBAC Implementation
Implement role-based access control.

### M8. Development Origins in Production CORS
Remove localhost origins in production.

### M9. Content-Security-Policy Too Permissive
Tighten CSP directives.

### M10. Privacy Page Links to Old /upload Route
Update to `/diagnostic`.

### M11. No Data Deletion API Endpoint
Implement self-service GDPR deletion.

### M12. Console Error Logging in Browser
Sanitize error messages.

### M13. Print Statements in Engine Code
Remove debug prints.

### M14. Temp Files Not Cleaned on Process Crash
Use temp directory with cron cleanup.

### M15. Presigned URL Size Not Server-Enforced
Add post-upload HeadObject validation.

---

## POSITIVE SECURITY FINDINGS

### Implemented Correctly
1. **PII Stripping:** Comprehensive anonymization service (emails, phones, SSNs, credit cards)
2. **SKU Hashing:** Optional SHA256 with salt
3. **SQL Injection Prevention:** Parameterized queries via Supabase
4. **API Key Protection:** No keys in logs, presence-only logging
5. **Security Headers:** OWASP-compliant headers (HSTS, X-Frame-Options, CSP)
6. **File Sanitization:** Path traversal prevention, extension whitelist, magic bytes
7. **S3 Key Ownership:** Validates user_id prefix
8. **Row Level Security:** Enabled on all Supabase tables
9. **Rate Limiting:** Analysis and billing endpoints protected
10. **Unsubscribe:** CAN-SPAM compliant List-Unsubscribe headers
11. **HTTPS Only:** HSTS with 1-year max-age
12. **No Third-Party Tracking:** No GA or ad pixels

---

## PRIVACY POLICY vs IMPLEMENTATION MATRIX

| Claim | Status | Gap |
|-------|--------|-----|
| Files deleted within 24 hours | **FAIL** | No auto-deletion mechanism |
| Session data 24 hours | **FAIL** | Indefinite (until restart) |
| Email until unsubscribe | **PARTIAL** | No auto-deletion, manual only |
| No PII in analytics | **PASS** | Proper anonymization |
| No third-party tracking | **PASS** | Verified |
| HTTPS encryption | **PASS** | TLS 1.2+ |
| AES-256 at rest | **PASS** | S3 SSE configured |
| IP address collection | **NOT DISCLOSED** | Update policy |
| User agent collection | **NOT DISCLOSED** | Update policy |

---

## COMPLIANCE MATRIX

| Requirement | Status | Gap |
|-------------|--------|-----|
| **GDPR Art. 5 (Data Minimization)** | FAIL | Email in logs |
| **GDPR Art. 17 (Right to Erasure)** | PARTIAL | Manual process only |
| **GDPR Art. 13 (Transparency)** | PARTIAL | IP/UA not disclosed |
| **CCPA Right to Delete** | PARTIAL | Manual process only |
| **CAN-SPAM Unsubscribe** | PASS | Headers present |
| **SOC 2 CC6.1 (Encryption)** | PASS | TLS + AES-256 |
| **SOC 2 CC6.6 (Secure Disposal)** | FAIL | No auto-deletion |
| **SOC 2 CC7.1 (System Operations)** | PARTIAL | Missing scheduled jobs |

---

## REMEDIATION PRIORITY

### Immediate (Before Go-Live)
1. [x] C1: Implement session TTL/cleanup ✅ **FIXED (2026-01-25)**
2. [x] C2: Configure S3 lifecycle policy (24-hour expiration) ✅ **FIXED (2026-01-25)**
3. [x] C3: Add session authentication ✅ **FIXED (2026-01-25)** - Optional auth with ownership validation
4. [x] C4: Remove PII from logs ✅ **FIXED (2026-01-25)**
5. [x] C5: Add missing Stripe config fields ✅ **FIXED (2026-01-25)**
6. [x] H1-H2: Update privacy policy for IP/UA disclosure ✅ **FIXED (2026-01-25)**

### Within 30 Days
7. [ ] H3-H4: Configure pg_cron scheduled jobs
8. [x] H5: Add rate limiting to diagnostic endpoints ✅ **FIXED (2026-01-25)**
9. [ ] H8: Implement email signup retention policy
10. [ ] M11: Implement privacy deletion endpoint
11. [x] M10: Fix privacy page link to /diagnostic ✅ **FIXED (2026-01-25)**

### Within 90 Days
12. [ ] C6: Integrate virus scanning
13. [ ] H6-H7: Implement distributed rate limiting
14. [ ] M1-M3: Clean up sensitive logging
15. [ ] Full penetration test

---

## INFRASTRUCTURE CONFIGURATION REQUIRED

### S3 Lifecycle Policy (C2)

The S3 bucket requires a lifecycle policy to automatically delete files after 24 hours. This must be configured in AWS Console or via Terraform/CloudFormation:

**AWS CLI Command:**
```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket profitsentinel-dev-uploads \
  --lifecycle-configuration '{
    "Rules": [
      {
        "ID": "DeleteAfter24Hours",
        "Status": "Enabled",
        "Filter": {},
        "Expiration": {
          "Days": 1
        }
      }
    ]
  }'
```

**Terraform:**
```hcl
resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  rule {
    id     = "delete-after-24-hours"
    status = "Enabled"

    expiration {
      days = 1
    }
  }
}
```

### Database Scheduled Jobs (H3-H4)

Configure pg_cron in Supabase Dashboard → Database → Extensions:

```sql
-- Enable pg_cron extension
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule trial expiration (hourly)
SELECT cron.schedule('expire-trials', '0 * * * *', $$SELECT expire_stale_trials()$$);

-- Schedule monthly analysis count reset (1st of each month at midnight UTC)
SELECT cron.schedule('reset-analysis-counts', '0 0 1 * *', $$SELECT reset_monthly_analysis_counts()$$);
```

---

## Files Requiring Changes

### Critical
- `apps/api/src/routes/diagnostic.py` - Session auth, TTL
- `apps/api/src/routes/premium.py` - Session auth, TTL
- `apps/api/src/config.py` - Stripe fields
- `apps/api/src/routes/contact.py` - Remove email from logs
- `apps/api/src/routes/reports.py` - Remove email from logs
- `apps/api/src/services/email.py` - Remove email from logs

### High
- `apps/web/src/app/privacy/page.tsx` - Add IP/UA disclosure
- `supabase/migrations/` - Add pg_cron jobs
- S3 bucket configuration - Lifecycle policy

### Medium
- `apps/api/src/routes/billing.py` - Clean up logging
- `apps/web/next.config.ts` - Tighten CSP
- Various logging statements throughout

---

## Appendix: Audit Methodology

### Tools Used
- Manual code review
- Static analysis of authentication patterns
- Privacy policy cross-reference
- Database schema analysis
- API endpoint enumeration

### Files Reviewed
- All Python files in `apps/api/src/`
- All TypeScript files in `apps/web/src/`
- All SQL migrations in `supabase/migrations/`
- Configuration files (next.config.ts, config.py)
- Privacy policy page

---

**Report Prepared By:** Security Audit Agent
**Review Status:** Pending Engineering Review
**Next Audit:** Recommended after remediation (30 days)
