# SECURITY_ISSUES.md - Complete Security Findings

## Security Audit Date: 2026-01-21
## Auditor: Claude Code (Automated Analysis)

---

## SEVERITY SUMMARY

| Severity | Count | Immediate Action Required |
|----------|-------|--------------------------|
| CRITICAL | 1 | YES - Before any customer use |
| HIGH | 2 | YES - Before Stripe integration |
| MEDIUM | 3 | Fix before public launch |
| LOW | 2 | Address when convenient |
| INFORMATIONAL | 3 | Best practices |

---

## CRITICAL ISSUES (Fix Immediately)

### CRIT-001: S3 Versioning Retains "Deleted" User Data for 90 Days

**Location:** `infrastructure/modules/s3/main.tf:23-50`

**Description:**
S3 bucket versioning is enabled with a 90-day retention policy for non-current versions. When files are "deleted" via the API, only a delete marker is created. The actual file data persists as a recoverable version for 90 days.

**Impact:**
- User POS data (potentially containing business-sensitive information) persists long after promised deletion
- Violates privacy policy claims ("data deleted after processing")
- Legal/compliance exposure if data contains PII
- Your test data with potential fraud evidence is still in S3

**Evidence:**
```hcl
# infrastructure/modules/s3/main.tf
resource "aws_s3_bucket_versioning" "uploads" {
  versioning_configuration {
    status = "Enabled"  # Problem: versioning retains "deleted" files
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  rule {
    noncurrent_version_expiration {
      noncurrent_days = 90  # Problem: 90-day retention
    }
  }
}
```

**Remediation:**
1. **Immediate:** Run `scripts/destroy_test_data.py` to purge all versions
2. **Infrastructure Fix:**
   ```hcl
   # Option A: Reduce version retention to 1 day
   noncurrent_version_expiration {
     noncurrent_days = 1
   }

   # Option B: Disable versioning entirely (recommended for user uploads)
   versioning_configuration {
     status = "Disabled"
   }
   ```
3. **Application Fix:** Update deletion code to explicitly delete versions:
   ```python
   # In anonymization.py cleanup_s3_file()
   versions = s3_client.list_object_versions(Bucket=bucket_name, Prefix=key)
   for version in versions.get('Versions', []):
       s3_client.delete_object(
           Bucket=bucket_name,
           Key=key,
           VersionId=version['VersionId']
       )
   ```

**Timeline:** Must fix before any customer data is processed

---

## HIGH SEVERITY ISSUES (Fix Before Stripe)

### HIGH-001: CloudWatch Logs Have No Retention Policy

**Location:** `infrastructure/modules/ecs/main.tf:68-74`

**Description:**
CloudWatch log groups are created without a retention policy, meaning logs are retained indefinitely. Logs contain filenames and potentially identifying information.

**Impact:**
- Unlimited log storage costs over time
- Filenames in logs could identify customers/uploads
- Compliance risk if logs contain any PII

**Evidence:**
```hcl
resource "aws_cloudwatch_log_group" "ecs" {
  name = "/ecs/${var.name_prefix}"
  # No retention_in_days specified = infinite retention
}
```

**Remediation:**
```hcl
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.name_prefix}"
  retention_in_days = 7  # Add retention policy
}
```

Or via AWS CLI:
```bash
aws logs put-retention-policy \
  --log-group-name /ecs/profitsentinel-dev \
  --retention-in-days 7
```

**Timeline:** Fix before Stripe integration

---

### HIGH-002: Email Reports Contain Full SKU Data

**Location:** `apps/api/src/services/email.py:386-414`

**Description:**
Email reports sent to users contain full item-level data including SKUs, quantities, costs, and prices. This data is retained by email providers (Resend/SendGrid) for ~30 days.

**Impact:**
- Sensitive business data retained by third parties
- No control over data once email is sent
- User's email could be compromised, exposing their business data

**Evidence:**
```python
# email.py - full item details in email
for detail in item_details[:5]:
    sku = detail.get("sku", "Unknown")
    qty = detail.get("quantity", 0)
    cost = detail.get("cost", 0)
    # All sent to user's email
```

**Remediation Options:**
1. **Reduce data in emails:** Send summary only, provide secure link to full report
2. **Implement time-limited secure reports:** Generate temporary signed URLs
3. **Document in privacy policy:** Clearly state that emailed reports contain business data
4. **Consider self-hosted email:** Run own SMTP to control retention

**Timeline:** Document risk before Stripe; implement fix before scaling

---

## MEDIUM SEVERITY ISSUES

### MED-001: RDS skip_final_snapshot Enabled

**Location:** `infrastructure/modules/rds/main.tf:62`

**Description:**
RDS cluster has `skip_final_snapshot = true`, meaning no backup is created if the database is deleted.

**Impact:**
- Data loss risk if database accidentally deleted
- No recovery point for incident response

**Evidence:**
```hcl
resource "aws_rds_cluster" "aurora" {
  skip_final_snapshot = true  # Dev setting, dangerous in prod
  deletion_protection = false  # Also concerning
}
```

**Remediation:**
```hcl
# For production
skip_final_snapshot  = false
final_snapshot_identifier = "${var.name_prefix}-final-snapshot"
deletion_protection = true
```

**Timeline:** Fix before production deployment

---

### MED-002: No WAF/Rate Limiting at ALB Level

**Location:** `infrastructure/modules/alb/main.tf`

**Description:**
No AWS WAF or ALB-level rate limiting configured. Application-level rate limiting exists (slowapi) but can be bypassed.

**Impact:**
- DDoS vulnerability
- Application-level rate limiting less effective against distributed attacks
- No protection against common web attacks (SQLi, XSS)

**Remediation:**
1. Add AWS WAF with managed rule sets
2. Configure rate-based rules at WAF level
3. Enable AWS Shield Standard (free)

**Timeline:** Recommended before public launch

---

### MED-003: Anonymous Uploads Without Verification

**Location:** `apps/api/src/routes/uploads.py:93`

**Description:**
The system allows anonymous uploads without any form of verification (CAPTCHA, email verification). This enables abuse potential.

**Impact:**
- Resource abuse by automated scripts
- Storage costs from malicious uploads
- Potential for processing malicious files

**Evidence:**
```python
@router.post("/presign")
async def presign_upload(
    user_id: str | None = Depends(get_current_user),  # Optional - anonymous allowed
):
```

**Remediation:**
1. Implement CAPTCHA for anonymous uploads
2. Require email verification before processing
3. Implement fingerprinting/browser verification
4. Lower rate limits for anonymous users

**Timeline:** Address before scaling

---

## LOW SEVERITY ISSUES

### LOW-001: Debug Mode Potentially Exposable

**Location:** `apps/api/src/config.py:22`

**Description:**
Debug flag is configurable via environment variable. If accidentally set to true in production, could expose sensitive information.

**Evidence:**
```python
debug: bool = Field(default=False)
```

**Remediation:**
- Add explicit production check
- Log warning if debug enabled in non-dev environment
- Consider removing debug mode entirely for production builds

---

### LOW-002: CORS Origins Include localhost

**Location:** `apps/api/src/config.py:75-89`

**Description:**
CORS configuration includes localhost origins, which should be removed in production.

**Evidence:**
```python
cors_origins: list[str] = Field(
    default=[
        "https://profitsentinel.com",
        "http://localhost:3000",  # Should not be in prod
        "http://localhost:5173",  # Should not be in prod
    ]
)
```

**Remediation:**
Use environment-specific CORS configuration:
```python
cors_origins: list[str] = Field(default_factory=lambda: [
    "https://profitsentinel.com",
    "https://www.profitsentinel.com",
] if os.getenv("ENV") == "production" else [
    "http://localhost:3000",
    # ... dev origins
])
```

---

## INFORMATIONAL

### INFO-001: Secrets Management Best Practices

**Status:** GOOD

- AWS Secrets Manager used for sensitive values
- No hardcoded credentials found
- .env properly gitignored
- GitHub Actions uses repository secrets

**Recommendation:** Consider rotating credentials quarterly.

---

### INFO-002: Network Security Configuration

**Status:** GOOD

- VPC with public/private subnet separation
- RDS in private subnets only
- ECS tasks in private subnets
- NAT Gateway for outbound traffic
- Security groups properly scoped

---

### INFO-003: Encryption Configuration

**Status:** GOOD

- S3 server-side encryption (AES256)
- RDS encryption at rest enabled
- HTTPS enforced (via ALB + ACM certificate)

See ENCRYPTION_AUDIT.md for details.

---

## PRE-STRIPE SECURITY REQUIREMENTS

Before integrating Stripe for payments, ensure:

| Requirement | Status | Action |
|-------------|--------|--------|
| No user data retained beyond stated policy | FAIL | Fix S3 versioning |
| Logs don't contain PII | NEEDS REVIEW | Set retention, audit logs |
| No hardcoded secrets | PASS | - |
| HTTPS everywhere | PASS | - |
| Database encrypted | PASS | - |
| IAM least privilege | PASS | - |
| Rate limiting | PARTIAL | Add WAF |
| Input validation | PASS | - |
| Error messages don't leak info | PASS | - |

---

## AUDIT METHODOLOGY

This audit was performed via:
1. Static code analysis of all source files
2. Infrastructure code review (Terraform)
3. Pattern matching for secrets/credentials
4. Data flow analysis
5. Configuration review

**Limitations:**
- No runtime testing performed
- No penetration testing
- No access to actual AWS console
- No access to Supabase dashboard

**Recommended Follow-up:**
1. Run `find_test_data.py` to verify findings
2. Manual review of AWS console for resources
3. Consider third-party security audit before launch
