# DATA_PERSISTENCE.md - What Data Remains After "Deletion"

## Security Audit Date: 2026-01-21
## Status: CRITICAL ISSUES IDENTIFIED

---

## EXECUTIVE SUMMARY

| Question | Answer | Risk Level |
|----------|--------|------------|
| After S3 auto-deletes file (1 hour), what remains? | **FILE VERSIONS PERSIST 90 DAYS** | **CRITICAL** |
| Are analysis results stored? | Yes, anonymized aggregates only | LOW |
| What gets logged? | Filenames, row counts, errors | MEDIUM |
| Are email reports retained? | By email provider ~30 days | HIGH |

---

## DETAILED BREAKDOWN

### 1. S3 BUCKET - UPLOADED FILES

#### Expected Behavior (What You Thought)
- File uploaded to S3
- Analysis runs
- File deleted after 60 seconds
- Data gone

#### Actual Behavior (Reality)
```
Time 0:00  - File uploaded to S3
Time 0:01  - Analysis runs, reads file
Time 1:01  - "Deletion" runs → Creates DELETE MARKER
             └─ Actual file data becomes "non-current version"
             └─ Non-current version visible to anyone with ListObjectVersions

Time 90 days - Lifecycle rule finally deletes non-current version
```

#### Evidence in Code

**File: `infrastructure/modules/s3/main.tf:23-28`**
```hcl
resource "aws_s3_bucket_versioning" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  versioning_configuration {
    status = "Enabled"  # <-- VERSIONING IS ON
  }
}
```

**File: `infrastructure/modules/s3/main.tf:39-50`**
```hcl
resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    noncurrent_version_expiration {
      noncurrent_days = 90  # <-- 90 DAYS RETENTION!
    }
  }
}
```

**File: `apps/api/src/services/anonymization.py:256-278`**
```python
async def cleanup_s3_file(self, s3_client, bucket_name, key, delay_seconds=0):
    # ... delay logic ...
    s3_client.delete_object(Bucket=bucket_name, Key=key)  # Only creates delete marker!
```

#### REMEDIATION REQUIRED

1. **Immediate**: Run `destroy_test_data.py` to permanently delete all versions
2. **Short-term**: Update lifecycle rule to 1 day retention
3. **Long-term**: Consider disabling versioning for user uploads

---

### 2. SUPABASE DATABASE - ANALYTICS

#### What's Stored

**Table: `email_signups`** (File: `supabase/migrations/001_create_email_signups.sql`)

| Column | Content | PII Risk |
|--------|---------|----------|
| email | User email address | YES |
| company_name | Optional company name | YES |
| pos_system | "Paladin", "Square", etc. | NO |
| ip_address | User IP | YES |
| utm_* | Marketing attribution | NO |
| created_at | Timestamp | NO |

**Table: `analysis_synopses`** (File: `supabase/migrations/002_create_analysis_synopses.sql`)

| Column | Content | PII Risk |
|--------|---------|----------|
| file_hash | SHA256 of file | NO (one-way hash) |
| file_row_count | Number of rows | NO |
| detection_counts | {"low_stock": 42, ...} | NO |
| total_impact_estimate | Dollar amounts | NO |
| dataset_stats | {"avg_margin": 0.32, ...} | NO |

**SKUs and actual item data are NOT stored in Supabase.**

#### Evidence in Code

**File: `apps/api/src/services/anonymization.py:165-222`**
```python
def extract_aggregated_stats(self, results):
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "files_analyzed": len(results),
        "total_rows": 0,
        "total_items_flagged": 0,
        # ... only counts and averages, no item-level data
    }
```

#### RISK ASSESSMENT: LOW

Supabase contains:
- Email addresses (can be purged if needed)
- Aggregated statistics (no PII)
- No SKUs, no product names, no actual file content

---

### 3. CLOUDWATCH LOGS

#### What's Logged

Based on code review (`apps/api/src/routes/` and `apps/api/src/services/`):

| Log Type | Example | Risk |
|----------|---------|------|
| File keys | `"Loading file: anonymous/abc123-inventory.csv"` | MEDIUM |
| Row counts | `"Loaded DataFrame (15000 rows)"` | LOW |
| Column names | `"Original columns: ['SKU', 'Description', ...]"` | LOW |
| Errors | `"Analysis failed: KeyError: 'quantity'"` | LOW |
| Mapping warnings | `"Sanitized filename to: invoice_data.csv"` | LOW |

**SKUs and actual data values are NOT logged** (confirmed via code review).

#### Evidence in Code

**File: `apps/api/src/routes/uploads.py:151-154`**
```python
logger.info(
    f"Generated {len(presigned_urls)} presigned URLs for "
    f"{'user ' + user_id if user_id else 'anonymous user'}"
)
```

**File: `apps/api/src/routes/analysis.py:387-391`**
```python
logger.info(
    f"Loaded DataFrame ({len(df)} rows, {len(df.columns)} columns) "
    f"in {time.time() - load_start:.2f}s"
)
```

#### CURRENT RETENTION: NEVER EXPIRE (DEFAULT)

#### REMEDIATION REQUIRED

1. Set retention policy to 7 days
2. Review logs for any test data filenames
3. Delete log streams from test period

---

### 4. EMAIL REPORTS

#### What's Sent

When a user requests their full report, the email contains:

| Content | Example | Risk |
|---------|---------|------|
| SKU identifiers | "SKU-12345", "UPC-987654321" | **HIGH** |
| Item descriptions | "DeWalt 20V Drill" | **HIGH** |
| Quantities | "QOH: 15, Sold: 3" | **HIGH** |
| Cost/Price | "$45.00 cost, $89.99 price" | **HIGH** |
| Recommendations | "Consider clearance pricing" | LOW |

#### Evidence in Code

**File: `apps/api/src/services/email.py:386-414`**
```python
# Full item details sent in email
for detail in item_details[:5]:
    sku = detail.get("sku", "Unknown")
    qty = detail.get("quantity", 0)
    cost = detail.get("cost", 0)
    # ... all this goes in the email
```

#### RETENTION

| Provider | Retention | Access |
|----------|-----------|--------|
| Resend | ~30 days | Via Resend dashboard |
| SendGrid | ~30 days | Via SendGrid dashboard |

#### REMEDIATION

If you sent test reports:
1. Check your email provider dashboard for sent messages
2. Delete/purge messages containing test data
3. Consider if test email addresses should be purged from email_signups table

---

### 5. RDS AURORA (POTENTIALLY UNUSED)

Based on code analysis, the application uses **Supabase** for database operations, not the Aurora cluster defined in Terraform.

**Evidence:**
- `anonymization.py` uses `self.supabase_client`
- No direct PostgreSQL connection strings in app code
- Aurora appears to be infrastructure "in waiting"

#### CHECK IF USED

Run at home:
```bash
# Check if Aurora has any tables
aws rds describe-db-cluster-endpoints --db-cluster-identifier profitsentinel-dev-cluster

# Or connect directly and check
psql -h <cluster-endpoint> -U adminuser -d profitsentinel -c "\dt"
```

If Aurora is unused, there's no contamination risk there.

---

## CONTAMINATION TIMELINE

If you uploaded test data, here's where it might still exist:

| Time Since Upload | S3 File | S3 Versions | CloudWatch | Email | Supabase |
|-------------------|---------|-------------|------------|-------|----------|
| 1 minute | EXISTS | N/A | Logged | N/A | Stats saved |
| 1 hour | DELETED* | **EXISTS** | Logged | If sent | Stats exist |
| 1 day | DELETED* | **EXISTS** | Logged | If sent | Stats exist |
| 30 days | DELETED* | **EXISTS** | Logged | Maybe | Stats exist |
| 90 days | DELETED* | DELETED | Maybe** | Maybe | Stats exist |

\* "Deleted" means delete marker exists, version still recoverable

\*\* Depends on log retention setting (currently: never expire)

---

## COMPLETE PURGE CHECKLIST

To ensure ALL test data is removed:

- [ ] **S3**: Delete all object versions (not just current objects)
- [ ] **S3**: Delete all delete markers
- [ ] **CloudWatch**: Delete log streams from test period
- [ ] **CloudWatch**: Set 7-day retention going forward
- [ ] **Email Provider**: Delete sent messages containing test data
- [ ] **Supabase email_signups**: Purge test email addresses if any
- [ ] **Supabase analysis_synopses**: These are anonymized, but can purge if concerned
- [ ] **RDS**: Check if used; if so, check for snapshots

Use the scripts in `scripts/` directory for automated cleanup.

---

## POST-CLEANUP VERIFICATION

After running cleanup scripts:

1. Run `verify_cleanup.py`
2. Manually check:
   - AWS S3 console → bucket → Show versions
   - CloudWatch Logs console → log groups
   - Email provider dashboard → sent messages
   - Supabase dashboard → tables
