# ARCHITECTURE_MAP.md - Profit Sentinel Data Flow Analysis

## Security Audit Date: 2026-01-21
## Status: CRITICAL ISSUES IDENTIFIED

---

## EXECUTIVE SUMMARY

### CRITICAL FINDING: S3 Versioning Retains "Deleted" Files

**Your 1-hour auto-delete does NOT actually delete files!**

S3 versioning is ENABLED, which means:
- When files are "deleted", only a delete marker is created
- The actual file data persists as a non-current version for **90 DAYS**
- Anyone with bucket access can recover "deleted" files

**Immediate Action Required:** See DATA_PERSISTENCE.md for remediation steps.

---

## DATA FLOW TRACE

### Complete Upload-to-Results Path

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           USER BROWSER (Frontend)                               │
│  apps/web/src/app/upload/page.tsx                                               │
│  - File selection (CSV, XLSX, XLS)                                              │
│  - Max 5 files per request, 50MB each                                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: REQUEST PRESIGNED URL                                                   │
│  POST /uploads/presign                                                          │
│  - apps/api/src/routes/uploads.py:88                                            │
│  - Rate limited: 20/minute per IP                                               │
│  - Returns S3 presigned PUT URLs                                                │
│  - Key format: {user_id|anonymous}/{uuid}-{sanitized_filename}                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DIRECT UPLOAD TO S3                                                    │
│  PUT {presigned_url}                                                            │
│  - Direct browser → S3 upload (bypasses backend)                                │
│  - Bucket: profitsentinel-dev-uploads (from terraform)                          │
│  - Stored with AES256 server-side encryption                                    │
│                                                                                 │
│  ⚠️  CRITICAL: VERSIONING ENABLED - Files persist 90 days after "deletion"!    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: COLUMN MAPPING SUGGESTION                                              │
│  POST /uploads/suggest-mapping                                                  │
│  - apps/api/src/routes/uploads.py:165                                           │
│  - Loads sample (50 rows) from S3                                               │
│  - Uses Grok AI for intelligent mapping (falls back to heuristics)              │
│  - Returns suggested column → standard field mappings                           │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: ANALYSIS                                                               │
│  POST /analysis/analyze                                                         │
│  - apps/api/src/routes/analysis.py:313                                          │
│  - Rate limited: 10/minute per IP                                               │
│  - Loads full file from S3 (up to 500K rows)                                    │
│  - Applies column mapping                                                       │
│  - Runs 8 detection primitives (VSA resonator)                                  │
│  - Returns leak results with scores and SKUs                                    │
│                                                                                 │
│  BACKGROUND TASKS (scheduled after response):                                   │
│  1. cleanup_s3_file: Delete from S3 after 60s delay                             │
│  2. store_anonymized_analytics: Save aggregated stats to Supabase               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
┌──────────────────────────────────┐   ┌──────────────────────────────────┐
│  S3 "DELETION" (60s delay)       │   │  SUPABASE STORAGE                │
│  apps/api/src/services/          │   │  apps/api/src/services/          │
│  anonymization.py:256            │   │  anonymization.py:224            │
│                                  │   │                                  │
│  ⚠️  CREATES DELETE MARKER ONLY  │   │  Stores to 'analytics' table:    │
│  Actual data persists as         │   │  - files_analyzed (count)        │
│  non-current version for 90 DAYS │   │  - total_rows (count)            │
│                                  │   │  - leak_counts (JSON object)     │
│  See: infrastructure/modules/    │   │  - avg_scores (JSON object)      │
│  s3/main.tf:39-50               │   │  - impact estimates              │
│                                  │   │  NO SKUs or PII stored           │
└──────────────────────────────────┘   └──────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: EMAIL REPORT (Optional - user provides email)                          │
│  POST /reports/send                                                             │
│  - apps/api/src/services/email.py                                               │
│  - Sends via Resend or SendGrid                                                 │
│                                                                                 │
│  ⚠️  EMAIL CONTAINS FULL SKU DATA                                               │
│  - All detected SKUs with scores                                                │
│  - Cost/quantity/margin data                                                    │
│  - Recommendations per item                                                     │
│                                                                                 │
│  RETENTION: Email provider logs (Resend/SendGrid retain ~30 days)               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## PERSISTENCE POINTS SUMMARY

| Location | What's Stored | Retention | Contains Test Data? |
|----------|--------------|-----------|---------------------|
| **S3 Bucket** | Raw uploaded files | **90 DAYS** (versioning!) | **YES - CRITICAL** |
| **S3 Versions** | All "deleted" file versions | 90 days | **YES - CRITICAL** |
| **Supabase: email_signups** | Email, company name, UTM params | Indefinite | Maybe (email addresses) |
| **Supabase: analysis_synopses** | Aggregated stats only | Indefinite | No (anonymized) |
| **CloudWatch Logs** | Application logs | Default (never expire) | Filenames, row counts |
| **Email (Resend/SendGrid)** | Full reports with SKUs | ~30 days | **YES if reports sent** |
| **RDS Aurora** | Not used for file storage | N/A | No |

---

## FILE REFERENCES

### Frontend Upload Flow
- `apps/web/src/app/upload/page.tsx` - Main upload page
- `apps/web/src/lib/api.ts` - API client functions
- `apps/web/src/lib/supabase.ts` - Supabase auth/client

### Backend API Routes
- `apps/api/src/routes/uploads.py` - Presign URL generation
- `apps/api/src/routes/analysis.py` - Analysis endpoint
- `apps/api/src/routes/reports.py` - Email report sending

### Backend Services
- `apps/api/src/services/s3.py` - S3 operations (upload, download, delete)
- `apps/api/src/services/analysis.py` - VSA resonator analysis
- `apps/api/src/services/anonymization.py` - PII stripping, cleanup
- `apps/api/src/services/email.py` - Report email generation

### Infrastructure
- `infrastructure/modules/s3/main.tf` - S3 bucket config (VERSIONING ENABLED!)
- `infrastructure/modules/ecs/main.tf` - ECS task definitions
- `infrastructure/modules/rds/main.tf` - Aurora PostgreSQL

### Configuration
- `apps/api/src/config.py` - Environment variables, settings
- `.env.example` - Environment template

---

## SECURITY NOTES

### Positive Findings
- S3 public access is blocked
- Server-side encryption (AES256) enabled
- RDS encryption at rest enabled
- RDS only accessible from ECS tasks (private subnets)
- No hardcoded credentials found in code
- Rate limiting on upload/analysis endpoints
- S3 key validation prevents unauthorized access

### Critical Issues
1. **S3 Versioning**: "Deleted" files persist for 90 days
2. **CloudWatch Logs**: No retention policy, may contain sensitive filenames
3. **Email Reports**: Full SKU data sent to user emails, retained by provider

---

## NEEDS VERIFICATION

- [ ] What is the actual S3 bucket name in production? (Check AWS console)
- [ ] Are CloudWatch log groups configured with retention policies?
- [ ] Are there any RDS snapshots containing test data?
- [ ] Were any email reports sent during testing that contain real data?
- [ ] Is the Supabase 'analytics' table actually being populated?
