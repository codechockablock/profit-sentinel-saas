# ENV_AUDIT.md - Environment Variable Security Audit

## Security Audit Date: 2026-01-21
## Status: SECURE - No hardcoded secrets found

---

## AUDIT SUMMARY

| Check | Status | Notes |
|-------|--------|-------|
| Hardcoded AWS credentials | PASS | Only example values in .env.example |
| Hardcoded API keys | PASS | No production keys in code |
| .env files in repo | PASS | None found (properly gitignored) |
| Secrets in GitHub workflows | PASS | Uses `${{ secrets.* }}` correctly |
| Terraform secrets | PASS | Uses Secrets Manager, no hardcoded values |

---

## ENVIRONMENT VARIABLES INVENTORY

### AWS Configuration (REQUIRED)

| Variable | Used In | Source | Security Notes |
|----------|---------|--------|----------------|
| `AWS_ACCESS_KEY_ID` | Backend API, Docker | .env / GitHub Secrets | Rotatable, use IAM roles in production |
| `AWS_SECRET_ACCESS_KEY` | Backend API, Docker | .env / GitHub Secrets | Rotatable, use IAM roles in production |
| `AWS_REGION` | Backend API, ECS | .env / Terraform | Default: us-east-1 |
| `S3_BUCKET_NAME` | Backend API, ECS | .env / Terraform | Default: profitsentinel-dev-uploads |

**Terraform Reference:** `infrastructure/modules/ecs/main.tf:186-192`

### Supabase Configuration (REQUIRED)

| Variable | Used In | Source | Security Notes |
|----------|---------|--------|----------------|
| `SUPABASE_URL` | Backend, Frontend | .env / Vercel Env | Public URL, safe to expose |
| `SUPABASE_ANON_KEY` | Frontend only | .env / Vercel Env | Limited permissions (RLS enforced) |
| `SUPABASE_SERVICE_KEY` | Backend only | .env / Secrets Manager | Full access - NEVER expose to frontend |

**Code References:**
- Backend: `apps/api/src/services/anonymization.py:64-65`
- Frontend: `apps/web/src/lib/supabase.ts`

### AI API Keys (RECOMMENDED)

| Variable | Used In | Source | Security Notes |
|----------|---------|--------|----------------|
| `XAI_API_KEY` | Backend API | .env / Secrets Manager | For Grok AI column mapping |
| `GROK_API_KEY` | Legacy alias | .env | Alias for XAI_API_KEY |

**Code Reference:** `apps/api/src/config.py:102-118`

### Email Service (OPTIONAL)

| Variable | Used In | Source | Security Notes |
|----------|---------|--------|----------------|
| `RESEND_API_KEY` | Backend API | .env | For Resend email delivery |
| `SENDGRID_API_KEY` | Backend API | .env | Alternative email provider |
| `EMAIL_FROM` | Backend API | .env | Default: reports@profitsentinel.com |
| `EMAIL_FROM_NAME` | Backend API | .env | Default: Profit Sentinel |

**Code Reference:** `apps/api/src/services/email.py:21-39`

### Deployment (CI/CD)

| Variable | Used In | Source | Security Notes |
|----------|---------|--------|----------------|
| `VERCEL_TOKEN` | GitHub Actions | GitHub Secrets | Deployment only |
| `VERCEL_ORG_ID` | GitHub Actions | GitHub Secrets | Deployment only |
| `VERCEL_PROJECT_ID` | GitHub Actions | GitHub Secrets | Deployment only |
| `CORE_REPO_PAT` | GitHub Actions | GitHub Secrets | For private repo access |

**Code Reference:** `.github/workflows/deploy.yml:77-89`

### Privacy/Anonymization

| Variable | Used In | Source | Security Notes |
|----------|---------|--------|----------------|
| `ANONYMIZATION_SALT` | Backend API | .env | Change in production! |

**Code Reference:** `apps/api/src/services/anonymization.py:63`

### Analysis Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_VSA_GROUNDING` | true | Enable VSA evidence grounding |
| `INCLUDE_CAUSE_DIAGNOSIS` | true | Include root cause analysis |
| `VSA_CONFIDENCE_THRESHOLD` | 0.6 | Hot path routing threshold |
| `VSA_AMBIGUITY_THRESHOLD` | 0.5 | Ambiguity routing threshold |
| `MARGIN_LEAK_THRESHOLD` | 0.25 | Margin below this triggers alert |
| `DEAD_STOCK_DAYS` | 90 | Days without sale = dead stock |
| `LOW_STOCK_THRESHOLD` | 10 | Quantity below = low stock |
| `OVERSTOCK_DAYS_SUPPLY` | 180 | Days supply above = overstock |

**Code Reference:** `apps/api/src/config.py:16-71`

---

## SECRETS STORAGE HIERARCHY

### Production (AWS/Vercel)

```
├── AWS Secrets Manager
│   ├── XAI_API_KEY (referenced by ECS task)
│   ├── SUPABASE_SERVICE_KEY (referenced by ECS task)
│   └── RDS master password (auto-managed)
│
├── Vercel Environment Variables
│   ├── NEXT_PUBLIC_SUPABASE_URL
│   ├── NEXT_PUBLIC_SUPABASE_ANON_KEY
│   ├── NEXT_PUBLIC_API_URL
│   └── XAI_API_KEY (for serverless functions)
│
└── GitHub Repository Secrets
    ├── AWS_ACCESS_KEY_ID
    ├── AWS_SECRET_ACCESS_KEY
    ├── VERCEL_TOKEN
    ├── VERCEL_ORG_ID
    ├── VERCEL_PROJECT_ID
    └── CORE_REPO_PAT
```

### Local Development

```
.env (gitignored)
├── AWS_ACCESS_KEY_ID
├── AWS_SECRET_ACCESS_KEY
├── S3_BUCKET_NAME
├── SUPABASE_URL
├── SUPABASE_ANON_KEY
├── SUPABASE_SERVICE_KEY
├── XAI_API_KEY
├── RESEND_API_KEY (optional)
└── ANONYMIZATION_SALT
```

---

## GITIGNORE VERIFICATION

The following patterns are correctly in `.gitignore`:

| Pattern | Present | Notes |
|---------|---------|-------|
| `.env` | YES | Main env file |
| `.env.local` | YES | Local overrides |
| `.env.*.local` | YES | Environment-specific locals |
| `.env.development` | YES | Dev environment |
| `.env.production` | YES | Prod environment |
| `*.tfvars` | YES | Terraform variables |
| `*.tfstate` | YES | Terraform state |
| `credentials.json` | YES | Generic credentials |
| `*.pem` | YES | Private keys |
| `*.key` | YES | Private keys |

---

## FILES CHECKED FOR HARDCODED SECRETS

### Patterns Searched

| Pattern | Matches | Status |
|---------|---------|--------|
| `AKIA[A-Z0-9]{16}` | 2 (examples only) | SAFE |
| `sk-[a-zA-Z0-9]{24,}` | 0 | SAFE |
| `xai-[a-zA-Z0-9]{20,}` | 0 | SAFE |
| `password.*=.*"[^"]{8,}"` | 0 | SAFE |
| `eyJ[a-zA-Z0-9]{100,}` (JWT) | 2 (examples only) | SAFE |

### Example Values Found (SAFE)

1. `.env.example:27` - `AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE` (AWS doc example)
2. `.env.example:28` - `AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` (AWS doc example)
3. `.env.example:48-49` - Supabase JWT examples (placeholder format)

---

## RECOMMENDATIONS

### Immediate Actions

1. **Rotate all credentials** if any real values were ever committed (check git history)
2. **Verify GitHub secrets** are properly scoped to this repository only
3. **Check Vercel environment variables** match production values

### Best Practices to Maintain

1. Never commit `.env` files (verified: working)
2. Use Secrets Manager for ECS tasks (verified: working)
3. Use `NEXT_PUBLIC_` prefix only for safe-to-expose vars (verified: working)
4. Rotate credentials quarterly
5. Use IAM roles instead of access keys where possible

### Potential Improvements

1. **Move to IAM roles** for ECS tasks (instead of access keys)
2. **Enable AWS CloudTrail** for credential usage auditing
3. **Add git-secrets** pre-commit hook to prevent accidental commits
4. **Implement credential rotation** automation

---

## VERIFICATION COMMANDS

Run these at home to verify production secrets are not exposed:

```bash
# Check git history for accidentally committed secrets
git log -p --all -- '*.env' '*.tfvars' 2>/dev/null | grep -E '(AKIA|sk-|xai-|password)' || echo "SAFE: No secrets in git history"

# Check for any .env files that shouldn't exist
find . -name ".env*" -type f ! -name ".env.example" ! -name ".env*.example"

# Verify GitHub secrets are configured (requires gh CLI)
gh secret list --repo yourusername/profit-sentinel-saas
```
