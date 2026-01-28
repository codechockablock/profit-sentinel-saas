# Environment Separation Strategy

## Executive Summary

This document outlines a complete environment separation strategy to prevent future data contamination issues and enable safe testing.

## Current State Problems

| Issue | Impact | Status |
|-------|--------|--------|
| Single AWS account for all environments | Test data mixed with production | CRITICAL |
| No isolated testing environment | Real data used for testing | CRITICAL |
| Shared S3 bucket for all uploads | Contamination risk | HIGH |
| No environment tagging | Can't distinguish data origin | MEDIUM |

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEVELOPMENT (Local)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  - Docker Compose environment                                               │
│  - PostgreSQL (local)                                                        │
│  - MinIO (local S3-compatible)                                               │
│  - Synthetic data ONLY                                                       │
│  - No AWS credentials                                                        │
│  - No Supabase credentials                                                   │
│                                                                             │
│  Cost: $0                                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STAGING (New AWS Account)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  - Separate AWS account (profitsentinel-staging)                             │
│  - Isolated VPC, S3 bucket, RDS                                              │
│  - Can use realistic test data safely                                        │
│  - Mirrors production architecture                                           │
│  - CI/CD deploys here first                                                  │
│                                                                             │
│  Cost: ~$100-150/month (can scale down when not testing)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION (Current AWS Account)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  - Customer data ONLY                                                        │
│  - Never used for testing                                                    │
│  - Strict access controls                                                    │
│  - Automated backups                                                         │
│  - Monitoring and alerting                                                   │
│                                                                             │
│  Cost: ~$150-200/month + usage                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Local Development (Immediate)

**Already Done:**
- [x] Created `docker-compose.local.yml`
- [x] Created synthetic data generator
- [x] Documented local setup

**Actions Required:**
- [ ] Team adopts local development workflow
- [ ] Remove AWS credentials from local development

### Phase 2: Clean Production (This Week)

**Actions Required:**
- [ ] Run `find_test_data.py` to identify test data
- [ ] Run `destroy_test_data.py` to remove test data
- [ ] Run `verify_cleanup.py` to confirm
- [ ] Update S3 lifecycle to 1-day version retention
- [ ] Set CloudWatch log retention to 7 days

### Phase 3: Create Staging Account (Next 2-4 Weeks)

**Setup Steps:**

1. **Create new AWS account**
   ```
   AWS Organization → Create Account → "profitsentinel-staging"
   ```

2. **Apply Terraform with staging config**
   ```bash
   cd infrastructure/environments
   mkdir staging
   # Copy from dev, update name_prefix to "profitsentinel-staging"
   terraform init
   terraform apply
   ```

3. **Create separate Supabase project**
   - supabase.com → New Project → "profit-sentinel-staging"
   - Apply migrations

4. **Update CI/CD pipeline**
   ```yaml
   # .github/workflows/deploy.yml
   jobs:
     deploy-staging:
       if: github.ref == 'refs/heads/develop'
       environment: staging
       # ... deploy to staging account

     deploy-production:
       if: github.ref == 'refs/heads/main'
       needs: [deploy-staging, e2e-tests]
       environment: production
       # ... deploy to production account
   ```

### Phase 4: Enforce Separation (Ongoing)

**Technical Controls:**

1. **IAM Policies** - Developers can't access production directly
2. **Branch Protection** - Production only deployed from main
3. **Environment Variables** - Different secrets per environment
4. **Network Isolation** - Staging can't reach production

**Process Controls:**

1. **Code Review** - All changes reviewed before production
2. **Staging Testing** - All features tested in staging first
3. **Data Policy** - Real customer data NEVER leaves production
4. **Incident Response** - Clear escalation for production issues

## Resource Naming Convention

| Environment | Prefix | Example S3 Bucket |
|-------------|--------|-------------------|
| Local | `local-` | (MinIO only) |
| Staging | `profitsentinel-staging-` | `profitsentinel-staging-uploads` |
| Production | `profitsentinel-prod-` | `profitsentinel-prod-uploads` |

Current production uses `profitsentinel-dev-` prefix - consider renaming to `profitsentinel-prod-` for clarity.

## Environment Configuration

### Local (.env.local)
```bash
# NO AWS/SUPABASE CREDENTIALS
DATABASE_URL=postgresql://sentinel_dev:local@localhost:5432/profit_sentinel_dev
S3_ENDPOINT_URL=http://localhost:9000
S3_BUCKET_NAME=profit-sentinel-uploads
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin123
```

### Staging (AWS Secrets Manager / GitHub Secrets)
```bash
AWS_ACCOUNT_ID=222222222222  # Staging account
S3_BUCKET_NAME=profitsentinel-staging-uploads
SUPABASE_URL=https://staging-xxxxx.supabase.co
# All other staging credentials
```

### Production (AWS Secrets Manager / GitHub Secrets)
```bash
AWS_ACCOUNT_ID=111111111111  # Production account
S3_BUCKET_NAME=profitsentinel-prod-uploads
SUPABASE_URL=https://prod-xxxxx.supabase.co
# All other production credentials
```

## Git Workflow

```
main (production)
  │
  ├── deploy → production
  │
develop (staging)
  │
  ├── deploy → staging
  │
feature/xxx
  │
  └── local development only
```

**Rules:**
1. Feature branches created from `develop`
2. PRs merged to `develop` → auto-deploy to staging
3. After staging validation, PR from `develop` to `main`
4. Merges to `main` → auto-deploy to production

## Cost Comparison

| Environment | Monthly Cost | Data Type |
|-------------|--------------|-----------|
| Local | $0 | Synthetic only |
| Staging | $100-150 | Test data (real-ish) |
| Production | $150-200+ | Customer data only |

**Staging Cost Optimization:**
- Use smaller instance sizes
- Scale down when not actively testing
- Use Aurora Serverless v2 with min 0.5 ACU
- Set aggressive lifecycle policies

## Migration Checklist

### Immediate (Before Customer Launch)
- [ ] Clean all test data from production
- [ ] Update S3 lifecycle rules
- [ ] Set log retention policies
- [ ] Document current architecture

### Short-term (Within 2 Weeks)
- [ ] Create staging AWS account
- [ ] Deploy staging infrastructure
- [ ] Set up staging Supabase project
- [ ] Update CI/CD for multi-environment

### Medium-term (Before Stripe Integration)
- [ ] Implement IAM policies for environment isolation
- [ ] Set up monitoring and alerting
- [ ] Document deployment procedures
- [ ] Train team on new workflow

### Long-term (Ongoing)
- [ ] Regular security audits
- [ ] Credential rotation schedule
- [ ] Disaster recovery testing
- [ ] Penetration testing (annual)

## Verification

After implementation, verify separation by:

1. **Local Test**: Can develop and test with no AWS credentials
2. **Staging Test**: Can deploy to staging without affecting production
3. **Production Test**: Cannot accidentally deploy test data to production
4. **Access Test**: Developers cannot directly access production database

## Contact

For questions about this architecture:
- Security concerns: Review SECURITY_ISSUES.md
- AWS resources: Review AWS_RESOURCES.md
- Data flow: Review ARCHITECTURE_MAP.md
