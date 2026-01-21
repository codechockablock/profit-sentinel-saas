# AWS_RESOURCES.md - Complete AWS Infrastructure Inventory

## Security Audit Date: 2026-01-21
## Environment: profitsentinel-dev

---

## RESOURCE SUMMARY

| Resource Type | Name/Identifier | Data Exposure Risk | Action Required |
|---------------|-----------------|-------------------|-----------------|
| S3 Bucket | profitsentinel-dev-uploads | **CRITICAL** | Clear all versions |
| Aurora PostgreSQL | profitsentinel-dev-cluster | LOW | Review snapshots |
| ECS Cluster | profitsentinel-dev-cluster | NONE | N/A |
| ECR Repository | profitsentinel-dev-api | NONE | N/A |
| CloudWatch Logs | /ecs/profitsentinel-dev | **MEDIUM** | Set retention, review |
| VPC | profitsentinel-dev-vpc | NONE | N/A |
| ALB | profitsentinel-dev-alb | NONE | N/A |
| Secrets Manager | (multiple) | NONE | N/A |

---

## S3 BUCKET (CRITICAL)

### Configuration

| Property | Value | Security Status |
|----------|-------|-----------------|
| **Bucket Name** | `profitsentinel-dev-uploads` | |
| **Region** | us-east-1 | |
| **Encryption** | AES256 (SSE-S3) | GOOD |
| **Public Access** | All blocked | GOOD |
| **Versioning** | **ENABLED** | **CRITICAL ISSUE** |
| **Lifecycle** | Non-current expire 90 days | **INSUFFICIENT** |

### Infrastructure Code Reference
`infrastructure/modules/s3/main.tf`

```hcl
resource "aws_s3_bucket_versioning" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  versioning_configuration {
    status = "Enabled"  # <-- THIS IS THE PROBLEM
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    noncurrent_version_expiration {
      noncurrent_days = 90  # <-- Versions persist 90 days!
    }
  }
}
```

### CRITICAL FINDING

**Your test data may still exist in S3 even if you think it was deleted!**

When a file is "deleted" with versioning enabled:
1. S3 creates a "delete marker" (current version)
2. The actual file data becomes a "non-current version"
3. Non-current versions persist for 90 days per lifecycle rule

**To verify:** Run `find_test_data.py` script (see scripts/ directory)

### Remediation Steps

1. **List all object versions:**
   ```bash
   aws s3api list-object-versions --bucket profitsentinel-dev-uploads
   ```

2. **Delete all versions permanently:**
   ```bash
   # Use destroy_test_data.py script for safe deletion with logging
   python scripts/destroy_test_data.py
   ```

3. **Update lifecycle rule (RECOMMENDED):**
   ```hcl
   noncurrent_version_expiration {
     noncurrent_days = 1  # Expire versions after 1 day
   }
   ```

4. **Consider disabling versioning** for uploads bucket (user data should not be retained)

---

## RDS AURORA POSTGRESQL

### Configuration

| Property | Value | Security Status |
|----------|-------|-----------------|
| **Cluster ID** | profitsentinel-dev-cluster | |
| **Engine** | Aurora PostgreSQL 15 | |
| **Instance Class** | db.serverless | |
| **Encryption at Rest** | Enabled | GOOD |
| **Public Access** | Disabled | GOOD |
| **VPC** | Private subnets only | GOOD |
| **Password Management** | Secrets Manager | GOOD |
| **Skip Final Snapshot** | true (dev) | NEEDS REVIEW |
| **Deletion Protection** | false (dev) | NEEDS REVIEW |

### Infrastructure Code Reference
`infrastructure/modules/rds/main.tf`

### Security Group Rules

```hcl
ingress {
  description     = "Postgres from ECS"
  from_port       = 5432
  to_port         = 5432
  protocol        = "tcp"
  security_groups = [var.ecs_sg_id]  # Only ECS can access
}
```

**Status: GOOD** - Database only accessible from ECS tasks.

### NEEDS VERIFICATION

- [ ] Are there any RDS snapshots that might contain test data?
- [ ] Is the database actually used by the application? (Appears to use Supabase instead)

Run this to check snapshots:
```bash
aws rds describe-db-snapshots --db-instance-identifier profitsentinel-dev-cluster
```

---

## ECS (Elastic Container Service)

### Cluster Configuration

| Property | Value |
|----------|-------|
| **Cluster Name** | profitsentinel-dev-cluster |
| **Container Insights** | Enabled |
| **Launch Type** | Fargate |

### Task Definition

| Property | Value |
|----------|-------|
| **Family** | profitsentinel-dev-api |
| **CPU** | 1024 (1 vCPU) |
| **Memory** | 2048 MB |
| **Network Mode** | awsvpc |

### Environment Variables in Task

```json
{
  "S3_BUCKET_NAME": "profitsentinel-dev-uploads",
  "AWS_REGION": "us-east-1",
  "SUPABASE_URL": "(from variable)",
  "USE_VSA_GROUNDING": "true",
  "INCLUDE_CAUSE_DIAGNOSIS": "true"
}
```

### Secrets (from Secrets Manager)

- `XAI_API_KEY`
- `SUPABASE_SERVICE_KEY`

### Infrastructure Code Reference
`infrastructure/modules/ecs/main.tf`

**Status: GOOD** - No sensitive data stored in task definition.

---

## ECR (Container Registry)

### Repository

| Property | Value |
|----------|-------|
| **Repository Name** | profitsentinel-dev-api |
| **Image Scanning** | On push (recommended) |
| **Encryption** | AES256 |

**Status: GOOD** - Only Docker images stored, no user data.

---

## CLOUDWATCH LOGS (MEDIUM RISK)

### Log Groups

| Log Group | Retention | Data Risk |
|-----------|-----------|-----------|
| `/ecs/profitsentinel-dev` | **NEVER EXPIRE** | **MEDIUM** |

### What's Logged

Based on code analysis (`apps/api/src/` logging):
- Request URLs and methods
- **File keys** (includes sanitized filenames)
- Row counts from uploads
- Error messages (may include data samples)
- Processing times

### CRITICAL ACTIONS

1. **Set retention policy:**
   ```bash
   aws logs put-retention-policy \
     --log-group-name /ecs/profitsentinel-dev \
     --retention-in-days 7
   ```

2. **Review existing logs for sensitive data:**
   ```bash
   aws logs filter-log-events \
     --log-group-name /ecs/profitsentinel-dev \
     --start-time $(date -d '7 days ago' +%s000) \
     --filter-pattern "inventory" # or other test data identifiers
   ```

3. **Consider deleting old log streams:**
   ```bash
   # See destroy_test_data.py for safe deletion
   ```

---

## VPC NETWORKING

### Configuration

| Resource | CIDR/Details | Security Status |
|----------|--------------|-----------------|
| **VPC** | 10.0.0.0/16 | GOOD |
| **Public Subnets** | 10.0.0.0/24, 10.0.1.0/24 | For ALB only |
| **Private Subnets** | 10.0.2.0/24, 10.0.3.0/24 | ECS + RDS |
| **NAT Gateway** | 1 (single AZ) | GOOD |
| **Internet Gateway** | Attached | GOOD |

### Infrastructure Code Reference
`infrastructure/modules/vpc/main.tf`

**Status: GOOD** - Standard secure VPC configuration.

---

## APPLICATION LOAD BALANCER

### Configuration

| Property | Value |
|----------|-------|
| **Name** | profitsentinel-dev-alb |
| **Scheme** | Internet-facing |
| **Subnets** | Public subnets |
| **Target Group** | ECS service on port 8000 |

### Security Group

| Rule | Port | Source |
|------|------|--------|
| HTTPS Inbound | 443 | 0.0.0.0/0 |
| HTTP Inbound | 80 | 0.0.0.0/0 (redirects to HTTPS) |

**Status: GOOD** - Standard ALB configuration.

---

## SECRETS MANAGER

### Managed Secrets

| Secret | Purpose | Auto-Rotation |
|--------|---------|---------------|
| RDS Master Password | Database access | Yes (managed by RDS) |
| XAI_API_KEY | Grok AI access | No |
| SUPABASE_SERVICE_KEY | Supabase full access | No |

**Status: GOOD** - Secrets properly managed, not hardcoded.

---

## IAM ROLES

### ECS Task Execution Role

**Name:** `profitsentinel-dev-ecs-execution-role`

**Permissions:**
- `AmazonECSTaskExecutionRolePolicy` (managed)
- Custom: `secretsmanager:GetSecretValue` on specific ARNs

### ECS Task Role

**Name:** `profitsentinel-dev-ecs-task-role`

**Permissions:**
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:DeleteObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::profitsentinel-dev-uploads",
    "arn:aws:s3:::profitsentinel-dev-uploads/*"
  ]
}
```

**Status: GOOD** - Minimal required permissions.

---

## COST CONSIDERATIONS

Based on infrastructure code:

| Resource | Estimated Monthly Cost |
|----------|----------------------|
| ECS Fargate (1024 CPU, 2GB) | ~$30-50 |
| Aurora Serverless v2 (0.5-2 ACU) | ~$45-90 |
| NAT Gateway | ~$32 + data |
| ALB | ~$16 + data |
| S3 | < $5 |
| CloudWatch Logs | Varies |
| **Total Estimated** | **~$130-200/month** |

---

## VERIFICATION COMMANDS

Run these at home with AWS credentials:

```bash
# List all S3 objects and versions
aws s3api list-object-versions --bucket profitsentinel-dev-uploads

# Check RDS snapshots
aws rds describe-db-snapshots

# List CloudWatch log groups
aws logs describe-log-groups

# Check ECS task definition
aws ecs describe-task-definition --task-definition profitsentinel-dev-api

# List Secrets Manager secrets
aws secretsmanager list-secrets
```
