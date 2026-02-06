# Infrastructure Inventory

**Date:** 2026-02-06
**Environment:** dev (profitsentinel-dev-*)

## AWS Resources

### ECR (Elastic Container Registry)
- **Repository:** `profitsentinel-dev-api`
- **Module:** `infrastructure/modules/ecr/`

### ECS (Elastic Container Service)
- **Cluster:** `profitsentinel-dev-cluster`
- **Service:** `profitsentinel-dev-api-service`
- **Task Definition:** `profitsentinel-dev-api`
- **Launch Type:** Fargate
- **CPU/Memory:** 4096 / 16384 (4 vCPU / 16 GB)
- **Container Port:** 8000
- **Desired Count:** 1
- **Module:** `infrastructure/modules/ecs/`

### ALB (Application Load Balancer)
- **Name:** `profitsentinel-dev-alb`
- **Target Group:** `profitsentinel-dev-tg` (port 8000, HTTP, /health)
- **Listeners:** HTTPS :443 (TLS 1.2+), HTTP :80 (redirect to HTTPS)
- **Idle Timeout:** 300s (for large CSV processing)
- **Module:** `infrastructure/modules/alb/`

### VPC
- **Name:** `profitsentinel-dev`
- **Subnets:** Public (ALB) + Private (ECS tasks)
- **Module:** `infrastructure/modules/vpc/`

### RDS
- **Type:** Aurora PostgreSQL
- **Module:** `infrastructure/modules/rds/`

### S3
- **Bucket:** `profitsentinel-dev-uploads` (or similar)
- **Module:** `infrastructure/modules/s3/`

### Secrets Manager
- `xai_api_key_secret_arn` — XAI/Grok API key
- `supabase_service_key_secret_arn` — Supabase service role key
- `resend_api_key_secret_arn` — Resend email API key

### ACM
- Certificate ARN configured via `var.acm_certificate_arn`
- Used by ALB HTTPS listener

## Environment Variables (ECS Task)
| Variable | Source | Description |
|----------|--------|-------------|
| S3_BUCKET_NAME | env | Upload bucket name |
| AWS_REGION | env | us-east-1 |
| SUPABASE_URL | env | Supabase project URL |
| USE_VSA_GROUNDING | env | "true" |
| INCLUDE_CAUSE_DIAGNOSIS | env | "true" |
| XAI_API_KEY | secret | Grok API key |
| SUPABASE_SERVICE_KEY | secret | Supabase service key |
| RESEND_API_KEY | secret | Email delivery key |

## Terraform State
- **Backend:** S3 bucket `profitsentinel-terraform-state-codechockablock`
- **State Key:** `dev/terraform.tfstate`
- **Lock Table:** `terraform-locks` (DynamoDB)
- **Region:** us-east-1

## CI/CD
- **Backend deploy:** `.github/workflows/deploy.yml` → ECR push → ECS update
- **Frontend deploy:** Vercel (via Vercel CLI in same workflow)
- **Tests:** `.github/workflows/test.yml`, `backend-ci.yml`, `frontend-ci.yml`

## Domain Configuration
- **Frontend (production):** https://www.profitsentinel.com (Vercel)
- **API (production):** https://api.profitsentinel.com (ALB → ECS)

## Changes Needed for Staging

### New Resources (staging environment)
1. **ECS Task Definition** — new container using `Dockerfile.sidecar` (port 8001)
2. **ECR Repository** — reuse `profitsentinel-dev-api` or create `profitsentinel-staging-sidecar`
3. **ALB Target Group** — update to port 8001 with /health check
4. **Security Group** — allow port 8001 ingress from ALB
5. **Environment variables** — add SENTINEL_DEFAULT_STORE, SENTINEL_TOP_K, USE_NEW_ENGINE

### Preserved Resources (no changes)
- VPC, subnets
- RDS cluster
- S3 bucket
- Secrets Manager entries
- ACM certificate
