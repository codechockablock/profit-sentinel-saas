# Profit Sentinel v2.1.0 - GPU Deployment Architecture

**Version:** 2.1.0
**Target Instance:** AWS EC2 g4dn.xlarge (NVIDIA T4 GPU)
**Architecture:** Hybrid Baseline + VSA Resonator with GPU Acceleration

---

## Executive Summary

This document describes the production deployment architecture for Profit Sentinel's hybrid anomaly detection pipeline on AWS GPU infrastructure. The deployment leverages the NVIDIA T4 GPU for VSA Resonator acceleration while maintaining the CPU-based Baseline Detector as the source of truth.

### Key Metrics (Validated)
- **Baseline Avg F1:** 82.4%
- **Baseline Avg Recall:** 97.1%
- **Resonator Convergence:** 100%
- **Contradictions Detected:** 410
- **Hallucinations:** 0
- **Expected GPU Speedup:** 5-10x for Resonator

---

## Deployment Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            AWS REGION: us-east-1                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                              VPC (10.0.0.0/16)                               │   │
│  │                                                                              │   │
│  │  ┌──────────────────────┐          ┌──────────────────────┐                │   │
│  │  │   PUBLIC SUBNETS     │          │   PRIVATE SUBNETS    │                │   │
│  │  │   (10.0.0.0/24)      │          │   (10.0.2.0/24)      │                │   │
│  │  │   (10.0.1.0/24)      │          │   (10.0.3.0/24)      │                │   │
│  │  │                      │          │                      │                │   │
│  │  │  ┌────────────────┐  │          │  ┌────────────────┐  │                │   │
│  │  │  │ Application    │  │   ──►    │  │  Auto Scaling  │  │                │   │
│  │  │  │ Load Balancer  │  │          │  │  Group (GPU)   │  │                │   │
│  │  │  │ (HTTPS/443)    │  │          │  │                │  │                │   │
│  │  │  └────────────────┘  │          │  │ ┌────────────┐ │  │                │   │
│  │  │                      │          │  │ │g4dn.xlarge │ │  │                │   │
│  │  │  ┌────────────────┐  │          │  │ │ T4 GPU     │ │  │                │   │
│  │  │  │ NAT Gateway    │  │   ◄──    │  │ │ Docker     │ │  │                │   │
│  │  │  └────────────────┘  │          │  │ └────────────┘ │  │                │   │
│  │  │                      │          │  │                │  │                │   │
│  │  └──────────────────────┘          │  │ ┌────────────┐ │  │                │   │
│  │                                    │  │ │g4dn.xlarge │ │  │                │   │
│  │  ┌──────────────────────┐          │  │ │ (Spot)     │ │  │                │   │
│  │  │   Internet Gateway   │          │  │ └────────────┘ │  │                │   │
│  │  └──────────────────────┘          │  │                │  │                │   │
│  │           │                        │  └────────────────┘  │                │   │
│  │           │                        │          │           │                │   │
│  └───────────│────────────────────────│──────────│───────────│────────────────┘   │
│              │                        │          │           │                     │
│              ▼                        │          ▼           │                     │
│  ┌──────────────────┐                 │  ┌──────────────────┐│                     │
│  │    Route 53      │                 │  │  Aurora RDS      ││                     │
│  │ api.sentinel.ai  │                 │  │  PostgreSQL 15   ││                     │
│  └──────────────────┘                 │  │  (Serverless v2) ││                     │
│                                       │  └──────────────────┘│                     │
│  ┌──────────────────┐                 │                      │                     │
│  │    CloudWatch    │◄────────────────┘                      │                     │
│  │  • Logs          │                                        │                     │
│  │  • Metrics       │         ┌──────────────────┐           │                     │
│  │  • Alarms        │         │      S3          │           │                     │
│  └──────────────────┘         │  • Uploads       │◄──────────┘                     │
│                               │  • Models        │                                 │
│  ┌──────────────────┐         └──────────────────┘                                 │
│  │  Secrets Manager │                                                              │
│  │  • API Keys      │         ┌──────────────────┐                                 │
│  │  • DB Passwords  │         │      ECR         │                                 │
│  └──────────────────┘         │  Docker Registry │                                 │
│                               └──────────────────┘                                 │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Application Container (Docker)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROFIT SENTINEL CONTAINER                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [FastAPI Application - Port 8000]                            │
│          │                                                      │
│          ├──► /health          (Health check)                  │
│          ├──► /v1/analyze      (Analysis endpoint)             │
│          ├──► /v1/reports      (Report generation)             │
│          └──► /v1/metrics      (Pipeline metrics)              │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              HYBRID DETECTION PIPELINE                   │  │
│   │                                                          │  │
│   │  [CPU] Baseline Detector                                 │  │
│   │    • 8 detection primitives                              │  │
│   │    • Rule-based, deterministic                           │  │
│   │    • 34ms / 10K rows                                     │  │
│   │                                                          │  │
│   │  [GPU] VSA Resonator (T4 Accelerated)                    │  │
│   │    • Symbolic validation                                 │  │
│   │    • PyTorch CUDA tensors                                │  │
│   │    • ~2-4s / 10K rows (5-10x faster than CPU)           │  │
│   │                                                          │  │
│   │  [CPU] Contradiction Detector                            │  │
│   │    • Logical conflict detection                          │  │
│   │    • Manual review flagging                              │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Runtime: Python 3.12 + PyTorch 2.x + CUDA 12.1              │
│   Base: NVIDIA CUDA Runtime Image                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Components

### Compute Layer

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **Primary Instances** | g4dn.xlarge (On-Demand) | Production workloads |
| **Spot Instances** | g4dn.xlarge (Spot) | Cost optimization |
| **GPU** | NVIDIA T4 (16GB VRAM) | VSA Resonator acceleration |
| **vCPU** | 4 cores | Baseline detector, API |
| **Memory** | 16 GB | Large dataset processing |
| **Storage** | 125 GB gp3 SSD | Docker images, temp data |

### Auto Scaling Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Min Capacity | 1 | Cost efficiency |
| Desired Capacity | 2 | High availability |
| Max Capacity | 5 | Handle traffic spikes |
| Scale-up Threshold | CPU > 70% | Performance-based |
| Scale-down Threshold | CPU < 30% | Cost optimization |
| Spot/On-Demand Mix | 50/50 | Balance cost & reliability |

### Networking

| Component | Configuration | Notes |
|-----------|---------------|-------|
| VPC | 10.0.0.0/16 | Existing VPC |
| Public Subnets | 10.0.0.0/24, 10.0.1.0/24 | ALB, NAT Gateway |
| Private Subnets | 10.0.2.0/24, 10.0.3.0/24 | EC2 instances |
| ALB | HTTPS (443) | TLS termination |
| Health Check | /health | 30s interval |

### Security

| Layer | Implementation |
|-------|----------------|
| Network | Security Groups (ALB→EC2→RDS) |
| Secrets | AWS Secrets Manager |
| IAM | Least-privilege instance profile |
| TLS | ACM certificate on ALB |
| Data | S3 encryption (AES-256) |

---

## Deployment Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │────►│   GitHub    │────►│    ECR      │────►│   EC2 ASG   │
│   Push      │     │   Actions   │     │   Push      │     │   Deploy    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │                   │
      │                   │                   │                   │
      ▼                   ▼                   ▼                   ▼
  main branch        Build Docker       Store Image        Pull & Run
  trigger            + CUDA             with tags          Container
                     + Tests            (SHA, latest)      + GPU
```

### CI/CD Pipeline Stages

1. **Security Scan** - Check for hardcoded secrets
2. **Build** - Multi-stage Docker build with CUDA
3. **Test** - Unit tests + validation pipeline
4. **Push** - ECR with commit SHA tags
5. **Deploy** - Rolling update via ASG refresh
6. **Health Check** - Verify /health endpoint
7. **Notify** - Deployment summary

---

## Cost Analysis

### Monthly Estimates (us-east-1)

| Component | Configuration | Monthly Cost |
|-----------|---------------|--------------|
| EC2 (On-Demand) | 1x g4dn.xlarge | ~$379 |
| EC2 (Spot) | 1x g4dn.xlarge | ~$114 (70% savings) |
| ALB | Standard | ~$22 |
| NAT Gateway | Per-GB | ~$32 |
| Aurora RDS | Serverless v2 | ~$86 (0.5-2 ACU avg) |
| S3 | 10GB storage | ~$0.23 |
| CloudWatch | Logs + Metrics | ~$10 |
| **Total (Spot)** | | **~$264/mo** |
| **Total (On-Demand)** | | **~$529/mo** |

### Cost Optimization Strategies

1. **Spot Instances** - 50-70% savings on GPU compute
2. **Aurora Serverless** - Scale to zero when idle
3. **S3 Lifecycle** - Auto-delete old uploads
4. **Reserved Instances** - 1-year commitment for baseline
5. **Scheduled Scaling** - Scale down off-hours

---

## Monitoring & Observability

### CloudWatch Alarms

| Alarm | Threshold | Action |
|-------|-----------|--------|
| CPU Utilization | > 80% for 5 min | Scale up |
| Memory Utilization | > 85% for 5 min | Alert |
| GPU Utilization | > 90% for 5 min | Scale up |
| Unhealthy Hosts | > 0 | Alert |
| 5xx Errors | > 10/min | Alert |
| Latency P99 | > 30s | Alert |

### Custom Metrics (from hybrid pipeline)

```python
metrics = {
    "baseline_time_ms": 34,
    "resonator_time_ms": 2000,  # With GPU
    "total_candidates": 5656,
    "convergence_rate": 1.0,
    "contradiction_count": 410,
    "avg_f1": 0.824,
}
```

---

## Disaster Recovery

| Scenario | RTO | RPO | Strategy |
|----------|-----|-----|----------|
| Instance Failure | 2 min | 0 | ASG auto-replace |
| AZ Failure | 5 min | 0 | Multi-AZ deployment |
| Region Failure | 4 hr | 1 hr | Cross-region replica |
| Data Corruption | 1 hr | 5 min | S3 versioning, RDS snapshots |

---

## File Structure

```
infrastructure/
├── environments/
│   └── gpu-prod/
│       ├── main.tf           # Root module
│       ├── variables.tf      # Input variables
│       ├── outputs.tf        # Outputs
│       ├── backend.tf        # S3 state backend
│       └── terraform.tfvars  # Environment values
├── modules/
│   ├── vpc/                  # Existing
│   ├── alb/                  # Existing
│   ├── rds/                  # Existing
│   ├── s3/                   # Existing
│   ├── ecr/                  # Existing
│   └── gpu-asg/              # NEW: GPU Auto Scaling Group
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       └── user_data.sh
├── Dockerfile.gpu            # GPU-enabled container
└── docker-compose.gpu.yml    # Local GPU testing

.github/workflows/
└── deploy-gpu.yml            # GPU deployment workflow
```

---

## Security Checklist

- [x] No hardcoded secrets in code
- [x] Secrets in AWS Secrets Manager
- [x] IAM roles with least privilege
- [x] Security groups restrict traffic
- [x] HTTPS enforced via ALB
- [x] VPC flow logs enabled
- [x] CloudWatch logs encrypted
- [x] S3 buckets encrypted
- [x] RDS encrypted at rest
- [x] ECR image scanning enabled

---

## Deployment Instructions

### Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** v1.5+ installed
3. **Docker** with BuildKit support
4. Existing VPC, ALB, RDS, and S3 infrastructure

### Step 1: Create Backend Resources

```bash
# Create S3 bucket for Terraform state
aws s3api create-bucket \
  --bucket profitsentinel-terraform-state \
  --region us-east-1

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket profitsentinel-terraform-state \
  --versioning-configuration Status=Enabled

# Create DynamoDB table for state locking
aws dynamodb create-table \
  --table-name profitsentinel-terraform-locks \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

### Step 2: Create Secrets in AWS Secrets Manager

```bash
# Database credentials
aws secretsmanager create-secret \
  --name profitsentinel/db-credentials \
  --secret-string '{"host":"your-rds-endpoint","port":"5432","dbname":"profitsentinel","username":"admin","password":"your-password"}'

# API key
aws secretsmanager create-secret \
  --name profitsentinel/api-key \
  --secret-string 'your-api-key'

# Supabase service key
aws secretsmanager create-secret \
  --name profitsentinel/supabase-service-key \
  --secret-string 'your-supabase-service-key'
```

### Step 3: Deploy Infrastructure

```bash
cd infrastructure/environments/gpu-prod

# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the configuration
terraform apply
```

### Step 4: Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build the GPU image
docker build -f Dockerfile.gpu -t profit-sentinel-gpu:latest .

# Tag and push
docker tag profit-sentinel-gpu:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/profit-sentinel-gpu:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/profit-sentinel-gpu:latest
```

### Step 5: Trigger Deployment

Push to the main branch or manually trigger the GitHub Actions workflow:

```bash
git push origin main
```

Or use the GitHub CLI:

```bash
gh workflow run deploy-gpu.yml -f environment=prod
```

### Step 6: Verify Deployment

```bash
# Check ASG status
aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names profitsentinel-gpu-asg-prod

# Check instance health
aws elbv2 describe-target-health \
  --target-group-arn YOUR_TARGET_GROUP_ARN

# Test health endpoint
curl https://api.profitsentinel.ai/health
```

---

## Troubleshooting

### GPU Not Detected

```bash
# SSH to instance and check NVIDIA drivers
nvidia-smi

# Check container GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Container Fails to Start

```bash
# Check CloudWatch logs
aws logs tail /profitsentinel/gpu-api --follow

# Check user-data script output
cat /var/log/user-data.log
```

### Instance Not Joining Target Group

```bash
# Check security group rules
aws ec2 describe-security-groups --group-ids YOUR_SG_ID

# Check health check configuration
aws elbv2 describe-target-groups --target-group-arns YOUR_TG_ARN
```

---

## References

- [config/hybrid_pipeline_config.yaml](../config/hybrid_pipeline_config.yaml) - Pipeline configuration
- [docs/HYBRID_PIPELINE_API.md](./HYBRID_PIPELINE_API.md) - API specification
- [docs/PROFIT_SENTINEL_HYBRID_VALIDATION.md](./PROFIT_SENTINEL_HYBRID_VALIDATION.md) - Validation report
- [tests/hybrid_validation_results.json](../tests/hybrid_validation_results.json) - Metrics

---

*Architecture designed for Profit Sentinel v2.1.0*
*Last Updated: 2026-01-16*
