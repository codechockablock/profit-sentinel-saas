# =============================================================================
# Profit Sentinel v2.1.0 - GPU Production Environment
# =============================================================================
# This is the root module for the GPU production deployment.
# It references existing VPC, ALB, RDS, and other modules.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "ProfitSentinel"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Version     = "2.1.0"
    }
  }
}

# -----------------------------------------------------------------------------
# Data Sources - Reference existing infrastructure
# -----------------------------------------------------------------------------

data "aws_vpc" "main" {
  tags = {
    Name = "${var.name_prefix}-vpc"
  }
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }

  tags = {
    Tier = "private"
  }
}

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.main.id]
  }

  tags = {
    Tier = "public"
  }
}

# Reference existing ALB
data "aws_lb" "main" {
  tags = {
    Name = "${var.name_prefix}-alb"
  }
}

data "aws_lb_target_group" "gpu" {
  name = "${var.name_prefix}-gpu-tg"
}

data "aws_security_group" "alb" {
  tags = {
    Name = "${var.name_prefix}-alb-sg"
  }
}

# Reference existing ECR repository
data "aws_ecr_repository" "gpu" {
  name = "profit-sentinel-gpu"
}

# Reference existing S3 bucket
data "aws_s3_bucket" "uploads" {
  bucket = "${var.name_prefix}-uploads-${var.environment}"
}

# Reference secrets
data "aws_secretsmanager_secret" "db" {
  name = "${var.name_prefix}/db-credentials"
}

data "aws_secretsmanager_secret" "api_key" {
  name = "${var.name_prefix}/api-key"
}

data "aws_secretsmanager_secret" "supabase" {
  name = "${var.name_prefix}/supabase-service-key"
}

# Optional: SNS topic for alarms
data "aws_sns_topic" "alerts" {
  count = var.enable_alerts ? 1 : 0
  name  = "${var.name_prefix}-alerts"
}

# -----------------------------------------------------------------------------
# GPU Auto Scaling Group Module
# -----------------------------------------------------------------------------

module "gpu_asg" {
  source = "../../modules/gpu-asg"

  name_prefix = var.name_prefix
  environment = var.environment

  # Networking
  vpc_id             = data.aws_vpc.main.id
  private_subnet_ids = data.aws_subnets.private.ids

  # Load Balancer
  alb_security_group_id   = data.aws_security_group.alb.id
  target_group_arn        = data.aws_lb_target_group.gpu.arn
  target_group_arn_suffix = data.aws_lb_target_group.gpu.arn_suffix
  alb_arn_suffix          = data.aws_lb.main.arn_suffix

  # Container Registry
  ecr_repository_url = data.aws_ecr_repository.gpu.repository_url
  image_tag          = var.image_tag

  # Storage
  s3_bucket_arn  = data.aws_s3_bucket.uploads.arn
  s3_bucket_name = data.aws_s3_bucket.uploads.id

  # Secrets
  secret_arns = [
    data.aws_secretsmanager_secret.db.arn,
    data.aws_secretsmanager_secret.api_key.arn,
    data.aws_secretsmanager_secret.supabase.arn,
  ]
  db_secret_arn       = data.aws_secretsmanager_secret.db.arn
  api_key_secret_arn  = data.aws_secretsmanager_secret.api_key.arn
  supabase_url        = var.supabase_url
  supabase_secret_arn = data.aws_secretsmanager_secret.supabase.arn

  # Instance Configuration
  instance_type    = var.instance_type
  desired_capacity = var.desired_capacity
  min_size         = var.min_size
  max_size         = var.max_size

  # Cost Optimization
  on_demand_base_capacity = var.on_demand_base_capacity
  on_demand_percentage    = var.on_demand_percentage

  # Storage
  root_volume_size = var.root_volume_size

  # Logging
  log_group_name     = var.log_group_name
  log_retention_days = var.log_retention_days

  # SSH (disabled in production)
  enable_ssh      = var.enable_ssh
  ssh_cidr_blocks = var.ssh_cidr_blocks

  # Alerting
  alarm_sns_topic_arn = var.enable_alerts ? data.aws_sns_topic.alerts[0].arn : ""

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "asg_name" {
  description = "Name of the GPU Auto Scaling Group"
  value       = module.gpu_asg.asg_name
}

output "asg_arn" {
  description = "ARN of the GPU Auto Scaling Group"
  value       = module.gpu_asg.asg_arn
}

output "launch_template_id" {
  description = "ID of the Launch Template"
  value       = module.gpu_asg.launch_template_id
}

output "security_group_id" {
  description = "ID of the instance security group"
  value       = module.gpu_asg.security_group_id
}

output "log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = module.gpu_asg.log_group_name
}

output "instance_role_arn" {
  description = "ARN of the IAM instance role"
  value       = module.gpu_asg.instance_role_arn
}
