# Profit Sentinel — Staging Environment
#
# Runs the new Rust+Python sidecar container (port 8001)
# alongside the existing dev environment (port 8000).
# Uses the same VPC and shared infrastructure as dev.

module "vpc" {
  source      = "../../modules/vpc"
  name_prefix = "profitsentinel-staging"
}

module "alb" {
  source          = "../../modules/alb"
  name_prefix     = "profitsentinel-staging"
  vpc_id          = module.vpc.vpc_id
  public_subnets  = module.vpc.public_subnets
  certificate_arn = var.acm_certificate_arn
  target_port     = 8001
}

module "ecs" {
  source                          = "../../modules/ecs"
  name_prefix                     = "profitsentinel-staging"
  vpc_id                          = module.vpc.vpc_id
  private_subnets                 = module.vpc.private_subnets
  alb_target_group_arn            = module.alb.target_group_arn
  alb_sg_id                       = module.alb.alb_sg_id
  ecr_repository_url              = module.ecr.repository_url
  s3_bucket_name                  = module.s3.bucket_name
  xai_api_key_secret_arn          = var.xai_api_key_secret_arn
  supabase_url                    = var.supabase_url
  supabase_service_key_secret_arn = var.supabase_service_key_secret_arn
  resend_api_key_secret_arn       = var.resend_api_key_secret_arn

  # Sidecar-specific config
  container_port   = 8001
  container_cpu    = "2048"  # 2 vCPU (Rust pipeline is fast enough)
  container_memory = "8192"  # 8GB (sufficient for 36K rows in ~3s)

  extra_environment = [
    {
      name  = "SIDECAR_DEV_MODE"
      value = "false"
    },
    {
      name  = "SENTINEL_BIN"
      value = "/app/sentinel-server"
    }
  ]
}

module "ecr" {
  source      = "../../modules/ecr"
  name_prefix = "profitsentinel-staging"
}

module "s3" {
  source      = "../../modules/s3"
  name_prefix = "profitsentinel-staging"
}

# Staging does not need its own RDS — uses Supabase for auth/DB

output "alb_dns_name" {
  value       = module.alb.alb_dns_name
  description = "Staging ALB DNS for domain pointing"
}

output "repository_url" {
  value       = module.ecr.repository_url
  description = "Staging ECR repo URL for Docker push"
}

output "s3_bucket_name" {
  value       = module.s3.bucket_name
  description = "Staging S3 uploads bucket"
}
