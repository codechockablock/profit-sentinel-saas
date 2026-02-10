# Profit Sentinel — Production Environment
#
# Runs the Rust+Python sidecar container (port 8001).
# Uses Supabase for auth/DB (no RDS needed).
# Minimal compute: 1 vCPU / 4GB — sufficient for production traffic.

module "vpc" {
  source      = "../../modules/vpc"
  name_prefix = "profitsentinel-prod"
}

module "alb" {
  source          = "../../modules/alb"
  name_prefix     = "profitsentinel-prod"
  vpc_id          = module.vpc.vpc_id
  public_subnets  = module.vpc.public_subnets
  certificate_arn = var.acm_certificate_arn
  target_port     = 8001
}

module "ecs" {
  source                          = "../../modules/ecs"
  name_prefix                     = "profitsentinel-prod"
  vpc_id                          = module.vpc.vpc_id
  private_subnets                 = module.vpc.private_subnets
  alb_target_group_arn            = module.alb.target_group_arn
  alb_sg_id                       = module.alb.alb_sg_id
  ecr_repository_url              = module.ecr.repository_url
  s3_bucket_name                  = module.s3.bucket_name
  anthropic_api_key_secret_arn    = var.anthropic_api_key_secret_arn
  supabase_url                    = var.supabase_url
  supabase_service_key_secret_arn = var.supabase_service_key_secret_arn
  resend_api_key_secret_arn       = var.resend_api_key_secret_arn

  # Production config — minimal compute
  container_port   = 8001
  container_cpu    = "1024"  # 1 vCPU
  container_memory = "4096"  # 4GB

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
  name_prefix = "profitsentinel-prod"
}

module "s3" {
  source      = "../../modules/s3"
  name_prefix = "profitsentinel-prod"
}

# Production does not need its own RDS — uses Supabase for auth/DB

output "alb_dns_name" {
  value       = module.alb.alb_dns_name
  description = "Production ALB DNS for api.profitsentinel.com CNAME"
}

output "repository_url" {
  value       = module.ecr.repository_url
  description = "Production ECR repo URL for Docker push"
}

output "s3_bucket_name" {
  value       = module.s3.bucket_name
  description = "Production S3 uploads bucket"
}
