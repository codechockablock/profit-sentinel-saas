module "vpc" {
  source      = "../../modules/vpc"
  name_prefix = "profitsentinel-dev"
}

module "alb" {
  source          = "../../modules/alb"
  name_prefix     = "profitsentinel-dev"
  vpc_id          = module.vpc.vpc_id
  public_subnets  = module.vpc.public_subnets
  certificate_arn = var.acm_certificate_arn
}

module "ecs" {
  source                          = "../../modules/ecs"
  name_prefix                     = "profitsentinel-dev"
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
}

module "rds" {
  source          = "../../modules/rds"
  name_prefix     = "profitsentinel-dev"
  vpc_id          = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnets
  ecs_sg_id       = module.ecs.ecs_sg_id # ECS must output this
}

module "ecr" {
  source      = "../../modules/ecr"
  name_prefix = "profitsentinel-dev"
}

module "s3" {
  source      = "../../modules/s3"
  name_prefix = "profitsentinel-dev"
}

output "alb_dns_name" {
  value       = module.alb.alb_dns_name
  description = "ALB DNS for domain pointing"
}

output "repository_url" {
  value       = module.ecr.repository_url
  description = "ECR repo URL for Docker push"
}

output "db_endpoint" {
  value       = module.rds.cluster_endpoint
  description = "RDS Aurora endpoint"
}

output "s3_bucket_name" {
  value       = module.s3.bucket_name
  description = "S3 uploads bucket"
}