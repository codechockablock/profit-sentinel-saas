module "rds" {
  source          = "../../modules/rds"
  name_prefix     = "profitsentinel-dev"
  vpc_id          = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnets
  ecs_sg_id       = module.ecs.ecs_security_group_id  # Add output in ECS module if needed
}

module "s3" {
  source      = "../../modules/s3"
  name_prefix = "profitsentinel-dev"
}

# ECS task role/policy for S3/RDS access (add later)