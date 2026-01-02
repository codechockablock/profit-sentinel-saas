module "vpc" {
  source      = "../../modules/vpc"
  name_prefix = "profitsentinel-dev"
}

module "alb" {
  source          = "../../modules/alb"
  name_prefix     = "profitsentinel-dev"
  vpc_id          = module.vpc.vpc_id
  public_subnets  = module.vpc.public_subnets
  certificate_arn = "arn:aws:acm:us-east-1:133608785306:certificate/6c016bb3-b1f6-4928-b233-c5c690d4fbce"
}

module "ecs" {
  source               = "../../modules/ecs"
  name_prefix          = "profitsentinel-dev"
  vpc_id               = module.vpc.vpc_id
  private_subnets      = module.vpc.private_subnets
  alb_target_group_arn = module.alb.target_group_arn
  alb_sg_id            = module.alb.alb_sg_id
}

module "rds" {
  source          = "../../modules/rds"
  name_prefix     = "profitsentinel-dev"
  vpc_id          = module.vpc.vpc_id
  private_subnets = module.vpc.private_subnets
  ecs_sg_id       = module.ecs.ecs_sg_id  # ECS must output this
}

module "ecr" {
  source      = "../../modules/ecr"
  name_prefix = "profitsentinel-dev"
}

module "s3" {
  source      = "../../modules/s3"
  name_prefix = "profitsentinel-dev"
}