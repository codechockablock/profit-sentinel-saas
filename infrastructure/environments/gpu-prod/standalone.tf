# =============================================================================
# Profit Sentinel v2.1.0 - Standalone GPU Deployment
# =============================================================================
# Creates all necessary infrastructure for GPU deployment including:
# - VPC with public/private subnets
# - NAT Gateway
# - Application Load Balancer
# - GPU Auto Scaling Group
# =============================================================================

# -----------------------------------------------------------------------------
# VPC and Networking
# -----------------------------------------------------------------------------

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.name_prefix}-vpc"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.name_prefix}-igw"
  }
}

# Public Subnets
resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.0.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.name_prefix}-public-a"
    Tier = "public"
  }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.name_prefix}-public-b"
    Tier = "public"
  }
}

# Private Subnets - Using b and c for better GPU Spot availability
resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "${var.aws_region}b"

  tags = {
    Name = "${var.name_prefix}-private-b"
    Tier = "private"
  }
}

resource "aws_subnet" "private_b" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.3.0/24"
  availability_zone = "${var.aws_region}c"

  tags = {
    Name = "${var.name_prefix}-private-c"
    Tier = "private"
  }
}

# Elastic IP for NAT Gateway
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name = "${var.name_prefix}-nat-eip"
  }

  depends_on = [aws_internet_gateway.main]
}

# NAT Gateway
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public_a.id

  tags = {
    Name = "${var.name_prefix}-nat"
  }

  depends_on = [aws_internet_gateway.main]
}

# Public Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.name_prefix}-public-rt"
  }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

# Private Route Table
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "${var.name_prefix}-private-rt"
  }
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private.id
}

# -----------------------------------------------------------------------------
# S3 Bucket for uploads
# -----------------------------------------------------------------------------

resource "aws_s3_bucket" "uploads" {
  bucket = "${var.name_prefix}-uploads-${var.environment}"

  tags = {
    Name = "${var.name_prefix}-uploads"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# -----------------------------------------------------------------------------
# Application Load Balancer
# -----------------------------------------------------------------------------

resource "aws_security_group" "alb" {
  name        = "${var.name_prefix}-alb-sg"
  description = "Security group for ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-alb-sg"
  }
}

resource "aws_lb" "main" {
  name               = "${var.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = [aws_subnet.public_a.id, aws_subnet.public_b.id]

  tags = {
    Name = "${var.name_prefix}-alb"
  }
}

resource "aws_lb_target_group" "gpu" {
  name        = "${var.name_prefix}-gpu-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "instance"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 10
    unhealthy_threshold = 3
  }

  tags = {
    Name = "${var.name_prefix}-gpu-tg"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.gpu.arn
  }
}

# -----------------------------------------------------------------------------
# Reference secrets
# -----------------------------------------------------------------------------

data "aws_secretsmanager_secret" "db" {
  name = "profitsentinel/db-credentials"
}

data "aws_secretsmanager_secret" "api_key" {
  name = "profitsentinel/api-key"
}

data "aws_secretsmanager_secret" "supabase" {
  name = "profitsentinel/supabase-service-key"
}

# -----------------------------------------------------------------------------
# GPU Auto Scaling Group Module
# -----------------------------------------------------------------------------

module "gpu_asg" {
  source = "../../modules/gpu-asg"

  name_prefix = var.name_prefix
  environment = var.environment

  # Networking
  vpc_id             = aws_vpc.main.id
  private_subnet_ids = [aws_subnet.private_a.id, aws_subnet.private_b.id]

  # Load Balancer
  alb_security_group_id   = aws_security_group.alb.id
  target_group_arn        = aws_lb_target_group.gpu.arn
  target_group_arn_suffix = aws_lb_target_group.gpu.arn_suffix
  alb_arn_suffix          = aws_lb.main.arn_suffix

  # Container Registry
  ecr_repository_url = "PLACEHOLDER_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/profit-sentinel-gpu"
  image_tag          = var.image_tag

  # Storage
  s3_bucket_arn  = aws_s3_bucket.uploads.arn
  s3_bucket_name = aws_s3_bucket.uploads.id

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

  # Instance Configuration - Cost optimized
  instance_type    = var.instance_type
  desired_capacity = 1  # Start with 1 for cost savings
  min_size         = 1
  max_size         = 3

  # Using On-Demand instances (Spot quota exceeded)
  # To switch back to Spot instances for cost savings, request quota increase:
  # https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas
  on_demand_base_capacity = 1
  on_demand_percentage    = 100

  # Storage
  root_volume_size = 100  # Reduced from 125

  # Logging
  log_group_name     = var.log_group_name
  log_retention_days = var.log_retention_days

  # SSH (disabled in production)
  enable_ssh      = var.enable_ssh
  ssh_cidr_blocks = var.ssh_cidr_blocks

  # Alerting (disabled for cost)
  alarm_sns_topic_arn = ""

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "alb_dns_name" {
  description = "ALB DNS name - use this to access the API"
  value       = aws_lb.main.dns_name
}

output "alb_url" {
  description = "Full ALB URL"
  value       = "http://${aws_lb.main.dns_name}"
}

output "asg_name" {
  description = "Name of the GPU Auto Scaling Group"
  value       = module.gpu_asg.asg_name
}

output "s3_bucket" {
  description = "S3 bucket for uploads"
  value       = aws_s3_bucket.uploads.id
}

output "ecr_repository" {
  description = "ECR repository URL"
  value       = "PLACEHOLDER_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/profit-sentinel-gpu"
}
