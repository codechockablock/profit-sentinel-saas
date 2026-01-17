variable "name_prefix" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "private_subnets" {
  type = list(string)
}

variable "ecs_sg_id" { # Allow access from ECS tasks
  type = string
}

resource "aws_security_group" "rds" {
  name        = "${var.name_prefix}-rds-sg"
  description = "Allow inbound from ECS"
  vpc_id      = var.vpc_id

  ingress {
    description     = "Postgres from ECS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [var.ecs_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-rds-sg"
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "${var.name_prefix}-db-subnet-group"
  subnet_ids = var.private_subnets

  tags = {
    Name = "${var.name_prefix}-db-subnet-group"
  }
}

resource "aws_rds_cluster" "aurora" {
  cluster_identifier          = "${var.name_prefix}-cluster"
  engine                      = "aurora-postgresql"
  engine_mode                 = "provisioned"
  engine_version              = "15" # Latest compatible
  database_name               = "profitsentinel"
  master_username             = "adminuser"
  manage_master_user_password = true # Uses Secrets Manager automatically
  db_subnet_group_name        = aws_db_subnet_group.main.name
  vpc_security_group_ids      = [aws_security_group.rds.id]
  storage_encrypted           = true
  skip_final_snapshot         = true # Dev only—set false in prod
  deletion_protection         = false

  serverlessv2_scaling_configuration {
    min_capacity = 0.5
    max_capacity = 8.0 # Adjust for growth
  }

  tags = {
    Name = "${var.name_prefix}-aurora-cluster"
  }
}

resource "aws_rds_cluster_instance" "aurora_instances" {
  count               = 1 # Start with 1, scale later
  identifier          = "${var.name_prefix}-instance-${count.index}"
  cluster_identifier  = aws_rds_cluster.aurora.id
  instance_class      = "db.serverless"
  engine              = "aurora-postgresql"
  publicly_accessible = false

  tags = {
    Name = "${var.name_prefix}-aurora-instance-${count.index}"
  }
}

output "cluster_endpoint" {
  value = aws_rds_cluster.aurora.endpoint
}

output "reader_endpoint" {
  value = aws_rds_cluster.aurora.reader_endpoint
}

output "secret_arn" {
  value = aws_rds_cluster.aurora.master_user_secret[0].secret_arn
}