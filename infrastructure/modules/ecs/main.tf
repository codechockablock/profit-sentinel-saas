variable "name_prefix" {
  type = string
}

variable "vpc_id" {
  type = string
}

variable "private_subnets" {
  type = list(string)
}

variable "alb_target_group_arn" {
  type = string
}

variable "alb_sg_id" {
  type = string
}

variable "ecr_repository_url" {
  description = "ECR repository URL for the API image (without tag)"
  type        = string
}

variable "container_image_tag" {
  description = "Container image tag to deploy"
  type        = string
  default     = "latest"
}

variable "s3_bucket_name" {
  description = "S3 bucket name for file uploads"
  type        = string
}

variable "supabase_url" {
  description = "Supabase project URL"
  type        = string
  default     = ""
}

variable "supabase_service_key_secret_arn" {
  description = "ARN of the Secrets Manager secret containing SUPABASE_SERVICE_KEY"
  type        = string
  default     = ""
}

variable "resend_api_key_secret_arn" {
  description = "ARN of the Secrets Manager secret containing RESEND_API_KEY for email delivery"
  type        = string
  default     = ""
}

variable "anthropic_api_key_secret_arn" {
  description = "ARN of the Secrets Manager secret containing ANTHROPIC_API_KEY"
  type        = string
  default     = ""
}

variable "container_port" {
  description = "Port the container listens on"
  type        = number
  default     = 8000
}

variable "container_cpu" {
  description = "CPU units for the task (1024 = 1 vCPU)"
  type        = string
  default     = "4096"
}

variable "container_memory" {
  description = "Memory in MiB for the task"
  type        = string
  default     = "16384"
}

variable "extra_environment" {
  description = "Additional environment variables for the container"
  type = list(object({
    name  = string
    value = string
  }))
  default = []
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
}

variable "enable_autoscaling" {
  description = "Enable ECS service autoscaling"
  type        = bool
  default     = false
}

variable "autoscaling_min_capacity" {
  description = "Minimum number of ECS tasks when autoscaling is enabled"
  type        = number
  default     = 2
}

variable "autoscaling_max_capacity" {
  description = "Maximum number of ECS tasks when autoscaling is enabled"
  type        = number
  default     = 4
}

variable "autoscaling_cpu_target" {
  description = "Target CPU utilization percentage for autoscaling"
  type        = number
  default     = 70
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

resource "aws_ecs_cluster" "main" {
  name = "${var.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.name_prefix}-cluster"
  }
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.name_prefix}"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.name_prefix}-ecs-logs"
  }
}

# IAM Role for ECS Task Execution (access to ECR, logs, etc.)
resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.name_prefix}-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role for ECS Task (runtime permissions - S3, etc.)
resource "aws_iam_role" "ecs_task" {
  name = "${var.name_prefix}-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

# S3 access policy for task role
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "${var.name_prefix}-ecs-task-s3"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_name}",
          "arn:aws:s3:::${var.s3_bucket_name}/*"
        ]
      }
    ]
  })
}

# Policy to allow execution role to read secrets
resource "aws_iam_role_policy" "ecs_execution_secrets" {
  count = var.supabase_service_key_secret_arn != "" || var.resend_api_key_secret_arn != "" || var.anthropic_api_key_secret_arn != "" ? 1 : 0
  name  = "${var.name_prefix}-ecs-execution-secrets"
  role  = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = compact([
          var.supabase_service_key_secret_arn,
          var.resend_api_key_secret_arn,
          var.anthropic_api_key_secret_arn
        ])
      }
    ]
  })
}

# ECS Task Definition with environment variables and secrets
# PERFORMANCE OPTIMIZED (v3.3): 4 vCPU / 16GB for 156K+ row processing
# Single-pass bundling with pre-normalized alias lookup targets <15s
# Valid Fargate combos: 4096 CPU supports 8192-30720 memory
resource "aws_ecs_task_definition" "api" {
  family                   = "${var.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.container_cpu
  memory                   = var.container_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${var.ecr_repository_url}:${var.container_image_tag}"
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]
      environment = concat([
        {
          name  = "S3_BUCKET_NAME"
          value = var.s3_bucket_name
        },
        {
          name  = "AWS_REGION"
          value = data.aws_region.current.name
        },
        {
          name  = "SUPABASE_URL"
          value = var.supabase_url
        },
        {
          name  = "USE_VSA_GROUNDING"
          value = "true"
        },
        {
          name  = "INCLUDE_CAUSE_DIAGNOSIS"
          value = "true"
        }
      ], var.extra_environment)
      secrets = concat(
        var.supabase_service_key_secret_arn != "" ? [
          {
            name      = "SUPABASE_SERVICE_KEY"
            valueFrom = var.supabase_service_key_secret_arn
          }
        ] : [],
        var.resend_api_key_secret_arn != "" ? [
          {
            name      = "RESEND_API_KEY"
            valueFrom = var.resend_api_key_secret_arn
          }
        ] : [],
        var.anthropic_api_key_secret_arn != "" ? [
          {
            name      = "ANTHROPIC_API_KEY"
            valueFrom = var.anthropic_api_key_secret_arn
          }
        ] : []
      )
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = data.aws_region.current.name
          awslogs-stream-prefix = "ecs"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:${var.container_port}/health/shallow || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  tags = {
    Name = "${var.name_prefix}-task-def"
  }
}

data "aws_region" "current" {}

resource "aws_security_group" "ecs" {
  name   = "${var.name_prefix}-ecs-sg"
  vpc_id = var.vpc_id

  ingress {
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [var.alb_sg_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-ecs-sg"
  }
}

resource "aws_ecs_service" "api" {
  name            = "${var.name_prefix}-api-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = var.alb_target_group_arn
    container_name   = "api"
    container_port   = var.container_port
  }

  tags = {
    Name = "${var.name_prefix}-service"
  }

  depends_on = [var.alb_target_group_arn]
}

# Autoscaling
resource "aws_appautoscaling_target" "ecs" {
  count              = var.enable_autoscaling ? 1 : 0
  max_capacity       = var.autoscaling_max_capacity
  min_capacity       = var.autoscaling_min_capacity
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  count              = var.enable_autoscaling ? 1 : 0
  name               = "${var.name_prefix}-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs[0].resource_id
  scalable_dimension = aws_appautoscaling_target.ecs[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = var.autoscaling_cpu_target
  }
}

output "ecs_sg_id" {
  value = aws_security_group.ecs.id
}

output "cluster_id" {
  value = aws_ecs_cluster.main.id
}

output "service_name" {
  value = aws_ecs_service.api.name
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}
