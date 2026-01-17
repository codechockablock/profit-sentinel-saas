# =============================================================================
# Profit Sentinel v2.1.0 - GPU Auto Scaling Group Module
# =============================================================================
#
# Creates an Auto Scaling Group with NVIDIA T4 GPU instances (g4dn.xlarge)
# for running the hybrid anomaly detection pipeline.
#
# Key Metrics:
# - Baseline Avg F1: 82.4%
# - Baseline Avg Recall: 97.1%
# - Resonator Convergence: 100%
# - Expected GPU Speedup: 5-10x
#
# =============================================================================

# -----------------------------------------------------------------------------
# Data Sources
# -----------------------------------------------------------------------------

# Get the latest AWS Deep Learning AMI with NVIDIA drivers
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch * (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "state"
    values = ["available"]
  }
}

# Get current AWS region
data "aws_region" "current" {}

# Get current AWS account ID
data "aws_caller_identity" "current" {}

# -----------------------------------------------------------------------------
# IAM Role for EC2 Instances
# -----------------------------------------------------------------------------

resource "aws_iam_role" "gpu_instance" {
  name = "${var.name_prefix}-gpu-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# ECR Pull Policy
resource "aws_iam_role_policy" "ecr_pull" {
  name = "${var.name_prefix}-ecr-pull"
  role = aws_iam_role.gpu_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# S3 Access Policy
resource "aws_iam_role_policy" "s3_access" {
  name = "${var.name_prefix}-s3-access"
  role = aws_iam_role.gpu_instance.id

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
          var.s3_bucket_arn,
          "${var.s3_bucket_arn}/*"
        ]
      }
    ]
  })
}

# Secrets Manager Access Policy
resource "aws_iam_role_policy" "secrets_access" {
  name = "${var.name_prefix}-secrets-access"
  role = aws_iam_role.gpu_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = var.secret_arns
      }
    ]
  })
}

# CloudWatch Logs Policy
resource "aws_iam_role_policy" "cloudwatch_logs" {
  name = "${var.name_prefix}-cloudwatch-logs"
  role = aws_iam_role.gpu_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.id}:${data.aws_caller_identity.current.account_id}:log-group:/profitsentinel/*"
      }
    ]
  })
}

# CloudWatch Metrics Policy
resource "aws_iam_role_policy" "cloudwatch_metrics" {
  name = "${var.name_prefix}-cloudwatch-metrics"
  role = aws_iam_role.gpu_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "ProfitSentinel"
          }
        }
      }
    ]
  })
}

# SSM for Parameter Store access (optional)
resource "aws_iam_role_policy_attachment" "ssm_managed" {
  role       = aws_iam_role.gpu_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Instance Profile
resource "aws_iam_instance_profile" "gpu_instance" {
  name = "${var.name_prefix}-gpu-instance-profile"
  role = aws_iam_role.gpu_instance.name

  tags = var.tags
}

# -----------------------------------------------------------------------------
# Security Group
# -----------------------------------------------------------------------------

resource "aws_security_group" "gpu_instances" {
  name        = "${var.name_prefix}-gpu-instances-sg"
  description = "Security group for GPU instances running Profit Sentinel"
  vpc_id      = var.vpc_id

  # Allow inbound from ALB
  ingress {
    description     = "HTTP from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [var.alb_security_group_id]
  }

  # Allow SSH from bastion (optional, for debugging)
  dynamic "ingress" {
    for_each = var.enable_ssh ? [1] : []
    content {
      description = "SSH from bastion"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = var.ssh_cidr_blocks
    }
  }

  # Allow all outbound
  egress {
    description = "Allow all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-gpu-instances-sg"
  })
}

# -----------------------------------------------------------------------------
# Launch Template
# -----------------------------------------------------------------------------

resource "aws_launch_template" "gpu" {
  name          = "${var.name_prefix}-gpu-launch-template"
  description   = "Launch template for Profit Sentinel GPU instances"
  image_id      = data.aws_ami.deep_learning.id
  instance_type = var.instance_type

  # IAM Instance Profile
  iam_instance_profile {
    arn = aws_iam_instance_profile.gpu_instance.arn
  }

  # Network configuration
  network_interfaces {
    associate_public_ip_address = false
    security_groups             = [aws_security_group.gpu_instances.id]
    delete_on_termination       = true
  }

  # Block device mappings
  block_device_mappings {
    device_name = "/dev/sda1"

    ebs {
      volume_size           = var.root_volume_size
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
      encrypted             = true
    }
  }

  # User data script
  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    aws_region          = data.aws_region.current.id
    ecr_repository_url  = var.ecr_repository_url
    image_tag           = var.image_tag
    s3_bucket_name      = var.s3_bucket_name
    log_group_name      = var.log_group_name
    environment         = var.environment
    db_secret_arn       = var.db_secret_arn
    api_key_secret_arn  = var.api_key_secret_arn
    supabase_url        = var.supabase_url
    supabase_secret_arn = var.supabase_secret_arn
  }))

  # Monitoring
  monitoring {
    enabled = true
  }

  # Metadata options (IMDSv2 required for security)
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
    instance_metadata_tags      = "enabled"
  }

  # Tags
  tag_specifications {
    resource_type = "instance"
    tags = merge(var.tags, {
      Name = "${var.name_prefix}-gpu-instance"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(var.tags, {
      Name = "${var.name_prefix}-gpu-volume"
    })
  }

  tags = var.tags

  lifecycle {
    create_before_destroy = true
  }
}

# -----------------------------------------------------------------------------
# Auto Scaling Group
# -----------------------------------------------------------------------------

resource "aws_autoscaling_group" "gpu" {
  name                      = "${var.name_prefix}-gpu-asg"
  desired_capacity          = var.desired_capacity
  min_size                  = var.min_size
  max_size                  = var.max_size
  vpc_zone_identifier       = var.private_subnet_ids
  target_group_arns         = [var.target_group_arn]
  health_check_type         = "ELB"
  health_check_grace_period = 300

  # Mixed instances policy for Spot + On-Demand
  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = var.on_demand_base_capacity
      on_demand_percentage_above_base_capacity = var.on_demand_percentage
      spot_allocation_strategy                 = "lowest-price"
      spot_instance_pools                      = 2
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.gpu.id
        version            = "$Latest"
      }

      override {
        instance_type = "g4dn.xlarge"
      }

      override {
        instance_type = "g4dn.2xlarge"
      }
    }
  }

  # Instance refresh for rolling deployments
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
      instance_warmup        = 300
    }
  }

  # Tags
  dynamic "tag" {
    for_each = merge(var.tags, {
      Name = "${var.name_prefix}-gpu-instance"
    })
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }

  lifecycle {
    create_before_destroy = true
    ignore_changes        = [desired_capacity]
  }
}

# -----------------------------------------------------------------------------
# Auto Scaling Policies
# -----------------------------------------------------------------------------

# Scale up policy
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "${var.name_prefix}-scale-up"
  autoscaling_group_name = aws_autoscaling_group.gpu.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = 1
  cooldown               = 300
}

# Scale down policy
resource "aws_autoscaling_policy" "scale_down" {
  name                   = "${var.name_prefix}-scale-down"
  autoscaling_group_name = aws_autoscaling_group.gpu.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = -1
  cooldown               = 300
}

# CPU high alarm
resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.name_prefix}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 70
  alarm_description   = "Scale up when CPU exceeds 70%"
  alarm_actions       = [aws_autoscaling_policy.scale_up.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.gpu.name
  }

  tags = var.tags
}

# CPU low alarm
resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name          = "${var.name_prefix}-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 30
  alarm_description   = "Scale down when CPU below 30%"
  alarm_actions       = [aws_autoscaling_policy.scale_down.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.gpu.name
  }

  tags = var.tags
}

# -----------------------------------------------------------------------------
# CloudWatch Log Group
# -----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "app" {
  name              = var.log_group_name
  retention_in_days = var.log_retention_days

  tags = var.tags
}

# -----------------------------------------------------------------------------
# CloudWatch Alarms for Application Metrics
# -----------------------------------------------------------------------------

# Unhealthy host alarm
resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${var.name_prefix}-unhealthy-hosts"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "UnHealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Alert when any host is unhealthy"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    TargetGroup  = var.target_group_arn_suffix
    LoadBalancer = var.alb_arn_suffix
  }

  tags = var.tags
}

# 5xx errors alarm
resource "aws_cloudwatch_metric_alarm" "http_5xx" {
  alarm_name          = "${var.name_prefix}-http-5xx"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "HTTPCode_Target_5XX_Count"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Alert when 5xx errors exceed 10 per minute"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []
  treat_missing_data  = "notBreaching"

  dimensions = {
    TargetGroup  = var.target_group_arn_suffix
    LoadBalancer = var.alb_arn_suffix
  }

  tags = var.tags
}

# High latency alarm
resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${var.name_prefix}-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 300
  extended_statistic  = "p99"
  threshold           = 30
  alarm_description   = "Alert when P99 latency exceeds 30 seconds"
  alarm_actions       = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []

  dimensions = {
    TargetGroup  = var.target_group_arn_suffix
    LoadBalancer = var.alb_arn_suffix
  }

  tags = var.tags
}
