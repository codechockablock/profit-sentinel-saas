# =============================================================================
# Profit Sentinel v2.1.0 - GPU ASG Module Outputs
# =============================================================================

# -----------------------------------------------------------------------------
# Auto Scaling Group Outputs
# -----------------------------------------------------------------------------

output "asg_id" {
  description = "ID of the Auto Scaling Group"
  value       = aws_autoscaling_group.gpu.id
}

output "asg_arn" {
  description = "ARN of the Auto Scaling Group"
  value       = aws_autoscaling_group.gpu.arn
}

output "asg_name" {
  description = "Name of the Auto Scaling Group"
  value       = aws_autoscaling_group.gpu.name
}

# -----------------------------------------------------------------------------
# Launch Template Outputs
# -----------------------------------------------------------------------------

output "launch_template_id" {
  description = "ID of the Launch Template"
  value       = aws_launch_template.gpu.id
}

output "launch_template_arn" {
  description = "ARN of the Launch Template"
  value       = aws_launch_template.gpu.arn
}

output "launch_template_latest_version" {
  description = "Latest version of the Launch Template"
  value       = aws_launch_template.gpu.latest_version
}

# -----------------------------------------------------------------------------
# IAM Outputs
# -----------------------------------------------------------------------------

output "instance_role_arn" {
  description = "ARN of the IAM instance role"
  value       = aws_iam_role.gpu_instance.arn
}

output "instance_role_name" {
  description = "Name of the IAM instance role"
  value       = aws_iam_role.gpu_instance.name
}

output "instance_profile_arn" {
  description = "ARN of the IAM instance profile"
  value       = aws_iam_instance_profile.gpu_instance.arn
}

output "instance_profile_name" {
  description = "Name of the IAM instance profile"
  value       = aws_iam_instance_profile.gpu_instance.name
}

# -----------------------------------------------------------------------------
# Security Group Outputs
# -----------------------------------------------------------------------------

output "security_group_id" {
  description = "ID of the instance security group"
  value       = aws_security_group.gpu_instances.id
}

output "security_group_arn" {
  description = "ARN of the instance security group"
  value       = aws_security_group.gpu_instances.arn
}

# -----------------------------------------------------------------------------
# CloudWatch Outputs
# -----------------------------------------------------------------------------

output "log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.app.name
}

output "log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.app.arn
}

# -----------------------------------------------------------------------------
# Scaling Policy Outputs
# -----------------------------------------------------------------------------

output "scale_up_policy_arn" {
  description = "ARN of the scale-up policy"
  value       = aws_autoscaling_policy.scale_up.arn
}

output "scale_down_policy_arn" {
  description = "ARN of the scale-down policy"
  value       = aws_autoscaling_policy.scale_down.arn
}

# -----------------------------------------------------------------------------
# CloudWatch Alarm Outputs
# -----------------------------------------------------------------------------

output "cpu_high_alarm_arn" {
  description = "ARN of the high CPU alarm"
  value       = aws_cloudwatch_metric_alarm.cpu_high.arn
}

output "cpu_low_alarm_arn" {
  description = "ARN of the low CPU alarm"
  value       = aws_cloudwatch_metric_alarm.cpu_low.arn
}

output "unhealthy_hosts_alarm_arn" {
  description = "ARN of the unhealthy hosts alarm"
  value       = aws_cloudwatch_metric_alarm.unhealthy_hosts.arn
}

output "high_latency_alarm_arn" {
  description = "ARN of the high latency alarm"
  value       = aws_cloudwatch_metric_alarm.high_latency.arn
}
