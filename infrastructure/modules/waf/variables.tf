variable "name_prefix" {
  description = "Prefix for resource names (e.g. profitsentinel-prod)"
  type        = string
}

variable "alb_arn" {
  description = "ARN of the ALB to associate with the WAF WebACL"
  type        = string
}
