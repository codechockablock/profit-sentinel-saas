variable "acm_certificate_arn" {
  description = "ARN of the ACM certificate for HTTPS"
  type        = string
  sensitive   = true
}

variable "xai_api_key_secret_arn" {
  description = "ARN of the Secrets Manager secret containing XAI_API_KEY"
  type        = string
  default     = ""
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
