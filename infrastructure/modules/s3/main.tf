variable "name_prefix" {
  type = string
}

variable "enable_access_logging" {
  description = "Enable S3 access logging for audit compliance"
  type        = bool
  default     = false
}

variable "logging_bucket" {
  description = "Target S3 bucket for access logs (required when enable_access_logging = true)"
  type        = string
  default     = ""
}

resource "aws_s3_bucket" "uploads" {
  bucket = "${var.name_prefix}-uploads"

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

resource "aws_s3_bucket_versioning" "uploads" {
  bucket = aws_s3_bucket.uploads.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "uploads" {
  bucket = aws_s3_bucket.uploads.id

  # PRIVACY COMPLIANCE: Auto-delete uploaded files within 24 hours
  # This ensures files are deleted even if application-level deletion fails
  rule {
    id     = "delete-uploads-24h"
    status = "Enabled"

    filter {
      prefix = ""  # Apply to all objects
    }

    expiration {
      days = 1  # S3 minimum granularity is 1 day (24 hours)
    }
  }

  # Clean up old versions quickly for privacy
  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 1  # Changed from 90 to 1 day for privacy
    }
  }
}

resource "aws_s3_bucket_logging" "uploads" {
  count  = var.enable_access_logging ? 1 : 0
  bucket = aws_s3_bucket.uploads.id

  target_bucket = var.logging_bucket
  target_prefix = "${aws_s3_bucket.uploads.id}/"
}

output "bucket_name" {
  value = aws_s3_bucket.uploads.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.uploads.arn
}