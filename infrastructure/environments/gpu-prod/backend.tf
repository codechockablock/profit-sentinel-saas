# =============================================================================
# Profit Sentinel v2.1.0 - Terraform Backend Configuration
# =============================================================================
# State is stored in S3 with DynamoDB locking for team collaboration.
# =============================================================================

terraform {
  backend "s3" {
    bucket         = "profitsentinel-terraform-state"
    key            = "gpu-prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "profitsentinel-terraform-locks"

    # Optional: Enable versioning on the S3 bucket for state history
    # versioning = true
  }
}

# =============================================================================
# Backend Setup Instructions
# =============================================================================
#
# Before running terraform init, create the S3 bucket and DynamoDB table:
#
# 1. Create S3 bucket for state:
#    aws s3api create-bucket \
#      --bucket profitsentinel-terraform-state \
#      --region us-east-1
#
# 2. Enable versioning:
#    aws s3api put-bucket-versioning \
#      --bucket profitsentinel-terraform-state \
#      --versioning-configuration Status=Enabled
#
# 3. Enable encryption:
#    aws s3api put-bucket-encryption \
#      --bucket profitsentinel-terraform-state \
#      --server-side-encryption-configuration '{
#        "Rules": [{
#          "ApplyServerSideEncryptionByDefault": {
#            "SSEAlgorithm": "AES256"
#          }
#        }]
#      }'
#
# 4. Create DynamoDB table for state locking:
#    aws dynamodb create-table \
#      --table-name profitsentinel-terraform-locks \
#      --attribute-definitions AttributeName=LockID,AttributeType=S \
#      --key-schema AttributeName=LockID,KeyType=HASH \
#      --billing-mode PAY_PER_REQUEST \
#      --region us-east-1
#
# =============================================================================
