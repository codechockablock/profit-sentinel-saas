terraform {
  backend "s3" {
    bucket         = "profitsentinel-terraform-state-codechockablock"  # 
    key            = "dev/terraform.tfstate"
    region         = "us-east-1"  # bucket region
    dynamodb_table = "terraform-locks"  # Your table name
    encrypt        = true
  }
}