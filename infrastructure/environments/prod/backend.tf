terraform {
  backend "s3" {
    bucket         = "profitsentinel-terraform-state-codechockablock"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}
