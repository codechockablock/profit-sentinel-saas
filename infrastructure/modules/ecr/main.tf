variable "name_prefix" {
  type = string
}

resource "aws_ecr_repository" "api" {
  name = "${var.name_prefix}-api"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "${var.name_prefix}-api"
  }
}

output "repository_url" {
  value = aws_ecr_repository.api.repository_url
}