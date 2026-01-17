#!/bin/bash
# =============================================================================
# Profit Sentinel v2.1.0 - GPU Instance User Data Script
# =============================================================================
# This script runs on first boot of GPU instances to:
# 1. Configure NVIDIA GPU drivers and Docker runtime
# 2. Authenticate with ECR
# 3. Pull and run the Profit Sentinel container
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration (injected by Terraform)
# -----------------------------------------------------------------------------
AWS_REGION="${aws_region}"
ECR_REPO_URL="${ecr_repository_url}"
IMAGE_TAG="${image_tag}"
LOG_GROUP="${log_group_name}"
S3_BUCKET="${s3_bucket_name}"
SUPABASE_URL="${supabase_url}"
DB_SECRET_ARN="${db_secret_arn}"
API_KEY_SECRET_ARN="${api_key_secret_arn}"
SUPABASE_SECRET_ARN="${supabase_secret_arn}"
ENVIRONMENT="${environment}"

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting Profit Sentinel GPU instance setup..."

# -----------------------------------------------------------------------------
# System Updates
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Updating system packages..."
yum update -y

# -----------------------------------------------------------------------------
# Install CloudWatch Agent
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Installing CloudWatch agent..."
yum install -y amazon-cloudwatch-agent

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<EOF
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "root"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/user-data.log",
            "log_group_name": "$LOG_GROUP",
            "log_stream_name": "{instance_id}/user-data",
            "retention_in_days": 30
          },
          {
            "file_path": "/var/log/docker.log",
            "log_group_name": "$LOG_GROUP",
            "log_stream_name": "{instance_id}/docker",
            "retention_in_days": 30
          }
        ]
      }
    }
  },
  "metrics": {
    "namespace": "ProfitSentinel/GPU",
    "metrics_collected": {
      "cpu": {
        "measurement": ["cpu_usage_idle", "cpu_usage_user", "cpu_usage_system"],
        "metrics_collection_interval": 60
      },
      "mem": {
        "measurement": ["mem_used_percent"],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": ["disk_used_percent"],
        "metrics_collection_interval": 60
      }
    }
  }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# -----------------------------------------------------------------------------
# NVIDIA GPU Setup (Deep Learning AMI has drivers pre-installed)
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Verifying NVIDIA drivers..."

# Verify NVIDIA drivers are working
if ! nvidia-smi; then
  echo "ERROR: NVIDIA drivers not functioning properly"
  exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') - NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# -----------------------------------------------------------------------------
# Docker Configuration
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Configuring Docker..."

# Ensure Docker is installed and running
if ! command -v docker &> /dev/null; then
  echo "Installing Docker..."
  amazon-linux-extras install docker -y
fi

# Start and enable Docker
systemctl start docker
systemctl enable docker

# Add ec2-user to docker group
usermod -aG docker ec2-user

# Configure NVIDIA Container Toolkit (should be pre-installed on Deep Learning AMI)
echo "$(date '+%Y-%m-%d %H:%M:%S') - Configuring NVIDIA Container Toolkit..."

# Verify nvidia-docker is available
if ! docker info 2>/dev/null | grep -q "nvidia"; then
  echo "Installing NVIDIA Container Toolkit..."

  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
    tee /etc/yum.repos.d/nvidia-docker.repo

  yum clean expire-cache
  yum install -y nvidia-docker2

  systemctl restart docker
fi

# Verify GPU is accessible from Docker
echo "$(date '+%Y-%m-%d %H:%M:%S') - Verifying Docker GPU access..."
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi || {
  echo "WARNING: Docker GPU access test failed, continuing anyway..."
}

# -----------------------------------------------------------------------------
# ECR Authentication
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Authenticating with ECR..."

# Get ECR login token
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $(echo $ECR_REPO_URL | cut -d'/' -f1)

# -----------------------------------------------------------------------------
# Fetch Secrets from Secrets Manager
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Fetching secrets..."

# Function to get secret value
get_secret() {
  local secret_arn=$1
  aws secretsmanager get-secret-value \
    --secret-id "$secret_arn" \
    --region "$AWS_REGION" \
    --query 'SecretString' \
    --output text
}

# Fetch database credentials
DB_CREDS=$(get_secret "$DB_SECRET_ARN")
DB_HOST=$(echo $DB_CREDS | jq -r '.host // .DB_HOST // empty')
DB_PORT=$(echo $DB_CREDS | jq -r '.port // .DB_PORT // "5432"')
DB_NAME=$(echo $DB_CREDS | jq -r '.dbname // .database // .DB_NAME // "profitsentinel"')
DB_USER=$(echo $DB_CREDS | jq -r '.username // .DB_USER // empty')
DB_PASS=$(echo $DB_CREDS | jq -r '.password // .DB_PASSWORD // empty')

# Fetch API key
API_KEY=$(get_secret "$API_KEY_SECRET_ARN")

# Fetch Supabase service key
SUPABASE_KEY=$(get_secret "$SUPABASE_SECRET_ARN")

# Construct DATABASE_URL
DATABASE_URL="postgresql://$DB_USER:$DB_PASS@$DB_HOST:$DB_PORT/$DB_NAME"

# -----------------------------------------------------------------------------
# Pull Docker Image
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Pulling Docker image: $ECR_REPO_URL:$IMAGE_TAG"

docker pull "$ECR_REPO_URL:$IMAGE_TAG"

# -----------------------------------------------------------------------------
# Run Container
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting Profit Sentinel container..."

# Stop existing container if running
docker stop profit-sentinel 2>/dev/null || true
docker rm profit-sentinel 2>/dev/null || true

# Run the container with GPU support
docker run -d \
  --name profit-sentinel \
  --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  --log-driver=awslogs \
  --log-opt awslogs-region=$AWS_REGION \
  --log-opt awslogs-group=$LOG_GROUP \
  --log-opt awslogs-stream=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)/app \
  --log-opt awslogs-create-group=true \
  -e ENVIRONMENT=$ENVIRONMENT \
  -e DATABASE_URL="$DATABASE_URL" \
  -e API_KEY="$API_KEY" \
  -e SUPABASE_URL="$SUPABASE_URL" \
  -e SUPABASE_SERVICE_KEY="$SUPABASE_KEY" \
  -e S3_BUCKET="$S3_BUCKET" \
  -e AWS_REGION="$AWS_REGION" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
  "$ECR_REPO_URL:$IMAGE_TAG"

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Waiting for application to become healthy..."

MAX_RETRIES=30
RETRY_INTERVAL=10
HEALTH_ENDPOINT="http://localhost:8000/health"

for i in $(seq 1 $MAX_RETRIES); do
  if curl -sf "$HEALTH_ENDPOINT" > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Application is healthy!"
    break
  fi

  if [ $i -eq $MAX_RETRIES ]; then
    echo "ERROR: Application failed to become healthy after $((MAX_RETRIES * RETRY_INTERVAL)) seconds"
    echo "Container logs:"
    docker logs profit-sentinel --tail 50
    exit 1
  fi

  echo "Health check attempt $i/$MAX_RETRIES failed, retrying in $RETRY_INTERVAL seconds..."
  sleep $RETRY_INTERVAL
done

# -----------------------------------------------------------------------------
# Setup Container Auto-restart on Failure
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - Setting up container monitoring..."

# Create systemd service for container health monitoring
cat > /etc/systemd/system/profit-sentinel-monitor.service <<EOF
[Unit]
Description=Profit Sentinel Container Health Monitor
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/local/bin/container-monitor.sh
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF

# Create monitor script
cat > /usr/local/bin/container-monitor.sh <<'MONITOR'
#!/bin/bash
while true; do
  if ! docker ps --filter "name=profit-sentinel" --filter "status=running" | grep -q profit-sentinel; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Container not running, restarting..."
    docker start profit-sentinel || docker restart profit-sentinel
  fi

  # Check container health
  if ! curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Health check failed, restarting container..."
    docker restart profit-sentinel
  fi

  sleep 30
done
MONITOR

chmod +x /usr/local/bin/container-monitor.sh
systemctl daemon-reload
systemctl enable profit-sentinel-monitor
systemctl start profit-sentinel-monitor

# -----------------------------------------------------------------------------
# Final Status
# -----------------------------------------------------------------------------
echo "$(date '+%Y-%m-%d %H:%M:%S') - GPU Instance setup complete!"
echo ""
echo "Instance Details:"
echo "  - Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "  - Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "  - Availability Zone: $(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)"
echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  - Container Image: $ECR_REPO_URL:$IMAGE_TAG"
echo "  - Health Endpoint: $HEALTH_ENDPOINT"
echo ""
echo "Container Status:"
docker ps --filter "name=profit-sentinel"
