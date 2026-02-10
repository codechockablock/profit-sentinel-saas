#!/usr/bin/env bash
set -euo pipefail

# teardown-eval.sh — Tear down staging evaluation environment
#
# Brings costs back to near-zero by:
#   1. Scaling ECS staging service to 0 tasks
#   2. Destroying NAT Gateway, ALB, and EIP via Terraform
#   3. Stopping Aurora RDS if it's running
#
# Usage: ./scripts/teardown-eval.sh
# Cost after teardown: ~$1-2/month (Route53, Secrets Manager, ECR storage)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
STAGING_DIR="$REPO_ROOT/infrastructure/environments/staging"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "  Profit Sentinel — Eval Teardown"
echo "========================================="
echo ""

# Pre-flight
for cmd in aws terraform; do
  if ! command -v "$cmd" &>/dev/null; then
    echo -e "${RED}Error: $cmd not found${NC}"
    exit 1
  fi
done

# 1. Scale ECS to 0
echo -e "${YELLOW}[1/4] Scaling ECS staging service to 0 tasks...${NC}"
aws ecs update-service \
  --cluster profitsentinel-staging-cluster \
  --service profitsentinel-staging-api-service \
  --desired-count 0 \
  --query 'service.desiredCount' \
  --output text
echo -e "${GREEN}  ✓ ECS desired count set to 0${NC}"

# 2. Wait for tasks to drain
echo -e "${YELLOW}[2/4] Waiting for running tasks to drain (up to 60s)...${NC}"
for i in $(seq 1 12); do
  RUNNING=$(aws ecs describe-services \
    --cluster profitsentinel-staging-cluster \
    --services profitsentinel-staging-api-service \
    --query 'services[0].runningCount' --output text 2>/dev/null || echo "0")
  if [ "$RUNNING" = "0" ]; then
    echo -e "${GREEN}  ✓ All tasks drained${NC}"
    break
  fi
  echo "  Waiting... ($RUNNING tasks still running)"
  sleep 5
done

# 3. Terraform destroy NAT + ALB (targeted)
echo -e "${YELLOW}[3/4] Destroying NAT Gateway, ALB, and EIP via Terraform...${NC}"
cd "$STAGING_DIR"
terraform destroy \
  -var-file=terraform.tfvars \
  -target=module.alb.aws_lb_listener.https \
  -target=module.alb.aws_lb_listener.http_redirect \
  -target=module.alb.aws_lb.main \
  -target=module.alb.aws_lb_target_group.api \
  -target=module.vpc.aws_route_table_association.private[0] \
  -target=module.vpc.aws_route_table_association.private[1] \
  -target=module.vpc.aws_route_table.private[0] \
  -target=module.vpc.aws_nat_gateway.nat[0] \
  -target=module.vpc.aws_eip.nat[0] \
  -auto-approve
echo -e "${GREEN}  ✓ NAT Gateway, ALB, and EIP destroyed${NC}"

# 4. Stop Aurora if running
echo -e "${YELLOW}[4/4] Checking Aurora RDS cluster...${NC}"
AURORA_STATUS=$(aws rds describe-db-clusters \
  --db-cluster-identifier profitsentinel-dev-cluster \
  --query 'DBClusters[0].Status' --output text 2>/dev/null || echo "not-found")

if [ "$AURORA_STATUS" = "available" ]; then
  aws rds stop-db-cluster --db-cluster-identifier profitsentinel-dev-cluster >/dev/null 2>&1
  echo -e "${GREEN}  ✓ Aurora cluster stop initiated${NC}"
elif [ "$AURORA_STATUS" = "stopped" ]; then
  echo -e "${GREEN}  ✓ Aurora already stopped${NC}"
else
  echo -e "${YELLOW}  ⚠ Aurora status: $AURORA_STATUS (no action taken)${NC}"
fi

echo ""
echo "========================================="
echo -e "${GREEN}  Teardown complete!${NC}"
echo "========================================="
echo ""
echo "Remaining monthly costs:"
echo "  Route 53:        ~\$0.50 (hosted zone)"
echo "  Secrets Manager: ~\$1.20 (3 secrets)"
echo "  ECR:             ~\$2-3  (stored images)"
echo "  S3:              ~\$0.01 (minimal storage)"
echo "  Total:           ~\$4-5/month"
echo ""
echo "To spin up again:"
echo "  cd infrastructure/environments/staging"
echo "  terraform apply -var-file=terraform.tfvars"
