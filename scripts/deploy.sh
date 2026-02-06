#!/usr/bin/env bash
# ==============================================================================
# PROFIT SENTINEL - PRODUCTION DEPLOYMENT SCRIPT
# ==============================================================================
#
# This script safely deploys both frontend (Vercel) and backend (AWS ECS).
#
# Fixed for reliable latest revision deployment.
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# CONFIGURATION
# ==============================================================================

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_NAME="${ECR_REPO_NAME:-profitsentinel-dev-api}"

ECS_CLUSTER="${ECS_CLUSTER:-profitsentinel-dev-cluster}"
ECS_SERVICE="${ECS_SERVICE:-profitsentinel-dev-api-service}"

TF_DIR="infrastructure/environments/dev"
TF_VAR_FILE="terraform.tfvars"

# ==============================================================================
# COLORS AND FORMATTING
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================="
    echo -e " $1"
    echo -e "==============================================================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
print_step() { echo -e "${BOLD}>>> $1${NC}"; }

confirm() {
    if [ "$AUTO_CONFIRM" = true ]; then return 0; fi
    read -p "$1 [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed or not in PATH"
        exit 1
    fi
}

# ==============================================================================
# ARGUMENTS
# ==============================================================================

AUTO_CONFIRM=false
SKIP_TESTS=false
BACKEND_ONLY=false
FRONTEND_ONLY=false
SKIP_TERRAFORM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto) AUTO_CONFIRM=true; shift ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        --backend-only) BACKEND_ONLY=true; shift ;;
        --frontend-only) FRONTEND_ONLY=true; shift ;;
        --skip-terraform) SKIP_TERRAFORM=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  --auto            Auto-confirm"
            echo "  --skip-tests      Skip tests"
            echo "  --backend-only    Backend only"
            echo "  --frontend-only   Frontend only"
            echo "  --skip-terraform  Skip Terraform"
            exit 0 ;;
        *) print_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ==============================================================================
# PRE-FLIGHT
# ==============================================================================

print_header "PROFIT SENTINEL DEPLOYMENT SCRIPT"

print_step "Running pre-flight checks..."

check_command git
check_command docker
check_command aws

if [ "$SKIP_TERRAFORM" = false ] && [ "$FRONTEND_ONLY" = false ]; then
    check_command terraform
fi

if [ -z "$AWS_ACCOUNT_ID" ] && [ "$FRONTEND_ONLY" = false ]; then
    print_error "AWS_ACCOUNT_ID not set! export AWS_ACCOUNT_ID=PLACEHOLDER_AWS_ACCOUNT_ID"
    exit 1
fi

if [ "$FRONTEND_ONLY" = false ]; then
    print_info "Verifying AWS credentials..."
    aws sts get-caller-identity &> /dev/null || { print_error "AWS credentials invalid"; exit 1; }
    print_success "AWS credentials valid"

    print_info "Checking Docker..."
    docker info &> /dev/null || { print_error "Docker not running"; exit 1; }
    print_success "Docker running"
fi

print_success "Pre-flight checks passed"

# ==============================================================================
# SECURITY
# ==============================================================================

print_header "STEP 1: SECURITY VERIFICATION"

print_step "Checking for secrets..."
if git diff --cached --name-only | grep -qE '\.env$|\.env\.|\.tfvars$|credentials|\.pem$|\.key$|secret'; then
    print_error "Secrets in staged files!"
    exit 1
fi
print_success "No secrets staged"

print_step "Verifying .gitignore..."
REQUIRED=(" .env" "*.tfvars" "*.tfstate" "credentials.json" "*.pem" "*.key")
for p in "${REQUIRED[@]}"; do
    grep -q "$p" .gitignore || print_warning "Missing .gitignore pattern: $p"
done
print_success ".gitignore verified"

# ==============================================================================
# TESTS
# ==============================================================================

if [ "$SKIP_TESTS" = false ]; then
    print_header "STEP 2: RUNNING TESTS"

    # Backend tests
    if [ -d "apps/api" ]; then
        print_step "Running backend tests..."
        (cd apps/api && poetry run pytest tests/ -v) && print_success "Backend tests passed" || {
            print_error "Backend tests failed"
            confirm "Continue anyway?" || exit 1
        }
    fi

    # Frontend tests
    if [ -d "apps/web" ]; then
        print_step "Running frontend tests..."
        (cd apps/web && npm test -- --passWithNoTests) && print_success "Frontend tests passed" || {
            print_error "Frontend tests failed"
            confirm "Continue anyway?" || exit 1
        }
    fi
else
    print_info "Skipping tests"
fi

# ==============================================================================
# GIT COMMIT & PUSH
# ==============================================================================

print_header "STEP 3: GIT COMMIT AND PUSH"

git status --short

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
print_info "Branch: $CURRENT_BRANCH"

if [ -n "$(git status --porcelain)" ]; then
    git add -A

    # Post-stage secrets scan — catches files missed by the pre-stage check
    if git diff --cached --name-only | grep -qE '\.env$|\.env\.|\.tfvars$|credentials|\.pem$|\.key$|secret'; then
        print_error "Secrets detected in staged files after git add!"
        git reset HEAD -- . > /dev/null
        exit 1
    fi

    IMAGE_TAG=$(git rev-parse --short HEAD)
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M")

    DEFAULT_MESSAGE="deploy: production release $(date +%Y%m%d-%H%M)

- Backend: image $IMAGE_TAG
- Frontend: Vercel deploy
- Deployed: $TIMESTAMP"

    if confirm "Use default commit message?"; then
        git commit -m "$DEFAULT_MESSAGE"
    else
        echo "Enter message:"
        git commit
    fi
    print_success "Committed"
fi

if confirm "Push to origin/$CURRENT_BRANCH? (Triggers Vercel)"; then
    git push origin "$CURRENT_BRANCH"
    print_success "Pushed — Vercel deploying frontend"
fi

[ "$FRONTEND_ONLY" = true ] && { print_header "FRONTEND DEPLOY COMPLETE"; exit 0; }

# ==============================================================================
# DOCKER BUILD & PUSH
# ==============================================================================

print_header "STEP 4: DOCKER BUILD AND PUSH"

IMAGE_TAG=$(git rev-parse --short HEAD)
print_info "Image tag: $IMAGE_TAG"

ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
FULL_IMAGE_NAME="$ECR_URI/$ECR_REPO_NAME"

print_step "ECR login..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URI"
print_success "Logged in"

print_step "Building image (amd64)..."
docker build --platform linux/amd64 -f apps/api/Dockerfile -t "$ECR_REPO_NAME:$IMAGE_TAG" .
print_success "Built"

print_step "Tagging..."
docker tag "$ECR_REPO_NAME:$IMAGE_TAG" "$FULL_IMAGE_NAME:$IMAGE_TAG"
docker tag "$ECR_REPO_NAME:$IMAGE_TAG" "$FULL_IMAGE_NAME:latest"

print_step "Pushing..."
docker push "$FULL_IMAGE_NAME:$IMAGE_TAG"
docker push "$FULL_IMAGE_NAME:latest"
print_success "Pushed to ECR"

# ==============================================================================
# REGISTER NEW TASK DEF & UPDATE SERVICE
# ==============================================================================

print_header "STEP 5: NEW TASK DEFINITION & SERVICE UPDATE"

CURRENT_TASK_DEF=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "$ECS_SERVICE" \
  --query 'services[0].taskDefinition' \
  --output text \
  --region "$AWS_REGION")

print_info "Current task def: $CURRENT_TASK_DEF"

print_step "Registering new revision with latest image..."
NEW_TASK_DEF=$(aws ecs register-task-definition \
  --family profitsentinel-dev-api \
  --requires-compatibilities FARGATE \
  --network-mode awsvpc \
  --cpu "512" \
  --memory "1024" \
  --execution-role-arn "arn:aws:iam::$AWS_ACCOUNT_ID:role/profitsentinel-dev-ecs-execution-role" \
  --task-role-arn "arn:aws:iam::$AWS_ACCOUNT_ID:role/ProfitSentinelECSTaskRole" \
  --container-definitions "[
    {
      \"name\": \"api\",
      \"image\": \"$FULL_IMAGE_NAME:$IMAGE_TAG\",
      \"portMappings\": [{\"containerPort\": 8000, \"protocol\": \"tcp\"}],
      \"essential\": true,
      \"logConfiguration\": {
        \"logDriver\": \"awslogs\",
        \"options\": {
          \"awslogs-group\": \"/ecs/profitsentinel-dev\",
          \"awslogs-region\": \"$AWS_REGION\",
          \"awslogs-stream-prefix\": \"api\"
        }
      }
    }
  ]" \
  --query 'taskDefinition.taskDefinitionArn' \
  --output text \
  --region "$AWS_REGION")

print_success "New task def: $NEW_TASK_DEF"

print_step "Updating service..."
aws ecs update-service \
  --cluster "$ECS_CLUSTER" \
  --service "$ECS_SERVICE" \
  --task-definition "$NEW_TASK_DEF" \
  --force-new-deployment \
  --region "$AWS_REGION"

print_success "Service updated — latest code deploying"

# ==============================================================================
# TERRAFORM (Optional)
# ==============================================================================

if [ "$SKIP_TERRAFORM" = false ]; then
    print_header "STEP 6: TERRAFORM UPDATE"

    if [ -d "$TF_DIR" ]; then
        cd "$TF_DIR"
        terraform init -input=false
        terraform plan -out=tfplan $TF_VAR_ARG || true
        if confirm "Apply Terraform?"; then
            terraform apply tfplan
            rm tfplan
            print_success "Terraform applied"
        fi
        cd - > /dev/null
    fi
fi

# ==============================================================================
# VERIFICATION
# ==============================================================================

print_header "DEPLOYMENT COMPLETE"

echo "Summary:"
echo "  Branch: $CURRENT_BRANCH"
echo "  Image: $FULL_IMAGE_NAME:$IMAGE_TAG"
echo "  Task Def: $NEW_TASK_DEF"

echo "Verify:"
echo "  curl https://api.profitsentinel.com/health"
echo "  aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE --region $AWS_REGION"

echo "Profit Sentinel deployed — leaks have nowhere to hide!"