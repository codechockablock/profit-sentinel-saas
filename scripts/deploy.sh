#!/usr/bin/env bash
# ==============================================================================
# PROFIT SENTINEL - PRODUCTION DEPLOYMENT SCRIPT
# ==============================================================================
#
# This script safely deploys both frontend (Vercel) and backend (AWS ECS).
#
# PREREQUISITES:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed and running
#   - Terraform installed (for infrastructure updates)
#   - Git repository with no uncommitted secrets
#
# USAGE:
#   ./scripts/deploy.sh                    # Interactive mode (prompts for confirmation)
#   ./scripts/deploy.sh --auto             # Auto-confirm (for CI/CD)
#   ./scripts/deploy.sh --skip-tests       # Skip test verification
#   ./scripts/deploy.sh --backend-only     # Only deploy backend
#   ./scripts/deploy.sh --frontend-only    # Only deploy frontend (git push)
#
# ==============================================================================

set -euo pipefail

# ==============================================================================
# CONFIGURATION - CUSTOMIZE THESE VALUES
# ==============================================================================

# AWS Configuration
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"           # Your AWS Account ID
AWS_REGION="${AWS_REGION:-us-east-1}"          # AWS Region
ECR_REPO_NAME="${ECR_REPO_NAME:-profitsentinel-dev-api}"

# ECS Configuration
ECS_CLUSTER="${ECS_CLUSTER:-profitsentinel-dev-cluster}"
ECS_SERVICE="${ECS_SERVICE:-profitsentinel-dev-api-service}"

# Terraform Configuration
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
NC='\033[0m' # No Color
BOLD='\033[1m'

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}=============================================================================="
    echo -e " $1"
    echo -e "==============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_step() {
    echo -e "${BOLD}>>> $1${NC}"
}

confirm() {
    if [ "$AUTO_CONFIRM" = true ]; then
        return 0
    fi
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
# PARSE COMMAND LINE ARGUMENTS
# ==============================================================================

AUTO_CONFIRM=false
SKIP_TESTS=false
BACKEND_ONLY=false
FRONTEND_ONLY=false
SKIP_TERRAFORM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_CONFIRM=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --frontend-only)
            FRONTEND_ONLY=true
            shift
            ;;
        --skip-terraform)
            SKIP_TERRAFORM=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --auto            Auto-confirm all prompts"
            echo "  --skip-tests      Skip test verification"
            echo "  --backend-only    Only deploy backend (Docker/ECS)"
            echo "  --frontend-only   Only deploy frontend (git push)"
            echo "  --skip-terraform  Skip Terraform apply step"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================

print_header "PROFIT SENTINEL DEPLOYMENT SCRIPT"

print_step "Running pre-flight checks..."

# Check required tools
check_command git
check_command docker
check_command aws

if [ "$SKIP_TERRAFORM" = false ] && [ "$FRONTEND_ONLY" = false ]; then
    check_command terraform
fi

# Verify AWS Account ID is set
if [ -z "$AWS_ACCOUNT_ID" ] && [ "$FRONTEND_ONLY" = false ]; then
    print_error "AWS_ACCOUNT_ID is not set!"
    echo ""
    echo "Set it with:"
    echo "  export AWS_ACCOUNT_ID=\"123456789012\""
    echo ""
    echo "Or find it in AWS Console (top-right dropdown)"
    exit 1
fi

# Verify AWS credentials
if [ "$FRONTEND_ONLY" = false ]; then
    print_info "Verifying AWS credentials..."
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials are not configured or invalid"
        echo "Run: aws configure"
        exit 1
    fi
    print_success "AWS credentials valid"
fi

# Verify Docker is running
if [ "$FRONTEND_ONLY" = false ]; then
    print_info "Checking Docker daemon..."
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        echo "Start Docker Desktop or run: sudo systemctl start docker"
        exit 1
    fi
    print_success "Docker is running"
fi

print_success "All pre-flight checks passed"

# ==============================================================================
# SECURITY VERIFICATION
# ==============================================================================

print_header "STEP 1: SECURITY VERIFICATION"

print_step "Checking for secrets in staged files..."

# Check if there are any staged changes
if git diff --cached --name-only | grep -qE '\.env$|\.env\.|\.tfvars$|credentials|\.pem$|\.key$|secret'; then
    print_error "DANGER: Potential secrets detected in staged files!"
    echo ""
    git diff --cached --name-only | grep -E '\.env$|\.env\.|\.tfvars$|credentials|\.pem$|\.key$|secret' || true
    echo ""
    print_error "Unstage these files before continuing:"
    echo "  git reset HEAD <filename>"
    exit 1
fi

print_success "No secrets detected in staged files"

# Verify .gitignore has critical patterns
print_step "Verifying .gitignore coverage..."

REQUIRED_PATTERNS=(".env" "*.tfvars" "*.tfstate" "credentials.json" "*.pem" "*.key")
MISSING_PATTERNS=()

for pattern in "${REQUIRED_PATTERNS[@]}"; do
    if ! grep -q "^${pattern}$\|^${pattern}/" .gitignore 2>/dev/null; then
        # Check for similar patterns
        if ! grep -q "${pattern}" .gitignore 2>/dev/null; then
            MISSING_PATTERNS+=("$pattern")
        fi
    fi
done

if [ ${#MISSING_PATTERNS[@]} -gt 0 ]; then
    print_warning "Some security patterns might be missing from .gitignore:"
    for pattern in "${MISSING_PATTERNS[@]}"; do
        echo "  - $pattern"
    done
    if ! confirm "Continue anyway?"; then
        exit 1
    fi
else
    print_success ".gitignore has all required security patterns"
fi

# ==============================================================================
# RUN TESTS (Optional)
# ==============================================================================

if [ "$SKIP_TESTS" = false ]; then
    print_header "STEP 2: RUNNING TESTS"

    print_step "Running backend tests..."
    if [ -d "apps/api" ]; then
        (
            cd apps/api
            if [ -f "venv/bin/activate" ]; then
                source venv/bin/activate
            fi
            export PYTHONPATH="$PWD:$PWD/src:$PWD/../../packages/vsa-core/src:$PWD/../../packages/sentinel-engine/src"
            if python -m pytest tests/ -v --tb=short 2>/dev/null; then
                print_success "Backend tests passed"
            else
                print_error "Backend tests failed!"
                if ! confirm "Continue deployment anyway?"; then
                    exit 1
                fi
            fi
        )
    else
        print_warning "Backend directory not found, skipping tests"
    fi

    print_step "Running frontend tests..."
    if [ -d "apps/web" ]; then
        (
            cd apps/web
            if npm run test -- --passWithNoTests 2>/dev/null; then
                print_success "Frontend tests passed"
            else
                print_error "Frontend tests failed!"
                if ! confirm "Continue deployment anyway?"; then
                    exit 1
                fi
            fi
        )
    else
        print_warning "Frontend directory not found, skipping tests"
    fi
else
    print_info "Skipping tests (--skip-tests flag)"
fi

# ==============================================================================
# GIT COMMIT AND PUSH
# ==============================================================================

print_header "STEP 3: GIT COMMIT AND PUSH"

# Get current git status
print_step "Checking git status..."
git status --short

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
print_info "Current branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "main" ]; then
    print_warning "You are not on the 'main' branch!"
    if ! confirm "Continue on branch '$CURRENT_BRANCH'?"; then
        exit 1
    fi
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    print_step "Staging changes..."
    git add -A

    # Generate commit message
    IMAGE_TAG=$(git rev-parse --short HEAD 2>/dev/null || echo "initial")
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    DEFAULT_MESSAGE="deploy: production release $(date +%Y%m%d-%H%M)

- Backend: Docker image tag ${IMAGE_TAG}
- Frontend: Vercel auto-deploy triggered
- All security checks passed
- Deployed at: ${TIMESTAMP}"

    echo ""
    echo "Default commit message:"
    echo "---"
    echo "$DEFAULT_MESSAGE"
    echo "---"
    echo ""

    if confirm "Use this commit message?"; then
        COMMIT_MESSAGE="$DEFAULT_MESSAGE"
    else
        echo "Enter your commit message (end with Ctrl+D):"
        COMMIT_MESSAGE=$(cat)
    fi

    print_step "Creating commit..."
    git commit -m "$COMMIT_MESSAGE"
    print_success "Commit created"
else
    print_info "No changes to commit"
fi

# Push to remote
if ! confirm "Push to origin/$CURRENT_BRANCH? (This triggers Vercel auto-deploy)"; then
    print_warning "Skipping git push"
else
    print_step "Pushing to origin/$CURRENT_BRANCH..."
    git push origin "$CURRENT_BRANCH"
    print_success "Pushed to origin/$CURRENT_BRANCH"
    print_info "Frontend deployment triggered on Vercel"
fi

# If frontend only, we're done
if [ "$FRONTEND_ONLY" = true ]; then
    print_header "DEPLOYMENT COMPLETE (FRONTEND ONLY)"
    echo "Frontend will be deployed automatically by Vercel."
    echo ""
    echo "Check deployment status at: https://vercel.com/dashboard"
    exit 0
fi

# ==============================================================================
# DOCKER BUILD AND PUSH TO ECR
# ==============================================================================

if [ "$BACKEND_ONLY" = true ] || [ "$FRONTEND_ONLY" = false ]; then
    print_header "STEP 4: DOCKER BUILD AND PUSH TO ECR"

    # Get image tag from git commit
    IMAGE_TAG=$(git rev-parse --short HEAD)
    print_info "Image tag (from git commit): $IMAGE_TAG"

    ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    FULL_IMAGE_NAME="$ECR_URI/$ECR_REPO_NAME"

    # Authenticate to ECR
    print_step "Authenticating to Amazon ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URI"
    print_success "ECR authentication successful"

    # Build Docker image
    print_step "Building Docker image..."
    docker build -t "$ECR_REPO_NAME:$IMAGE_TAG" apps/api/
    print_success "Docker image built: $ECR_REPO_NAME:$IMAGE_TAG"

    # Tag for ECR
    print_step "Tagging image for ECR..."
    docker tag "$ECR_REPO_NAME:$IMAGE_TAG" "$FULL_IMAGE_NAME:$IMAGE_TAG"
    docker tag "$ECR_REPO_NAME:$IMAGE_TAG" "$FULL_IMAGE_NAME:latest"

    # Push to ECR
    print_step "Pushing image to ECR..."
    docker push "$FULL_IMAGE_NAME:$IMAGE_TAG"
    docker push "$FULL_IMAGE_NAME:latest"
    print_success "Image pushed to ECR"
    print_info "Image URI: $FULL_IMAGE_NAME:$IMAGE_TAG"
fi

# ==============================================================================
# TERRAFORM APPLY (Optional)
# ==============================================================================

if [ "$SKIP_TERRAFORM" = false ] && [ "$FRONTEND_ONLY" = false ]; then
    print_header "STEP 5: TERRAFORM INFRASTRUCTURE UPDATE"

    if [ -d "$TF_DIR" ]; then
        print_step "Changing to Terraform directory: $TF_DIR"
        cd "$TF_DIR"

        # Check for terraform.tfvars
        if [ ! -f "$TF_VAR_FILE" ]; then
            print_warning "terraform.tfvars not found!"
            echo ""
            echo "Create it with your ACM certificate ARN:"
            echo "  acm_certificate_arn = \"arn:aws:acm:$AWS_REGION:$AWS_ACCOUNT_ID:certificate/xxx\""
            echo ""
            if ! confirm "Continue without terraform.tfvars?"; then
                cd - > /dev/null
                exit 1
            fi
            TF_VAR_ARG=""
        else
            TF_VAR_ARG="-var-file=$TF_VAR_FILE"
        fi

        # Terraform init
        print_step "Initializing Terraform..."
        terraform init -input=false

        # Terraform plan
        print_step "Planning infrastructure changes..."
        terraform plan $TF_VAR_ARG -out=tfplan

        if confirm "Apply Terraform changes?"; then
            print_step "Applying Terraform changes..."
            terraform apply tfplan
            rm -f tfplan
            print_success "Terraform apply complete"
        else
            rm -f tfplan
            print_warning "Skipping Terraform apply"
        fi

        cd - > /dev/null
    else
        print_warning "Terraform directory not found: $TF_DIR"
        print_info "Skipping Terraform step"
    fi
fi

# ==============================================================================
# FORCE ECS SERVICE UPDATE
# ==============================================================================

if [ "$FRONTEND_ONLY" = false ]; then
    print_header "STEP 6: ECS SERVICE UPDATE"

    if confirm "Force ECS service to deploy new image?"; then
        print_step "Updating ECS service..."
        aws ecs update-service \
            --cluster "$ECS_CLUSTER" \
            --service "$ECS_SERVICE" \
            --force-new-deployment \
            --region "$AWS_REGION" \
            --output text \
            --query 'service.serviceName'
        print_success "ECS service update triggered"
        print_info "New tasks will be deployed with image: $IMAGE_TAG"
    else
        print_warning "Skipping ECS service update"
    fi
fi

# ==============================================================================
# POST-DEPLOYMENT VERIFICATION
# ==============================================================================

print_header "STEP 7: POST-DEPLOYMENT VERIFICATION"

echo ""
echo -e "${BOLD}Deployment Summary:${NC}"
echo "-------------------"
echo "  Git Branch:     $CURRENT_BRANCH"
echo "  Git Commit:     $(git rev-parse --short HEAD)"
if [ "$FRONTEND_ONLY" = false ]; then
    echo "  Docker Image:   $ECR_REPO_NAME:$IMAGE_TAG"
    echo "  ECR URI:        $FULL_IMAGE_NAME:$IMAGE_TAG"
    echo "  ECS Cluster:    $ECS_CLUSTER"
    echo "  ECS Service:    $ECS_SERVICE"
fi
echo ""

echo -e "${BOLD}Verification Commands:${NC}"
echo ""
echo "# Check frontend deployment (Vercel)"
echo "curl -I https://your-vercel-domain.vercel.app"
echo ""

if [ "$FRONTEND_ONLY" = false ]; then
    echo "# Check backend health"
    echo "curl https://api.yourdomain.com/health"
    echo ""
    echo "# Check ECS service status"
    echo "aws ecs describe-services --cluster $ECS_CLUSTER --services $ECS_SERVICE --query 'services[0].{running:runningCount,desired:desiredCount}' --region $AWS_REGION"
    echo ""
    echo "# View CloudWatch logs"
    echo "aws logs tail /ecs/profitsentinel-dev --follow --region $AWS_REGION"
fi

echo ""
print_header "DEPLOYMENT COMPLETE"
echo ""
echo -e "${GREEN}Your Profit Sentinel application has been deployed!${NC}"
echo ""
echo "Next steps:"
echo "  1. Check Vercel dashboard for frontend status"
if [ "$FRONTEND_ONLY" = false ]; then
    echo "  2. Check ECS console for backend task status"
    echo "  3. Monitor CloudWatch logs for any errors"
    echo "  4. Run health checks on both endpoints"
fi
echo ""
echo "For rollback instructions, see README.md -> Final Deployment Guide -> Rollback Procedures"
echo ""
