#!/bin/bash
# Profit Sentinel - Development Environment Setup Script

set -e

echo "=========================================="
echo "  Profit Sentinel Development Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is required but not installed.${NC}"
        exit 1
    fi
}

echo -e "\n${YELLOW}Checking prerequisites...${NC}"
check_command python3
check_command node
check_command npm

# Python version check
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Node version check
NODE_VERSION=$(node --version)
echo "Node version: $NODE_VERSION"

# Create .env if not exists
if [ ! -f .env ]; then
    echo -e "\n${YELLOW}Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}Created .env file. Please update it with your credentials.${NC}"
fi

# Setup Backend
echo -e "\n${YELLOW}Setting up Backend...${NC}"
cd apps/api

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install local packages in development mode
echo "Installing local packages..."
pip install -e ../../packages/vsa-core
pip install -e ../../packages/reasoning
pip install -e ../../packages/sentinel-engine

cd ../..

# Setup Frontend
echo -e "\n${YELLOW}Setting up Frontend...${NC}"
cd apps/web

echo "Installing Node dependencies..."
npm install

cd ../..

# Run initial tests
echo -e "\n${YELLOW}Running initial tests...${NC}"
cd apps/api
source venv/bin/activate
python -m pytest tests/ -v --tb=short || echo -e "${YELLOW}Some tests may fail without proper configuration${NC}"
cd ../..

echo -e "\n${GREEN}=========================================="
echo "  Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Update .env with your credentials"
echo "  2. Run 'npm run dev' in apps/web for frontend"
echo "  3. Run 'uvicorn src.main:app --reload' in apps/api for backend"
echo "  4. Or use 'docker-compose up' for full stack"
echo ""
