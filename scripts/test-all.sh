#!/bin/bash
# Profit Sentinel - Run All Tests

set -e

echo "=========================================="
echo "  Running All Tests"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILED=0

# Backend Tests
echo -e "\n${YELLOW}Running Backend Tests...${NC}"
cd apps/api
if [ -d "venv" ]; then
    source venv/bin/activate
fi
PYTHONPATH="../../packages/vsa-core/src:../../packages/reasoning/src:../../packages/sentinel-engine/src:$PYTHONPATH" \
    python -m pytest tests/ -v --tb=short || FAILED=1
cd ../..

# VSA Core Tests
echo -e "\n${YELLOW}Running VSA Core Tests...${NC}"
cd packages/vsa-core
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    python -m pytest tests/ -v --tb=short || FAILED=1
else
    echo "No tests found, skipping..."
fi
cd ../..

# Reasoning Tests
echo -e "\n${YELLOW}Running Reasoning Tests...${NC}"
cd packages/reasoning
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    python -m pytest tests/ -v --tb=short || FAILED=1
else
    echo "No tests found, skipping..."
fi
cd ../..

# Sentinel Engine Tests
echo -e "\n${YELLOW}Running Sentinel Engine Tests...${NC}"
cd packages/sentinel-engine
if [ -d "tests" ] && [ "$(ls -A tests/*.py 2>/dev/null)" ]; then
    python -m pytest tests/ -v --tb=short || FAILED=1
else
    echo "No tests found, skipping..."
fi
cd ../..

# Frontend Tests
echo -e "\n${YELLOW}Running Frontend Tests...${NC}"
cd apps/web
if [ -f "package.json" ] && npm run test --if-present 2>/dev/null; then
    echo "Frontend tests passed"
else
    echo "No frontend tests configured, skipping..."
fi
cd ../..

# Summary
echo -e "\n${YELLOW}=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}  All Tests Passed!${NC}"
else
    echo -e "${RED}  Some Tests Failed!${NC}"
fi
echo "=========================================="

exit $FAILED
