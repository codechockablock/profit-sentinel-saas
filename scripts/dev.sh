#!/bin/bash
# Profit Sentinel - Start Development Servers

set -e

echo "Starting Profit Sentinel Development Servers..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start Backend
echo -e "${GREEN}Starting Backend (FastAPI)...${NC}"
cd apps/api
source venv/bin/activate
PYTHONPATH="../../packages/vsa-core/src:../../packages/reasoning/src:../../packages/sentinel-engine/src:$PYTHONPATH" \
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
sleep 3

# Start Frontend
echo -e "${GREEN}Starting Frontend (Next.js)...${NC}"
cd apps/web
npm run dev &
FRONTEND_PID=$!
cd ../..

echo -e "\n${GREEN}=========================================="
echo "  Development servers running!"
echo "==========================================${NC}"
echo ""
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:3000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for processes
wait
