# Profit Sentinel

**Your AI Profit Sentinel - Vigilant Protection for Retail Margins**

Profit Sentinel is a powerful SaaS platform designed to uncover hidden profit leaks in your POS data using advanced **Vector Symbolic Architecture (VSA)** and **symbolic reasoning**. Upload your exports, and get instant forensic insights: unrecorded COGS, negative inventory, margin leaks, dead stock, and smart recommendations.

Built for independent retailers (hardware, lumber, garden centers, auto parts, and more) - a lightweight, non-disruptive add-on to your existing POS system.

## Why Profit Sentinel?

Retail margins are thin (typically 3-5%), and silent leaks from skipped receiving, negatives, or unrecorded purchases can cost tens or hundreds of thousands annually. Most POS tools assume clean data - Profit Sentinel exposes the truth.

- **Forensic Leak Detection**: Find $100k+ hidden shrink others miss
- **Hyperdimensional Computing**: VSA-powered pattern detection for complex anomalies
- **Symbolic Reasoning**: Explainable AI that shows *why* anomalies occur
- **AI-Powered Column Mapping**: Grok AI intelligently maps messy POS exports
- **Privacy-Focused**: Secure AWS infrastructure with encryption everywhere
- **No Disruption**: Works alongside your current POS - no migration needed

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4 |
| **Backend** | FastAPI (Python 3.12), Uvicorn |
| **AI/ML** | PyTorch, VSA/Hyperdimensional Computing, Grok API |
| **Database** | Supabase (PostgreSQL), AWS RDS |
| **Storage** | AWS S3 |
| **Auth** | Supabase Auth (JWT) |
| **Infrastructure** | Terraform, AWS ECS, Docker |
| **CI/CD** | GitHub Actions, Vercel |

## Repository Structure

```
profit-sentinel-saas/
├── apps/                          # Application code
│   ├── api/                       # FastAPI backend
│   │   ├── src/                   # Source code
│   │   │   ├── main.py           # Entry point
│   │   │   ├── config.py         # Settings
│   │   │   ├── routes/           # API endpoints
│   │   │   └── services/         # Business logic
│   │   ├── tests/                # Backend tests
│   │   ├── Dockerfile            # Container config
│   │   └── requirements.txt      # Dependencies
│   │
│   └── web/                       # Next.js frontend
│       └── src/
│           ├── app/              # App Router
│           ├── components/       # React components
│           └── lib/              # Utilities
│
├── packages/                      # Shared libraries
│   ├── vsa-core/                 # VSA/Hyperdimensional Computing
│   ├── reasoning/                # Symbolic reasoning engine
│   └── sentinel-engine/          # Analysis pipeline
│
├── config/                        # Shared configuration
│   ├── primitives/               # Semantic primitive definitions
│   ├── magnitude_buckets/        # Value range mappings
│   └── rules/                    # Anomaly detection rules
│
├── infrastructure/                # Terraform IaC
│   ├── modules/                  # AWS modules
│   └── environments/             # Environment configs
│
├── scripts/                       # Developer scripts
├── docker-compose.yml            # Local development
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker (optional, for containers)
- AWS Account (for S3 storage)
- Supabase Account (for auth/database)
- xAI API Key (recommended, for AI-powered features)

### Getting Your xAI API Key

The xAI API (Grok) powers intelligent column mapping and the chat assistant:

1. Visit **[https://x.ai/api](https://x.ai/api)**
2. Sign up or log in to your account
3. Navigate to **API Keys** and create a new key
4. Copy the key (format: `xai-...`)
5. Add to your `.env` file as `XAI_API_KEY`

> **Note**: Without an API key, the app uses heuristic fallback for column mapping (functional but less accurate).

### 1. Clone and Setup

```bash
git clone https://github.com/your-org/profit-sentinel-saas.git
cd profit-sentinel-saas

# Run setup script (creates venv, installs deps)
./scripts/setup.sh
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit with your credentials
# Required: AWS keys, S3 bucket, Supabase URL/keys
# Recommended: XAI_API_KEY for AI-powered column mapping
```

> **Warning**: Never commit your `.env` file or expose API keys in code. The `.env` file is already in `.gitignore`.

### 3. Run Development Servers

**Option A: Using scripts**
```bash
./scripts/dev.sh
```

**Option B: Manual**
```bash
# Terminal 1: Backend
cd apps/api
source venv/bin/activate
uvicorn src.main:app --reload --port 8000

# Terminal 2: Frontend
cd apps/web
npm run dev
```

**Option C: Docker Compose**
```bash
docker-compose up
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/uploads/presign` | POST | Generate S3 presigned URLs |
| `/uploads/suggest-mapping` | POST | AI-powered column mapping |
| `/analysis/analyze` | POST | Run profit leak analysis |

### Example: Analyze POS Data

```bash
# 1. Get presigned URL
curl -X POST http://localhost:8000/uploads/presign \
  -F "filenames=sales_export.csv"

# 2. Upload file to S3 (use returned presigned URL)

# 3. Get column mapping suggestions
curl -X POST http://localhost:8000/uploads/suggest-mapping \
  -F "key=anonymous/uuid-sales_export.csv" \
  -F "filename=sales_export.csv"

# 4. Run analysis
curl -X POST http://localhost:8000/analysis/analyze \
  -F "key=anonymous/uuid-sales_export.csv" \
  -F 'mapping={"Date":"date","SKU":"sku","Qty":"quantity","Price":"revenue","Cost":"cost"}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js)                      │
│   Channel View → Grok Chat → File Upload → Results Display  │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST API
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    BACKEND (FastAPI)                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Routes: /presign, /suggest-mapping, /analyze           │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Services: S3Service, MappingService, AnalysisService   │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────┬─────────────────────┬─────────────────┬─────────┘
            │                     │                 │
      ┌─────▼─────┐         ┌─────▼─────┐     ┌─────▼─────┐
      │  AWS S3   │         │  Supabase │     │  Grok AI  │
      │  Storage  │         │  Auth/DB  │     │  Mapping  │
      └───────────┘         └───────────┘     └───────────┘
            │
            │ Data Flow
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   VSA ANALYSIS ENGINE                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  vsa-core   │  │  reasoning  │  │  sentinel-engine    │  │
│  │  (vectors,  │  │  (symbolic  │  │  (pipeline,         │  │
│  │   bind,     │──│   logic,    │──│   codebook,         │  │
│  │   resonator)│  │   proofs)   │  │   batch processing) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Testing

Profit Sentinel includes a comprehensive test suite covering backend services, frontend components, and end-to-end user flows.

### Test Architecture

| Test Type | Framework | Directory | Coverage |
|-----------|-----------|-----------|----------|
| Backend Unit/Integration | Pytest | `apps/api/tests/` | 80%+ |
| VSA Core | Pytest | `packages/vsa-core/tests/` | 85%+ |
| Frontend Unit | Jest + RTL | `apps/web/__tests__/` | 70%+ |
| End-to-End | Playwright | `apps/web/e2e/` | Critical flows |

### Quick Commands

```bash
# Run ALL tests (backend + frontend + e2e)
npm run test:all --prefix apps/web  # Frontend
cd apps/api && pytest tests/ -v     # Backend

# Backend Tests (Python)
cd apps/api
pip install pytest pytest-asyncio pytest-cov httpx pytest-mock
pytest tests/ -v                              # Run all
pytest tests/ --cov=src --cov-report=html    # With coverage
pytest tests/test_routes.py -v               # Specific file
pytest tests/ -k "test_health"               # Specific test pattern

# Frontend Unit Tests (Jest)
cd apps/web
npm install                     # First time only
npm run test                   # Run once
npm run test:watch             # Watch mode
npm run test:coverage          # With coverage report

# Frontend E2E Tests (Playwright)
cd apps/web
npx playwright install         # Install browsers (first time)
npm run test:e2e               # Run headless
npm run test:e2e:headed        # Run with browser visible
npm run test:e2e:ui            # Interactive UI mode
npm run test:e2e:report        # View HTML report
```

### Running Tests Locally

#### Prerequisites

1. **Backend**: Python 3.11+, pip
2. **Frontend**: Node.js 20+, npm
3. **E2E**: Chromium (installed via Playwright)

#### Environment Setup for Testing

Tests use mocked external services (S3, Supabase, Grok API) to avoid real API calls and costs. Create a test environment file:

```bash
# apps/api/.env.test (optional - defaults work for mocks)
AWS_ACCESS_KEY_ID=test-access-key
AWS_SECRET_ACCESS_KEY=test-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=test-bucket
SUPABASE_URL=https://test.supabase.co
SUPABASE_SERVICE_KEY=test-service-key
XAI_API_KEY=test-xai-key
```

#### Backend Test Execution

```bash
cd apps/api

# Create virtual environment (first time)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install test dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov httpx pytest-mock

# Set PYTHONPATH (important!)
export PYTHONPATH=$PWD:$PWD/src:$PWD/../../packages/vsa-core/src:$PWD/../../packages/sentinel-engine/src

# Run tests
pytest tests/ -v --tb=short
```

#### Frontend Test Execution

```bash
cd apps/web

# Install dependencies
npm install

# Run Jest unit tests
npm run test

# Run Playwright E2E tests
npx playwright install chromium  # First time only
npm run test:e2e
```

### Test Files Overview

#### Backend Tests (`apps/api/tests/`)

| File | Description | Tests |
|------|-------------|-------|
| `conftest.py` | Pytest fixtures, mocks for S3/Supabase/Grok | - |
| `test_routes.py` | API endpoint tests (health, uploads, analysis) | 20+ |
| `test_services.py` | Service class unit tests (S3, Mapping, Analysis) | 25+ |

#### Frontend Tests (`apps/web/__tests__/`)

| File | Description | Tests |
|------|-------------|-------|
| `components/grok-chat.test.tsx` | GrokChat component tests | 20+ |
| `components/sidebar.test.tsx` | Sidebar component tests | 10+ |
| `api/grok.test.ts` | API route handler tests | 10+ |
| `lib/supabase.test.ts` | Supabase client tests | 5+ |

#### E2E Tests (`apps/web/e2e/`)

| File | Description | Scenarios |
|------|-------------|-----------|
| `home.spec.ts` | Home page, sidebar, theme toggle, Grok chat | 15+ |

### CI/CD Integration

Tests run automatically on GitHub Actions for every push and pull request:

```yaml
# .github/workflows/test.yml
# Triggered on: push to main/develop, PRs to main/develop

Jobs:
  - backend-tests      # Python API tests
  - frontend-unit-tests # Jest tests
  - frontend-e2e-tests  # Playwright tests
  - vsa-core-tests     # VSA package tests
  - lint-and-typecheck # Code quality
```

#### Viewing CI Results

1. Go to repository **Actions** tab
2. Click on workflow run
3. View individual job logs
4. Download test artifacts (Playwright reports)

### Coverage Reports

#### Backend Coverage

```bash
cd apps/api
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

#### Frontend Coverage

```bash
cd apps/web
npm run test:coverage
# Open coverage/lcov-report/index.html in browser
```

### Mocking External Services

All external services are mocked in tests:

```python
# Backend - Mock S3
@pytest.fixture
def mock_s3_client():
    mock = MagicMock()
    mock.generate_presigned_url.return_value = "https://..."
    return mock

# Backend - Mock Grok
@pytest.fixture
def mock_grok_client():
    mock = MagicMock()
    mock.chat.completions.create.return_value = MockResponse(...)
    return mock
```

```typescript
// Frontend - Mock fetch
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ content: 'Mock response' })
  })
)

// Frontend - Mock next-themes
jest.mock('next-themes', () => ({
  useTheme: () => ({ theme: 'light', setTheme: jest.fn() })
}))
```

### Writing New Tests

#### Backend Test Template

```python
# apps/api/tests/test_my_feature.py
import pytest
from unittest.mock import MagicMock, patch

class TestMyFeature:
    """Tests for my new feature."""

    def test_basic_functionality(self, client):
        """Test basic case."""
        response = client.get("/my-endpoint")
        assert response.status_code == 200

    def test_with_mock_dependency(self, client, mock_s3_client):
        """Test with mocked S3."""
        with patch('apps.api.src.dependencies.get_s3_client', return_value=mock_s3_client):
            response = client.post("/upload", data={"file": "test.csv"})
            assert response.status_code == 200
```

#### Frontend Test Template

```typescript
// apps/web/__tests__/components/my-component.test.tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MyComponent from '@/components/my-component'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Expected Text')).toBeInTheDocument()
  })

  it('handles user interaction', async () => {
    const user = userEvent.setup()
    render(<MyComponent />)

    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Clicked!')).toBeInTheDocument()
  })
})
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: apps.api` | Set `PYTHONPATH` correctly |
| Jest can't find modules | Check `moduleNameMapper` in jest.config.js |
| Playwright timeout | Increase timeout or check if dev server is running |
| Mock not working | Verify mock path matches import path |
| Coverage too low | Check `collectCoverageFrom` patterns |

### Best Practices

1. **Test naming**: Use descriptive names (`test_analyze_returns_leak_categories`)
2. **AAA pattern**: Arrange, Act, Assert in each test
3. **Mock at boundaries**: Mock external services, not internal logic
4. **Test edge cases**: Empty inputs, errors, auth failures
5. **Keep tests fast**: Use mocks, avoid real network calls
6. **Run tests before commit**: `npm run test && pytest tests/`

## Deployment

### Backend (AWS ECS)

```bash
# Build and push to ECR
docker build -t profit-sentinel-api apps/api/
docker tag profit-sentinel-api:latest <account>.dkr.ecr.us-east-1.amazonaws.com/profit-sentinel-api:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/profit-sentinel-api:latest

# Deploy via Terraform
cd infrastructure/environments/prod
terraform apply
```

### Frontend (Vercel)

The frontend auto-deploys via Vercel on push to `main`.

## VSA Engine Overview

The core analysis uses **Vector Symbolic Architecture** (Hyperdimensional Computing):

- **16,384-dimensional complex vectors** on the unit hypersphere
- **Deterministic seeding**: SHA256(string) → reproducible vectors
- **Algebraic operations**: bind (×), bundle (+), permute (shift)
- **Resonator network**: Iterative query cleanup for pattern matching

### Detected Anomalies

| Primitive | Description | Severity |
|-----------|-------------|----------|
| `low_stock` | Inventory below safety threshold | Medium |
| `high_margin_leak` | Margin below target (<20%) | High |
| `dead_item` | Zero sales in 90+ days | Medium |
| `negative_inventory` | System shows impossible negative quantity | Critical |

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AWS_ACCESS_KEY_ID` | Yes | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Yes | AWS secret key |
| `S3_BUCKET_NAME` | Yes | S3 bucket for uploads |
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service key |
| `XAI_API_KEY` | Recommended | xAI API key for AI features ([get key](https://x.ai/api)) |

> **Security Note**: All API keys are loaded from environment variables only. Never hardcode secrets in source files.

### YAML Configuration

The analysis engine is configured via YAML files in `/config`:

- `primitives/retail_inventory.yaml` - Semantic primitive definitions
- `magnitude_buckets/retail_inventory.yaml` - Value range mappings
- `rules/anomaly_detection.yaml` - Detection rules and inference chains

---

## Full Setup and Configuration Guide

This section provides complete, step-by-step instructions to get Profit Sentinel fully configured and running.

### Local Development Setup

#### Step 1: Create Your Environment File

```bash
# In the project root directory
cp .env.example .env
```

#### Step 2: Get Your API Keys

You need credentials from multiple services. Here's exactly how to get each one:

##### AWS Credentials (Required)

1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam)
2. Click **Users** → **Create user**
3. Enter username: `profit-sentinel-dev`
4. Click **Next** → **Attach policies directly**
5. Search and select: `AmazonS3FullAccess`
6. Click **Next** → **Create user**
7. Click on the user → **Security credentials** tab
8. Click **Create access key** → **Application running outside AWS**
9. **Copy both keys immediately** (you won't see the secret again!)

Add to your `.env`:
```bash
AWS_ACCESS_KEY_ID=AKIA...your-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-east-1
```

##### S3 Bucket (Required)

1. Go to [S3 Console](https://s3.console.aws.amazon.com/s3)
2. Click **Create bucket**
3. Name: `profitsentinel-yourname-uploads`
4. Region: `us-east-1` (or match AWS_REGION)
5. Uncheck "Block all public access" (for presigned URLs)
6. Click **Create bucket**
7. Click on bucket → **Permissions** → **CORS configuration**
8. Add this CORS policy:
```json
[
  {
    "AllowedHeaders": ["*"],
    "AllowedMethods": ["GET", "PUT", "POST"],
    "AllowedOrigins": ["http://localhost:3000", "https://yourdomain.com"],
    "ExposeHeaders": ["ETag"]
  }
]
```

Add to your `.env`:
```bash
S3_BUCKET_NAME=profitsentinel-yourname-uploads
```

##### Supabase Credentials (Required)

1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Click **New project** (or select existing)
3. Wait for project to initialize
4. Go to **Settings** → **API**
5. Copy the following:
   - **Project URL** → `SUPABASE_URL`
   - **anon public** key → `SUPABASE_ANON_KEY`
   - **service_role** key → `SUPABASE_SERVICE_KEY`

Add to your `.env`:
```bash
SUPABASE_URL=https://abc123.supabase.co
SUPABASE_ANON_KEY=eyJ...your-anon-key
SUPABASE_SERVICE_KEY=eyJ...your-service-key

# MUST match above (for Next.js client-side)
NEXT_PUBLIC_SUPABASE_URL=https://abc123.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...your-anon-key
```

##### xAI / Grok API Key (Recommended)

1. Go to [https://x.ai/api](https://x.ai/api)
2. Sign up or log in
3. Navigate to **API Keys**
4. Click **Create new key**
5. Copy the key (format: `xai-...`)

Add to your `.env`:
```bash
XAI_API_KEY=xai-your-key-here
```

#### Step 3: Verify Your .env File

Your `.env` should look like this (with your real values):

```bash
# AWS
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/EXAMPLE
AWS_REGION=us-east-1
S3_BUCKET_NAME=profitsentinel-yourname-uploads

# Supabase
SUPABASE_URL=https://abc123.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...
NEXT_PUBLIC_SUPABASE_URL=https://abc123.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...

# xAI
XAI_API_KEY=xai-...

# API URL (local)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

> **Warning**: No spaces around `=` signs! No quotes around values unless they contain spaces!

#### Step 4: Start Development Servers

```bash
# Terminal 1: Backend API
cd apps/api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8000

# Terminal 2: Frontend
cd apps/web
npm install
npm run dev
```

#### Step 5: Verify Everything Works

1. Backend health: http://localhost:8000/health → Should return `{"status": "healthy"}`
2. API docs: http://localhost:8000/docs → Should show Swagger UI
3. Frontend: http://localhost:3000 → Should load the app
4. Check console for any "missing key" warnings

---

### Networking and Connectivity

#### Frontend → Backend Communication

The frontend calls the backend API via `NEXT_PUBLIC_API_URL`.

| Environment | NEXT_PUBLIC_API_URL Value |
|-------------|---------------------------|
| Local Dev   | `http://localhost:8000`   |
| Production  | `https://api.yourdomain.com` |

#### CORS Configuration

The backend allows requests from origins defined in `config.py`. Default allowed origins:

```python
cors_origins = [
    "https://profitsentinel.com",
    "https://www.profitsentinel.com",
    "https://profit-sentinel-saas.vercel.app",
    "http://localhost:3000",
]
```

To add your domain, edit `apps/api/src/config.py` and add your domain to the list.

#### API Endpoint Matching

| Frontend calls | Backend endpoint |
|----------------|------------------|
| `${API_URL}/health` | `/health` |
| `${API_URL}/uploads/presign` | `/uploads/presign` |
| `${API_URL}/uploads/suggest-mapping` | `/uploads/suggest-mapping` |
| `${API_URL}/analysis/analyze` | `/analysis/analyze` |

---

### Deployment and Production Configuration

#### Frontend Deployment (Vercel)

**Step 1: Connect to Vercel**

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **Add New...** → **Project**
3. Import your GitHub repository
4. Set **Root Directory** to `apps/web`
5. Click **Deploy**

**Step 2: Add Environment Variables in Vercel**

1. Go to your project → **Settings** → **Environment Variables**
2. Add each variable:

| Variable | Value | Environment | Required |
|----------|-------|-------------|----------|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://YOUR_PROJECT.supabase.co` | Production, Preview | **Yes** |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | `eyJ...` (your anon key) | Production, Preview | **Yes** |
| `NEXT_PUBLIC_API_URL` | `https://api.yourdomain.com` | Production | **Yes** |
| `XAI_API_KEY` | `xai-...` (your xAI key) | Production | Recommended |

> **IMPORTANT**: The `NEXT_PUBLIC_` prefix is required for these variables to be available in the browser at runtime. Without them, the Supabase client will fail to initialize and you'll see console errors.

**Step 3: Redeploy** after adding variables.

```bash
# Verify variables are set in Vercel CLI (optional)
vercel env ls
```

#### Backend Deployment (AWS ECS)

**Step 1: Set GitHub Secrets**

Go to GitHub repo → **Settings** → **Secrets and variables** → **Actions**

Add these secrets:

| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key |
| `VERCEL_TOKEN` | From Vercel account settings |
| `VERCEL_ORG_ID` | From Vercel project settings |
| `VERCEL_PROJECT_ID` | From Vercel project settings |

---

### Domain and DNS Setup (GoDaddy → Vercel/AWS)

#### Step 1: Deploy to Vercel First

Make sure your app is deployed and working at the `.vercel.app` URL.

#### Step 2: Add Custom Domain in Vercel

1. Go to Vercel → Your Project → **Settings** → **Domains**
2. Enter your domain: `profitsentinel.com`
3. Click **Add**

#### Step 3: Configure GoDaddy DNS

1. Go to [GoDaddy DNS Management](https://dcc.godaddy.com/manage/dns)
2. Select your domain
3. Add/modify these records:

**For apex domain (profitsentinel.com):**

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | @ | `76.76.21.21` | 600 |

**For www subdomain:**

| Type | Name | Value | TTL |
|------|------|-------|-----|
| CNAME | www | `cname.vercel-dns.com` | 600 |

**For API subdomain:**

| Type | Name | Value | TTL |
|------|------|-------|-----|
| CNAME | api | Your ALB DNS name | 600 |

#### Step 4: Verify DNS Propagation

1. Wait 5-30 minutes
2. Check: [whatsmydns.net](https://whatsmydns.net)
3. Vercel auto-provisions SSL certificate

---

### Security Warnings

> **CRITICAL SECURITY RULES**

| DO | DON'T |
|----|-------|
| Keep `.env` files local only | NEVER commit `.env` to git |
| Use GitHub Secrets for CI/CD | NEVER put secrets in code |
| Use Vercel Environment Variables | NEVER expose keys in client-side JS |
| Rotate keys if accidentally exposed | NEVER share keys in chat/email |
| Use least-privilege IAM roles | NEVER use root AWS credentials |

---

### Verification Checklist

#### Local Development
- [ ] `.env` file created from `.env.example`
- [ ] All required values filled in
- [ ] Backend starts: `curl http://localhost:8000/health`
- [ ] Frontend loads: http://localhost:3000
- [ ] No "missing key" warnings

#### Production
- [ ] GitHub Secrets configured
- [ ] Vercel environment variables set
- [ ] Custom domain resolves
- [ ] HTTPS certificate valid
- [ ] API calls work (no CORS errors)

---

## Final Deployment Guide

This section provides complete, production-ready deployment instructions for Profit Sentinel. Follow these steps to safely deploy both frontend (Vercel) and backend (AWS ECS) to production.

### Pre-Deployment Checklist

Before deploying, verify the following:

| Item | Command | Expected Result |
|------|---------|-----------------|
| Backend tests pass | `cd apps/api && pytest tests/ -v` | All tests green |
| Frontend tests pass | `cd apps/web && npm run test` | All tests green |
| No secrets in code | `git diff --cached` | No `.env`, `*.tfvars`, API keys |
| `.gitignore` complete | `cat .gitignore` | All secrets patterns listed |
| Docker builds | `docker build -t test apps/api/` | Build succeeds |

### Step 1: Verify .gitignore Security

Your `.gitignore` must block all sensitive files. Current coverage:

```bash
# Verify these patterns are in .gitignore (they should be):
.env
.env.*
*.tfvars
*.tfstate
*.tfstate.*
.terraform/
credentials.json
*.pem
*.key
secrets/
```

**Quick verification command:**
```bash
# Check for any staged secrets (should return empty)
git diff --cached --name-only | grep -E '\.env|\.tfvars|credentials|\.pem|\.key' || echo "SAFE: No secrets staged"
```

### Step 2: Configure Deployment Variables

Before running the deployment script, gather these values:

| Variable | Where to Find | Example |
|----------|---------------|---------|
| `AWS_ACCOUNT_ID` | AWS Console → Account ID (top right) | `123456789012` |
| `AWS_REGION` | Your infrastructure region | `us-east-1` |
| `ECR_REPO_NAME` | Terraform output or ECR console | `profitsentinel-dev-api` |
| `ECS_CLUSTER` | Terraform output or ECS console | `profitsentinel-dev-cluster` |
| `ECS_SERVICE` | Terraform output or ECS console | `profitsentinel-dev-api-service` |

**Get values from Terraform:**
```bash
cd infrastructure/environments/dev
terraform output repository_url  # ECR URL
terraform output alb_dns_name    # API endpoint
```

### Step 3: Local Manual Deployment

#### Option A: Use the Deploy Script (Recommended)

Create and run `scripts/deploy.sh` (provided below):

```bash
# Make executable and run
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

#### Option B: Step-by-Step Commands

**3.1 Git Commit and Push**

```bash
# 1. Verify what will be committed
git status

# 2. Check for secrets one more time
git diff --cached --name-only | grep -E '\.env|tfvars|credentials' && echo "DANGER: Secrets detected!" || echo "Safe to proceed"

# 3. Stage all changes (excluding gitignored files)
git add -A

# 4. Commit with descriptive message
git commit -m "deploy: production release v1.0.0

- All tests passing (backend + frontend + e2e)
- VSA analysis engine fully functional
- Grok AI column mapping integrated
- Infrastructure ready for production"

# 5. Push to main (triggers Vercel auto-deploy)
git push origin main
```

**3.2 Build and Push Docker Image to ECR**

```bash
# Set your AWS account details
export AWS_ACCOUNT_ID="YOUR_AWS_ACCOUNT_ID"
export AWS_REGION="us-east-1"
export ECR_REPO_NAME="profitsentinel-dev-api"

# Get the git commit hash for image tagging (traceability)
export IMAGE_TAG=$(git rev-parse --short HEAD)

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build the Docker image
docker build -t $ECR_REPO_NAME:$IMAGE_TAG apps/api/

# Tag for ECR
docker tag $ECR_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG
docker tag $ECR_REPO_NAME:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest

echo "Pushed image with tag: $IMAGE_TAG"
```

**3.3 Deploy Infrastructure with Terraform**

```bash
cd infrastructure/environments/dev

# Initialize (first time or after module changes)
terraform init

# Preview changes
terraform plan -var-file="terraform.tfvars"

# Apply (creates/updates AWS resources)
terraform apply -var-file="terraform.tfvars"

# Note: terraform.tfvars contains sensitive values and is gitignored
# Create it with: acm_certificate_arn = "arn:aws:acm:..."
```

**3.4 Force ECS Service Update**

```bash
# Trigger new deployment with the latest image
aws ecs update-service \
  --cluster profitsentinel-dev-cluster \
  --service profitsentinel-dev-api-service \
  --force-new-deployment \
  --region us-east-1
```

### Step 4: CI/CD Automatic Deployment (Alternative)

If you have GitHub Actions configured, pushing to `main` automatically triggers:

1. **Frontend**: Vercel auto-deploys on push
2. **Backend**: GitHub Actions builds Docker image, pushes to ECR, updates ECS

**Required GitHub Secrets** (Settings → Secrets → Actions):

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `VERCEL_TOKEN` | From vercel.com/account/tokens |
| `VERCEL_ORG_ID` | From Vercel project settings |
| `VERCEL_PROJECT_ID` | From Vercel project settings |

### Step 5: Post-Deployment Verification

#### Frontend (Vercel)

1. **Vercel Dashboard**: https://vercel.com/dashboard
   - Check deployment status (should be "Ready")
   - Click deployment to see build logs

2. **Live Site Check**:
   ```bash
   curl -I https://your-domain.vercel.app
   # Should return: HTTP/2 200
   ```

#### Backend (AWS ECS)

1. **ECS Console**: https://console.aws.amazon.com/ecs
   - Navigate to Clusters → profitsentinel-dev-cluster
   - Check service "Running tasks" count (should be 1+)
   - View "Events" tab for deployment status

2. **Health Check**:
   ```bash
   # Replace with your ALB DNS or custom domain
   curl https://api.yourdomain.com/health
   # Should return: {"status":"healthy","environment":"production"}
   ```

3. **CloudWatch Logs**: https://console.aws.amazon.com/cloudwatch
   - Log groups → /ecs/profitsentinel-dev
   - Check for startup logs and any errors

#### Complete Verification Commands

```bash
# Frontend health
curl -s https://your-frontend-domain.com | head -20

# Backend health
curl -s https://api.yourdomain.com/health | jq .

# Backend API docs
curl -s https://api.yourdomain.com/docs | head -5

# ECS task status
aws ecs describe-services \
  --cluster profitsentinel-dev-cluster \
  --services profitsentinel-dev-api-service \
  --query 'services[0].{running:runningCount,desired:desiredCount,status:status}' \
  --region us-east-1
```

### Rollback Procedures

#### Frontend Rollback (Vercel)

1. Go to Vercel Dashboard → Deployments
2. Find previous working deployment
3. Click "..." → "Promote to Production"

#### Backend Rollback (ECS)

```bash
# List recent images
aws ecr describe-images \
  --repository-name profitsentinel-dev-api \
  --query 'imageDetails[*].{tag:imageTags[0],pushed:imagePushedAt}' \
  --region us-east-1

# Update task definition to use previous image tag
# Then force new deployment
aws ecs update-service \
  --cluster profitsentinel-dev-cluster \
  --service profitsentinel-dev-api-service \
  --force-new-deployment
```

### Troubleshooting

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| ECS task keeps failing | CloudWatch Logs → /ecs/profitsentinel-dev | Check for missing env vars or startup errors |
| 503 from ALB | Target group health checks | Verify security groups allow port 8000 |
| Vercel build fails | Vercel deployment logs | Check package.json scripts and dependencies |
| CORS errors | Browser console | Add domain to CORS_ORIGINS in backend config |
| Docker push denied | ECR authentication | Re-run `aws ecr get-login-password` |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Security

- All data encrypted at rest (S3/RDS) and in transit (HTTPS)
- Multi-tenant isolation with row-level security
- AWS VPC with private networking
- No data sold or shared

Report security issues to: security@profitsentinel.com

## License

MIT License - see [LICENSE.md](LICENSE.md)

## Support

- Documentation: https://docs.profitsentinel.com
- Issues: https://github.com/your-org/profit-sentinel-saas/issues
- Email: support@profitsentinel.com

---

**Profit Sentinel** - Protecting retail margins with AI-powered forensic analysis.
