# Profit Sentinel

**Your AI Profit Sentinel - Vigilant Protection for Retail Margins**

Profit Sentinel is a SaaS platform that uncovers hidden profit leaks in POS data using a high-performance **Rust analysis pipeline** with **Vector Symbolic Architecture (VSA)** and **symbolic reasoning**. Upload your exports, and get instant forensic insights: unrecorded COGS, negative inventory, margin leaks, dead stock, and smart recommendations.

Built for independent retailers (hardware, lumber, garden centers, auto parts, and more) - a lightweight, non-disruptive add-on to your existing POS system.

## Architecture (v2 — Rust Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js)                      │
│   Channel View → File Upload → Column Mapping → Results     │
└───────────────────────────┬─────────────────────────────────┘
                            │ REST API
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                 PYTHON SIDECAR (FastAPI :8001)               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Routes: /health, /suggest-mapping, /analyze            │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Adapters: Column Mapping (M1) + Result Adapter (M2)    │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Engine Bridge: Python ↔ Rust subprocess (CSV stdin)    │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────┬─────────────────────┬─────────────────┬─────────┘
            │                     │                 │
      ┌─────▼─────┐         ┌────▼──────┐     ┌────▼──────┐
      │  AWS S3   │         │ Supabase  │     │ Anthropic │
      │  Storage  │         │ Auth/DB   │     │  Mapping  │
      └───────────┘         └───────────┘     └───────────┘
            │
            │ CSV Data
            ▼
┌─────────────────────────────────────────────────────────────┐
│               RUST ANALYSIS ENGINE (sentinel-server)        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │sentinel-vsa │  │  sentinel-  │  │  sentinel-server    │ │
│  │ (1024-dim   │  │  pipeline   │  │  (JSON output,      │ │
│  │  Complex64, │──│  (classify, │──│   evidence,         │ │
│  │  bundling)  │  │   score)    │  │   enrichment)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
│  Performance: 36K rows in ~280ms (warm) / 3.3s E2E         │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Next.js 16, React 19, Tailwind CSS 4 |
| **Python Sidecar** | FastAPI, Uvicorn, Python 3.13 |
| **Rust Pipeline** | Rust 1.84, sentinel-vsa, sentinel-pipeline, sentinel-server |
| **AI/ML** | VSA/Hyperdimensional Computing (1024-dim Complex64), Anthropic Claude API |
| **Database** | Supabase (PostgreSQL), AWS RDS |
| **Storage** | AWS S3 |
| **Auth** | Supabase Auth (JWT) |
| **Infrastructure** | Terraform, AWS ECS (Fargate), Docker |
| **CI/CD** | GitHub Actions, Vercel |

## Repository Structure

```
profit-sentinel-saas/
├── profit-sentinel-rs/            # Rust + Python system (active)
│   ├── sentinel-vsa/              # VSA encoding (Complex64 vectors)
│   ├── sentinel-pipeline/         # Issue classification & scoring
│   ├── sentinel-server/           # CLI binary (reads CSV, outputs JSON)
│   ├── python/                    # Python package (sentinel_agent)
│   │   ├── sentinel_agent/
│   │   │   ├── sidecar.py         # FastAPI app
│   │   │   ├── engine.py          # Rust subprocess bridge
│   │   │   ├── adapters/          # POS data adapters (Orgill, Do It Best, Ace, etc.)
│   │   │   ├── category_mix.py    # Category mix optimizer
│   │   │   ├── coop_models.py     # Co-op alert models
│   │   │   ├── llm_layer.py       # LLM digest rendering
│   │   │   └── world_model/       # VSA world model (Engine 2)
│   │   └── tests/                 # Python tests
│   └── Dockerfile.sidecar         # Multi-stage Rust+Python container
│
├── web/                           # Next.js 16 frontend (active)
│   ├── src/app/                   # App router pages
│   └── src/components/            # React components
│
├── config/                        # Analysis configuration (YAML)
│   ├── primitives/                # Semantic primitive definitions
│   ├── magnitude_buckets/         # Value range mappings
│   └── rules/                     # Anomaly detection rules
│
├── infrastructure/                # Terraform IaC
│   ├── modules/                   # AWS modules (VPC, ALB, ECS, ECR, RDS, S3)
│   └── environments/              # Environment configs (dev, staging, prod)
│
├── docs/                          # Project documentation
│   ├── ARCHITECTURE_ANALYSIS.md
│   ├── INFRASTRUCTURE_INVENTORY.md
│   └── POS_COLUMN_MAPPING_REFERENCE.md
│
├── docker-compose.yml             # Local development
└── README.md
```

## Quick Start

### Prerequisites

- Rust 1.84+ (for pipeline)
- Python 3.13+
- Node.js 20+
- Docker (optional, for containers)
- AWS Account (for S3 storage)
- Supabase Account (for auth/database)
- Anthropic API Key (recommended, for AI-powered column mapping)

### 1. Build the Rust Pipeline

```bash
cd profit-sentinel-rs
cargo build --release
# Binary: target/release/sentinel-server
```

### 2. Set Up the Python Sidecar

```bash
cd profit-sentinel-rs
python -m venv .venv
source .venv/bin/activate
pip install -e "python/[dev]"
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit with your credentials:
# AWS keys, S3 bucket, Supabase URL/keys, ANTHROPIC_API_KEY
```

### 4. Run Development Servers

**Option A: Docker Compose**
```bash
docker-compose up
```

**Option B: Manual**
```bash
# Terminal 1: Python sidecar (port 8001)
cd profit-sentinel-rs
python -m sentinel_agent serve

# Terminal 2: Frontend (port 3000)
cd web
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **Sidecar API**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health status with version info |
| `/uploads/presign` | POST | Generate S3 presigned URLs |
| `/uploads/suggest-mapping` | POST | AI-powered column mapping |
| `/analysis/analyze` | POST | Run Rust pipeline analysis |
| `/api/v1/findings` | GET | Paginated findings with acknowledge support |
| `/api/v1/dashboard` | GET | Dashboard summary (recovery $, department status) |
| `/api/v1/config` | GET | Fetch dead stock configuration |
| `/api/v1/config` | PUT | Save dead stock configuration |
| `/api/v1/transfers` | GET | Transfer recommendations for dead stock |
| `/api/v1/predictions` | GET | Active predictions with confidence |

## Detected Issue Types

| Issue Type | Description | Severity |
|-----------|-------------|----------|
| `NegativeInventory` | System shows impossible negative quantity | Critical |
| `MarginLeak` | Margin below target or cost > price | High |
| `DeadStock` | Configurable — watchlist at 60 days (default) | Medium |
| `Overstock` | Excessive inventory relative to sales velocity | Medium |
| `Shrinkage` | Unexplained inventory loss patterns | High |
| `CostSpike` | Sudden cost increase without price adjustment | High |
| `PhantomInventory` | Stock discrepancy between system and reality | High |
| `VendorAnomaly` | Multiple SKUs from same vendor showing pattern | Medium |
| `SeasonalMisalignment` | Stock levels mismatched with seasonal velocity | Medium |
| `UnrecordedCOGS` | Cost of goods sold not properly recorded | High |
| `PricingGap` | Significant gap between cost and selling price | Medium |

## Testing

### Test Suite (145 tests passing)

| Suite | Framework | Count | Command |
|-------|-----------|-------|---------|
| Rust Pipeline | `cargo test` | 132 | `cd profit-sentinel-rs && cargo test --workspace` |
| Python World Model | Pytest | 9 | `cd profit-sentinel-rs && python -m pytest python/sentinel_agent/world_model/tests/ -v` |
| Python Standalone | manual | 4 | See below |

See [docs/TEST_STATUS.md](docs/TEST_STATUS.md) for detailed breakdown and coverage gaps.

### Quick Test Commands

```bash
# All Rust tests (132 passing)
cd profit-sentinel-rs && cargo test --workspace

# Python world model tests (9 passing)
cd profit-sentinel-rs && python -m pytest python/sentinel_agent/world_model/tests/ -v

# Python standalone verification
cd profit-sentinel-rs && python -m sentinel_agent.world_model.config
```

## Performance

| Metric | Value |
|--------|-------|
| Rust pipeline (warm) | ~280ms for 36K rows |
| Rust pipeline (cold) | ~596ms for 36K rows |
| Full E2E (Python + Rust) | ~3.3s for 36K rows |
| Previous Python engine | ~10s (with broken output) |
| Speedup | **188x** (pipeline only) |

## Deployment

### Backend (AWS ECS via Dockerfile.sidecar)

```bash
# Build multi-stage image (Rust + Python)
docker build -f profit-sentinel-rs/Dockerfile.sidecar -t sentinel-sidecar .

# Push to ECR and deploy via Terraform
cd infrastructure/environments/staging
terraform apply
```

### Frontend (Vercel)

Auto-deploys via Vercel on push to `main`.

## Migration Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | Done | Column Mapping Adapter |
| M2 | Done | Result Adapter (dual-engine) |
| M3 | Done | Performance optimization (52.5s -> 280ms) |
| M4 | Done | Integration testing (355 tests) |
| M5 | In Progress | Staging deployment |
| M6 | Pending | Production cutover |
| M7 | Done | Legacy removal (`_legacy/` archived, tag: v1.0-legacy-final) |
| M8 | Pending | Monitoring & observability |

## Security

- All data encrypted at rest (S3/RDS) and in transit (HTTPS)
- Multi-tenant isolation with row-level security
- AWS VPC with private networking
- No data sold or shared

Report security issues to: security@profitsentinel.com

## License

MIT License - see [LICENSE.md](LICENSE.md)

---

**Profit Sentinel** — Protecting retail margins with deterministic forensic analysis.
