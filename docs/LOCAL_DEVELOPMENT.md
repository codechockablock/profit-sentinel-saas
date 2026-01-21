# Local Development Environment Setup

This guide explains how to run Profit Sentinel locally without any dependency on AWS, Supabase, or other production services.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for synthetic data generation)
- Node.js 20+ (optional, for direct frontend development)

## Quick Start

```bash
# 1. Start the local environment
docker-compose -f docker-compose.local.yml up -d

# 2. Wait for services to be healthy (30-60 seconds)
docker-compose -f docker-compose.local.yml ps

# 3. Generate synthetic test data
cd tools
pip install pandas numpy faker
python generate_synthetic_data.py

# 4. Open the application
open http://localhost:3000
```

## Services

| Service | Local URL | Description |
|---------|-----------|-------------|
| Frontend | http://localhost:3000 | Next.js web application |
| Backend API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Swagger documentation |
| MinIO Console | http://localhost:9001 | S3-compatible storage UI |
| PostgreSQL | localhost:5432 | Database (direct access) |

## Default Credentials

### PostgreSQL
```
Host: localhost
Port: 5432
Database: profit_sentinel_dev
Username: sentinel_dev
Password: local_dev_password_123
```

### MinIO (S3)
```
Console URL: http://localhost:9001
Access Key: minioadmin
Secret Key: minioadmin123
Bucket: profit-sentinel-uploads
```

## Architecture Comparison

| Component | Production | Local Development |
|-----------|------------|-------------------|
| Database | Supabase (PostgreSQL) | Local PostgreSQL |
| Object Storage | AWS S3 | MinIO |
| AI Column Mapping | Grok/XAI API | Heuristic fallback |
| Authentication | Supabase Auth | Disabled |
| Email Reports | Resend/SendGrid | Disabled (logged only) |

## Common Tasks

### View Logs

```bash
# All services
docker-compose -f docker-compose.local.yml logs -f

# Specific service
docker-compose -f docker-compose.local.yml logs -f api
docker-compose -f docker-compose.local.yml logs -f web
```

### Reset Database

```bash
# Stop and remove volumes
docker-compose -f docker-compose.local.yml down -v

# Restart (will reinitialize)
docker-compose -f docker-compose.local.yml up -d
```

### Access PostgreSQL Directly

```bash
# Via psql
docker exec -it profit-sentinel-local-db psql -U sentinel_dev -d profit_sentinel_dev

# List tables
\dt

# Query data
SELECT * FROM email_signups LIMIT 10;
```

### Access MinIO Directly

```bash
# Open web console
open http://localhost:9001

# Or use AWS CLI with MinIO endpoint
aws --endpoint-url http://localhost:9000 s3 ls s3://profit-sentinel-uploads/
```

### Generate More Test Data

```bash
cd tools
python generate_synthetic_data.py --items 5000 --transactions 20000
```

## Troubleshooting

### Services won't start

```bash
# Check what's using the ports
lsof -i :3000
lsof -i :8000
lsof -i :5432
lsof -i :9000

# Force recreate containers
docker-compose -f docker-compose.local.yml up -d --force-recreate
```

### Database connection errors

```bash
# Check postgres is healthy
docker exec profit-sentinel-local-db pg_isready -U sentinel_dev

# View postgres logs
docker logs profit-sentinel-local-db
```

### MinIO bucket not found

```bash
# Manually create bucket
docker exec profit-sentinel-minio-setup mc mb local/profit-sentinel-uploads

# Or via console
open http://localhost:9001
# Create bucket named "profit-sentinel-uploads"
```

### API returns 500 errors

```bash
# Check API logs
docker logs profit-sentinel-local-api -f

# Check if database is connected
curl http://localhost:8000/health
```

## Development Workflow

### Working on Backend

```bash
# Changes to apps/api/ are auto-reloaded
# No restart needed unless dependencies change

# If you add new dependencies:
docker-compose -f docker-compose.local.yml build api
docker-compose -f docker-compose.local.yml up -d api
```

### Working on Frontend

```bash
# Changes to apps/web/ are auto-reloaded via Next.js

# If you add new dependencies:
docker-compose -f docker-compose.local.yml build web
docker-compose -f docker-compose.local.yml up -d web
```

### Running Tests Locally

```bash
# Backend tests
docker exec profit-sentinel-local-api pytest

# Frontend tests
docker exec profit-sentinel-local-web npm test
```

## Key Differences from Production

1. **No authentication** - All uploads are anonymous
2. **No AI mapping** - Uses heuristic column detection only
3. **No email reports** - Reports logged to console only
4. **Local storage only** - Files never leave your machine
5. **Debug mode enabled** - More verbose logging

## Switching to Production Testing

If you need to test with real AWS/Supabase:

1. Copy `.env.example` to `.env`
2. Fill in production credentials
3. Use `docker-compose.yml` instead:
   ```bash
   docker-compose up -d
   ```

**WARNING:** Only do this with a separate staging account, never with production data!

## Cleanup

```bash
# Stop all services
docker-compose -f docker-compose.local.yml down

# Remove all data (full reset)
docker-compose -f docker-compose.local.yml down -v
docker volume rm profit-sentinel-local-postgres profit-sentinel-local-minio

# Remove images (fresh rebuild)
docker-compose -f docker-compose.local.yml down --rmi all
```
