# Legacy Codebase (Archived)

**Archived:** 2026-02-06
**Tag:** v1.0-legacy-final
**Reason:** Replaced by Rust+Python architecture in profit-sentinel-rs/

This code is preserved for reference and emergency rollback only.
Do not modify. See /profit-sentinel-rs for active development.

## What was here
- `apps/api/` — FastAPI backend (Python 3.12, uvicorn, port 8000)
- `apps/web/` — Next.js 16 frontend (React 19, Tailwind CSS 4, port 3000)
- `packages/vsa-core/` — Python VSA core library
- `packages/reasoning/` — Symbolic reasoning engine
- `packages/sentinel-engine/` — Analysis pipeline engine (v5.0.0 Dorian)

## Performance comparison
| Metric | Legacy | New (Rust) |
|--------|--------|------------|
| Analysis time (36K rows) | ~10.3s | ~3.3s |
| Dollar impact calculation | $0 (broken) | $77,877 (accurate) |
| Tests passing | ~566 Python | 355 (85 Rust + 243 Python + 27 Frontend) |

## Rollback procedure
See /docs/MIGRATION_PLAN.md section "Rollback Procedure"

1. Revert to tag: `git checkout v1.0-legacy-final`
2. Restore apps/api Dockerfile: `docker build -f apps/api/Dockerfile .`
3. Push to ECR with legacy tag
4. Update ECS task definition to use legacy image
5. Update ALB target group back to port 8000
