# Sprint 2 of 5: Security Hardening

**Status:** Complete
**Branch:** `main`
**Commit:** `security: close auth gaps, harden CORS, lock dev mode, fix S3 boundary`
**Tests:** 837/837 passing (Python). Rust tests require toolchain — Dockerfile change is trivial.

---

## Scope

Close every auth gap, fix CORS, lock down dev mode, secure the proxy route, fix S3 boundary check.

---

## Fixes

### 2.1 — counterfactual.py: Add auth requirement

**Vulnerability:** Counterfactual router was mountable without auth. The `require_auth` parameter was optional and the `/summary` endpoint used a conditional `Depends(require_auth) if require_auth else None` pattern, falling back to `"dev-user"`.

**Before:**
```python
def create_counterfactual_router(state: AppState, require_auth=None) -> APIRouter:
    router = APIRouter(prefix="/counterfactual", tags=["Engine 3"])

    @router.get("/summary")
    async def get_counterfactual_summary(
        ctx: UserContext = Depends(require_auth) if require_auth else None,
    ):
        user_id = ctx.user_id if ctx else "dev-user"
```

**After:**
```python
def create_counterfactual_router(state: AppState, require_auth=None) -> APIRouter:
    if require_auth is None:
        raise ValueError("require_auth dependency is required for counterfactual router")

    router = APIRouter(
        prefix="/counterfactual",
        tags=["Engine 3"],
        dependencies=[Depends(require_auth)],
    )

    @router.get("/summary")
    async def get_counterfactual_summary(
        ctx: UserContext = Depends(require_auth),
    ):
        user_id = ctx.user_id
```

**Files:** `profit-sentinel-rs/python/sentinel_agent/routes/counterfactual.py`

---

### 2.2 — Next.js proxy route (route.ts): Add auth and rate limiting

**Vulnerability:** Public API route proxied to the XAI API key with zero authentication. Anyone who discovered `/api/grok` could burn through API credits.

**Before:**
- No auth check
- No rate limiting
- No model restriction
- Any anonymous request proxied directly to XAI

**After:**
1. Requires valid Supabase Bearer token — returns 401 for unauthenticated requests
2. Per-user rate limiting: 10 requests/minute — returns 429 when exceeded
3. Server-side model allowlist (`grok-beta` only) — returns 400 for disallowed models
4. Periodic cleanup of rate limit map (every 5 min) to prevent memory leaks

**Files:** `web/src/app/api/grok/route.ts`

---

### 2.3 — sidecar.py: Dev mode startup guard

**Vulnerability:** `SIDECAR_DEV_MODE=true` could accidentally be enabled in production, bypassing all authentication.

**Before:** No guard. Dev mode could run anywhere.

**After:**
```python
if settings.sidecar_dev_mode:
    _is_production = bool(
        os.environ.get("ECS_CONTAINER_METADATA_URI")
        or os.environ.get("ECS_CONTAINER_METADATA_URI_V4")
        or os.environ.get("AWS_EXECUTION_ENV")
    )
    if _is_production:
        raise RuntimeError(
            "SIDECAR_DEV_MODE=true is not allowed in ECS/production. "
            "Detected production environment via ECS metadata or AWS_EXECUTION_ENV."
        )
```

**Files:** `profit-sentinel-rs/python/sentinel_agent/sidecar.py`

---

### 2.4 — sidecar.py: Remove localhost from production CORS

**Vulnerability:** `localhost` origins were always included in CORS allowed origins, even in production. Dev mode used `["*"]` (wildcard).

**Before:**
```python
origins = (
    ["*"]
    if settings.sidecar_dev_mode
    else [
        "https://www.profitsentinel.com",
        ...
        "http://localhost:3000",
        "http://localhost:5173",
        ...
    ]
)
```

**After:**
```python
origins = [
    "https://www.profitsentinel.com",
    "https://profitsentinel.com",
    "https://profit-sentinel-saas.vercel.app",
    "https://profit-sentinel.vercel.app",
]
if settings.sidecar_dev_mode:
    origins.extend([
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ])
```

**Files:** `profit-sentinel-rs/python/sentinel_agent/sidecar.py`

---

### 2.5 — upload_routes.py: Fix S3 prefix boundary check

**Vulnerability:** `key.startswith(ctx.s3_prefix)` allowed prefix `tenant/abc` to match `tenant/abcdef/file.csv`, leaking across tenant boundaries.

**Before:**
```python
if not key.startswith(ctx.s3_prefix):
    raise HTTPException(403)
```

**After:**
```python
expected_prefix = ctx.s3_prefix.rstrip("/") + "/"
if not key.startswith(expected_prefix):
    raise HTTPException(403)
```

Applied in both `/uploads/suggest-mapping` and `/analysis/analyze` endpoints.

**Files:** `profit-sentinel-rs/python/sentinel_agent/upload_routes.py`

---

### 2.6 — turnstile.py: Fail closed on exceptions

**Vulnerability:** Turnstile verification failed open — network errors or exceptions returned `True`, allowing bots through.

**Before:**
```python
except Exception as e:
    logger.error("Turnstile verification error: %s", e)
    # On network error, fail open to avoid blocking legitimate users
    return True
```

**After:**
```python
except Exception as e:
    logger.error("Turnstile verification failed: %s", e)
    # Fail closed — reject the request if we can't verify the token
    return False
```

**Files:** `profit-sentinel-rs/python/sentinel_agent/turnstile.py`

---

### 2.7 — layout.tsx: Fail closed when Supabase unconfigured

**Vulnerability:** When Supabase env vars were missing, `setIsAuthenticated(true)` granted full dashboard access to everyone.

**Before:**
```tsx
if (!supabase) {
  // No Supabase config — allow access (dev mode)
  setIsAuthenticated(true);
  return;
}
```

**After:**
```tsx
if (!supabase) {
  // No Supabase config — fail closed, require configuration
  setIsAuthenticated(false);
  return;
}
```

Also added a distinct error state when Supabase is unconfigured, showing a red message directing to set `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

**Files:** `web/src/app/dashboard/layout.tsx`

---

### 2.8 — upload_routes.py: Sanitize error responses

**Vulnerability:** Raw exception text (`str(e)`) was returned to the client in 500 responses, potentially leaking internal paths, stack traces, or configuration details.

**Before:**
```python
except Exception as e:
    logger.error(f"Analysis failed: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
```

**After:**
```python
except Exception as e:
    error_id = str(_uuid.uuid4())[:8]
    logger.error(f"Analysis failed [{error_id}]: {e}", exc_info=True)
    raise HTTPException(
        status_code=500,
        detail=f"Analysis failed. Please try again or contact support. (ref: {error_id})",
    )
```

Error ID is logged server-side with full traceback for debugging; only the opaque reference is sent to the client.

**Files:** `profit-sentinel-rs/python/sentinel_agent/upload_routes.py`

---

### 2.9 — Dockerfile.sidecar: Remove `|| true` from cargo build

**Vulnerability:** `cargo build ... || true` on the dependency-caching layer meant a broken Rust build could silently pass, producing a Docker image without a working binary.

**Before:**
```dockerfile
RUN cargo build --release -p sentinel-server 2>/dev/null || true
```

**After:**
```dockerfile
RUN cargo build --release -p sentinel-server
```

**Files:** `profit-sentinel-rs/Dockerfile.sidecar`

---

### 2.10 — dual_auth.py: Don't trust raw X-Forwarded-For

**Vulnerability:** `forwarded.split(",")[0]` trusted the leftmost IP in `X-Forwarded-For`, which is client-supplied and trivially spoofable. An attacker could bypass IP-based rate limiting by injecting a fake IP.

**Before:**
```python
def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
```

**After:**
```python
def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        if parts:
            return parts[-1]  # rightmost = ALB-appended (trusted)
    return request.client.host if request.client else "unknown"
```

**Files:** `profit-sentinel-rs/python/sentinel_agent/dual_auth.py`

---

### 2.11 — s3_service.py: Enforce upload size via presigned POST conditions

**Vulnerability:** Presigned PUT URLs had no server-side file size enforcement. A user could ignore the client-side limit and upload arbitrarily large files directly to S3.

**Before:**
```python
def generate_presigned_url(s3_client, bucket_name, key, expires_in=3600) -> str:
    return s3_client.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket_name, "Key": key, "ContentType": "application/octet-stream"},
        ExpiresIn=expires_in,
    )
```

**After:**
```python
def generate_presigned_post(s3_client, bucket_name, key, max_size_bytes, expires_in=3600) -> dict:
    return s3_client.generate_presigned_post(
        Bucket=bucket_name,
        Key=key,
        Conditions=[
            ["content-length-range", 1, max_size_bytes],
        ],
        ExpiresIn=expires_in,
    )
```

The `content-length-range` condition is enforced by S3 itself — uploads exceeding the limit are rejected at the infrastructure level. Response shape now includes `fields` (for multipart form upload) and `upload_method: "POST"`.

**Files:** `profit-sentinel-rs/python/sentinel_agent/s3_service.py`

---

### 2.12 — upload_routes.py: Rate limit presign endpoint

**Vulnerability:** `/uploads/presign` had no dedicated throttling. An attacker could generate unlimited presigned URLs, enabling S3 abuse.

**After:** Added `_check_presign_rate_limit()` — 20 presign requests per hour per user. Uses the same `_rate_lock` from `dual_auth.py` for thread safety. Returns 429 when exceeded.

```python
PRESIGN_RATE_LIMIT = 20  # per user per hour

async def _check_presign_rate_limit(user_id: str) -> None:
    """Raise 429 if the user has exceeded 20 presign requests/hour."""
    ...
```

Integrated at the top of the presign endpoint, before captcha verification.

**Files:** `profit-sentinel-rs/python/sentinel_agent/upload_routes.py`

---

## Test Updates

| Test File | Changes |
|-----------|---------|
| `test_upload_routes.py` | Updated all presign mocks from `generate_presigned_url` (string return) to `generate_presigned_post` (dict return with `url` + `fields`) |
| `test_dual_auth.py` | Updated `test_x_forwarded_for` to expect rightmost IP (`10.0.0.1`) instead of leftmost (`1.2.3.4`). Updated presign mocks to `generate_presigned_post`. |

---

## Verification Checklist

- [x] 2.1 — Counterfactual router requires auth (router-level + endpoint-level)
- [x] 2.2 — Grok proxy requires Supabase token, rate limited 10/min, model allowlist
- [x] 2.3 — Dev mode raises `RuntimeError` if ECS/AWS env vars detected
- [x] 2.4 — localhost origins only present when `sidecar_dev_mode=True`
- [x] 2.5 — S3 prefix check uses path-segment boundary (`/` suffix)
- [x] 2.6 — Turnstile exceptions return `False` (fail closed)
- [x] 2.7 — Missing Supabase env vars → `authenticated = false` + error message
- [x] 2.8 — 500 responses contain opaque `error_id`, not raw exception text
- [x] 2.9 — Dockerfile `cargo build` fails fast (no `|| true`)
- [x] 2.10 — `X-Forwarded-For` trusts rightmost IP (ALB-appended)
- [x] 2.11 — Presigned POST with `content-length-range` enforces file size at S3 level
- [x] 2.12 — Presign endpoint rate limited to 20 req/hour per user
- [x] All 837 Python tests passing
- [x] No changes to `world_model/` modules
- [x] No changes to Rust engine code (except Dockerfile `|| true` removal)

---

## Files Modified

### Python (sidecar)
- `sentinel_agent/routes/counterfactual.py` — auth requirement
- `sentinel_agent/sidecar.py` — dev mode guard, CORS hardening
- `sentinel_agent/upload_routes.py` — S3 boundary fix, error sanitization, presign rate limit
- `sentinel_agent/dual_auth.py` — X-Forwarded-For fix
- `sentinel_agent/turnstile.py` — fail closed
- `sentinel_agent/s3_service.py` — presigned POST with size enforcement

### Web (Next.js)
- `web/src/app/api/grok/route.ts` — auth + rate limiting
- `web/src/app/dashboard/layout.tsx` — fail closed on missing Supabase

### Infrastructure
- `Dockerfile.sidecar` — remove `|| true`

### Tests
- `tests/test_upload_routes.py` — presigned POST mocks
- `tests/test_dual_auth.py` — X-Forwarded-For + presigned POST mocks
