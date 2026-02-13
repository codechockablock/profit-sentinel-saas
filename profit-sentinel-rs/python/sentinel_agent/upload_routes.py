"""Upload & Analysis routes for the sidecar API.

Adds the 3 legacy endpoints required by the production frontend:
    POST /uploads/presign        — generate S3 presigned URLs
    POST /uploads/suggest-mapping — AI column mapping suggestions
    POST /analysis/analyze       — run Rust analysis pipeline

Public mode:  Anonymous users can use these endpoints (no JWT required).
              Rate-limited to 5 analyses/hour, 10 MB file limit.
Auth mode:    Authenticated users get 100 analyses/hour, 50 MB limit,
              results saved, no upgrade prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request

from .analysis_store import save_analysis
from .column_adapter import ColumnAdapter
from .dual_auth import UserContext, build_upgrade_prompt, check_rate_limit
from .engine import PipelineError, SentinelEngine
from .result_adapter import RustResultAdapter
from .s3_service import (
    SUPPORTED_POS_SYSTEMS,
    delete_file,
    generate_upload_urls,
    get_s3_client,
    load_dataframe,
    sanitize_filename,
)
from .sidecar_config import SidecarSettings
from .turnstile import verify_turnstile_token

logger = logging.getLogger("sentinel.upload_routes")

# Numeric column aliases for cleaning (subset of legacy)
NUMERIC_COLUMN_ALIASES = [
    "quantity",
    "qty",
    "Qty.",
    "qoh",
    "on_hand",
    "In Stock Qty.",
    "stock",
    "inventory",
    "qty_on_hand",
    "cost",
    "Cost",
    "unit_cost",
    "avg_cost",
    "cogs",
    "revenue",
    "retail",
    "Retail",
    "retail_price",
    "price",
    "sell_price",
    "msrp",
    "Sug. Retail",
    "sug. retail",
    "sold",
    "Sold",
    "units_sold",
    "qty_sold",
    "sales_qty",
    "movement",
    "margin",
    "Profit Margin %",
    "margin_pct",
    "gp_pct",
    "profit_margin",
    "sub_total",
    "subtotal",
    "total",
    "inventory_value",
    "discount",
    "discount_amt",
    "tax",
    "sales_tax",
    "reorder_point",
    "min_qty",
    "safety_stock",
    "on_order_qty",
    "on_order",
]


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert numeric columns from any POS system."""
    df_cols_lower = {
        col.lower().replace(" ", "").replace(".", ""): col for col in df.columns
    }
    existing_numeric = []

    for alias in NUMERIC_COLUMN_ALIASES:
        normalized = alias.lower().replace(" ", "").replace(".", "")
        if normalized in df_cols_lower:
            existing_numeric.append(df_cols_lower[normalized])

    for alias in NUMERIC_COLUMN_ALIASES:
        if alias in df.columns and alias not in existing_numeric:
            existing_numeric.append(alias)

    existing_numeric = list(dict.fromkeys(existing_numeric))

    for col in existing_numeric:
        try:
            df[col] = (
                df[col].astype(str).str.replace(r"[$,\%]", "", regex=True).str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0.0)
        except Exception as e:
            logger.warning(f"Failed to clean column {col}: {e}")

    return df


def _validate_and_apply_mapping(
    df: pd.DataFrame, mapping_dict: dict[str, str]
) -> tuple[pd.DataFrame, list[str]]:
    """Safely validate and apply column mapping to DataFrame."""
    warnings: list[str] = []
    df_columns = set(df.columns.tolist())

    valid_mapping: dict[str, str] = {}
    seen_targets: set[str] = set()

    for source_col, target_col in mapping_dict.items():
        if not target_col or not str(target_col).strip():
            continue

        target_col = str(target_col).strip()

        if source_col not in df_columns:
            warnings.append(f"Source column '{source_col}' not found in data, skipping")
            continue

        if target_col in seen_targets:
            warnings.append(
                f"Duplicate target column '{target_col}' for source '{source_col}', skipping"
            )
            continue

        valid_mapping[source_col] = target_col
        seen_targets.add(target_col)

    if valid_mapping:
        df = df.rename(columns=valid_mapping)

    return df, warnings


def create_upload_router(
    settings: SidecarSettings,
    engine: SentinelEngine | None,
    get_user_context,
    app_state=None,
) -> APIRouter:
    """Create the upload/analysis router with dual auth (anonymous + authenticated).

    Args:
        settings: Sidecar configuration.
        engine: Rust pipeline engine (may be None if binary not found).
        get_user_context: FastAPI dependency that returns a UserContext.
                          Allows anonymous users (no 401 for missing token).
        app_state: Optional AppState for Engine 1→2 bridging.
    """

    uploads_router = APIRouter(prefix="/uploads", tags=["uploads"])
    analysis_router = APIRouter(prefix="/analysis", tags=["analysis"])

    # -----------------------------------------------------------------
    # POST /uploads/presign
    # -----------------------------------------------------------------

    @uploads_router.post("/presign")
    async def presign_upload(
        request: Request,
        filenames: list[str] = Form(...),
        cf_turnstile_response: str = Form(default=""),
        ctx: UserContext = Depends(get_user_context),
    ):
        """Generate presigned URLs for direct S3 upload.

        Anonymous users: uploads/anonymous/{ip_hash}/{uuid}/{filename}
        Authenticated:   uploads/{user_id}/{uuid}/{filename}
        """
        # Captcha verification for anonymous users only
        if not ctx.is_authenticated and settings.turnstile_secret_key:
            client_ip = request.client.host if request.client else None
            is_valid = await verify_turnstile_token(
                cf_turnstile_response,
                settings.turnstile_secret_key,
                remote_ip=client_ip,
            )
            if not is_valid:
                raise HTTPException(
                    status_code=403,
                    detail="Captcha verification failed. Please try again.",
                )

        try:
            s3_client = get_s3_client()
            result = generate_upload_urls(
                s3_client,
                settings.s3_bucket_name,
                filenames,
                ctx.s3_prefix,
                max_file_size_mb=ctx.max_file_size_mb,
            )
            logger.info(
                f"Generated {len(result['presigned_urls'])} presigned URLs for {ctx!r}"
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # -----------------------------------------------------------------
    # POST /uploads/suggest-mapping
    # -----------------------------------------------------------------

    @uploads_router.post("/suggest-mapping")
    async def suggest_mapping(
        request: Request,
        key: str = Form(...),
        filename: str = Form(...),
        ctx: UserContext = Depends(get_user_context),
    ) -> dict:
        """Analyze uploaded file and suggest column mappings.

        Uses Anthropic Claude for intelligent mapping with heuristic fallback.
        """
        # Validate S3 key ownership
        if not key.startswith(ctx.s3_prefix):
            raise HTTPException(
                status_code=403,
                detail="Access denied: you can only access your own uploaded files",
            )

        try:
            from .mapping_service import MappingService

            s3_client = get_s3_client()
            df = load_dataframe(
                s3_client,
                settings.s3_bucket_name,
                key,
                sample_rows=50,
                max_size_mb=ctx.max_file_size_mb,
            )

            mapping_svc = MappingService()
            result = mapping_svc.suggest_mapping(
                df, filename, anthropic_api_key=settings.anthropic_api_key
            )
            return result

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Column mapping failed for {key}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Column mapping failed. Please check your file format and try again.",
            )

    # -----------------------------------------------------------------
    # POST /analysis/analyze
    # -----------------------------------------------------------------

    @analysis_router.post("/analyze")
    async def analyze_upload(
        request: Request,
        key: str = Form(...),
        mapping: str = Form(...),
        background_tasks: BackgroundTasks = BackgroundTasks(),
        ctx: UserContext = Depends(get_user_context),
    ) -> dict:
        """Analyze uploaded POS data for profit leaks using Rust pipeline.

        Rate-limited: 5/hour anonymous, 100/hour authenticated.
        Anonymous results include an upgrade_prompt.
        """
        # Rate limit check
        await check_rate_limit(ctx)

        # Parse mapping
        try:
            mapping_dict = json.loads(mapping)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=422, detail=f"Invalid mapping JSON: {str(e)}"
            )

        if not isinstance(mapping_dict, dict):
            raise HTTPException(
                status_code=400,
                detail="Mapping must be a JSON object with column name pairs",
            )

        # Validate S3 key ownership
        if not key.startswith(ctx.s3_prefix):
            raise HTTPException(
                status_code=403,
                detail="Access denied: you can only analyze your own uploaded files",
            )

        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Analysis engine not available. sentinel-server binary not found.",
            )

        try:
            overall_start = time.time()
            logger.info(f"Starting analysis for key: {key} ({ctx!r})")

            s3_client = get_s3_client()

            # Load full DataFrame (respects per-user file size limit)
            load_start = time.time()
            df = load_dataframe(
                s3_client,
                settings.s3_bucket_name,
                key,
                max_size_mb=ctx.max_file_size_mb,
            )
            logger.info(
                f"Loaded DataFrame ({len(df)} rows, {len(df.columns)} columns) "
                f"in {time.time() - load_start:.2f}s"
            )

            # Apply column mapping
            df, mapping_warnings = _validate_and_apply_mapping(df, mapping_dict)

            # Clean numeric columns
            df = _clean_numeric_columns(df)

            # Drop duplicate columns
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated(keep="first")]

            # Convert to records for enrichment
            rows = df.to_dict(orient="records")

            # Run Rust pipeline
            adapter_start = time.time()
            col_adapter = ColumnAdapter(
                default_store_id=settings.sentinel_default_store,
            )

            try:
                csv_path = col_adapter.to_rust_csv(df)
                adapter_time = time.time() - adapter_start
                logger.info(f"Column adapter: {len(df)} rows in {adapter_time:.2f}s")

                # Run sentinel-server
                rust_start = time.time()
                digest = engine.run(
                    csv_path,
                    top_k=settings.sentinel_top_k,
                    timeout_seconds=300,
                )
                rust_time = time.time() - rust_start
                logger.info(f"Rust pipeline: {rust_time:.2f}s")

                # Transform to legacy format
                total_time = time.time() - overall_start
                result_adapter = RustResultAdapter()
                result = result_adapter.transform(
                    digest=digest.model_dump(),
                    total_rows=len(df),
                    analysis_time=total_time,
                    original_rows=rows,
                )

            finally:
                col_adapter.cleanup()

            # Add metadata
            result["status"] = "success"
            result["warnings"] = mapping_warnings if mapping_warnings else None
            result["supported_pos_systems"] = SUPPORTED_POS_SYSTEMS
            result["is_authenticated"] = ctx.is_authenticated

            # Upgrade prompt for anonymous users
            if not ctx.is_authenticated:
                result["upgrade_prompt"] = build_upgrade_prompt()

            total_time = time.time() - overall_start
            logger.info(
                f"Analysis completed in {total_time:.2f}s "
                f"({len(df)} rows, {result['summary']['total_items_flagged']} issues, "
                f"auth={ctx.is_authenticated})"
            )

            # Engine 1→2 bridge: feed Rust pipeline results to world model
            if app_state is not None:
                try:
                    app_state.feed_engine2(result)
                except Exception as e:
                    # Bridge failure must NOT block the analysis response
                    logger.warning(f"Engine 1→2 bridge failed (non-fatal): {e}")

            # Engine 1→3: enrich findings with counterfactuals
            if app_state is not None and app_state.counterfactual_engine is not None:
                try:
                    app_state.counterfactual_engine.enrich(result)
                    logger.info(
                        "Engine 3: %s",
                        result.get("engine3_summary", {}).get("message", "no summary"),
                    )
                except Exception as e:
                    # Engine 3 failure must NOT block the analysis response
                    logger.warning(f"Engine 3 enrichment failed (non-fatal): {e}")

            if app_state is not None:
                # Populate digest_cache so dashboard endpoints can display data
                # without needing a local CSV file. Use a long TTL (1 hour).
                try:
                    from .routes.state import DigestCacheEntry

                    cache_key = f":{settings.sentinel_top_k}"
                    app_state.digest_cache[cache_key] = DigestCacheEntry(
                        digest,
                        ttl_seconds=3600,
                    )
                    logger.info(
                        "Cached analysis digest (key=%s, %d issues) for dashboard",
                        cache_key,
                        len(digest.issues),
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache digest (non-fatal): {e}")

            # Persist analysis for authenticated users
            if ctx.is_authenticated:
                try:
                    # Compute file hash from the S3 key (deterministic per upload)
                    file_hash = hashlib.sha256(key.encode()).hexdigest()
                    # Extract filename from S3 key (last path component)
                    original_filename = key.rsplit("/", 1)[-1] if "/" in key else key

                    saved = save_analysis(
                        user_id=ctx.user_id,
                        result=result,
                        file_hash=file_hash,
                        file_row_count=len(df),
                        file_column_count=len(df.columns),
                        original_filename=original_filename,
                        processing_time_seconds=round(total_time, 2),
                    )
                    result["analysis_id"] = saved.get("id")
                    logger.info(
                        f"Analysis persisted: {saved.get('id')} for {ctx.user_id}"
                    )
                except Exception as e:
                    # Persistence failure should NOT block the analysis response
                    logger.warning(f"Failed to persist analysis: {e}", exc_info=True)

            # Schedule S3 file cleanup (60s delay for retries)
            async def _cleanup_file():
                await asyncio.sleep(60)
                try:
                    delete_file(s3_client, settings.s3_bucket_name, key)
                    # Structured log for CloudWatch monitoring
                    logger.info(
                        "S3_FILE_DELETED key=%s bucket=%s user_type=%s "
                        "rows=%d delay_seconds=60 method=background_task",
                        key,
                        settings.s3_bucket_name,
                        "authenticated" if ctx.is_authenticated else "guest",
                        len(df),
                    )
                except Exception as e:
                    logger.error(
                        "S3_DELETE_FAILED key=%s bucket=%s error=%s "
                        "user_type=%s fallback=s3_lifecycle_24h",
                        key,
                        settings.s3_bucket_name,
                        str(e),
                        "authenticated" if ctx.is_authenticated else "guest",
                    )

            background_tasks.add_task(_cleanup_file)

            return result

        except PipelineError as e:
            logger.error(f"Rust pipeline failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Analysis pipeline failed: {str(e)}",
            )
        except pd.errors.EmptyDataError:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty or has no valid data",
            )
        except ValueError as e:
            error_msg = str(e)
            if "too large" in error_msg.lower() or "size" in error_msg.lower():
                raise HTTPException(status_code=413, detail=error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    # -----------------------------------------------------------------
    # GET /analysis/primitives
    # -----------------------------------------------------------------

    @analysis_router.get("/primitives")
    async def list_primitives() -> dict:
        """List all available analysis primitives."""
        from .result_adapter import ALL_PRIMITIVES, LEAK_DISPLAY, RECOMMENDATIONS

        primitives = {}
        for p in ALL_PRIMITIVES:
            display = LEAK_DISPLAY.get(p, {})
            primitives[p] = {
                "key": p,
                "title": display.get("title", p.replace("_", " ").title()),
                "severity": display.get("severity", "info"),
                "category": display.get("category", "Unknown"),
                "recommendations": RECOMMENDATIONS.get(p, []),
            }

        return {"primitives": primitives, "count": len(primitives)}

    # -----------------------------------------------------------------
    # GET /analysis/supported-pos
    # -----------------------------------------------------------------

    @analysis_router.get("/supported-pos")
    async def list_supported_pos() -> dict:
        """List all supported POS systems."""
        return {
            "supported_systems": SUPPORTED_POS_SYSTEMS,
            "count": len(SUPPORTED_POS_SYSTEMS),
            "notes": "Column mapping auto-detects formats from these systems",
        }

    # -----------------------------------------------------------------
    # POST /analysis/send-report
    # -----------------------------------------------------------------

    @analysis_router.post("/send-report")
    async def send_report(
        request: Request,
        background_tasks: BackgroundTasks = BackgroundTasks(),
    ) -> dict:
        """Send a full analysis report via email with PDF attachment.

        The frontend holds the analysis result in React state (not persisted
        for guests) and POSTs it back here along with the guest's email.

        After the email is sent, triggers the anonymization pipeline to
        extract aggregated patterns into dorian_facts.

        Request body (JSON):
            email: str          — recipient email address
            analysis_result: dict — the full result from /analysis/analyze

        Rate limit: Shares the analysis rate limit (prevents spam).
        """
        import re

        from .anonymizer import anonymize_analysis, store_anonymized_facts
        from .email_service import send_report_email
        from .pdf_report import generate_report_pdf

        body = await request.json()
        email = body.get("email", "").strip()
        analysis_result = body.get("analysis_result")

        # Validate email
        if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            raise HTTPException(status_code=422, detail="Valid email address required")

        if not analysis_result or not isinstance(analysis_result, dict):
            raise HTTPException(status_code=422, detail="analysis_result is required")

        if "leaks" not in analysis_result or "summary" not in analysis_result:
            raise HTTPException(
                status_code=422,
                detail="analysis_result must contain 'leaks' and 'summary' keys",
            )

        # Rate limit: reuse the analysis bucket to prevent email spam
        # (guests get 5/hour — sending a report counts as 1)
        try:
            ctx = await get_user_context(request)
            await check_rate_limit(ctx)
        except HTTPException:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later.",
            )

        summary = analysis_result.get("summary", {})
        leaks = analysis_result.get("leaks", {})
        total_items = summary.get("total_rows_analyzed", 0)
        total_flagged = summary.get("total_items_flagged", 0)
        active_leaks = sum(1 for v in leaks.values() if v.get("count", 0) > 0)

        # 1. Generate PDF
        try:
            pdf_bytes = generate_report_pdf(analysis_result)
            logger.info("Generated PDF report: %d bytes for %s", len(pdf_bytes), email)
        except Exception as e:
            logger.error("PDF generation failed: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to generate report PDF")

        # 2. Send email with PDF attachment
        resend_key = settings.resend_api_key
        if not resend_key:
            raise HTTPException(
                status_code=503,
                detail="Email service not configured",
            )

        try:
            email_result = await send_report_email(
                api_key=resend_key,
                to=email,
                pdf_bytes=pdf_bytes,
                total_items=total_items,
                total_flagged=total_flagged,
                leak_count=active_leaks,
            )
            logger.info("Report email sent to %s, id=%s", email, email_result.get("id"))
        except Exception as e:
            logger.error("Failed to send report email: %s", e, exc_info=True)
            raise HTTPException(
                status_code=502,
                detail="Failed to send email. Please try again.",
            )

        # 3. Trigger anonymization pipeline in background
        async def _anonymize_and_store():
            try:
                facts = anonymize_analysis(analysis_result)
                if facts and settings.supabase_url and settings.supabase_service_key:
                    stored = await store_anonymized_facts(
                        facts,
                        supabase_url=settings.supabase_url,
                        supabase_service_key=settings.supabase_service_key,
                    )
                    logger.info(
                        "Anonymization pipeline: stored %d facts from report to %s",
                        stored,
                        email,
                    )
                else:
                    logger.warning(
                        "Anonymization skipped: %d facts, supabase configured=%s",
                        len(facts),
                        bool(settings.supabase_url),
                    )
            except Exception as e:
                # Anonymization failure must NOT block the user response
                logger.error("Anonymization pipeline failed: %s", e, exc_info=True)

        background_tasks.add_task(_anonymize_and_store)

        return {
            "success": True,
            "message": f"Report sent to {email}",
            "email_id": email_result.get("id"),
        }

    # Combine into a parent router
    combined = APIRouter()
    combined.include_router(uploads_router)
    combined.include_router(analysis_router)

    return combined
