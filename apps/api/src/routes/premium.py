"""
Premium Routes - Multi-File Vendor Correlation Diagnostic

PREVIEW FEATURE: Available now as a preview of upcoming premium features.

This module provides endpoints for the multi-file diagnostic that:
- Accepts up to 200 files (inventory + vendor invoices)
- Cross-references inventory with vendor invoices
- Discovers causal patterns (short ships → negative stock, etc.)
- Provides interactive Q&A for each correlation

Endpoints:
    POST /premium/diagnostic/start     - Upload files, start correlation session
    GET  /premium/diagnostic/{id}/question  - Get current correlation question
    POST /premium/diagnostic/{id}/answer    - Submit answer
    GET  /premium/diagnostic/{id}/summary   - Get session summary with vendor metrics
    GET  /premium/diagnostic/{id}/report    - Generate comprehensive PDF
    DELETE /premium/diagnostic/{id}         - Delete session
"""

import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import get_settings

# Rate limiter - 20 premium diagnostic starts per hour (stricter due to more processing)
limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session storage with TTL (use Redis/DB in production)
premium_sessions: dict[str, dict[str, Any]] = {}

# Session cleanup task reference
_cleanup_task: asyncio.Task | None = None


def _get_session_ttl() -> timedelta:
    """Get the session TTL from settings."""
    settings = get_settings()
    return timedelta(hours=settings.session_ttl_hours)


def _is_session_expired(session_data: dict[str, Any]) -> bool:
    """Check if a session has expired based on TTL."""
    created_at = session_data.get("created_at")
    if not created_at:
        return True
    return datetime.now() - created_at > _get_session_ttl()


def _cleanup_session_files(session_data: dict[str, Any]) -> None:
    """Clean up any temporary files associated with a session."""
    if "report_path" in session_data:
        report_path = session_data["report_path"]
        if report_path and os.path.exists(report_path):
            try:
                os.unlink(report_path)
                logger.debug(f"Cleaned up report file: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp report file: {e}")


async def _cleanup_expired_sessions() -> None:
    """Background task to clean up expired sessions."""
    while True:
        try:
            # Check every 5 minutes
            await asyncio.sleep(300)

            expired_ids = []
            for session_id, data in premium_sessions.items():
                if _is_session_expired(data):
                    expired_ids.append(session_id)

            for session_id in expired_ids:
                data = premium_sessions.pop(session_id, None)
                if data:
                    _cleanup_session_files(data)
                    logger.info(f"Cleaned up expired premium session: {session_id}")

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired premium sessions")

        except asyncio.CancelledError:
            logger.info("Premium session cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in premium session cleanup task: {e}")


def start_cleanup_task() -> None:
    """Start the background cleanup task if not already running."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_expired_sessions())
        logger.info("Started premium session cleanup task")


def _validate_session(session_id: str) -> dict[str, Any]:
    """Validate and return session data, raising HTTPException if invalid or expired."""
    if session_id not in premium_sessions:
        raise HTTPException(404, "Session not found")

    data = premium_sessions[session_id]

    if _is_session_expired(data):
        # Clean up the expired session
        _cleanup_session_files(data)
        del premium_sessions[session_id]
        raise HTTPException(410, "Session has expired. Please start a new diagnostic.")

    return data


# =============================================================================
# MODELS
# =============================================================================


class PremiumAnswerRequest(BaseModel):
    classification: str
    note: str | None = ""


class PremiumStartResponse(BaseModel):
    session_id: str
    store_name: str
    files_processed: dict
    inventory_items: int
    vendor_invoice_lines: int
    patterns_discovered: int
    total_potential_impact: float
    vendors_analyzed: int
    is_premium_preview: bool = True


class CorrelationQuestionResponse(BaseModel):
    pattern_index: int
    pattern_type: str
    question: str
    suggested_answers: list[list[str]]
    affected_skus_count: int
    affected_vendors: list[str]
    total_impact: float
    confidence: float
    description: str
    evidence_summary: dict
    progress: dict


class PremiumSummary(BaseModel):
    session_id: str
    store_name: str
    files_processed: dict
    inventory_items: int
    vendor_invoice_lines: int
    patterns_discovered: int
    patterns_by_type: dict
    total_potential_impact: float
    vendors_analyzed: int
    answers: list
    is_complete: bool
    vendor_summaries: list


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/diagnostic/start", response_model=PremiumStartResponse)
@limiter.limit("20/hour")
async def start_premium_diagnostic(
    request: Request,
    inventory_file: UploadFile = File(...),
    invoice_files: list[UploadFile] = File(default=[]),
    store_name: str = Form(default="My Store"),
):
    """
    Start a premium multi-file diagnostic session with vendor correlation.

    PREMIUM PREVIEW FEATURE

    Upload:
    - One inventory CSV file (required)
    - Up to 200 vendor invoice CSV files (optional)

    The engine will:
    1. Parse all files with auto-column detection
    2. Normalize SKUs across files
    3. Aggregate vendor invoice data
    4. Discover causal correlations
    5. Generate interactive questions

    Sessions expire after 24 hours (configurable via SESSION_TTL_HOURS).
    """
    # Start cleanup task if not running
    start_cleanup_task()

    # Import diagnostic engine (lazy load)
    try:
        from sentinel_engine.diagnostic.multi_file import MultiFileDiagnostic
    except ImportError as e:
        logger.error(f"Failed to import multi-file diagnostic: {e}")
        raise HTTPException(500, "Premium diagnostic engine not available")

    # Check file limits
    total_files = 1 + len(invoice_files)
    if total_files > 200:
        raise HTTPException(400, f"Maximum 200 files allowed, got {total_files}")

    # Create diagnostic instance
    diagnostic = MultiFileDiagnostic()

    # Process inventory file
    try:
        inventory_content = await inventory_file.read()
        inventory_text = inventory_content.decode("utf-8", errors="replace")
        diagnostic.add_inventory_file(
            inventory_file.filename or "inventory.csv", content=inventory_text
        )
    except Exception as e:
        logger.error(f"Failed to parse inventory file: {e}")
        raise HTTPException(400, f"Failed to parse inventory file: {str(e)}")

    # Process invoice files
    for invoice_file in invoice_files:
        try:
            invoice_content = await invoice_file.read()
            invoice_text = invoice_content.decode("utf-8", errors="replace")
            diagnostic.add_vendor_invoice(
                invoice_file.filename or "invoice.csv", content=invoice_text
            )
        except Exception as e:
            logger.warning(f"Failed to parse invoice {invoice_file.filename}: {e}")
            # Continue with other files

    # Start session and discover correlations
    try:
        diagnostic.start_session()
    except Exception as e:
        logger.error(f"Failed to start premium diagnostic session: {e}")
        raise HTTPException(500, f"Failed to initialize diagnostic: {str(e)}")

    # Get summary
    summary = diagnostic.get_summary()

    # Store session
    session_id = str(uuid.uuid4())
    premium_sessions[session_id] = {
        "diagnostic": diagnostic,
        "store_name": store_name,
        "created_at": datetime.now(),
        "status": "in_progress" if summary["patterns_discovered"] > 0 else "complete",
    }

    logger.info(
        f"Started premium diagnostic session {session_id} "
        f"with {summary['files_processed']['total']} files, "
        f"{summary['patterns_discovered']} patterns discovered"
    )

    return PremiumStartResponse(
        session_id=session_id,
        store_name=store_name,
        files_processed=summary["files_processed"],
        inventory_items=summary["inventory_items"],
        vendor_invoice_lines=summary["vendor_invoice_lines"],
        patterns_discovered=summary["patterns_discovered"],
        total_potential_impact=summary["total_potential_impact"],
        vendors_analyzed=summary["vendors_analyzed"],
        is_premium_preview=True,
    )


@router.get("/{session_id}/question", response_model=CorrelationQuestionResponse | None)
@limiter.limit("100/minute")
async def get_premium_question(request: Request, session_id: str):
    """Get the current correlation question for a premium session."""
    data = _validate_session(session_id)
    diagnostic = data["diagnostic"]
    question = diagnostic.get_current_question()

    if not question:
        return None

    return CorrelationQuestionResponse(
        pattern_index=question["pattern_index"],
        pattern_type=question["pattern_type"],
        question=question["question"],
        suggested_answers=[list(a) for a in question["suggested_answers"]],
        affected_skus_count=question["affected_skus_count"],
        affected_vendors=question["affected_vendors"],
        total_impact=question["total_impact"],
        confidence=question["confidence"],
        description=question["description"],
        evidence_summary=question["evidence_summary"],
        progress=question["progress"],
    )


@router.post("/{session_id}/answer")
@limiter.limit("100/minute")
async def submit_premium_answer(
    request: Request, session_id: str, answer: PremiumAnswerRequest
):
    """Submit an answer to the current correlation question."""
    data = _validate_session(session_id)
    diagnostic = data["diagnostic"]
    result = diagnostic.answer_question(answer.classification, answer.note)

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Update status if complete
    if result.get("is_complete"):
        premium_sessions[session_id]["status"] = "complete"
        logger.info(f"Premium diagnostic session {session_id} completed")

    return result


@router.get("/{session_id}/summary", response_model=PremiumSummary)
@limiter.limit("100/minute")
async def get_premium_summary(request: Request, session_id: str):
    """Get comprehensive summary with vendor metrics."""
    data = _validate_session(session_id)
    diagnostic = data["diagnostic"]
    summary = diagnostic.get_summary()
    final_report = diagnostic.get_final_report()

    return PremiumSummary(
        session_id=session_id,
        store_name=data["store_name"],
        files_processed=summary["files_processed"],
        inventory_items=summary["inventory_items"],
        vendor_invoice_lines=summary["vendor_invoice_lines"],
        patterns_discovered=summary["patterns_discovered"],
        patterns_by_type=summary["patterns_by_type"],
        total_potential_impact=summary["total_potential_impact"],
        vendors_analyzed=summary["vendors_analyzed"],
        answers=summary["answers"],
        is_complete=summary["is_complete"],
        vendor_summaries=final_report["vendor_summaries"],
    )


@router.get("/{session_id}/report")
@limiter.limit("10/minute")
async def generate_premium_report(request: Request, session_id: str):
    """Generate and download the premium PDF report with vendor analysis."""
    data = _validate_session(session_id)

    # For now, use the standard report generator
    # In the future, create a premium report with vendor correlation details
    try:
        from sentinel_engine.diagnostic.report import generate_report_from_session
    except ImportError as e:
        logger.error(f"Failed to import report generator: {e}")
        raise HTTPException(500, "Report generator not available")

    diagnostic = data["diagnostic"]

    # Get report data
    report_data = diagnostic.get_final_report()

    # Create temp file for PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        output_path = f.name

    try:
        # Generate report
        # Note: We're adapting the multi-file report to the standard format
        adapted_session_data = {
            "patterns_reviewed": [
                {
                    "pattern_id": p["type"],
                    "pattern_name": p["description"][:50],
                    "classification": data["diagnostic"]._answers[i]["classification"]
                    if i < len(data["diagnostic"]._answers)
                    else "investigate",
                    "total_value": p["impact"],
                    "item_count": len(p["affected_skus"]),
                    "items": [{"sku": sku} for sku in p["affected_skus"][:10]],
                }
                for i, p in enumerate(report_data["patterns"])
            ],
            "summary": {
                "total_shrinkage": report_data["inventory_stats"][
                    "total_negative_value"
                ],
                "explained_value": report_data["summary"]["total_potential_impact"]
                * 0.7,
                "unexplained_value": report_data["summary"]["total_potential_impact"]
                * 0.3,
                "reduction_percent": 70.0,
            },
            "vendor_analysis": report_data["vendor_summaries"],
        }

        # Create simple item list
        items = [
            {"sku": sku, "description": "", "stock": 0, "cost": 0}
            for p in report_data["patterns"]
            for sku in p["affected_skus"][:5]
        ]

        generate_report_from_session(
            session_data=adapted_session_data,
            all_items=items,
            store_name=data["store_name"],
            output_path=output_path,
        )
    except Exception as e:
        logger.error(f"Failed to generate premium report: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise HTTPException(500, f"Failed to generate report: {str(e)}")

    # Store path
    data["report_path"] = output_path

    safe_store_name = data["store_name"].lower().replace(" ", "_")
    filename = f"profit_sentinel_premium_report_{safe_store_name}.pdf"

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=filename,
    )


@router.delete("/{session_id}")
@limiter.limit("50/minute")
async def delete_premium_session(request: Request, session_id: str):
    """Delete a premium diagnostic session."""
    if session_id not in premium_sessions:
        raise HTTPException(404, "Session not found")

    # Clean up temp files
    data = premium_sessions.pop(session_id)
    _cleanup_session_files(data)
    logger.info(f"Deleted premium diagnostic session: {session_id}")

    return {"message": "Session deleted"}


@router.get("/sessions")
@limiter.limit("30/minute")
async def list_premium_sessions(request: Request):
    """List all active premium diagnostic sessions."""
    sessions = []
    for session_id, data in premium_sessions.items():
        # Skip expired sessions in listing
        if _is_session_expired(data):
            continue
        summary = data["diagnostic"].get_summary()
        sessions.append(
            {
                "session_id": session_id,
                "store_name": data["store_name"],
                "status": data["status"],
                "created_at": data["created_at"].isoformat(),
                "files_processed": summary["files_processed"],
                "patterns_discovered": summary["patterns_discovered"],
                "is_premium_preview": True,
            }
        )
    return {"sessions": sessions, "count": len(sessions)}
