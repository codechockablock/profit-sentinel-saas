"""
Diagnostic Routes - Conversational Shrinkage Diagnostic System

Powered by the Dorian knowledge engine. Provides an interactive Q&A flow
that helps users understand and classify inventory discrepancies.

Endpoints:
    POST /diagnostic/start     - Upload CSV, start session
    GET  /diagnostic/{id}/question  - Get current question
    POST /diagnostic/{id}/answer    - Submit answer
    GET  /diagnostic/{id}/summary   - Get running totals
    GET  /diagnostic/{id}/report    - Generate PDF
    POST /diagnostic/{id}/email     - Email report
    DELETE /diagnostic/{id}         - Delete session
"""

import asyncio
import csv
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta
from io import StringIO
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..config import get_settings
from ..dependencies import get_current_user
from ..services.email import get_email_service

# Rate limiter - 30 diagnostic starts per hour, 100 requests per minute for other endpoints
limiter = Limiter(key_func=get_remote_address)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session storage with TTL (use Redis/DB in production)
diagnostic_sessions: dict[str, dict[str, Any]] = {}

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
            for session_id, data in diagnostic_sessions.items():
                if _is_session_expired(data):
                    expired_ids.append(session_id)

            for session_id in expired_ids:
                data = diagnostic_sessions.pop(session_id, None)
                if data:
                    _cleanup_session_files(data)
                    logger.info(f"Cleaned up expired diagnostic session: {session_id}")

            if expired_ids:
                logger.info(
                    f"Cleaned up {len(expired_ids)} expired diagnostic sessions"
                )

        except asyncio.CancelledError:
            logger.info("Session cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in session cleanup task: {e}")


def start_cleanup_task() -> None:
    """Start the background cleanup task if not already running."""
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_expired_sessions())
        logger.info("Started diagnostic session cleanup task")


def _validate_session(session_id: str, user_id: str | None = None) -> dict[str, Any]:
    """
    Validate and return session data, raising HTTPException if invalid or expired.

    If user_id is provided and the session has an owner, validates ownership.
    Anonymous sessions (no owner) can be accessed by anyone with the session ID.
    """
    if session_id not in diagnostic_sessions:
        raise HTTPException(404, "Session not found")

    data = diagnostic_sessions[session_id]

    if _is_session_expired(data):
        # Clean up the expired session
        _cleanup_session_files(data)
        del diagnostic_sessions[session_id]
        raise HTTPException(410, "Session has expired. Please start a new diagnostic.")

    # Validate ownership if session has an owner
    session_owner = data.get("user_id")
    if session_owner is not None:
        # Session is owned - only owner can access
        if user_id != session_owner:
            logger.warning(
                f"Session access denied: session owner mismatch "
                f"(session={session_id[:8]}...)"
            )
            raise HTTPException(403, "Not authorized to access this session")

    return data


# =============================================================================
# MODELS
# =============================================================================


class AnswerRequest(BaseModel):
    classification: str
    note: str | None = ""


class EmailRequest(BaseModel):
    email: EmailStr
    include_summary: bool = True


class SessionSummary(BaseModel):
    session_id: str
    store_name: str
    status: str
    total_items: int
    negative_items: int
    total_shrinkage: float
    explained_value: float
    unexplained_value: float
    reduction_percent: float
    patterns_total: int
    patterns_answered: int


class QuestionResponse(BaseModel):
    pattern_id: str
    pattern_name: str
    question: str
    suggested_answers: list[list[str]]
    item_count: int
    total_value: float
    sample_items: list[dict]
    progress: dict[str, int]
    running_totals: dict[str, Any]


class StartResponse(BaseModel):
    session_id: str
    store_name: str
    total_items: int
    negative_items: int
    total_shrinkage: float
    patterns_detected: int


# =============================================================================
# COLUMN MAPPING UTILITIES
# =============================================================================

# Flexible column name aliases for different POS systems
SKU_ALIASES = [
    "SKU",
    "sku",
    "Item Number",
    "Item No",
    "Item",
    "UPC",
    "Product Code",
    "Part Number",
    "Article Number",
    "ItemNumber",
    "ItemNo",
    "product_code",
    "item_number",
]

DESCRIPTION_ALIASES = [
    "Description",
    "Description ",  # Common trailing space
    "Name",
    "Item Name",
    "Product Name",
    "Item Description",
    "ProductName",
    "ItemName",
    "description",
    "name",
    "product_name",
]

QUANTITY_ALIASES = [
    "In Stock Qty.",
    "In Stock Qty",
    "Quantity",
    "Qty",
    "On Hand",
    "QOH",
    "Stock",
    "Stock Qty",
    "Inventory",
    "Balance",
    "quantity",
    "qty",
    "on_hand",
    "stock",
]

COST_ALIASES = [
    "Cost",
    "Unit Cost",
    "Avg Cost",
    "Average Cost",
    "UnitCost",
    "AvgCost",
    "cost",
    "unit_cost",
    "avg_cost",
    "Price",
    "Unit Price",
]


def find_column(row: dict, aliases: list) -> str | None:
    """Find a column value using multiple alias options."""
    for alias in aliases:
        if alias in row:
            return row[alias]
    return None


def parse_numeric(value: str) -> float:
    """Parse a numeric value, handling common formats."""
    if not value:
        return 0.0
    # Remove currency symbols, commas, and whitespace
    cleaned = value.replace(",", "").replace("$", "").replace(" ", "").strip()
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/start", response_model=StartResponse)
@limiter.limit("30/hour")
async def start_diagnostic(
    request: Request,
    file: UploadFile = File(...),
    store_name: str = "My Store",
    user_id: str | None = Depends(get_current_user),
):
    """
    Start a new diagnostic session.

    Upload a CSV file with inventory data. Supports flexible column mapping
    for various POS system exports.

    Required columns (with flexible naming):
    - SKU/Item Number/UPC
    - Description/Name
    - Quantity/In Stock Qty/On Hand
    - Cost/Unit Cost/Avg Cost

    Sessions expire after 24 hours (configurable via SESSION_TTL_HOURS).

    Authentication is optional - anonymous users can use the diagnostic,
    but authenticated users get session ownership protection.
    """
    # Start cleanup task if not running
    start_cleanup_task()

    # Import diagnostic engine (lazy load to avoid startup overhead)
    try:
        from sentinel_engine.diagnostic.engine import ConversationalDiagnostic
    except ImportError as e:
        logger.error(f"Failed to import diagnostic engine: {e}")
        raise HTTPException(500, "Diagnostic engine not available. Check installation.")

    # Read and parse CSV
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(StringIO(text))

        items = []
        for row in reader:
            # Flexible column mapping
            sku = find_column(row, SKU_ALIASES) or ""
            desc = find_column(row, DESCRIPTION_ALIASES) or ""
            stock_str = find_column(row, QUANTITY_ALIASES) or "0"
            cost_str = find_column(row, COST_ALIASES) or "0"

            stock = parse_numeric(stock_str)
            cost = parse_numeric(cost_str)

            if sku.strip():  # Only include items with a SKU
                items.append(
                    {
                        "sku": sku.strip(),
                        "description": desc.strip(),
                        "stock": stock,
                        "cost": cost,
                    }
                )

        if not items:
            raise HTTPException(400, "No valid inventory items found in CSV")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

    # Create session
    session_id = str(uuid.uuid4())

    try:
        diagnostic = ConversationalDiagnostic()
        session = diagnostic.start_session(items)
    except Exception as e:
        logger.error(f"Failed to start diagnostic session: {e}")
        raise HTTPException(500, f"Failed to initialize diagnostic: {str(e)}")

    # Store session with optional user ownership
    diagnostic_sessions[session_id] = {
        "diagnostic": diagnostic,
        "session": session,
        "items": items,
        "store_name": store_name,
        "created_at": datetime.now(),
        "status": "in_progress",
        "user_id": user_id,  # None for anonymous, user ID for authenticated
    }

    # Log without exposing user_id (privacy)
    auth_status = "authenticated" if user_id else "anonymous"
    logger.info(
        f"Started diagnostic session {session_id} for {store_name} "
        f"with {len(items)} items, {session.negative_items} negative ({auth_status})"
    )

    return StartResponse(
        session_id=session_id,
        store_name=store_name,
        total_items=len(items),
        negative_items=session.negative_items,
        total_shrinkage=session.total_shrinkage,
        patterns_detected=len(session.patterns),
    )


@router.get("/{session_id}/question", response_model=QuestionResponse | None)
@limiter.limit("100/minute")
async def get_question(
    request: Request,
    session_id: str,
    user_id: str | None = Depends(get_current_user),
):
    """Get the current question for a session."""
    data = _validate_session(session_id, user_id)
    diagnostic = data["diagnostic"]
    question = diagnostic.get_current_question()

    if not question:
        return None

    return QuestionResponse(**question)


@router.post("/{session_id}/answer")
@limiter.limit("100/minute")
async def submit_answer(
    request: Request,
    session_id: str,
    answer: AnswerRequest,
    user_id: str | None = Depends(get_current_user),
):
    """Submit an answer to the current question."""
    data = _validate_session(session_id, user_id)
    diagnostic = data["diagnostic"]
    result = diagnostic.answer_question(answer.classification, answer.note)

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Update status if complete
    if result.get("is_complete"):
        diagnostic_sessions[session_id]["status"] = "complete"
        logger.info(f"Diagnostic session {session_id} completed")

    return result


@router.get("/{session_id}/summary", response_model=SessionSummary)
@limiter.limit("100/minute")
async def get_summary(
    request: Request,
    session_id: str,
    user_id: str | None = Depends(get_current_user),
):
    """Get current session summary with running totals."""
    data = _validate_session(session_id, user_id)
    session = data["session"]
    summary = session.get_summary()

    return SessionSummary(
        session_id=session_id,
        store_name=data["store_name"],
        status=data["status"],
        total_items=session.items_analyzed,
        negative_items=session.negative_items,
        total_shrinkage=summary["total_shrinkage"],
        explained_value=summary["explained_value"],
        unexplained_value=summary["unexplained_value"],
        reduction_percent=summary["reduction_percent"],
        patterns_total=summary["patterns_total"],
        patterns_answered=summary["patterns_answered"],
    )


@router.get("/{session_id}/report")
@limiter.limit("10/minute")
async def generate_report(
    request: Request,
    session_id: str,
    user_id: str | None = Depends(get_current_user),
):
    """Generate and download the PDF report."""
    data = _validate_session(session_id, user_id)

    # Import report generator
    try:
        from sentinel_engine.diagnostic.report import generate_report_from_session
    except ImportError as e:
        logger.error(f"Failed to import report generator: {e}")
        raise HTTPException(500, "Report generator not available")
    diagnostic = data["diagnostic"]

    # Generate report data
    report_data = diagnostic.get_final_report()

    # Create temp file for PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        output_path = f.name

    try:
        generate_report_from_session(
            session_data=report_data,
            all_items=data["items"],
            store_name=data["store_name"],
            output_path=output_path,
        )
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise HTTPException(500, f"Failed to generate report: {str(e)}")

    # Store path for email
    data["report_path"] = output_path

    safe_store_name = data["store_name"].lower().replace(" ", "_")
    filename = f"profit_sentinel_report_{safe_store_name}.pdf"

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=filename,
    )


@router.post("/{session_id}/email")
@limiter.limit("5/minute")
async def email_diagnostic_report(
    request: Request,
    session_id: str,
    email_request: EmailRequest,
    background_tasks: BackgroundTasks,
    user_id: str | None = Depends(get_current_user),
):
    """Email the diagnostic report."""
    data = _validate_session(session_id, user_id)

    # Generate report if not already done
    if "report_path" not in data or not os.path.exists(data.get("report_path", "")):
        try:
            from sentinel_engine.diagnostic.report import generate_report_from_session
        except ImportError as e:
            logger.error(f"Failed to import report generator: {e}")
            raise HTTPException(500, "Report generator not available")

        diagnostic = data["diagnostic"]
        report_data = diagnostic.get_final_report()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        try:
            generate_report_from_session(
                session_data=report_data,
                all_items=data["items"],
                store_name=data["store_name"],
                output_path=output_path,
            )
            data["report_path"] = output_path
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            if os.path.exists(output_path):
                os.unlink(output_path)
            raise HTTPException(500, f"Failed to generate report: {str(e)}")

    # Send email in background using the existing email service
    email_service = get_email_service()

    async def send_diagnostic_email():
        try:
            await email_service.send_diagnostic_report(
                to_email=email_request.email,
                pdf_path=data["report_path"],
                store_name=data["store_name"],
                summary=(
                    data["session"].get_summary()
                    if email_request.include_summary
                    else None
                ),
            )
        except Exception as e:
            logger.error(f"Failed to send diagnostic email: {e}")

    background_tasks.add_task(send_diagnostic_email)

    return {
        "message": f"Report will be sent to {email_request.email}",
        "session_id": session_id,
    }


@router.delete("/{session_id}")
@limiter.limit("50/minute")
async def delete_session(
    request: Request,
    session_id: str,
    user_id: str | None = Depends(get_current_user),
):
    """Delete a diagnostic session."""
    # Validate session exists and user has access
    data = _validate_session(session_id, user_id)

    # Clean up temp files
    diagnostic_sessions.pop(session_id)
    _cleanup_session_files(data)
    logger.info(f"Deleted diagnostic session: {session_id}")

    return {"message": "Session deleted"}


@router.get("/sessions")
@limiter.limit("30/minute")
async def list_sessions(
    request: Request,
    user_id: str | None = Depends(get_current_user),
):
    """
    List active diagnostic sessions.

    If authenticated, returns only the user's sessions.
    If anonymous, returns an empty list (anonymous sessions are private by session ID).
    """
    sessions = []
    for session_id, data in diagnostic_sessions.items():
        # Only show sessions owned by this user (or skip if anonymous)
        session_owner = data.get("user_id")
        if session_owner is None or session_owner != user_id:
            continue

        sessions.append(
            {
                "session_id": session_id,
                "store_name": data["store_name"],
                "status": data["status"],
                "created_at": data["created_at"].isoformat(),
                "total_items": len(data["items"]),
            }
        )
    return {"sessions": sessions, "count": len(sessions)}
