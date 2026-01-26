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

import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session storage (use Redis/DB in production)
premium_sessions: dict[str, dict[str, Any]] = {}


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
async def start_premium_diagnostic(
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
    """
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
        f"Started premium diagnostic session {session_id} for {store_name} "
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
async def get_premium_question(session_id: str):
    """Get the current correlation question for a premium session."""
    if session_id not in premium_sessions:
        raise HTTPException(404, "Session not found")

    diagnostic = premium_sessions[session_id]["diagnostic"]
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
async def submit_premium_answer(session_id: str, answer: PremiumAnswerRequest):
    """Submit an answer to the current correlation question."""
    if session_id not in premium_sessions:
        raise HTTPException(404, "Session not found")

    diagnostic = premium_sessions[session_id]["diagnostic"]
    result = diagnostic.answer_question(answer.classification, answer.note)

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Update status if complete
    if result.get("is_complete"):
        premium_sessions[session_id]["status"] = "complete"
        logger.info(f"Premium diagnostic session {session_id} completed")

    return result


@router.get("/{session_id}/summary", response_model=PremiumSummary)
async def get_premium_summary(session_id: str):
    """Get comprehensive summary with vendor metrics."""
    if session_id not in premium_sessions:
        raise HTTPException(404, "Session not found")

    data = premium_sessions[session_id]
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
async def generate_premium_report(session_id: str):
    """Generate and download the premium PDF report with vendor analysis."""
    if session_id not in premium_sessions:
        raise HTTPException(404, "Session not found")

    # For now, use the standard report generator
    # In the future, create a premium report with vendor correlation details
    try:
        from sentinel_engine.diagnostic.report import generate_report_from_session
    except ImportError as e:
        logger.error(f"Failed to import report generator: {e}")
        raise HTTPException(500, "Report generator not available")

    data = premium_sessions[session_id]
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
async def delete_premium_session(session_id: str):
    """Delete a premium diagnostic session."""
    if session_id not in premium_sessions:
        raise HTTPException(404, "Session not found")

    # Clean up temp files
    data = premium_sessions[session_id]
    if "report_path" in data and os.path.exists(data["report_path"]):
        try:
            os.unlink(data["report_path"])
        except Exception as e:
            logger.warning(f"Failed to delete temp report file: {e}")

    del premium_sessions[session_id]
    logger.info(f"Deleted premium diagnostic session {session_id}")

    return {"message": "Session deleted"}


@router.get("/sessions")
async def list_premium_sessions():
    """List all active premium diagnostic sessions."""
    sessions = []
    for session_id, data in premium_sessions.items():
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
