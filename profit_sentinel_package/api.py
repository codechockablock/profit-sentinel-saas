"""
Profit Sentinel API

FastAPI backend for the diagnostic system.

Endpoints:
    POST /api/diagnostic/start     - Upload CSV, start session
    GET  /api/diagnostic/question  - Get current question
    POST /api/diagnostic/answer    - Submit answer
    GET  /api/diagnostic/summary   - Get running totals
    GET  /api/diagnostic/report    - Generate PDF
    POST /api/diagnostic/email     - Email report

Run:
    uvicorn api:app --reload
"""

import csv
import os
import tempfile
import uuid
from datetime import datetime
from io import StringIO
from typing import Any, Dict, List, Optional

from diagnostic.email import email_report, email_report_summary

# Import our modules
from diagnostic.engine import ConversationalDiagnostic, DiagnosticSession
from diagnostic.report import generate_report_from_session
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr

app = FastAPI(
    title="Profit Sentinel API",
    description="AI-powered shrinkage diagnostic system",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis/DB in production)
sessions: dict[str, dict[str, Any]] = {}


# =============================================================================
# MODELS
# =============================================================================


class StartSessionRequest(BaseModel):
    store_name: str


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


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.post("/api/diagnostic/start")
async def start_diagnostic(file: UploadFile = File(...), store_name: str = "My Store"):
    """
    Start a new diagnostic session.

    Upload a CSV file with inventory data. Required columns:
    - SKU or Item Number
    - Description
    - In Stock Qty or Quantity
    - Cost or Unit Cost
    """
    # Read and parse CSV
    try:
        content = await file.read()
        text = content.decode("utf-8", errors="replace")
        reader = csv.DictReader(StringIO(text))

        items = []
        for row in reader:
            # Flexible column mapping
            sku = (
                row.get("SKU")
                or row.get("Item Number")
                or row.get("Item")
                or row.get("UPC")
                or ""
            ).strip()

            desc = (
                row.get("Description")
                or row.get("Description ")
                or row.get("Name")
                or row.get("Item Name")
                or ""
            ).strip()

            stock_str = (
                row.get("In Stock Qty.")
                or row.get("In Stock Qty")
                or row.get("Quantity")
                or row.get("Qty")
                or row.get("On Hand")
                or "0"
            )
            stock_str = stock_str.replace(",", "").replace("$", "").strip() or "0"

            cost_str = (
                row.get("Cost") or row.get("Unit Cost") or row.get("Avg Cost") or "0"
            )
            cost_str = cost_str.replace(",", "").replace("$", "").strip() or "0"

            try:
                stock = float(stock_str)
                cost = float(cost_str)
                items.append(
                    {
                        "sku": sku,
                        "description": desc,
                        "stock": stock,
                        "cost": cost,
                    }
                )
            except ValueError:
                continue

        if not items:
            raise HTTPException(400, "No valid inventory items found in CSV")

    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

    # Create session
    session_id = str(uuid.uuid4())

    diagnostic = ConversationalDiagnostic()
    session = diagnostic.start_session(items)

    # Store session
    sessions[session_id] = {
        "diagnostic": diagnostic,
        "session": session,
        "items": items,
        "store_name": store_name,
        "created_at": datetime.now(),
        "status": "in_progress",
    }

    return {
        "session_id": session_id,
        "store_name": store_name,
        "total_items": len(items),
        "negative_items": session.negative_items,
        "total_shrinkage": session.total_shrinkage,
        "patterns_detected": len(session.patterns),
    }


@app.get("/api/diagnostic/{session_id}/question")
async def get_question(session_id: str) -> QuestionResponse | None:
    """Get the current question for a session."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    diagnostic = sessions[session_id]["diagnostic"]
    question = diagnostic.get_current_question()

    if not question:
        return None

    return QuestionResponse(**question)


@app.post("/api/diagnostic/{session_id}/answer")
async def submit_answer(session_id: str, answer: AnswerRequest):
    """Submit an answer to the current question."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    diagnostic = sessions[session_id]["diagnostic"]
    result = diagnostic.answer_question(answer.classification, answer.note)

    if "error" in result:
        raise HTTPException(400, result["error"])

    # Update status if complete
    if result.get("is_complete"):
        sessions[session_id]["status"] = "complete"

    return result


@app.get("/api/diagnostic/{session_id}/summary")
async def get_summary(session_id: str) -> SessionSummary:
    """Get current session summary with running totals."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    data = sessions[session_id]
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


@app.get("/api/diagnostic/{session_id}/report")
async def generate_report(session_id: str):
    """Generate and download the PDF report."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    data = sessions[session_id]
    diagnostic = data["diagnostic"]

    # Generate report
    report_data = diagnostic.get_final_report()

    # Create temp file for PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        output_path = f.name

    generate_report_from_session(
        session_data=report_data,
        all_items=data["items"],
        store_name=data["store_name"],
        output_path=output_path,
    )

    # Store path for email
    data["report_path"] = output_path

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename=f"profit_sentinel_report_{data['store_name'].lower().replace(' ', '_')}.pdf",
    )


@app.post("/api/diagnostic/{session_id}/email")
async def email_diagnostic_report(
    session_id: str, request: EmailRequest, background_tasks: BackgroundTasks
):
    """Email the diagnostic report."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    data = sessions[session_id]

    # Generate report if not already done
    if "report_path" not in data or not os.path.exists(data.get("report_path", "")):
        diagnostic = data["diagnostic"]
        report_data = diagnostic.get_final_report()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            output_path = f.name

        generate_report_from_session(
            session_data=report_data,
            all_items=data["items"],
            store_name=data["store_name"],
            output_path=output_path,
        )
        data["report_path"] = output_path

    # Send email in background
    def send_email():
        try:
            email_report(
                to_email=request.email,
                pdf_path=data["report_path"],
                store_name=data["store_name"],
            )
        except Exception as e:
            print(f"Email failed: {e}")

    background_tasks.add_task(send_email)

    return {
        "message": f"Report will be sent to {request.email}",
        "session_id": session_id,
    }


@app.delete("/api/diagnostic/{session_id}")
async def delete_session(session_id: str):
    """Delete a diagnostic session."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    # Clean up temp files
    data = sessions[session_id]
    if "report_path" in data and os.path.exists(data["report_path"]):
        os.unlink(data["report_path"])

    del sessions[session_id]

    return {"message": "Session deleted"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(sessions),
        "version": "1.0.0",
    }


# =============================================================================
# OPTIONAL: DORIAN KNOWLEDGE ENDPOINTS
# =============================================================================

# These are optional - only include if you want to expose knowledge management

dorian_instance = None


@app.post("/api/knowledge/init")
async def init_knowledge():
    """Initialize the Dorian knowledge engine."""
    global dorian_instance

    from dorian.core import DorianCore
    from dorian.pipeline import KnowledgePipeline

    dorian_instance = DorianCore(dimensions=10000)
    KnowledgePipeline(dorian_instance)

    # Load domain knowledge
    from domains import load_all_domains

    domain_count = load_all_domains(dorian_instance)

    return {
        "message": "Knowledge engine initialized",
        "facts_loaded": domain_count,
    }


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    """Get knowledge base statistics."""
    if not dorian_instance:
        raise HTTPException(400, "Knowledge engine not initialized")

    return {
        "total_facts": len(dorian_instance.fact_store.facts)
        if hasattr(dorian_instance, "fact_store")
        else 0,
        "dimensions": dorian_instance.dimensions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
