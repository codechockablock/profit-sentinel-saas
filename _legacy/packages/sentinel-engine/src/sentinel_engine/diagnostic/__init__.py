"""
Profit Sentinel Diagnostic Engine

Conversational shrinkage diagnostic system that:
1. Detects patterns in negative inventory
2. Asks the user about each pattern
3. Learns rules from user responses
4. Generates comprehensive PDF reports
5. Emails reports via Resend

Core Components:
- ConversationalDiagnostic: Interactive Q&A engine
- ProfitSentinelReport: PDF report generator
- email_report: Resend integration for delivery

Example:
    from diagnostic.engine import ConversationalDiagnostic
    from diagnostic.report import generate_report_from_session
    from diagnostic.email import email_report

    # Run diagnostic
    diag = ConversationalDiagnostic()
    session = diag.start_session(inventory_items)

    while not session.is_complete:
        q = diag.get_current_question()
        # ... present to user, get answer
        diag.answer_question(classification)

    # Generate and email report
    report = diag.get_final_report()
    pdf_path = generate_report_from_session(report, items, "Store Name", "report.pdf")
    email_report("user@example.com", pdf_path, "Store Name")
"""

from .engine import ConversationalDiagnostic, DetectedPattern, DiagnosticSession
from .multi_file import (
    CorrelationPattern,
    CorrelationType,
    MultiFileDiagnostic,
    VendorSummary,
    create_multi_file_diagnostic,
)
from .report import DiagnosticResult, ProfitSentinelReport, generate_report_from_session

__all__ = [
    # Core diagnostic
    "ConversationalDiagnostic",
    "DiagnosticSession",
    "DetectedPattern",
    "ProfitSentinelReport",
    "generate_report_from_session",
    "DiagnosticResult",
    # Premium: Multi-file vendor correlation
    "MultiFileDiagnostic",
    "CorrelationPattern",
    "CorrelationType",
    "VendorSummary",
    "create_multi_file_diagnostic",
]
