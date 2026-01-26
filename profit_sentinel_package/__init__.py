"""
Profit Sentinel
===============

AI-powered shrinkage diagnostic system for retail inventory.

Built on the Dorian Vector Symbolic Architecture knowledge engine,
Profit Sentinel identifies process issues in inventory data through
an interactive diagnostic conversation.

Quick Start:
    from profit_sentinel import run_diagnostic

    results = run_diagnostic(
        csv_path="inventory.csv",
        store_name="My Hardware Store"
    )

Components:
    - dorian: Vector Symbolic Architecture knowledge engine
    - loaders: Knowledge loaders (Wikidata, arXiv, ConceptNet)
    - domains: Pre-built domain knowledge
    - diagnostic: Conversational diagnostic engine
    - api: FastAPI backend

Author: Joseph
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Joseph"


def run_diagnostic(csv_path: str, store_name: str = "My Store") -> dict:
    """
    Run a complete diagnostic on an inventory file.

    This is a convenience function that runs the full diagnostic
    with default answers (typical behaviors). For interactive
    diagnostics, use the ConversationalDiagnostic class directly.

    Args:
        csv_path: Path to inventory CSV file
        store_name: Name of the store

    Returns:
        Diagnostic results including shrinkage analysis
    """
    import csv

    from diagnostic.engine import ConversationalDiagnostic

    # Load inventory
    items = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                stock = float(row.get("In Stock Qty.", "0").replace(",", "") or 0)
                cost = float(
                    row.get("Cost", "0").replace(",", "").replace("$", "") or 0
                )
                items.append(
                    {
                        "sku": row.get("SKU", "").strip(),
                        "description": row.get(
                            "Description", row.get("Description ", "")
                        ).strip(),
                        "stock": stock,
                        "cost": cost,
                    }
                )
            except:
                pass

    # Run diagnostic
    diagnostic = ConversationalDiagnostic()
    session = diagnostic.start_session(items)

    # Auto-answer with typical behaviors
    from diagnostic.engine import DETECTION_PATTERNS

    while not session.is_complete:
        pattern = session.current_pattern
        if pattern:
            config = DETECTION_PATTERNS.get(pattern.pattern_id, {})
            typical = config.get("typical_behavior", "investigate")
            diagnostic.answer_question(typical)

    return diagnostic.get_final_report()
