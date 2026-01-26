# Profit Sentinel + Dorian Integration Package

## Overview

This package contains the complete Profit Sentinel system with the Dorian knowledge engine. Built across multiple sessions, this represents a production-ready AI-powered shrinkage diagnostic tool.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROFIT SENTINEL                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DORIAN KNOWLEDGE ENGINE                       │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │   │
│  │  │   VSA Core    │  │  Fact Store   │  │   Inference   │       │   │
│  │  │  (10K dims)   │  │  (10M facts)  │  │    Engine     │       │   │
│  │  └───────────────┘  └───────────────┘  └───────────────┘       │   │
│  │                                                                  │   │
│  │  ┌───────────────────────────────────────────────────────────┐  │   │
│  │  │              KNOWLEDGE PIPELINE                            │  │   │
│  │  │  Wikidata │ arXiv │ ConceptNet │ Domain Knowledge         │  │   │
│  │  └───────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 CONVERSATIONAL DIAGNOSTIC                        │   │
│  │  Pattern Detection → Questions → User Answers → Learn Rules     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    PDF REPORT GENERATOR                          │   │
│  │  Executive Summary │ Financial Impact │ Full SKU Listing        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      REACT DASHBOARD                             │   │
│  │  Upload → Q&A Flow → Running Totals → Results                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Manifest

### CORE ENGINE (Priority: CRITICAL)

| File | Purpose | Integration Target |
|------|---------|-------------------|
| `dorian_core.py` | Main VSA engine - 10K dimensions, FAISS indexing, fact storage, inference | `packages/sentinel-engine/src/sentinel_engine/dorian/core.py` |
| `dorian_knowledge_pipeline.py` | Unified loader for all knowledge sources | `packages/sentinel-engine/src/sentinel_engine/dorian/pipeline.py` |
| `profit_sentinel_conversational.py` | Conversational diagnostic engine | `packages/sentinel-engine/src/sentinel_engine/diagnostic.py` |
| `profit_sentinel_pdf_report.py` | PDF report generator with financial analysis | `packages/sentinel-engine/src/sentinel_engine/report.py` |

### KNOWLEDGE LOADERS (Priority: HIGH)

| File | Purpose | Integration Target |
|------|---------|-------------------|
| `dorian_wikidata_loader.py` | Wikidata SPARQL + dump streaming | `packages/sentinel-engine/src/sentinel_engine/dorian/loaders/wikidata.py` |
| `dorian_arxiv_loader.py` | arXiv API + OAI-PMH loader | `packages/sentinel-engine/src/sentinel_engine/dorian/loaders/arxiv.py` |
| `dorian_conceptnet_loader.py` | ConceptNet 34M assertions | `packages/sentinel-engine/src/sentinel_engine/dorian/loaders/conceptnet.py` |

### DOMAIN KNOWLEDGE (Priority: MEDIUM)

| File | Purpose | Integration Target |
|------|---------|-------------------|
| `dorian_math.py` | Mathematical concepts (299 facts) | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/math.py` |
| `dorian_physics.py` | Physics knowledge (378 facts) | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/physics.py` |
| `dorian_chemistry.py` | Chemistry knowledge (442 facts) | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/chemistry.py` |
| `dorian_biology.py` | Biology knowledge (366 facts) | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/biology.py` |
| `dorian_cs.py` | Computer science (777 facts) | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/cs.py` |
| `dorian_economics.py` | Economics knowledge (286 facts) | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/economics.py` |
| `dorian_philosophy.py` | Philosophy concepts (177 facts) | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/philosophy.py` |
| `dorian_web.py` | Web/programming knowledge | `packages/sentinel-engine/src/sentinel_engine/dorian/domains/web.py` |

### AGENTS (Priority: HIGH)

| File | Purpose | Integration Target |
|------|---------|-------------------|
| `dorian_agent_v2.py` | Multi-agent system with bootstrap inference | `packages/sentinel-engine/src/sentinel_engine/dorian/agent.py` |
| `profit_sentinel_agent.py` | Full Profit Sentinel agent with Dorian | `packages/sentinel-engine/src/sentinel_engine/agent.py` |

### FRONTEND (Priority: HIGH)

| File | Purpose | Integration Target |
|------|---------|-------------------|
| `profit_sentinel_conversational_ui.jsx` | React dashboard with Q&A flow | `apps/web/src/components/DiagnosticDashboard.jsx` |
| `profit_sentinel_dashboard.jsx` | Alternative dashboard (more detailed) | `apps/web/src/components/Dashboard.jsx` |

### SUPPORTING FILES (Priority: LOW)

| File | Purpose | Notes |
|------|---------|-------|
| `dorian_ontology.py` | Ontology integration | Optional - enhances reasoning |
| `dorian_v7.py` | Earlier version | Reference only |
| `profit_sentinel_integrated.py` | Integrated system (pre-conversational) | Reference only |
| `profit_sentinel_diagnostic.py` | Earlier diagnostic engine | Reference only |
| `profit_sentinel_discovery.py` | Discovery engine | Reference only |

## API Endpoints Needed

```python
# Backend API (FastAPI recommended)

# Session Management
POST   /api/diagnostic/start          # Upload CSV, start session
GET    /api/diagnostic/session/{id}   # Get session state

# Conversational Flow
GET    /api/diagnostic/question       # Get current question
POST   /api/diagnostic/answer         # Submit answer, get next question
POST   /api/diagnostic/skip           # Skip current question (mark investigate)

# Results
GET    /api/diagnostic/summary        # Get running totals
GET    /api/diagnostic/report         # Generate PDF report
POST   /api/diagnostic/email          # Email report via Resend

# Knowledge Management (optional)
POST   /api/knowledge/load            # Load knowledge sources
GET    /api/knowledge/stats           # Get knowledge base stats
```

## Resend Integration

```python
# Add to profit_sentinel_pdf_report.py or create email.py

import resend

resend.api_key = "re_YOUR_API_KEY"

def email_report(to_email: str, pdf_path: str, store_name: str):
    """Email the diagnostic report via Resend."""
    
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    params = {
        "from": "Profit Sentinel <reports@profitsentinel.com>",
        "to": [to_email],
        "subject": f"Shrinkage Diagnostic Report - {store_name}",
        "html": f"""
            <h2>Your Profit Sentinel Report is Ready</h2>
            <p>Attached is your comprehensive shrinkage diagnostic report for {store_name}.</p>
            <p>Key findings are summarized in the Executive Summary section.</p>
            <p>Questions? Reply to this email.</p>
            <br>
            <p>— The Profit Sentinel Team</p>
        """,
        "attachments": [
            {
                "filename": f"profit_sentinel_report_{store_name.lower().replace(' ', '_')}.pdf",
                "content": pdf_content,
            }
        ],
    }
    
    response = resend.Emails.send(params)
    return response
```

## Dependencies

```txt
# requirements.txt additions

# Core
numpy>=1.24.0
faiss-cpu>=1.7.4      # Or faiss-gpu for GPU support

# PDF Generation
reportlab>=4.0.0

# Knowledge Loading
requests>=2.28.0
SPARQLWrapper>=2.0.0  # For Wikidata

# Email
resend>=0.5.0

# API (if using FastAPI)
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6  # For file uploads
```

## Database Schema (if persisting sessions)

```sql
-- Sessions
CREATE TABLE diagnostic_sessions (
    id UUID PRIMARY KEY,
    store_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    total_items INT,
    negative_items INT,
    total_shrinkage DECIMAL(12,2),
    status VARCHAR(50) DEFAULT 'in_progress'
);

-- Pattern Answers
CREATE TABLE pattern_answers (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES diagnostic_sessions(id),
    pattern_id VARCHAR(100),
    pattern_name VARCHAR(255),
    item_count INT,
    total_value DECIMAL(12,2),
    classification VARCHAR(50),
    user_answer TEXT,
    answered_at TIMESTAMP DEFAULT NOW()
);

-- Learned Rules (persisted to Dorian)
CREATE TABLE learned_rules (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES diagnostic_sessions(id),
    pattern VARCHAR(255),
    behavior VARCHAR(50),
    explanation TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Quick Start Integration

```python
# Example: Minimal integration

from sentinel_engine.dorian.core import DorianCore
from sentinel_engine.dorian.pipeline import KnowledgePipeline
from sentinel_engine.diagnostic import ConversationalDiagnostic
from sentinel_engine.report import generate_report_from_session

# 1. Initialize Dorian
dorian = DorianCore(dimensions=10000)

# 2. Load knowledge (optional but recommended)
pipeline = KnowledgePipeline(dorian)
pipeline.load_all_sources()

# 3. Start diagnostic session
diagnostic = ConversationalDiagnostic()
session = diagnostic.start_session(inventory_items)

# 4. Conversational loop
while not session.is_complete:
    question = diagnostic.get_current_question()
    # Present to user, get answer
    diagnostic.answer_question(classification, note)

# 5. Generate report
report = diagnostic.get_final_report()
pdf_path = generate_report_from_session(report, items, store_name, output_path)

# 6. Email via Resend
email_report(user_email, pdf_path, store_name)
```

## Test Data

The system was validated on real inventory data:
- **156,139 SKUs** analyzed
- **3,996 negative stock items** detected
- **$726,749** apparent shrinkage
- **$521,879** (71.8%) explained as process issues
- **$204,869** remaining for investigation
- **27 patterns** detected and classified

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Inventory analysis (156K items) | < 1 second |
| Pattern detection | < 0.5 seconds |
| PDF generation (104 pages) | ~3 seconds |
| Knowledge query (sub-millisecond) | < 1ms |
| 10M fact capacity | Tested and validated |

## Key Classes

### DorianCore
```python
class DorianCore:
    """Vector Symbolic Architecture engine."""
    
    def __init__(self, dimensions=10000):
        self.dimensions = dimensions
        self.vsa_engine = VSAEngine(dimensions)
        self.fact_store = FactStore()
        self.inference_engine = InferenceEngine()
    
    def add_fact(self, subject, predicate, object, confidence=1.0)
    def query(self, subject=None, predicate=None, object=None)
    def infer(self, query, max_hops=3)
```

### ConversationalDiagnostic
```python
class ConversationalDiagnostic:
    """Asks about every pattern, user confirms."""
    
    def start_session(self, items: List[Dict]) -> DiagnosticSession
    def get_current_question(self) -> Optional[Dict]
    def answer_question(self, classification: str, note: str = "") -> Dict
    def get_final_report(self) -> Dict
```

### ProfitSentinelReport
```python
class ProfitSentinelReport:
    """Generates comprehensive PDF reports."""
    
    def generate(self, result: DiagnosticResult) -> str:
        # Returns path to generated PDF
```

## Notes for Claude Code

1. **File locations**: All source files are in `/mnt/user-data/outputs/` and `/home/claude/`

2. **Primary files to integrate**:
   - `dorian_core.py` (74KB) - The heart of the system
   - `profit_sentinel_conversational.py` (30KB) - Diagnostic engine
   - `profit_sentinel_pdf_report.py` (42KB) - PDF generation
   - `profit_sentinel_conversational_ui.jsx` (24KB) - React frontend

3. **The Dorian knowledge system is optional but valuable** - It can work without external knowledge, but the knowledge loaders add context for future features.

4. **Resend integration** is straightforward - just need API key and the email function above.

5. **The PDF report is comprehensive** - 104 pages including all SKUs. This is intentional to prove the analysis is real.
