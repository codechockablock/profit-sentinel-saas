import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Force app folder on path for imports

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Form, Header, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from typing import List, Dict
import pandas as pd
import io
import boto3
import os
import uuid
import json
from openai import OpenAI  # Compatible with Grok API

# Private resonator — local only, .gitignore'd
from sentinel_engine import bundle_pos_facts, query_bundle  # In backend/app/sentinel_engine.py

# ------------------- FastAPI App Setup -------------------
app = FastAPI(
    title="Profit Sentinel",
    description="Forensic analysis of POS exports to detect hidden profit leaks",
    version="1.0.0",
    openapi_tags=[
        {"name": "uploads", "description": "File upload and mapping"},
        {"name": "analysis", "description": "Profit leak analysis"}
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://profitsentinel.com",
        "https://www.profitsentinel.com",
        "https://profit-sentinel-saas.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Clients & Config -------------------
S3_CLIENT = boto3.client('s3')
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_SERVICE_KEY else None

STANDARD_FIELDS = {
    "date": ["date", "transaction_date", "sale_date", "timestamp", "posted_date"],
    "sku": ["sku", "product_id", "item_id", "upc", "barcode", "plu"],
    "quantity": ["qty", "quantity", "units_sold", "qty_sold", "units", "on_hand"],
    "revenue": ["sale_price", "price", "total_sale", "revenue", "ext_price", "line_total", "amount", "gross_sales"],
    "cost": ["cost", "cogs", "cost_price", "unit_cost", "avg_cost", "standard_cost"],
    "vendor": ["vendor", "supplier", "vendor_name", "manufacturer"],
    "category": ["category", "department", "product_type", "class", "group"],
    "transaction_id": ["transaction_id", "order_id", "invoice", "receipt_id"],
    "customer_id": ["customer", "client_id", "member_id"],
    "discount": ["discount", "promo", "coupon", "markdown"],
    "tax": ["tax", "sales_tax", "vat"],
    "return_flag": ["return", "refund", "is_return", "negative_qty"]
}

# Grok client
GROK_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
grok_client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1") if GROK_API_KEY else None

# ------------------- Auth Helper -------------------
async def get_current_user(authorization: str | None = Header(None)):
    if not authorization or not supabase:
        return None
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        return user.user.id if user else None
    except Exception:
        return None

# ------------------- Utility Functions -------------------
def load_sample_df(key: str) -> pd.DataFrame:
    obj = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=key)
    contents = obj['Body'].read()
    if key.lower().endswith('.csv'):
        return pd.read_csv(io.BytesIO(contents), nrows=50, dtype=str, keep_default_na=False, encoding='latin1')
    else:
        return pd.read_excel(io.BytesIO(contents), nrows=50, dtype=str)

def load_full_df(key: str) -> pd.DataFrame:
    obj = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=key)
    contents = obj['Body'].read()
    if key.lower().endswith('.csv'):
        try:
            return pd.read_csv(io.BytesIO(contents), dtype=str)
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(contents), dtype=str, encoding='latin1', errors='replace')
    else:
        return pd.read_excel(io.BytesIO(contents), dtype=str)
def heuristic_mapping(columns: List[str]) -> Dict:
    mapping = {}
    confidence = {}
    for col in columns:
        clean = col.strip().lower().replace(' ', '').replace('$', '').replace('.', '')
        matched = False
        for std, examples in STANDARD_FIELDS.items():
            if any(ex.replace(' ', '') in clean for ex in examples):
                mapping[col] = std
                confidence[col] = 0.8
                matched = True
                break
        if not matched:
            mapping[col] = None
            confidence[col] = 0.0
    return {"mapping": mapping, "confidence": confidence, "notes": "Heuristic keyword match"}

# ------------------- Core Mapping Logic -------------------
def suggest_column_mapping(df: pd.DataFrame, filename: str) -> Dict:
    columns = list(df.columns)
    sample = df.head(10).to_dict(orient='records')
    
    if grok_client:
        prompt = f"""
You are Profit Sentinel's expert semi-agentic column mapper for messy POS/ERP exports.
Task: Suggest mapping from uploaded columns to our standard fields using BOTH column names AND sample values.

Uploaded file: {filename}
Columns: {columns}
First 10 rows sample:
{json.dumps(sample, indent=2)}

Standard fields (preferred first, with common aliases):
{json.dumps(STANDARD_FIELDS, indent=2)}

Rules:
- Use sample values to disambiguate (e.g., dates → date, $-numbers → revenue/cost, negative qty → return_flag).
- Common tricks: "Ext Price"/"Line Total"/"Amount" → revenue, "Avg Cost" → cost.
- Only map if confident (>0.6 internally).
- Use EXACT standard field name or null.

Return ONLY valid JSON:
{{
  "mapping": {{"Uploaded Column Name": "standard_field" or null}},
  "confidence": {{"Uploaded Column Name": 0.0-1.0}},
  "notes": "Brief explanation of guesses/unmapped"
}}
"""
        try:
            response = grok_client.chat.completions.create(
                model="grok-3",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            suggestions = json.loads(content)
        except Exception as e:
            print(f"Grok mapping failed: {e}")
            suggestions = heuristic_mapping(columns)
            suggestions["notes"] = f"Grok failed ({str(e)}) – heuristic fallback"
    else:
        suggestions = heuristic_mapping(columns)
        suggestions["notes"] = "No GROK_API_KEY – heuristic fallback"
    
    if "confidence" not in suggestions:
        suggestions["confidence"] = {col: 1.0 if suggestions["mapping"].get(col) else 0.0 for col in columns}
    
    return {
        "original_columns": columns,
        "sample_data": sample,
        "suggestions": suggestions["mapping"],
        "confidences": suggestions["confidence"],
        "notes": suggestions.get("notes", "")
    }

# ------------------- Endpoints -------------------

@app.get("/", tags=["health"])
async def root():
    return {"message": "Profit Sentinel backend is running"}

@app.get("/health", tags=["health"])
async def health():
    return {"status": "healthy"}

@app.post("/presign", tags=["uploads"])
async def presign_upload(
    filenames: List[str] = Form(...),
    user_id: str | None = Depends(get_current_user)
):
    presigned_urls = []
    for filename in filenames:
        key = f"{user_id or 'anonymous'}/{uuid.uuid4()}-{filename}"
        url = S3_CLIENT.generate_presigned_url(
            'put_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': key,
                # REMOVED 'ContentType': 'application/octet-stream'
                # REMOVED 'ACL': 'private' (private is default; ACLs are deprecated)
            },
            ExpiresIn=3600
        )
        presigned_urls.append({"filename": filename, "key": key, "url": url})
    return {"presigned_urls": presigned_urls}

@app.post("/suggest-mapping", tags=["uploads"])
async def suggest_mapping_endpoint(
    key: str = Form(...),
    filename: str = Form(...),
    user_id: str | None = Depends(get_current_user)
):
    try:
        df = load_sample_df(key)
        result = suggest_column_mapping(df, filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Column mapping failed: {str(e)}")

@app.post("/analyze", tags=["analysis"])
async def analyze_upload(
    key: str = Form(...),
    mapping: str = Form(...),  # Change to str — curl/Swagger sends as string
    user_id: str | None = Depends(get_current_user)
):
    try:
        # Parse mapping string to dict
        mapping_dict = json.loads(mapping)
        
        # Load full CSV/Excel
        df = load_full_df(key)
        
        # Apply confirmed mapping
        df = df.rename(columns={k: v for k, v in mapping_dict.items() if v})
        
        # Convert to rows for resonator
        rows = df.to_dict(orient='records')
        
        # Bundle facts with sentinel resonator (private)
        bundle = bundle_pos_facts(rows)
        
        # MVP predefined leak queries
        leaks = {}
        for primitive in ["low_stock", "high_margin_leak", "dead_item", "negative_inventory"]:
            items, scores = query_bundle(bundle, primitive)
            leaks[primitive] = {"top_items": items[:20], "scores": [float(s) for s in scores[:20]]}
        
        return {"status": "success", "leaks": leaks}
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid mapping JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")