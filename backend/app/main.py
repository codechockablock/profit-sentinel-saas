# backend/app/main.py
from fastapi import FastAPI, Form, Header, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from typing import List, Dict
import pandas as pd
import io
import boto3
import os
import uuid
import json
from openai import OpenAI

app = FastAPI(
    title="Profit Sentinel",
    description="Forensic analysis of POS exports to detect hidden profit leaks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://profitsentinel.com",
        "https://www.profitsentinel.com",
        "https://profit-sentinel-saas.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

S3_CLIENT = boto3.client('s3')
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://PLACEHOLDER_SUPABASE_PROJECT.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_SERVICE_KEY else None

STANDARD_FIELDS = {
    "date": ["date", "transaction_date", "sale_date", "timestamp"],
    "sku": ["sku", "product_id", "item_id", "upc"],
    "quantity": ["qty", "quantity", "units_sold", "qty_sold"],
    "revenue": ["sale_price", "price", "total_sale", "revenue", "ext_price", "line_total", "amount"],
    "cost": ["cost", "cogs", "cost_price", "unit_cost", "avg_cost"],
    "vendor": ["vendor", "supplier", "vendor_name"],
    "category": ["category", "department", "product_type"],
    "transaction_id": ["transaction_id", "order_id", "invoice"],
    "customer_id": ["customer", "client_id"],
    "discount": ["discount", "promo"],
    "tax": ["tax", "sales_tax"],
    "return_flag": ["return", "refund", "is_return"]
}

async def get_current_user(authorization: str | None = Header(None)):
    if not authorization or not supabase:
        return None
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        return user.user.id if user else None
    except Exception as e:
        print(f"JWT verification failed: {e}")
        return None

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
        return pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False, encoding='latin1', low_memory=False)
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
    return {"mapping": mapping, "confidence": confidence, "notes": "Heuristic keyword matching"}

def suggest_column_mapping(df: pd.DataFrame, filename: str) -> Dict:
    columns = list(df.columns)
    sample = df.head(10).to_dict(orient='records')

    api_key = os.getenv("GROK_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        prompt = f"""
You are Profit Sentinel's expert column mapper for messy POS/ERP exports.

Task: Map uploaded columns to our standard fields using column names AND sample values for context.

Uploaded file: {filename}
Columns: {columns}
First 10 rows sample:
{json.dumps(sample, indent=2)}

Standard fields (preferred name first, with common aliases):
{json.dumps(STANDARD_FIELDS, indent=2)}

Rules:
- Use sample values to disambiguate (e.g., dates → date, $-prefixed numbers → revenue/cost, negative quantities → return_flag).
- Common tricks: "Ext Price"/"Line Total"/"Amount" → revenue, "Avg Cost" → cost.
- Only map if reasonably confident (>0.6 internally).
- Use EXACT standard field name or null.

Return ONLY valid JSON:
{{
  "mapping": {{"Uploaded Column Name": "standard_field" or null}},
  "confidence": {{"Uploaded Column Name": 0.0-1.0}},
  "notes": "Brief explanation of guesses or unmapped columns"
}}
"""
        try:
            response = client.chat.completions.create(
                model="grok-beta",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            content = response.choices[0].message.content.strip()
            suggestions = json.loads(content)
        except Exception as e:
            print(f"Grok mapping failed: {e}")
            suggestions = heuristic_mapping(columns)
    else:
        suggestions = heuristic_mapping(columns)
        suggestions["notes"] = "No GROK_API_KEY set – used heuristic fallback"

    # Ensure confidence dict exists
    if "confidence" not in suggestions:
        suggestions["confidence"] = {col: 1.0 if suggestions["mapping"][col] else 0.0 for col in columns}

    return {
        "original_columns": columns,
        "sample_data": sample,
        "suggestions": suggestions["mapping"],
        "confidences": suggestions["confidence"],
        "notes": suggestions.get("notes", "")
    }

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/presign")
async def presign(
    filenames: List[str] = Form(...),
    email: str = Form(None),
    user_id: str | None = Depends(get_current_user)
):
    presigned = []
    for filename in filenames:
        if not filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(400, f"Invalid file type: {filename}")

        key = f"uploads/{uuid.uuid4()}_{filename}"
        url = S3_CLIENT.generate_presigned_post(
            Bucket=BUCKET_NAME,
            Key=key,
            ExpiresIn=3600
        )
        presigned.append({"url": url, "key": key, "filename": filename})

    print(f"Presigned URLs for {len(filenames)} files by user {user_id or 'guest'}")
    return {"presigned": presigned}

@app.post("/suggest-mapping")
async def suggest_mapping(
    keys: List[str] = Form(...),
    user_id: str | None = Depends(get_current_user)
):
    results = []
    for key in keys:
        try:
            df = load_sample_df(key)
            mapping_result = suggest_column_mapping(df, os.path.basename(key))
            results.append({
                "key": key,
                "filename": os.path.basename(key),
                "rows_sampled": len(df),
                **mapping_result
            })
        except Exception as e:
            print(f"Suggest mapping error for {key}: {e}")
            results.append({"key": key, "error": str(e)})
    return {"results": results}

@app.post("/analyze")
async def analyze(
    keys: List[str] = Form(...),
    confirmed_mappings: str = Form(None),  # JSON: {"s3_key": {"Original Col": "standard_field", ...}}
    email: str = Form(None),
    user_id: str | None = Depends(get_current_user)
):
    mappings_dict = {}
    if confirmed_mappings:
        try:
            mappings_dict = json.loads(confirmed_mappings)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in confirmed_mappings")

    results = []
    for key in keys:
        file_result = {"key": key}
        try:
            df = load_full_df(key)

            if df.empty:
                file_result.update({"status": "error", "detail": "File is empty"})
                results.append(file_result)
                continue

            # Apply user-confirmed mapping if provided
            key_mapping = mappings_dict.get(key, {})
            if key_mapping:
                rename_dict = {orig: std for orig, std in key_mapping.items() if std}
                df = df.rename(columns=rename_dict)
                mapped_with = "user_confirmed"
            else:
                # Fallback heuristic (improved with STANDARD_FIELDS)
                col_lower = {col: col.strip().lower().replace('$', '').replace('.', '').replace(' ', '') for col in df.columns}
                quantity_col = next((orig for orig, clean in col_lower.items() if any(w in clean for w in ['qty', 'quantity', 'units', 'sold', 'onhand', 'diff'])), None)
                cost_col = next((orig for orig, clean in col_lower.items() if any(w in clean for w in ['cost', 'cogs', 'avgcost', 'unitcost', 'stdcost'])), None)
                revenue_col = next((orig for orig, clean in col_lower.items() if any(w in clean for w in ['revenue', 'sale', 'price', 'total', 'amt', 'ext', 'line'])), None)

                if not all([quantity_col, cost_col, revenue_col]):
                    file_result.update({
                        "status": "partial_failure",
                        "detail": "Heuristic could not map required columns (quantity, cost, revenue)"
                    })
                    results.append(file_result)
                    continue

                df = df.rename(columns={
                    quantity_col: "quantity",
                    cost_col: "cost",
                    revenue_col: "revenue"
                })
                mapped_with = "heuristic"

            # Required columns check
            required = ["quantity", "cost", "revenue"]
            missing = [r for r in required if r not in df.columns]
            if missing:
                file_result.update({
                    "status": "partial_failure",
                    "detail": f"Missing required columns after mapping: {missing}"
                })
                results.append(file_result)
                continue

            # Numeric conversion
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(0)
            df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0)

            # Core negative inventory leak analysis
            negative = df[df['quantity'] < 0].copy()
            negative['abs_qty'] = negative['quantity'].abs()
            negative['hidden_cogs'] = negative['abs_qty'] * negative['cost']
            hidden_cogs_total = negative['hidden_cogs'].sum()

            total_revenue = df['revenue'].sum()
            reported_cogs = max(0, (df['quantity'] * df['cost']).sum())
            true_cogs = reported_cogs + hidden_cogs_total
            reported_margin = (total_revenue - reported_cogs) / total_revenue if total_revenue else 0
            true_margin = (total_revenue - true_cogs) / total_revenue if total_revenue else 0

            margin_impact = (reported_margin - true_margin) * 100

            file_result.update({
                "filename": os.path.basename(key),
                "rows": len(df),
                "mapped_with": mapped_with,
                "status": "success",
                "negative_items": int(len(negative)),
                "estimated_hidden_cogs": round(float(hidden_cogs_total), 2),
                "reported_margin_pct": round(reported_margin * 100, 2),
                "true_margin_pct": round(true_margin * 100, 2),
                "margin_impact_pct": round(margin_impact, 2),
                "summary": f"Found {len(negative)} items with negative inventory, hiding an estimated ${hidden_cogs_total:,.0f} in unrecorded COGS.",
                "recommendation": "Review receiving processes and inventory adjustments—negative quantities often indicate skipped steps."
            })

            if email:
                file_result["report_sent_to"] = email

            file_result["s3_status"] = "already_saved"

        except Exception as e:
            print(f"Analysis error for {key}: {e}")
            file_result.update({"status": "error", "detail": str(e)})

        results.append(file_result)

    print(f"Analysis complete for {len(keys)} files by user {user_id or 'guest'}")
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)