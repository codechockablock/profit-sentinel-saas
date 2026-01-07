# backend/app/main.py
from fastapi import FastAPI, Form, Header, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from typing import List
import pandas as pd
import io
import boto3
import os
import uuid

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

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Generate presigned POST URLs for direct S3 upload
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

# Analyze files from S3 keys
@app.post("/analyze")
async def analyze(
    keys: List[str] = Form(...),
    email: str = Form(None),
    user_id: str | None = Depends(get_current_user)
):
    results = []
    for key in keys:
        file_result = {"key": key}
        try:
            obj = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=key)
            contents = obj['Body'].read()

            if key.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False, encoding='latin1')
            else:
                df = pd.read_excel(io.BytesIO(contents), dtype=str)

            if df.empty:
                file_result.update({"status": "error", "detail": "File is empty"})
                results.append(file_result)
                continue

            col_lower = {col: col.strip().lower().replace('$', '').replace('.', '').replace(' ', '') for col in df.columns}

            quantity_col = next((orig for orig, clean in col_lower.items() if 'qty' in clean or 'quantity' in clean or 'onhand' in clean or 'diff' in clean), None)
            cost_col = next((orig for orig, clean in col_lower.items() if 'cost' in clean or 'avgcost' in clean or 'stdcost' in clean), None)
            sales_col = next((orig for orig, clean in col_lower.items() if 'sales' in clean or 'sold' in clean), None)

            file_result.update({
                "filename": os.path.basename(key),
                "rows": len(df),
                "mapped": {"quantity": quantity_col, "cost": cost_col, "sales": sales_col},
            })

            if not all([quantity_col, cost_col, sales_col]):
                file_result.update({"status": "partial_failure", "detail": "Could not map all required columns"})
                results.append(file_result)
                continue

            df = df.rename(columns={quantity_col: "quantity", cost_col: "cost", sales_col: "sales"})

            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(0)
            df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)

            negative = df[df['quantity'] < 0].copy()
            negative['abs_qty'] = negative['quantity'].abs()
            negative['hidden_cogs'] = negative['abs_qty'] * negative['cost']
            hidden_cogs_total = negative['hidden_cogs'].sum()

            total_sales = df['sales'].sum()
            reported_cogs = max(0, (df['quantity'] * df['cost']).sum())
            true_cogs = reported_cogs + hidden_cogs_total
            reported_margin = (total_sales - reported_cogs) / total_sales if total_sales else 0
            true_margin = (total_sales - true_cogs) / total_sales if total_sales else 0

            margin_impact = (reported_margin - true_margin) * 100

            file_result.update({
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