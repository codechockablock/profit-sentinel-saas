from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import io
import boto3
from botocore.exceptions import ClientError
import os

app = FastAPI(title="Profit Sentinel")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all for dev/beta — change to your Vercel URL later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

S3_CLIENT = boto3.client('s3') if os.getenv("AWS_DEFAULT_REGION") else None
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀 Uncover hidden profit leaks in your POS data."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type—CSV or Excel only")

    contents = await file.read()

    try:
        if file.filename.lower().endswith('.csv'):
            # Try utf-8 first, fallback to latin1 (common for POS CSVs)
            try:
                df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False, encoding='latin1')
        else:
            # Excel is binary — safe
            df = pd.read_excel(io.BytesIO(contents), dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")
    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty")

    # Ultra-aggressive column matching (great!)
    col_lower = {col: col.strip().lower().replace('$', '').replace('.', '').replace(' ', '') for col in df.columns}

    quantity_col = next((orig for orig, clean in col_lower.items() if 'qty' in clean or 'quantity' in clean or 'onhand' in clean or 'diff' in clean), None)
    cost_col = next((orig for orig, clean in col_lower.items() if 'cost' in clean or 'avgcost' in clean or 'stdcost' in clean), None)
    sales_col = next((orig for orig, clean in col_lower.items() if 'sales' in clean or 'sold' in clean), None)

    result = {
        "filename": file.filename,
        "rows": len(df),
        "found_columns": list(df.columns),
        "mapped": {"quantity": quantity_col, "cost": cost_col, "sales": sales_col},
    }

    if not all([quantity_col, cost_col, sales_col]):
        result.update({
            "status": "partial_failure",
            "detail": "Could not map all required columns—analysis skipped"
        })
        return JSONResponse(content=result)

    # Rename and analyze only if mapped
    df = df.rename(columns={
        quantity_col: "quantity",
        cost_col: "cost",
        sales_col: "sales"
    })

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

    result.update({
        "status": "success",
        "negative_items": int(len(negative)),
        "estimated_hidden_cogs": round(float(hidden_cogs_total), 2),
        "reported_margin_pct": round(reported_margin * 100, 2),
        "true_margin_pct": round(true_margin * 100, 2),
        "margin_impact_pct": round((reported_margin - true_margin) * 100, 2),
    })

    if S3_CLIENT:
        try:
            S3_CLIENT.put_object(Bucket=BUCKET_NAME, Key=f"uploads/{file.filename}", Body=contents)
            result["s3_status"] = "saved"
        except Exception as e:  # Better to catch and log
            result["s3_status"] = f"save failed: {str(e)}"

    return JSONResponse(content=result)