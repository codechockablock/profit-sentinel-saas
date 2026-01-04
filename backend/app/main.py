from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import io
import boto3
from botocore.exceptions import ClientError
import os
from typing import Dict, List

app = FastAPI(title="Profit Sentinel")

# AWS S3 config (production); fallback for local dev
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

    # Read file into bytes
    contents = await file.read()

    # Parse with pandas
    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty")

    # Normalize column names for flexible matching
    df.columns = df.columns.str.strip().str.lower()

    # Flexible column mapping
    column_map = {
        'quantity': next((c for c in df.columns if 'qty' in c), None),
        'cost': next((c for c in df.columns if 'cost' in c), None),
        'sales': next((c for c in df.columns if 'sales' in c or 'sold' in c), None),
        'category': next((c for c in df.columns if 'cat' in c or 'category' in c or 'dept' in c), None),
        'vendor': next((c for c in df.columns if 'vendor' in c or 'supplier' in c), None),
    }

    missing = [k for k, v in column_map.items() if v is None and k in ['quantity', 'cost', 'sales']]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns for analysis: {missing}. Found columns: {list(df.columns)}"
        )

    # Rename mapped columns for consistent analysis
    rename_dict = {v: k for k, v in column_map.items() if v is not None}
    df = df.rename(columns=rename_dict)

    # Core Leak Analysis
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    negative_qty = df[df['quantity'] < 0].copy()
    negative_qty['abs_quantity'] = negative_qty['quantity'].abs()

    df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(0)
    negative_qty['hidden_cogs'] = negative_qty['abs_quantity'] * negative_qty['cost']
    estimated_hidden_cogs = negative_qty['hidden_cogs'].sum()

    df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
    total_sales = df['sales'].sum()
    reported_cogs = (df['quantity'] * df['cost']).clip(lower=0).sum()
    true_cogs = reported_cogs + estimated_hidden_cogs
    reported_margin = (total_sales - reported_cogs) / total_sales if total_sales else 0
    true_margin = (total_sales - true_cogs) / total_sales if total_sales else 0
    margin_adjustment = reported_margin - true_margin

    # Top offenders
    top_categories: Dict[str, float] = {}
    top_vendors: Dict[str, float] = {}
    if 'category' in df.columns:
        cat_leaks = negative_qty.groupby('category')['hidden_cogs'].sum().sort_values(ascending=False).head(10)
        top_categories = cat_leaks.to_dict()
    if 'vendor' in df.columns:
        ven_leaks = negative_qty.groupby('vendor')['hidden_cogs'].sum().sort_values(ascending=False).head(10)
        top_vendors = ven_leaks.to_dict()

    # Results
    analysis_results = {
        "filename": file.filename,
        "rows_processed": len(df),
        "total_sales": float(total_sales),
        "reported_margin_pct": float(reported_margin * 100),
        "true_margin_pct": float(true_margin * 100),
        "margin_adjustment_pct": float(margin_adjustment * 100),
        "leak_insights": {
            "negative_inventory_items": int(len(negative_qty)),
            "estimated_hidden_cogs": float(estimated_hidden_cogs),
            "top_problem_categories": top_categories,
            "top_problem_vendors": top_vendors
        }
    }

    # Upload to S3 for history
    if S3_CLIENT:
        try:
            S3_CLIENT.put_object(
                Bucket=BUCKET_NAME,
                Key=f"uploads/{file.filename}",
                Body=contents
            )
            analysis_results["s3_status"] = "uploaded successfully"
        except ClientError as e:
            analysis_results["s3_status"] = f"upload failed: {str(e)}"

    return JSONResponse(content=analysis_results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)