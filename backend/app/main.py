from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import io
import boto3
from botocore.exceptions import ClientError
import os
import re  # For extra cleaning

app = FastAPI(title="Profit Sentinel")

S3_CLIENT = boto3.client('s3') if os.getenv("AWS_DEFAULT_REGION") else None
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀 Uncover hidden profit leaks in your POS data."}

@app.get("/health")
def health():
    return {"status": "healthy"}

def clean_column(name: str) -> str:
    """Remove punctuation/spaces for robust matching"""
    return re.sub(r'[^a-z0-9]', '', name.lower())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type—CSV or Excel only")

    contents = await file.read()

    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty")

    # Clean columns for matching
    cleaned_map = {col: clean_column(col) for col in df.columns}

    # Flexible mapping
    column_map = {
        'quantity': next((col for col, clean in cleaned_map.items() if 'qty' in clean), None),
        'cost': next((col for col, clean in cleaned_map.items() if 'cost' in clean), None),
        'sales': next((col for col, clean in cleaned_map.items() if 'sales' in clean or 'sold' in clean), None),
        'category': next((col for col, clean in cleaned_map.items() if 'cat' in clean or 'category' in clean or 'dept' in clean), None),
        'vendor': next((col for col, clean in cleaned_map.items() if 'vendor' in clean or 'supplier' in clean), None),
    }

    missing = [k for k, v in column_map.items() if v is None and k in ['quantity', 'cost', 'sales']]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}. Found (cleaned): {list(cleaned_map.values())}"
        )

    # Rename to standard
    rename_dict = {v: k for k, v in column_map.items() if v is not None}
    df = df.rename(columns=rename_dict)

    # Analysis (same as before)
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

    top_categories = {}
    top_vendors = {}
    if 'category' in df.columns:
        cat_leaks = negative_qty.groupby('category')['hidden_cogs'].sum().sort_values(ascending=False).head(10)
        top_categories = cat_leaks.to_dict()
    if 'vendor' in df.columns:
        ven_leaks = negative_qty.groupby('vendor')['hidden_cogs'].sum().sort_values(ascending=False).head(10)
        top_vendors = ven_leaks.to_dict()

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

    if S3_CLIENT:
        try:
            S3_CLIENT.put_object(Bucket=BUCKET_NAME, Key=f"uploads/{file.filename}", Body=contents)
            analysis_results["s3_status"] = "uploaded successfully"
        except ClientError as e:
            analysis_results["s3_status"] = f"upload failed: {str(e)}"

    return JSONResponse(content=analysis_results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)