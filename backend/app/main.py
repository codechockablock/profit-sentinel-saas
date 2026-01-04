from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import io
import boto3
from botocore.exceptions import ClientError
import os
import re

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
    return re.sub(r'[^a-z0-9]', '', name.lower())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type—CSV or Excel only")

    contents = await file.read()

    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), dtype=str)
        else:
            df = pd.read_excel(io.BytesIO(contents), dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty")

    cleaned_map = {col: clean_column(col) for col in df.columns}

    # Flexible mapping for this file
    column_map = {
        'quantity': next((col for col, clean in cleaned_map.items() if 'qtydifference' in clean or 'qtydiff' in clean or 'difference' in clean or 'inventoriedqty' in clean or 'invento' in clean or 'onhand' in clean), None),
        'cost': next((col for col, clean in cleaned_map.items() if 'cost' in clean), None),
        'sales': next((col for col, clean in cleaned_map.items() if 'retail' in clean or 'sold' in clean), None),  # Use Retail as proxy for sales value
        'category': next((col for col, clean in cleaned_map.items() if 'cat' in clean or 'category' in clean or 'dpt' in clean), None),
        'vendor': next((col for col, clean in cleaned_map.items() if 'vendor' in clean), None),
    }

    missing = [k for k, v in column_map.items() if v is None and k in ['quantity', 'cost']]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}. Cleaned columns: {sorted(cleaned_map.values())}"
        )

    rename_dict = {v: k for k, v in column_map.items() if v is not None}
    df = df.rename(columns=rename_dict)

    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(0)

    negative_qty = df[df['quantity'] < 0].copy()
    negative_qty['abs_quantity'] = negative_qty['quantity'].abs()
    negative_qty['hidden_cogs'] = negative_qty['abs_quantity'] * negative_qty['cost']
    estimated_hidden_cogs = negative_qty['hidden_cogs'].sum()

    total_cost = df['cost'].sum()
    reported_cogs = (df['quantity'] * df['cost']).clip(lower=0).sum()
    true_cogs = reported_cogs + estimated_hidden_cogs

    analysis_results = {
        "filename": file.filename,
        "rows_processed": len(df),
        "estimated_hidden_cogs": float(estimated_hidden_cogs),
        "negative_inventory_items": int(len(negative_qty)),
        "top_problem_items": negative_qty.sort_values('hidden_cogs', ascending=False).head(10)[['sku', 'descriptionfull', 'hidden_cogs']].to_dict('records') if 'sku' in df.columns else []
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