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
    """Remove punctuation/spaces/special chars for robust matching"""
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

    # === ADAPTIVE COLUMN MAPPING ===

    # All quantity-related candidates
    qty_candidates = [col for col, clean in cleaned_map.items() if 'qty' in clean]

    # Prefer current/on-hand stock column first (common in full inventory exports)
    quantity_col = next((
        col for col in qty_candidates
        if 'onhand' in cleaned_map[col] or 'hand' in cleaned_map[col] or 'current' in cleaned_map[col]
    ), None)

    # If preferred has no negatives (likely a shrink/offender report), fall back to the qty column with the most negatives
    if quantity_col:
        temp_qty = pd.to_numeric(df[quantity_col], errors='coerce').fillna(0)
        if (temp_qty < 0).sum() == 0 or temp_qty.sum() >= 0:
            fallback_candidates = [c for c in qty_candidates if c != quantity_col]
            if fallback_candidates:
                neg_sums = {}
                for c in fallback_candidates:
                    series = pd.to_numeric(df[c], errors='coerce').fillna(0)
                    neg_sums[c] = series.sum()
                if any(v < 0 for v in neg_sums.values()):
                    quantity_col = min(neg_sums, key=neg_sums.get)

    # Fallback if still none: broadest qty match
    if not quantity_col:
        quantity_col = next((col for col, clean in cleaned_map.items() if 'qty' in clean), None)

    # Cost: strongly prefer average cost (best for ongoing leak valuation), then standard, then any cost
    cost_col = next((col for col, clean in cleaned_map.items() if 'avg' in clean and 'cost' in clean), None)
    if not cost_col:
        cost_col = next((col for col, clean in cleaned_map.items() if ('std' in clean or 'standard' in clean) and 'cost' in clean), None)
    if not cost_col:
        cost_col = next((col for col, clean in cleaned_map.items() if 'cost' in clean), None)

    # Sales: strongly prefer dollar-prefixed/suffixed revenue columns, then common names
    sales_col = next((
        col for col in df.columns
        if '$' in col and ('sold' in cleaned_map[col] or 'sales' in cleaned_map[col] or 'revenue' in cleaned_map[col])
    ), None)
    if not sales_col:
        sales_col = next((
            col for col, clean in cleaned_map.items()
            if 'sales' in clean or 'sold' in clean or 'revenue' in clean
        ), None)

    # Category: broad matching
    category_col = next((
        col for col, clean in cleaned_map.items()
        if 'cat' in clean or 'dept' in clean or 'dpt' in clean or 'category' in clean or 'department' in clean
    ), None)

    # Vendor: broad matching including manufacturer
    vendor_col = next((
        col for col, clean in cleaned_map.items()
        if 'vendor' in clean or 'supplier' in clean or 'mfgr' in clean or 'manufacturer' in clean
    ), None)

    # Required columns check with helpful debug
    required_mapping = {
        'quantity': quantity_col,
        'cost': cost_col,
        'sales': sales_col
    }
    missing = [k for k, v in required_mapping.items() if v is None]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Auto-mapping failed for required columns: {missing}. "
                f"Qty candidates found: {qty_candidates}. "
                f"Sample original columns: {list(df.columns)[:20]}... "
                f"Cleaned columns: {sorted(cleaned_map.values())}"
            )
        )

    # Rename mapped columns to standard names
    rename_dict = {
        v: k for k, v in {
            'quantity': quantity_col,
            'cost': cost_col,
            'sales': sales_col,
            'category': category_col,
            'vendor': vendor_col
        }.items() if v is not None
    }
    df = df.rename(columns=rename_dict)

    # === ANALYSIS ===
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
    if 'category' in df.columns and len(negative_qty) > 0:
        cat_leaks = negative_qty.groupby('category')['hidden_cogs'].sum().sort_values(ascending=False).head(10)
        top_categories = cat_leaks.to_dict()
    if 'vendor' in df.columns and len(negative_qty) > 0:
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