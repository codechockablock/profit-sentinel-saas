# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import boto3
import os

app = FastAPI(
    title="Profit Sentinel",
    description="Forensic analysis of POS exports to detect hidden profit leaks",
    version="1.0.0",
)

# === SECURITY: Lock down CORS to your real domain ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://profitsentinel.com",
        "https://www.profitsentinel.com",
    ],
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
async def upload_file(
    file: UploadFile = File(...),
    email: str = Form(None)  # Optional: for future report emailing
):
    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type—CSV or Excel only")

    contents = await file.read()

    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False, encoding='latin1')
        else:
            df = pd.read_excel(io.BytesIO(contents), dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty")

    # Your excellent aggressive column matching
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

    # Add human-friendly summary
    result.update({
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
        result["report_sent_to"] = email  # Placeholder for future email send

    # Your solid S3 upload
    if S3_CLIENT:
        try:
            S3_CLIENT.put_object(Bucket=BUCKET_NAME, Key=f"uploads/{file.filename}", Body=contents)
            result["s3_status"] = "saved"
        except Exception as e:
            result["s3_status"] = f"save failed: {str(e)}"

    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)