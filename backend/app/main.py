# backend/app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import pandas as pd
import io
import boto3
import os
import uuid  # For unique S3 keys

app = FastAPI(
    title="Profit Sentinel",
    description="Forensic analysis of POS exports to detect hidden profit leaks",
    version="1.0.0",
)

# === SECURITY: Tight CORS for production + Vercel preview ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://profitsentinel.com",
        "https://www.profitsentinel.com",
        "https://profit-sentinel-saas.vercel.app",  # Your Vercel production
        # Add preview domains if needed, e.g., "https://profit-sentinel-saas-git-main-yourusername.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# S3 client
S3_CLIENT = boto3.client('s3') if os.getenv("AWS_DEFAULT_REGION") else None
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")

# Supabase client for server-side JWT verification (service_role key - secret!)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://PLACEHOLDER_SUPABASE_PROJECT.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Add this in ECS task env vars
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_SERVICE_KEY else None

# Optional JWT verification dependency (guest allowed)
async def get_current_user(authorization: str | None = Header(None)):
    if not authorization or not supabase:
        return None  # Guest
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        return user.user.id if user else None
    except Exception as e:
        print(f"JWT verification failed: {e}")
        return None  # Invalid token → treat as guest

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀 Uncover hidden profit leaks in your POS data."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    email: str = Form(None),
    user_id: str | None = Depends(get_current_user)  # Optional auth - guest if None
):
    # File size limit (50MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (>50MB)")

    if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type—CSV or Excel only")

    try:
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False, encoding='latin1')
        else:
            df = pd.read_excel(io.BytesIO(contents), dtype=str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="File is empty")

    # Aggressive column matching (unchanged)
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

    # Human-friendly summary
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
        result["report_sent_to"] = email

    # Unique S3 key + save
    if S3_CLIENT:
        try:
            unique_key = f"uploads/{uuid.uuid4()}_{file.filename}"
            S3_CLIENT.put_object(Bucket=BUCKET_NAME, Key=unique_key, Body=contents)
            result["s3_status"] = "saved"
        except Exception as e:
            result["s3_status"] = f"save failed: {str(e)}"

    # Log authenticated user (for future dashboard/history)
    if user_id:
        result["user_id"] = user_id  # Or save to DB later
        print(f"Upload by authenticated user: {user_id}")
    else:
        print("Guest upload")

    return JSONResponse(content=result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)