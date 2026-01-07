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

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://kbjiejqotrjsdeuxhtcx.supabase.co")
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
            Fields={"acl": "private"},
            Conditions=[{"acl": "private"}, ["content-length-range", 1, 200 * 1024 * 1024]],  # Max 200MB per file
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
        try:
            obj = S3_CLIENT.get_object(Bucket=BUCKET_NAME, Key=key)
            contents = obj['Body'].read()

            # Your existing processing
            if key.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(contents), dtype=str, keep_default_na=False, encoding='latin1')
            else:
                df = pd.read_excel(io.BytesIO(contents), dtype=str)

            # ... your column matching, analysis, result dict ...

            results.append(result)
        except Exception as e:
            results.append({"key": key, "status": "error", "detail": str(e)})

    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)