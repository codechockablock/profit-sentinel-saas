from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import io
import boto3
from botocore.exceptions import ClientError
import os

app = FastAPI(title="Profit Sentinel")

# AWS S3 client (for production; fallback to local for dev)
s3_client = boto3.client('s3') if os.getenv("AWS_DEFAULT_REGION") else None
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "profitsentinel-dev-uploads")  # Set in env or Terraform

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀 Uncover hidden profit leaks in your POS data."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="Invalid file type—CSV or Excel only")

    # Read file into bytes
    contents = await file.read()

    # Parse with pandas
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    # Placeholder for your real analysis logic (from original scripts)
    # Example: Detect negatives, calculate hidden COGS, true margins
    analysis_results = {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "summary": "Analysis complete—placeholder results",
        "leak_insights": {
            "negative_inventory_items": 0,  # Replace with real count
            "estimated_hidden_cogs": 0.0,   # Your calculation
            "true_margin_adjustment": 0.0,
            "top_problem_categories": [],   # List from groupby
            "top_problem_vendors": []
        }
    }

    # Real analysis example (customize with your scripts)
    if 'Quantity' in df.columns and 'Cost' in df.columns:
        negative_qty = df[df['Quantity'] < 0]
        analysis_results["leak_insights"]["negative_inventory_items"] = len(negative_qty)
        analysis_results["leak_insights"]["estimated_hidden_cogs"] = (negative_qty['Quantity'].abs() * negative_qty['Cost']).sum()

    # Optional: Upload to S3 for storage/history
    if s3_client:
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=f"uploads/{file.filename}",
                Body=contents
            )
            analysis_results["s3_status"] = "uploaded"
        except ClientError as e:
            analysis_results["s3_status"] = f"upload failed: {str(e)}"

    return JSONResponse(content=analysis_results)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)