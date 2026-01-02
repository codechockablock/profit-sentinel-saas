from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(
    title="Profit Sentinel API",
    description="Vigilant guardian for retail profits",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀 Uncover hidden leaks in your POS data."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Placeholder: Later upload to S3 + run analysis
    contents = await file.read()
    return JSONResponse(content={
        "filename": file.filename,
        "size_bytes": len(contents),
        "status": "uploaded (analysis coming soon)"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)