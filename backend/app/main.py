from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Profit Sentinel")

@app.get("/")
def root():
    return {"message": "Profit Sentinel is live! 🚀 Uncover hidden profit leaks in your POS data."}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Placeholder—later S3 + analysis
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents), "status": "uploaded"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)