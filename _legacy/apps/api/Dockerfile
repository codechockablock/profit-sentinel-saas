# Profit Sentinel API Dockerfile (Legacy)
# NOTE: Build from repo root with: docker build -f _legacy/apps/api/Dockerfile .
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY _legacy/apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU-only FIRST to prevent sentinel-engine from pulling CUDA version
# This reduces image size from ~4.5GB to ~1.5GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install sentinel-engine (v5.0.0 - Dorian, self-contained)
# Using --no-deps to prevent re-installing torch with CUDA
COPY _legacy/packages/sentinel-engine /packages/sentinel-engine
RUN pip install --no-cache-dir --no-deps /packages/sentinel-engine

# Install remaining sentinel-engine dependencies (without torch)
RUN pip install --no-cache-dir numpy>=1.24.0 faiss-cpu>=1.7.4 pydantic>=2.5.0 \
    reportlab>=4.0.0 requests>=2.28.0 SPARQLWrapper>=2.0.0

# Copy application code
COPY _legacy/apps/api/ .

# Copy configuration files (POS mappings for column detection)
COPY config/ /app/config/

# Add app source to Python path
ENV PYTHONPATH="/app:/app/src:${PYTHONPATH}"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
