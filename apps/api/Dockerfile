# Profit Sentinel API Dockerfile
# NOTE: Build from repo root with: docker build -f apps/api/Dockerfile .
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
COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU-only (required by sentinel-engine Dorian)
# Using CPU-only to reduce image size; use cuda version if GPU needed
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install sentinel-engine (v5.0.0 - Dorian, self-contained)
COPY packages/sentinel-engine /packages/sentinel-engine
RUN pip install --no-cache-dir /packages/sentinel-engine

# Copy application code
COPY apps/api/ .

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
