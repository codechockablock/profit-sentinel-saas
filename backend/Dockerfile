# Use slim for smaller image
FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Copy requirements first for better layer caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory
COPY backend/app /app

# Expose port
EXPOSE 8000

# Run as non-root user for security (optional but good)
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Run uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]