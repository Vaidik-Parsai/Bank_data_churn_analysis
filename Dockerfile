# ==============================================================================
# Production Dockerfile for Bank Customer Churn Prediction Service
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user for secure container execution
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, trained model artifact, and metadata
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/
COPY reports/ /app/reports/
COPY data/ /app/data/

# Set ownership of application files to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose FastAPI service port
EXPOSE 8000

# Container liveness check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch uvicorn ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
