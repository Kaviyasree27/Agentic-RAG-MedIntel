# 1. Base Image with Python 3.11 Slim for minimal attack surface and size
FROM python:3.11-slim

# Set environment variables for clean container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    SENTENCE_TRANSFORMERS_HOME=/app/models_cache

# Set working directory
WORKDIR /app

# Install essential system dependencies for building C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Sentence-Transformers and FlashRank models so startup is instantaneous
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code into container
COPY . .

# Ensure storage directories exist
RUN mkdir -p chroma_db models_cache sample_data

# Expose web application port
EXPOSE 5000

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/documents || exit 1

# Production command using Gunicorn with 2 workers and thread concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
