# RAG Brain - Local Multimodal RAG System
# CPU-optimized Dockerfile

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CPU support first (separate layer for caching)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download models during build for faster startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY config/ ./config/

# Create data directories
RUN mkdir -p /app/data/embeddings /app/data/files /app/data/metadata /app/logs /app/mlflow

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CPU_ONLY=1
ENV TOKENIZERS_PARALLELISM=false

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
