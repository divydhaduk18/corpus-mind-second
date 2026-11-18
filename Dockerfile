# ==========================================
# Corpus Mind Multi-Stage Production Dockerfile
# ==========================================

# Stage 1: Build Dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements into wheels directory
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Final Runtime Image
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime system packages for OCR, PDF rendering & Computer Vision
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /root/.local /home/corpususer/.local

# Create non-root system user for security
RUN useradd -m -u 1000 -s /bin/bash corpususer && \
    mkdir -p /app/data /app/uploads /app/vector_store /app/logs /app/data/cache /app/data/index_backups && \
    chown -R corpususer:corpususer /app

# Add local bin to PATH
ENV PATH=/home/corpususer/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSERACT_CMD_PATH=/usr/bin/tesseract

# Copy application source code
COPY --chown=corpususer:corpususer . /app

# Switch to non-root user
USER corpususer

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || curl -f http://localhost:8000/health || exit 1

# Default command: Streamlit Web UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
