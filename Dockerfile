# ============================================
# Flood Watermark Measurement API - Dockerfile
# ============================================
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FORCE_CUDA=0 \
    TORCH_CUDA_ARCH_LIST=""

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt requirements.docker.txt

# Install Python dependencies (CPU-only PyTorch for Docker)
# We create a modified requirements file without the +cpu suffix
RUN sed 's/+cpu//g' requirements.docker.txt > requirements_clean.txt && \
    sed -i '/--extra-index-url/d' requirements_clean.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_clean.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p backend/uploads backend/data

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Run the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
