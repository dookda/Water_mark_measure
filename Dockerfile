FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.docker.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY .env .

# Create directories for data and uploads
RUN mkdir -p backend/data backend/uploads

# Expose the application port
EXPOSE 7860

# Run the application
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
