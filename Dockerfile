FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (must be before other deps to set index correctly)
RUN pip install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install core API dependencies (slim, no ML training libs)
COPY requirements-api.txt /app/
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application source
COPY . /app

# Expose port (Render uses 10000 for free tier)
EXPOSE 10000

# Start Uvicorn on port 10000
CMD ["uvicorn", "traffic_prediction.api.app:create_app", "--host", "0.0.0.0", "--port", "10000", "--factory"]
