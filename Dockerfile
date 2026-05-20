FROM python:3.10-slim

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

# Install PyTorch CPU-only first (smaller, ~800MB vs 2GB GPU version)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . /app

# Expose port (Render uses 10000 by default for free tier)
EXPOSE 10000

# Start Uvicorn on port 10000 (Render's expected port)
CMD ["uvicorn", "traffic_prediction.api.app:create_app", "--host", "0.0.0.0", "--port", "10000", "--factory"]
