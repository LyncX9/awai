FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies (if any are needed)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . /app

# Expose port
EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "traffic_prediction.api.app:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]
