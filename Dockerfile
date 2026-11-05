
FROM python:3.12-slim

# Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Native runtime dependency used by numerical/ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately for better layer caching
COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source
COPY . .

# Render provides PORT; 8000 is the local default
EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
