FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SPLADE_MODEL_DIR=/app/models/Splade_PP_en_v2 \
    SPLADE_AUTO_DOWNLOAD_MODEL=true

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p /app/models /app/app/data

# Expose the port the app runs on
EXPOSE 3000

# Command to run the application
CMD ["uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "3000"]