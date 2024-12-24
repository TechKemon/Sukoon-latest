# FOR MYCA
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    curl \  
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
# RUN mkdir -p /app/prompts
# RUN mkdir -p /app/storage

# Copy additional configuration files
# COPY prompts.yaml .
# COPY .env .

# Set environment variables
ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8000
# ENV SUPABASE_API_KEY=""  # Added for feedback endpoint
# ENV SUPABASE_AUTHORIZATION_TOKEN=""  # Added for feedback endpoint

# Expose the port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "myca_api:app", "--host", "0.0.0.0", "--port", "8000"]