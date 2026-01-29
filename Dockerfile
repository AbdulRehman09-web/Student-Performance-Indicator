FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port (Render injects $PORT automatically)
EXPOSE $PORT

# Run Gunicorn
CMD ["sh", "-c", "gunicorn flask_app.app:app --bind 0.0.0.0:$PORT --workers 1"]