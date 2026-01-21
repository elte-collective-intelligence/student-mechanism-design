FROM python:3.12-slim

WORKDIR /app

# Install system dependencies in a single layer and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    dos2unix \
    jq \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only requirements first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Set Python path
ENV PYTHONPATH=/app/src

# Use non-root user for security (optional but recommended)
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser

# Default to bash for interactive use
CMD ["/bin/bash"]