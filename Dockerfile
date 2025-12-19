# AlphaEvolve-Mini Docker Setup
# 
# Build:
#   docker build -t alphaevolve-mini .
#
# Run with Ollama (recommended):
#   # First, run Ollama on host or in another container
#   docker run -it --network host alphaevolve-mini python examples/local_demo.py
#
# Run with built-in Hugging Face model:
#   docker run -it --gpus all alphaevolve-mini python examples/local_demo.py --provider huggingface

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt requirements-local.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p checkpoints outputs

# Default command
CMD ["python", "examples/local_demo.py", "--help"]
