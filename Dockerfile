# Multi-stage Dockerfile for Insurance LLM Fine-Tuning Pipeline
# Base: Ubuntu 22.04 + CUDA 12.1 + Python 3.10

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================
# Training Stage
# ============================================
FROM base AS training

WORKDIR /workspace

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p outputs/checkpoints outputs/merged_models outputs/logs outputs/evaluation outputs/reports

# Expose ports for monitoring
EXPOSE 6006  # TensorBoard
EXPOSE 8000  # vLLM
EXPOSE 8001  # FastAPI

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1

# Default command
CMD ["/bin/bash"]

# ============================================
# Development Stage (with extra tools)
# ============================================
FROM training AS dev

# Install development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov

# Expose Jupyter
EXPOSE 8888

# ============================================
# Production Stage (minimal, inference only)
# ============================================
FROM base AS production

WORKDIR /app

# Copy only necessary files for inference
COPY config ./config
COPY src ./src
COPY requirements.txt .

# Install only core dependencies (no dev tools)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    transformers==4.36.2 \
    peft==0.7.1 \
    bitsandbytes==0.42.0 \
    vllm==0.3.1 \
    fastapi==0.109.0 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    python-dotenv==1.0.0

EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "src/api/main.py"]