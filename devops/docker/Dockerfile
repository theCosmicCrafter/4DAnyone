# ─── Stage 1: Base image with CUDA 12.8 + Python 3.11 ───
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# ─── Stage 2: Install dependencies ───
FROM base AS deps

# Copy requirements first for layer caching
COPY requirements.txt .

# Create venv and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN uv pip install -r requirements.txt

# ─── Stage 3: Application ───
FROM deps AS app

# Copy application code
COPY . /app/

# Initialize GVHMR submodule (requires .git)
RUN git submodule update --init third_party/GVHMR || true

# Download models (non-interactive, skips SMPL-X which requires license)
RUN python scripts/download_model.py || true

# Volumes for persistent data
VOLUME ["/app/data", "/app/models"]

# Default entrypoint
ENTRYPOINT ["python", "inference.py"]

# Default: 6-view quick test with example video
CMD ["--video_path", "data/source/pexels/2785536-uhd_2160_3840_25fps.mp4", "--views_per_layer", "6"]
