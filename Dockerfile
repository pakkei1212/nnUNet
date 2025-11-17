# =============================
# Base image and build args
# =============================
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.3.0
ARG TORCH_CHANNEL=https://download.pytorch.org/whl/cu121

# =============================
# Environment variables
# =============================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JUPYTER_TOKEN=nnunet \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# =============================
# Install OS-level dependencies
# =============================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \               
        python-is-python3 \
        python3-venv \
        build-essential \           
        git \
        graphviz \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libffi-dev \                
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# =============================
# Install Python dependencies
# =============================
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        jupyterlab \
        ipywidgets && \
    python3 -m pip install --no-cache-dir \
        --extra-index-url ${TORCH_CHANNEL} \
        torch==${TORCH_VERSION}

# =============================
# Copy project and install
# =============================
COPY . /workspace
RUN python3 -m pip install --no-cache-dir -e .

EXPOSE 8888

# =============================
# Launch Jupyter Lab
# =============================
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \
    --ServerApp.allow_remote_access=True \
    --ServerApp.preferred_dir=/workspace \
    --ServerApp.token=${JUPYTER_TOKEN:-nnunet} \
    --ServerApp.password='' \
    --allow-root"]
