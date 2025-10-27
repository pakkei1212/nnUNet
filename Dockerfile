ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn9-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.3.0
ARG TORCH_CHANNEL=https://download.pytorch.org/whl/cu121

# Prevent Python from writing .pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JUPYTER_TOKEN=nnunet \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install OS-level dependencies required by nnU-Net and visualization libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
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
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first to leverage Docker layer caching
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        jupyterlab \
        ipywidgets && \
    python3 -m pip install --no-cache-dir \
        --extra-index-url ${TORCH_CHANNEL} \
        torch==${TORCH_VERSION}

# Copy the project into the image and install it in editable mode together with Jupyter
COPY . /workspace

RUN python3 -m pip install --no-cache-dir -e .

EXPOSE 8888

# Launch Jupyter Lab, allowing remote access with the configured token
CMD ["sh", "-c", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --ServerApp.allow_remote_access=True --ServerApp.preferred_dir=/workspace --ServerApp.token=${JUPYTER_TOKEN:-nnunet} --ServerApp.password=''"]
