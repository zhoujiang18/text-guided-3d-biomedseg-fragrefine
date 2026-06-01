# Start from a lightweight CUDA runtime base
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install minimal system dependencies + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    libaio-dev \
    ffmpeg \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip & install core Python packages
COPY requirements.txt .

RUN python -m pip install --upgrade pip setuptools==65.6.3 packaging==23.0 \
 && pip install --no-cache-dir torch==2.3.1+cu118 torchvision==0.18.1+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir git+https://github.com/facebookresearch/detectron2.git

# Copy project files
COPY . .

# Make script executable
RUN chmod +x predict.sh

# Entrypoint
CMD ["sh", "predict.sh"]
