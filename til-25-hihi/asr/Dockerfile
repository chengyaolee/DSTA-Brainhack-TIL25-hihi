# Dockerfile for building the ASR image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
# FROM python:3.12-slim
# FROM gcr.io/deeplearning-platform-release/base-cu122.py310
# FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
# FROM asia-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest #no
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*
    
# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.10 python3.10-venv python3-pip python3-dev wget git \
#     libgl1-mesa-glx libglib2.0-0 build-essential \
#     && rm -rf /var/lib/apt/lists/* \
#     && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
#     && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
    
# Installs your dependencies.
RUN pip install -U pip 
# RUN pip install wheel
RUN pip install 'numpy<2.0'
# RUN pip install torch==2.1.0+cu121 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# RUN pip install torch==2.7.0+cu128 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install torch==2.7.0+cu128 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install flash-attn==2.3.6 --no-build-isolation

# Copies your source files.
COPY src .
#COPY models .
#COPY whisper_tiny_finetuned .
COPY whisper_ct2_small .

# Starts your model server.
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0
# CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001"]