FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf \
    TRANSFORMERS_CACHE=/runpod-volume/hf/transformers \
    TORCH_HOME=/runpod-volume/torch

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth=1 https://github.com/rumaihi111/GAP.git /app

COPY requirements-runpod.txt requirements-runpod.txt
RUN pip install --upgrade pip && pip install -r requirements-runpod.txt
RUN pip install runpod==1.7.0 diffusers==0.29.2 transformers==4.44.2 accelerate==0.34.2 safetensors==0.4.4 pillow==10.4.0 imageio==2.34.1 imageio-ffmpeg==0.4.9 einops==0.7.0 opencv-python-headless==4.10.0.84

CMD ["python","-u","diffusion/runpod_animatediff_handler.py"]