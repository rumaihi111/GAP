FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf \
    TRANSFORMERS_CACHE=/runpod-volume/hf/transformers \
    TORCH_HOME=/runpod-volume/torch

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-runpod.txt /tmp/requirements-runpod.txt
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-runpod.txt

COPY diffusion/ ./diffusion/

CMD ["python3","-u","diffusion/runpod_animatediff_handler.py"]
