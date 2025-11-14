FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-runtime-ubuntu22.04

ENV HF_HOME=/runpod-volume/hf \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf \
    TRANSFORMERS_CACHE=/runpod-volume/hf/transformers \
    TORCH_HOME=/runpod-volume/torch \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-runpod.txt /tmp/requirements-runpod.txt
RUN pip install --no-cache-dir -r /tmp/requirements-runpod.txt && \
    rm -rf /root/.cache/pip

# Copy only what you need to run the handler
COPY diffusion/ ./diffusion/

CMD ["python3","-u","diffusion/runpod_animatediff_handler.py"]

