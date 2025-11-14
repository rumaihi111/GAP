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

name: build-and-push-serverless-image
on:
  push:
    branches: [ main ]
    paths:
      - docs/Orumaihinew.dockerfile
      - requirements-runpod.txt
      - diffusion/**

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Free disk space
        run: |
          sudo rm -rf /usr/share/dotnet
          sudo rm -rf /opt/ghc
          sudo rm -rf /usr/local/share/boost
          sudo rm -rf /usr/local/lib/android
          sudo apt-get clean
          df -h

      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docs/Orumaihinew.dockerfile
          push: true
          tags: orumaihi/gap-animatediff:prod
          # enable cache only after first successful build
