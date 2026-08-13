#!/usr/bin/env bash
# One-time environment setup. Targets the RunPod PyTorch base image
# (runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04), which ships
# system Python 3.11 + torch/torchvision already built for CUDA 12.4 and has
# no conda — install straight into system site-packages with uv, same as
# Dockerfile.runpod.
set -e

command -v uv >/dev/null 2>&1 || pip install uv

uv pip install --system spconv-cu124
uv pip install --system -r requirements.txt

echo "Environment ready: $(python3 --version), torch $(python3 -c 'import torch; print(torch.__version__)')"
