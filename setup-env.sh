#!/usr/bin/env bash
# One-time environment setup. Targets CUDA 12.4 (RunPod PyTorch base image).
set -e

CONDA_ENV=encroachnet
CUDA_TAG=cu124

conda create -n $CONDA_ENV python=3.11 -y
source activate $CONDA_ENV

pip install torch torchvision --index-url https://download.pytorch.org/whl/$CUDA_TAG
pip install spconv-$CUDA_TAG
pip install pytorch-lightning tensorboard numpy scipy tqdm
pip install opencv-python transformers timm
pip install "laspy[lazrs]"
pip install open3d
pip install pycolmap
pip install scikit-image scikit-learn

echo "Environment '$CONDA_ENV' ready. Activate with: conda activate $CONDA_ENV"
