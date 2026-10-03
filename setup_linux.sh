#!/usr/bin/env bash
# One-command env setup for a Linux CUDA GPU box (RunPod / Vast / Lambda / etc).
#   bash setup_linux.sh
set -euo pipefail

# --- Python ---
PY=${PYTHON:-python3}
$PY -m pip install --upgrade pip

# --- PyTorch (CUDA build) ---
# Pin to a recent CUDA build; triton wheels need torch>=2.1.
$PY -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# --- Triton (GPU kernels) ---
$PY -m pip install triton

# --- Project Python deps ---
$PY -m pip install -r requirements.txt

# --- Optional plotting deps ---
$PY -m pip install matplotlib numpy

echo "---- versions ----"
$PY -c "import torch,triton,numpy; print('torch', torch.__version__); print('cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print('triton', triton.__version__)"
echo "done. run:  python -m pytest tests/ -v   &&  python benches/bench_attention.py"
