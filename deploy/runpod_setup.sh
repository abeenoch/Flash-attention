#!/usr/bin/env bash
# Run inside a RunPod GPU pod (use the "RunPod PyTorch 2.x" template so torch+CUDA
# are preinstalled - otherwise you pay GPU rates while pip downloads ~2.5GB of torch).
#
# Recommended pod: RTX 4090 (24GB), Community Cloud ~$0.34/hr or Secure ~$0.74/hr.
#   1) create pod, attach a Network Volume mounted at /workspace (persists the repo)
#   2) open the web terminal or SSH
#   3) bash runpod_setup.sh
#   4) STOP the pod when done:  runpodctl stop pod "$RUNPOD_POD_ID"
#
# The 4090 is Ada Lovelace, the same arch family as Modal's L4, so tile-size tuning
# transfers between the two. It has ~1008 GB/s bandwidth (vs L4's ~300 GB/s), so
# absolute TFLOP/s will be much higher here.
set -euo pipefail

REPO_URL="${REPO_URL:-}"          # set to your git remote, or copy the folder manually
WORKDIR="${WORKDIR:-/workspace}"
PROJECT="triton-attention"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ -n "$REPO_URL" ] && [ ! -d "$PROJECT" ]; then
  git clone "$REPO_URL" "$PROJECT"
fi
cd "$PROJECT"

# triton + test/bench deps (torch is provided by the pod template)
python -m pip install -q --upgrade pip
python -m pip install -q triton pytest matplotlib

echo "---- versions ----"
python -c "import torch, triton; print('torch', torch.__version__); print('triton', triton.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0))"

# persist compiled kernels across pod restarts when a network volume is attached
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORKDIR/.triton_cache}"
mkdir -p "$TRITON_CACHE_DIR"

echo "---- correctness ----"
python -m pytest tests/ -v -p no:cacheprovider

echo "---- benchmark ----"
python benches/bench_attention.py

echo
echo "Results written to $PROJECT/results/ ."
echo "When finished:  runpodctl stop pod \"\$RUNPOD_POD_ID\"   (stopped pods bill disk only)"
