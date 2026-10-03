"""Probe what the Triton autotuner really picks for D=256 (and D=128/D=64).

Run:  modal run deploy/probe_autotune.py
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/project"
GPU = os.environ.get("ATTN_GPU", "L4")

app = modal.App("triton-attention-probe")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "triton==3.0.0", "numpy")
    .env({
        "TRITON_CACHE_DIR": "/kernelcache",
        "PYTHONPATH": REMOTE_ROOT,
        "TRITON_PRINT_AUTOTUNING": "1",
    })
    .add_local_dir(str(HERE / "attention"), remote_path=f"{REMOTE_ROOT}/attention")
)

kernel_cache_vol = modal.Volume.from_name("triton-attention-kernelcache", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=1800, volumes={"/kernelcache": kernel_cache_vol})
def probe():
    import torch
    from attention.triton_kernel_pipelined import _flash_attn_fwd_kernel as K

    print("torch", torch.__version__, "| gpu", torch.cuda.get_device_name(0), flush=True)
    for D in (128, 64, 256):
        print(f"\n===== D={D} =====", flush=True)
        q = torch.randn(1, 4, 512, D, dtype=torch.float16, device="cuda")
        k = q.clone()
        v = q.clone()
        from attention.triton_kernel_pipelined import flash_attn_fwd_triton
        for i in range(3):
            flash_attn_fwd_triton(q, k, v, causal=True)
        torch.cuda.synchronize()
        print(f"--- after D={D} ---", flush=True)

    cache = getattr(K, "cache", None)
    print("\nAutotuner cache type:", type(cache), flush=True)
    try:
        keys = list(cache.keys())
        print(f"num cache entries: {len(keys)}", flush=True)
        for kk in keys:
            print("KEY:", repr(kk)[:500], flush=True)
    except Exception as e:
        print("could not list cache keys:", e, flush=True)


@app.local_entrypoint()
def main():
    print(f"launching probe on Modal GPU={GPU} ...")
    probe.remote()
    print("probe done.")
