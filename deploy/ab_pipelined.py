"""A/B experiment: v1 (while-loop) vs v2 (pipelined two-phase) Triton kernel.

Run:
    modal run deploy/ab_pipelined.py
    modal volume get triton-attention-results / ./results

What it does on the GPU:
  1. Correctness: v1 and v2 vs the fused SDPA reference on shapes chosen to
     stress the differences (GQA, ragged M, non-pow2 head dims, non-causal).
  2. Perf: v1 vs v2 vs PyTorch FA2 on the headline configs; writes
     results/ab_pipelined.csv.

The kernel-cache Volume is shared with modal_run.py, so v1's autotune
results are already warm; v2 autotunes once and is then cached too.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent  # project root
REMOTE_ROOT = "/root/project"
GPU = os.environ.get("ATTN_GPU", "L4")

app = modal.App("triton-attention-ab")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "triton==3.0.0")
    .env({
        "TRITON_CACHE_DIR": "/kernelcache",
        "PYTHONPATH": REMOTE_ROOT,
    })
    .add_local_dir(str(HERE / "attention"), remote_path=f"{REMOTE_ROOT}/attention")
)

results_vol = modal.Volume.from_name("triton-attention-results", create_if_missing=True)
kernel_cache_vol = modal.Volume.from_name("triton-attention-kernelcache", create_if_missing=True)

# B, Hq, Hkv, M, D, causal, tag
CORRECTNESS = [
    (1, 16, 16, 1024, 128, True, "mha_causal"),
    (1, 16, 4, 1024, 128, True, "gqa4_causal"),
    (1, 16, 16, 2048, 128, True, "mha2048_causal"),
    (1, 8, 8, 777, 96, True, "raggedM_hdim96_causal"),
    (2, 16, 16, 333, 80, True, "raggedM_hdim80_causal"),
    (1, 8, 8, 1000, 128, False, "raggedM_noncausal"),
    (1, 16, 4, 1000, 64, False, "gqa4_raggedM_noncausal"),
    (1, 8, 8, 512, 64, False, "aligned_noncausal"),
]

# B, Hq, Hkv, M, D, causal
PERF = [
    (1, 16, 16, 1024, 128, True),
    (1, 16, 16, 2048, 128, True),
    (1, 16, 16, 4096, 128, True),
    (1, 16, 4, 2048, 128, True),
    (1, 16, 2, 2048, 128, True),
    (1, 16, 16, 2048, 128, False),
]


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600,
    volumes={
        f"{REMOTE_ROOT}/results": results_vol,
        "/kernelcache": kernel_cache_vol,
    },
)
def ab():
    import time

    import torch

    from attention.reference import reference_attention
    from attention.triton_kernel import flash_attn_fwd_triton as v1
    from attention.triton_kernel_pipelined import flash_attn_fwd_triton as v2

    dev = torch.device("cuda")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"torch {torch.__version__} | triton kernel A/B: v1=while, v2=pipelined")

    def alloc(B, Hq, Hkv, M, D, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed)
        q = torch.randn(B, Hq, M, D, generator=g, device=dev, dtype=torch.float16)
        k = torch.randn(B, Hkv, M, D, generator=g, device=dev, dtype=torch.float16) * 0.5
        v = torch.randn(B, Hkv, M, D, generator=g, device=dev, dtype=torch.float16) * 0.5
        return q, k, v

    print("\n== correctness: max abs diff vs fused reference ==")
    ok_all = True
    for B, Hq, Hkv, M, D, causal, tag in CORRECTNESS:
        q, k, v = alloc(B, Hq, Hkv, M, D)
        ref = reference_attention(q, k, v, causal=causal)
        print(f"  {tag:26s} v1 ... ", end="", flush=True)
        d1 = (v1(q, k, v, causal=causal) - ref).abs().max().item()
        print("v2 ... ", end="", flush=True)
        d2 = (v2(q, k, v, causal=causal) - ref).abs().max().item()
        ok = d2 < 2e-2 and d1 < 2e-2
        ok_all = ok_all and ok
        print(f"v1={d1:.5f}  v2={d2:.5f}  {'OK' if ok else 'FAIL'}", flush=True)
    if not ok_all:
        raise AssertionError("correctness failed -- see table above")

    def timeit(fn, repeat=20, warmup=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / repeat * 1e3  # ms

    print("\n== perf: FA2 vs v1 (while) vs v2 (pipelined) ==")
    print(f"{'config':26s} {'FA2 TF/s':>9s} {'v1 TF/s':>9s} {'v2 TF/s':>9s} "
          f"{'v2/v1':>6s} {'v2/FA2':>7s}")
    rows = []
    for B, Hq, Hkv, M, D, causal in PERF:
        q, k, v = alloc(B, Hq, Hkv, M, D)
        tf = 4.0 * B * Hq * M * M * D / 1e12  # FA-paper flop convention
        t_fa = timeit(lambda: reference_attention(q, k, v, causal=causal), repeat=10)
        t1 = timeit(lambda: v1(q, k, v, causal=causal))
        t2 = timeit(lambda: v2(q, k, v, causal=causal))
        tag = f"M={M} D={D}" + (" GQA" if Hq != Hkv else "") + (" causal" if causal else "")
        fa, f1, f2 = tf / t_fa * 1e3, tf / t1 * 1e3, tf / t2 * 1e3
        print(f"{tag:26s} {fa:9.1f} {f1:9.1f} {f2:9.1f} {t1 / t2:5.2f}x {f2 / fa * 100:6.0f}%")
        rows.append((tag, t_fa, t1, t2, fa, f1, f2))

    csv_path = f"{REMOTE_ROOT}/results/ab_pipelined.csv"
    with open(csv_path, "w") as f:
        f.write("config,fa2_ms,v1_ms,v2_ms,fa2_tflops,v1_tflops,v2_tflops\n")
        for tag, t_fa, t1, t2, fa, f1, f2 in rows:
            f.write(f"{tag},{t_fa:.4f},{t1:.4f},{t2:.4f},{fa:.2f},{f1:.2f},{f2:.2f}\n")
    print(f"\nwrote {csv_path}")

    results_vol.commit()
    kernel_cache_vol.commit()


@app.local_entrypoint()
def main():
    print(f"launching A/B on Modal GPU={GPU} ...")
    ab.remote()
    print("done. pull artefacts with:")
    print("  modal volume get triton-attention-results / ./results")
