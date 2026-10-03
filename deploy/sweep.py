"""Fan out benchmark configs across Modal's concurrent GPUs (parallel sweep).

Each config runs in its own container on its own GPU, so a whole sequence-length
sweep finishes in roughly the wall-clock of a single config instead of N x that.
The Starter plan allows 10 concurrent GPUs.

Run:
    modal run deploy/sweep.py
    $env:ATTN_GPU='L40S'; modal run deploy/sweep.py

Outputs:
    results/sweep.csv   (written locally, and committed to the Volume)
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/project"
GPU = os.environ.get("ATTN_GPU", "L4")
LOCAL_OUT = HERE / "results"

app = modal.App("triton-attention-sweep")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "triton==3.0.0", "numpy", "pytest", "matplotlib")
    .env({"TRITON_CACHE_DIR": "/kernelcache"})
    .add_local_dir(str(HERE / "attention"), remote_path=f"{REMOTE_ROOT}/attention")
    .add_local_dir(str(HERE / "benches"), remote_path=f"{REMOTE_ROOT}/benches")
)

results_vol = modal.Volume.from_name("triton-attention-results", create_if_missing=True)
kernel_cache_vol = modal.Volume.from_name("triton-attention-kernelcache", create_if_missing=True)

# (tag, B, Hq, Hkv, M, D, causal) - sequence-length sweep plus satellites.
SWEEP = [
    ("seqlen_512", 1, 16, 16, 512, 128, True),
    ("seqlen_1024", 1, 16, 16, 1024, 128, True),
    ("seqlen_2048", 1, 16, 16, 2048, 128, True),
    ("seqlen_4096", 1, 16, 16, 4096, 128, True),
    ("gqa_g4_2048", 1, 16, 4, 2048, 128, True),
    ("gqa_g2_2048", 1, 16, 8, 2048, 128, True),
    ("decode_b16_128", 16, 16, 16, 128, 128, False),
    ("hdim96_1024", 1, 16, 16, 1024, 96, True),
    ("noncausal_2048", 1, 16, 16, 2048, 128, False),
]


@app.function(
    image=image,
    gpu=GPU,
    timeout=2400,
    volumes={
        f"{REMOTE_ROOT}/results": results_vol,
        "/kernelcache": kernel_cache_vol,
    },
)
def bench_one(cfg):
    """Run one sweep config on this container's GPU and return its rows."""
    import sys

    sys.path.insert(0, REMOTE_ROOT)
    tag, B, Hq, Hkv, M, D, causal = cfg
    from benches.bench_attention import run_case

    rows = [r.__dict__ for r in run_case(B, Hq, Hkv, M, D, causal)]
    for r in rows:
        r["tag"] = tag
    print(f"[{tag}] " + ", ".join(f"{r['backend']}={r['tflops']:.1f}TFLOP/s" for r in rows), flush=True)
    return rows


@app.local_entrypoint()
def main():
    print(f"fanning out {len(SWEEP)} configs on Modal GPU={GPU} ...", flush=True)
    # .map() runs each element in its own container, concurrently (10 GPUs on Starter).
    packed = list(bench_one.map(SWEEP))
    all_rows = [row for rows in packed for row in rows]

    # A prior `modal volume get ... ./results` can squash the whole volume into a
    # single FILE named "results"; move it aside before making the directory.
    if LOCAL_OUT.is_file():
        LOCAL_OUT.rename(LOCAL_OUT.with_name("results_volume_dump.bin"))
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    csv_path = LOCAL_OUT / "sweep.csv"
    cols = ["tag", "backend", "B", "Hq", "Hkv", "M", "D", "causal", "mean_ms", "tflops", "max_mem_mb"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {csv_path}")

    print(f"\n{'tag':16s} {'backend':11s} {'M':>5} {'D':>4} {'ms':>9} {'TFLOP/s':>8} {'memMB':>7}")
    for r in sorted(all_rows, key=lambda x: (x["tag"], x["backend"])):
        print(f"{r['tag']:16s} {r['backend']:11s} {r['M']:>5} {r['D']:>4} "
              f"{r['mean_ms']:>9.3f} {r['tflops']:>8.1f} {r['max_mem_mb']:>7.0f}")
