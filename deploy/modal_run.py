"""Run the triton-attention correctness suite + benchmarks on Modal (serverless GPU).

One-time setup:
    pip install modal
    modal setup                     # browser auth (sets up ~/.modal.toml)

Run:
    modal run deploy/modal_run.py                      # tests + bench on an L4 (~$0.80/hr)
    $env:ATTN_GPU='L40S'; modal run deploy/modal_run.py
    modal shell deploy/modal_run.py::run_and_bench      # interactive GPU shell
    modal volume get triton-attention-results / ./results   # fetch artefacts

Why these choices:
  * Image builds are CPU-only -> NOT billed as GPU time, and cached after the first
    run, so every later run starts warm.
  * TRITON_CACHE_DIR is set on the image and backed by a Volume, so the JIT-compiled
    kernels + autotune results survive teardown. Autotune compilation dominates the
    first run, so this makes iteration much faster and cheaper.
  * L4 is Ada Lovelace - same architecture family as the RTX 4090 - so kernel
    behaviour and tile-size tuning transfer. Absolute TFLOP/s differ (L4 has ~1/3
    the memory bandwidth), but speedups vs the baselines still hold.
  * Modal has no consumer 4090; the cheap options are T4 ($0.59/hr), L4 ($0.80/hr),
    A10 ($1.10/hr). Starter includes $30/mo compute = ~37 hours of L4.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent.parent  # project root
REMOTE_ROOT = "/root/project"
GPU = os.environ.get("ATTN_GPU", "L4")  # T4 | L4 | A10 | L40S | A100-80GB | H100

app = modal.App("triton-attention")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "triton==3.0.0", "numpy", "pytest", "matplotlib")
    .env({
        "TRITON_CACHE_DIR": "/kernelcache",
        "PYTHONPATH": REMOTE_ROOT,
    })
    .add_local_dir(str(HERE / "attention"), remote_path=f"{REMOTE_ROOT}/attention")
    .add_local_dir(str(HERE / "tests"), remote_path=f"{REMOTE_ROOT}/tests")
    .add_local_dir(str(HERE / "benches"), remote_path=f"{REMOTE_ROOT}/benches")
    .add_local_file(str(HERE / "requirements.txt"), remote_path=f"{REMOTE_ROOT}/requirements.txt")
)

results_vol = modal.Volume.from_name("triton-attention-results", create_if_missing=True)
kernel_cache_vol = modal.Volume.from_name("triton-attention-kernelcache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=3600,  # 1h hard stop so a hung run can never bill indefinitely
    volumes={
        f"{REMOTE_ROOT}/results": results_vol,
        "/kernelcache": kernel_cache_vol,
    },
)
def run_and_bench(pytest_args: str = "-v -rs -p no:cacheprovider"):
    """Run pytest, then the benchmark, and persist results to the Volume."""
    import subprocess
    import sys

    def sh(*args):
        print(f"\n$ {' '.join(args)}", flush=True)
        subprocess.run(list(args), cwd=REMOTE_ROOT, check=True)

    # sanity: confirm GPU + that the real Triton kernel path is live
    sh(sys.executable, "-c",
       "import torch, triton;"
       "print('torch', torch.__version__, '| triton', triton.__version__);"
       "print('gpu', torch.cuda.get_device_name(0));"
       "print('cuda', torch.cuda.is_available())")

    # diagnostic: surface the *full* Triton import traceback, never a silent skip
    sh(sys.executable, "-c",
       "import sys; sys.path.insert(0, '.');"
       "from attention.kernel import kernel_available, triton_import_error;"
       "print('triton kernel live:', kernel_available());"
       "err = triton_import_error();"
       "print('import traceback:' if err else 'no import error');"
       "print(err or '')")

    sh(sys.executable, "-m", "pytest", "tests/", *pytest_args.split())
    sh(sys.executable, "benches/bench_attention.py")

    results_vol.commit()        # persist results/bench.csv + roofline.png
    kernel_cache_vol.commit()   # persist compiled kernels for the next run


@app.local_entrypoint()
def main(pytest_args: str = "-v -p no:cacheprovider"):
    print(f"launching on Modal GPU={GPU} ...")
    run_and_bench.remote(pytest_args=pytest_args)
    print("done. pull artefacts with:")
    print("  modal volume get triton-attention-results / ./results")
