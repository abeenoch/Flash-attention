# triton-attention

A from-scratch, **FlashAttention-2-style** attention kernel implemented in **Triton**, with grouped-query attention (GQA), causal masking, online softmax, and a numerically-stable accumulation that never materializes the full `O(N²)` attention matrix.

This is a portfolio / learning project oriented toward **LLM inference infrastructure** work. The goals:

1. A **correct** kernel — verified against `torch.nn.functional.scaled_dot_product_attention` (the cuDNN/FlashAttention reference) across many configs on CPU and GPU.
2. A **fast** kernel — benchmarked vs PyTorch SDPA (and a naive matmul implementation) with a roofline-style writeup.
3. A **clear story** — every design choice (tiling, masking, softmax recurrence) documented so it's defensible in interview.

## Results (verified on a Modal L4, fp16)

33/33 correctness tests pass against the fused PyTorch SDPA reference. After two
diagnosed optimisations — causal block-skipping (1.6–1.8×) and a
software-pipelined two-phase KV walk (a further 1.05–1.27×) — the kernel reaches
**66–91% of PyTorch FA2** on prefill configs (best on GQA: 91% at M=2048), and
holds O(M·D) memory out to M=16384: **67 MB** where a naive implementation would
need **~53 GB** (measured 3.3 GB → 17 MB at M=4096).

See **[docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md)** for the full measured
results tables and a log of every bug hit — including three distinct Triton 3.0
software-pipeliner compiler crashes (found by bisection, worked around without
giving up the optimisation) and a latent ragged-tail correctness bug caught
during the v2 audit.

## Project layout

```
attention/
  reference.py      # PyTorch reference (fused SDPA) + naive O(N^2) implementation
  triton_kernel.py          # v1: FlashAttention-2 fwd, while-loop (kept for the A/B)
  triton_kernel_pipelined.py  # v2 (production): software-pipelined, two-phase KV walk
  kernel.py         # public entry point: picks Triton on GPU, naive on CPU
tests/
  test_correctness.py   # pytest: triton/naive vs torch reference, many configs
benches/
  bench_attention.py    # perf + memory benchmarks, writes CSV + plots
setup_linux.sh       # one-command env setup for a Linux GPU box (RunPod / Vast)
```

## Quick start (CPU — no GPU needed for correctness)

```bash
pip install -r requirements.txt        # torch, numpy, pytest, matplotlib
python -m pytest tests/ -v             # runs the full correctness suite on CPU
```

The Triton kernel is imported lazily; on a CPU-only host the dispatch falls back to the
naive torch implementation, so the **test suite still validates the API and reference
comparison** without any GPU.

## Quick start (Linux + GPU)

```bash
bash setup_linux.sh      # installs torch+cuda, triton, bench deps
python -m pytest tests/ -v
python benches/bench_attention.py      # perf vs baseline, roofline plot
```

## Run on the cloud (cheapest path)

The kernel needs Linux + an NVIDIA GPU. Easiest: a `g4dn.xlarge` (V100) or any
Linux 4090 box, ~$0.25–0.40/hr. See the "Cloud runbook" section below.

## Cloud runbook

1. Spin up a Linux GPU box (RunPod / Vast / Lambda Labs / Modal / your favorite).
2. `git clone` this repo, `bash setup_linux.sh`.
3. `python -m pytest tests/ -v` → all green = correctness verified.
4. `python benches/bench_attention.py` → `results/` CSV + PNG.
5. Iterate on tile sizes / `BLOCK_M` / `BLOCK_N` via `@triton.autotune`.

## Running on Modal (serverless, recommended default)

Modal bills per second of actual compute, image builds are CPU-only (not billed as
GPU time) and cached, and the Starter plan includes $30/month of compute (~37 hours
of an L4). There is no idle billing, so this is the cheapest way to iterate.

```bash
pip install modal
modal setup                                  # one-time browser auth
modal run deploy/modal_run.py                # tests + benchmark on an L4
modal run deploy/sweep.py                    # parallel sequence-length sweep
modal volume get triton-attention-results / ./results
```

`deploy/modal_run.py` also sets `TRITON_CACHE_DIR` on a Volume, so JIT-compiled
kernels and autotune results persist across runs (autotune compilation dominates
the first run). Choose the GPU with `ATTN_GPU` (T4 | L4 | A10 | L40S | A100-80GB | H100).
L4 is Ada Lovelace, the same architecture family as the RTX 4090, so tile-size
tuning transfers between the two.

## Running on RunPod (interactive / profiling)

RunPod gives you a real pod with SSH/root, which is what you want for `ncu`/`nsys`
profiling of the kernel:

1. Create a pod with the **PyTorch 2.x** template (torch+CUDA preinstalled, so you
   don't pay GPU rates for a pip download), attach a network volume at `/workspace`,
   pick an **RTX 4090** (Community ~$0.34/hr, Secure ~$0.74/hr).
2. In the pod: `REPO_URL=<your-remote> bash deploy/runpod_setup.sh`
3. **Stop the pod when done** — you are billed while it is up:
   `runpodctl stop pod "$RUNPOD_POD_ID"`

