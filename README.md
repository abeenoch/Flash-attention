# triton-attention

A from-scratch, **FlashAttention-2-style** attention kernel implemented in **Triton**, with grouped-query attention (GQA), causal masking, online softmax, and a numerically-stable accumulation that never materializes the full `O(N²)` attention matrix.

This is a portfolio / learning project oriented toward **LLM inference infrastructure** work. The goals:

1. A **correct** kernel — verified against `torch.nn.functional.scaled_dot_product_attention` (the cuDNN/FlashAttention reference) across many configs on CPU and GPU.
2. A **fast** kernel — benchmarked vs PyTorch SDPA (and a naive matmul implementation) with a roofline-style writeup.
3. A **clear story** — every design choice (tiling, masking, softmax recurrence) documented so it's defensible in interview.

## Results (verified on a Modal L4, fp16)

33/33 correctness tests pass against the fused PyTorch SDPA reference — across
MHA and GQA (groups 2 and 4), causal and non-causal, and non-power-of-2 head
dims (D=80, 96).

After two diagnosed optimisations — causal block-skipping (1.6–1.8×) and a
software-pipelined two-phase KV walk (a further 1.05–1.27×) — the kernel's
**v2 / FA2 throughput ratios, as min–max ranges over three independent L4 runs**
(per-run tables in `docs/DEVELOPMENT_LOG.md`):

| config | v2 / FA2 (3-run range) | note |
|---|---|---|
| GQA g4, M=2048 causal | **97–104%** | ties or beats FA2 |
| D=64, M=2048 causal | **101–139%** | tie-or-better; margin is FA2 noise |
| M=4096 causal | 83–88% | |
| M=2048 causal | 70–82% | |
| M=1024 causal | 73–79% | |
| M=16384 causal | 65–66% | long-context (falloff vs 8k open question) |
| M=8192 causal | 59–74% | |
| B=16, M=2048 causal | 53–58% | batch scaling: gap is memory-bound |
| D=256, M=2048 causal | 54–57% | wide heads: smem-capped tiles (structural) |
| hdim96 (padded), M=512 | 48–51% | masked-lane tax (structural) |

**Measurement honesty:** all numbers come from *shared* Modal L4 instances
where both backends vary ±10–15% run-to-run and independently of each other —
that is why results are ranges, not single ratios. A dedicated-GPU re-run and
`ncu` profiling were planned and **retracted; no dedicated-hardware numbers
exist.** Causal TFLOP/s use the FA-paper convention (4·B·H·M²·D) and are
inflated ~2× vs actually-executed ops; ratios are unaffected.

**Memory:** O(M·D) out to M=16384 — **67 MB** where naive needs **~53 GB**
(3.3 GB → 17 MB at M=4096; 13.7 GB → 134 MB at B=16/M=2048).



## Project layout

```
attention/
  reference.py      # PyTorch reference (fused SDPA) + naive O(N^2) implementation
  triton_kernel.py          # v1: FlashAttention-2 fwd, while-loop (kept for the A/B)
  triton_kernel_pipelined.py  # v2 (production): software-pipelined, two-phase KV walk
  kernel.py         # public entry point: picks Triton on GPU, naive on CPU
tests/
  test_correctness.py   # pytest: triton/naive vs torch reference, 33 configs
benches/
  bench_attention.py    # perf + memory benchmarks (12 configs), writes CSV + plots
deploy/
  modal_run.py          # tests + bench on Modal (serverless L4/GPU)
  ab_pipelined.py       # v1 vs v2 vs FA2 A/B tool
  probe_autotune.py     # prints true autotune winners (TRITON_PRINT_AUTOTUNING=1)
  runpod_setup.sh       # (optional, not run) dedicated-GPU runbook + ncu
docs/
  DEVELOPMENT_LOG.md    # full engineering log: bugs, diagnosis, all measurements
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

## Running on RunPod (optional — not used for any reported result)

> **Status:** the dedicated-GPU re-run and `ncu` profiling were planned and
> **retracted** — every number in this README comes from shared Modal L4
> runs (see ranges above). This section is kept as a runbook for anyone who
> wants to profile or reproduce on dedicated hardware.

RunPod gives you a real pod with SSH/root, which is what you want for `ncu`/`nsys`
profiling of the kernel:

1. Create a pod with the **PyTorch 2.x** template (torch+CUDA preinstalled, so you
   don't pay GPU rates for a pip download), attach a network volume at `/workspace`,
   pick an **RTX 4090** (Community ~$0.34/hr, Secure ~$0.74/hr).
2. In the pod: `REPO_URL=<your-remote> bash deploy/runpod_setup.sh`
3. **Stop the pod when done** — you are billed while it is up:
   `runpodctl stop pod "$RUNPOD_POD_ID"`

