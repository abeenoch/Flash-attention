"""Benchmark: Triton kernel vs PyTorch FlashAttention vs naive, plus memory.

Writes results/bench.csv and results/roofline.png (a grouped bar chart).

Run after `bash setup_linux.sh` on a CUDA box, or via `modal run deploy/modal_run.py`.
On CPU-only it runs the naive-vs-reference comparison (Triton path is skipped).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from attention.kernel import flash_attn_fwd, kernel_available
from attention.reference import naive_attention, reference_attention


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BenchResult:
    backend: str
    B: int
    Hq: int
    Hkv: int
    M: int
    D: int
    causal: bool
    mean_ms: float
    tflops: float
    max_mem_mb: float
    ok: bool
    best_cfg: str = ""  # autotune winner for the triton backend, e.g. "BM128/BN64/w4/s3"


def _alloc(B, Hq, Hkv, M, D, dtype=torch.float16, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if DEVICE.type == "cpu":
        dtype = torch.float32
    q = torch.randn(B, Hq, M, D, generator=g, device=DEVICE, dtype=dtype)
    k = torch.randn(B, Hkv, M, D, generator=g, device=DEVICE, dtype=dtype) * 0.5
    v = torch.randn(B, Hkv, M, D, generator=g, device=DEVICE, dtype=dtype) * 0.5
    return q, k, v


def _throughput_tflops(B, Hq, M, D):
    # FA-paper convention: 4 * B * H * M * M * D total flops.
    return 4.0 * B * Hq * M * M * D / 1e12


def _activation_mem_mb(fn):
    """Peak memory allocated by this call alone (the activation footprint)."""
    if DEVICE.type != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / 1e6


def timeit(fn, repeat=20, warmup=3):
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e3


def _triton_winner(D, causal):
    """Return the autotune winner for this (D, causal) as a short string.

    Reads Triton's Autotuner cache directly (no extra benchmark runs): the
    kernel is keyed on ["D", "causal"], but the runtime key is a tuple of
    (D, causal, *dtypes) -- e.g. (128, True, 'torch.float16', ...) -- and
    holds ONE winning Config per (D, causal) pair, tuned on the first shape
    seen and reused for all later M/B with the same D. Matched by membership
    so key-order/dtype-suffix changes across Triton versions don't break the
    lookup. Returns "" on CPU / fallback.
    """
    try:
        from attention.triton_kernel_pipelined import _flash_attn_fwd_kernel
        cache = _flash_attn_fwd_kernel.cache  # {key-tuple: Config, ...}
        cfg = None
        try:
            cfg = cache.get((D, bool(causal)))
        except Exception:
            cfg = None
        if cfg is None:
            # Key order/repr differs across Triton versions (e.g. the
            # decorator lists key=["D", "causal"] but the runtime may store
            # (causal, D) or normalised constexpr values). Match by
            # membership instead of exact tuple equality.
            for k, v in cache.items():
                try:
                    vals = tuple(k)
                except TypeError:
                    continue
                has_d = D in vals
                has_c = (causal in vals or bool(causal) in vals
                         or int(causal) in vals or str(bool(causal)) in vals)
                if has_d and has_c:
                    cfg = v
                    break
        if cfg is None and cache:
            # Fallback: single-entry cache from an identical-shape run.
            cfg = next(iter(cache.values()))
        if cfg is None:
            return ""
        kw = cfg.kwargs
        return (f"BM{kw.get('BLOCK_M')}/BN{kw.get('BLOCK_N')}/"
                f"w{cfg.num_warps}/s{cfg.num_stages}")
    except Exception:
        return ""


def run_case(B, Hq, Hkv, M, D, causal, with_naive=True):
    q, k, v = _alloc(B, Hq, Hkv, M, D)
    tf = _throughput_tflops(B, Hq, M, D)
    results = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        ref_call = lambda: reference_attention(q, k, v, causal=causal)
        t = timeit(ref_call, repeat=10)
        results.append(BenchResult("flashattn", B, Hq, Hkv, M, D, causal, t, tf / t * 1e3,
                                   _activation_mem_mb(ref_call), True))

    if with_naive:
        naive_call = lambda: naive_attention(q, k, v, causal=causal)
        t = timeit(naive_call, repeat=5)
        results.append(BenchResult("naive", B, Hq, Hkv, M, D, causal, t, tf / t * 1e3,
                                   _activation_mem_mb(naive_call), True))

    if torch.cuda.is_available() and kernel_available():
        torch.cuda.empty_cache()
        tri_call = lambda: flash_attn_fwd(q, k, v, causal=causal)
        # Warmup (also triggers autotune) BEFORE reading the winner below.
        # OutOfResources guard: a brand-new (D, causal) key can hold configs
        # that overflow shared memory (seen at D=256 on L4). Triton then raises
        # at launch instead of falling back, so record ok=False and continue
        # with the remaining cases rather than killing the whole bench run.
        try:
            t = timeit(tri_call, repeat=20)
            results.append(BenchResult("triton", B, Hq, Hkv, M, D, causal, t, tf / t * 1e3,
                                       _activation_mem_mb(tri_call), True,
                                       _triton_winner(D, causal)))
        except Exception as e:
            print(f"    triton FAILED on (B={B},M={M},D={D},causal={causal}): "
                  f"{type(e).__name__}: {e}", flush=True)
            results.append(BenchResult("triton", B, Hq, Hkv, M, D, causal,
                                       float("nan"), 0.0, float("nan"), False, ""))

    return results


CASES = [
    (1, 16, 16, 1024, 128, True),    # prefill, MHA
    (1, 16, 16, 2048, 128, True),    # long prefill
    (1, 16, 16, 4096, 128, True),    # longer prefill
    (1, 16, 4, 2048, 128, True),     # GQA long prefill
    (1, 16, 16, 8192, 128, True, False),   # 8k long-context (naive OOMs: skipped)
    (1, 16, 16, 16384, 128, True, False),  # 16k long-context (naive skipped)
    (8, 16, 16, 128, 128, False),    # batched decode (non-causal)
    (1, 16, 16, 512, 96, True),      # non-power-of-2 D
    # --- B-sweep: batch scaling at fixed M/D (grid-size & occupancy effects) ---
    (4, 16, 16, 2048, 128, True),    # batch-4 prefill
    (16, 16, 16, 2048, 128, True),   # batch-16 prefill
    # --- D-sweep: head-dim scaling at fixed M/B (on-chip working set) ---
    (1, 16, 16, 2048, 64, True),     # narrow heads: tensor-core mapping, mem-bound?
    (1, 16, 16, 2048, 256, True),    # wide heads: register pressure, occupancy?
]

def main():
    print(f"device={DEVICE}  triton_kernel={'yes' if kernel_available() else 'no (fallback)'}")
    # Report which fused SDPA backends exist WITHOUT changing global state.
    if DEVICE.type == "cuda":
        b = torch.backends.cuda
        print(f"  flash_sdp_enabled       = {b.flash_sdp_enabled()}")
        print(f"  mem_efficient_sdp_enabled = {b.mem_efficient_sdp_enabled()}")
        print(f"  math_sdp_enabled          = {b.math_sdp_enabled()}")
        try:
            print(f"  cuda = {torch.version.cuda}, cudnn = {torch.backends.cudnn.version()}")
        except Exception:
            pass
    all_results = []
    for case in CASES:
        print(f"running {case} ...", flush=True)
        all_results.extend(run_case(*case))

    # CSV (best_cfg = autotune winner for the triton row, "" for other backends)
    csv_path = os.path.join(OUT_DIR, "bench.csv")
    with open(csv_path, "w") as f:
        f.write("backend,B,Hq,Hkv,M,D,causal,mean_ms,tflops,max_mem_mb,ok,best_cfg\n")
        for r in all_results:
            f.write(f"{r.backend},{r.B},{r.Hq},{r.Hkv},{r.M},{r.D},{r.causal},"
                    f"{r.mean_ms:.4f},{r.tflops:.2f},{r.max_mem_mb:.1f},{r.ok},{r.best_cfg}\n")
    print(f"wrote {csv_path}")

    # Grouped bar chart: one group per config, one colored bar per backend.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        order = ["flashattn", "naive", "triton"]
        colors = {"flashattn": "#4C72B0", "naive": "#DD8452", "triton": "#55A868"}
        labels = {"flashattn": "PyTorch FlashAttention", "naive": "naive torch (O(M^2))",
                  "triton": "this kernel (Triton)"}

        groups = {}
        cfg_of = {}  # group tag -> autotune winner string (triton rows only)
        for r in all_results:
            tag = f"B={r.B}, M={r.M}, D={r.D}" + (" GQA" if r.Hq != r.Hkv else "") + \
                  ("" if r.causal else " noncausal")
            groups.setdefault(tag, {})[r.backend] = r.tflops
            if r.backend == "triton" and r.best_cfg:
                cfg_of[tag] = r.best_cfg

        group_names = list(groups.keys())
        n_groups = len(group_names)
        n_back = len(order)
        x = np.arange(n_groups)
        width = 0.8 / n_back

        fig, ax = plt.subplots(figsize=(max(9, n_groups * 1.7), 6))
        for i, backend in enumerate(order):
            vals = [groups[g].get(backend, 0.0) for g in group_names]
            # annotate triton bars with the autotune winner for that config
            cfgs = [cfg_of.get(g, "") for g in group_names] if backend == "triton" else [""] * n_groups
            xs = x + (i - (n_back - 1) / 2.0) * width
            bars = ax.bar(xs, vals, width * 0.9, label=labels[backend],
                          color=colors[backend], edgecolor="black", linewidth=0.4)
            if backend in ("triton", "naive"):
                ymax = max(max(d.values()) for d in groups.values())
                for k, (b, v) in enumerate(zip(bars, vals)):
                    if v <= 0.0:
                        continue  # backend not run for this config (e.g. naive OOM at 8k+)
                    label = f"{v:.0f}" if backend == "triton" else f"{v:.1f}"
                    if backend == "triton" and cfgs[k]:
                        label += f"\n{cfgs[k]}"
                    ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.01, label,
                            ha="center", va="bottom", fontsize=7, fontweight="bold",
                            color="#2c5f2e" if backend == "triton" else "#8a3b12")

        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=18, ha="right", fontsize=9)
        ax.set_ylabel("Throughput (TFLOP/s)")
        ax.set_title("FlashAttention vs naive vs this Triton kernel — Modal L4, fp16")
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        png = os.path.join(OUT_DIR, "roofline.png")
        fig.savefig(png, dpi=140)
        print(f"wrote {png}")
    except Exception as e:
        print(f"(skipped plot: {e})")

    # pretty stdout table (best_cfg = autotune winner, triton rows)
    print()
    print(f"{'backend':10s} {'B':>2} {'Hq':>3} {'Hkv':>3} {'M':>5} {'D':>3} {'causal':>6} "
          f"{'ms':>9} {'TFLOP/s':>8} {'memMB':>8}  best_cfg")
    for r in all_results:
        print(f"{r.backend:10s} {r.B:>2} {r.Hq:>3} {r.Hkv:>3} {r.M:>5} {r.D:>3} "
              f"{str(r.causal):>6} {r.mean_ms:>9.3f} {r.tflops:>8.1f} {r.max_mem_mb:>8.0f}  {r.best_cfg}")


if __name__ == "__main__":
    main()

