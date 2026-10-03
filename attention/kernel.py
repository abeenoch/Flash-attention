"""Public dispatch entry point for the Triton attention kernel.

On a CUDA GPU this calls the FlashAttention-2-style Triton kernel.
On CPU (or where Triton is unavailable) it transparently falls back to the
naive torch implementation so the test/bench API is identical everywhere.
"""

from __future__ import annotations

import torch

from .reference import naive_attention

_TRITON_ERROR = None
_HAS_TRITON = False

if torch.cuda.is_available():
    try:
        import triton  # noqa: F401
        # v2: software-pipelined, two-phase KV walk. The original while-loop
        # kernel is kept in triton_kernel.py for the A/B (see dev_log.md).
        from .triton_kernel_pipelined import flash_attn_fwd_triton

        _HAS_TRITON = True
    except Exception:
        import traceback

        _TRITON_ERROR = traceback.format_exc()
else:
    _TRITON_ERROR = "torch.cuda.is_available() returned False"


def flash_attn_fwd(q, k, v, causal=True, scale=None):
    """Forward-only (prefill/decode-style) attention; no backward pass yet.

    Args:
        q: (B, Hq, M, D)
        k: (B, Hkv, M, D)
        v: (B, Hkv, M, D)
        causal: masked softmax.
        scale: 1/sqrt(D) by default.
    """
    if q.device.type == "cuda" and _HAS_TRITON:
        return flash_attn_fwd_triton(q, k, v, causal=causal, scale=scale)
    return naive_attention(q, k, v, causal=causal, scale=scale)


def kernel_available():
    """True when the real Triton kernel will run (i.e. not the CPU fallback)."""
    return _HAS_TRITON


def triton_import_error():
    """Full traceback explaining why the Triton kernel did not load (or None)."""
    return _TRITON_ERROR
