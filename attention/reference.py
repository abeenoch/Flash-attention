"""Reference and naive (baseline) attention in pure PyTorch.

  * `reference`  -> golden output via the framework's fused kernel
                    (cuDNN / FlashAttention-2) `F.scaled_dot_product_attention`.
  * `naive`      -> hand-rolled matmul+softmax (O(N^2) memory); both a second
                    correctness oracle and the slow baseline for benchmarks.

Math inside `naive` runs in float32, with the result cast back to the input
dtype, so softmax stays stable in fp16/fp32 and results match the reference.

Layouts everywhere: Q (B,Hq,M,D), K (B,Hkv,M,D), V (B,Hkv,M,D).
With Hq != Hkv this is grouped-query attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def reference_attention(q, k, v, causal=True, scale=None, dropout_p=0.0):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    B, Hq, M, D = q.shape
    Hkv = k.shape[1]
    if Hq != Hkv:
        groups = Hq // Hkv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        # Keep tensors 4-D (B, Hq, M, D): the fused SDPA backends (flash /
    # mem-efficient) require 4-D inputs and will silently fall back to the math
    # backend if given (B*H, M, D). k/v were already expanded to Hq heads above.
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=causal, dropout_p=dropout_p, scale=scale)
    return out.contiguous()


@torch.no_grad()
def naive_attention(q, k, v, causal=True, scale=None):
    """Naive (correct) attention: materialises the full MxM matrix. O(N^2)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    B, Hq, M, D = q.shape
    Hkv = k.shape[1]
    out_dtype = q.dtype
    if Hq != Hkv:
        groups = Hq // Hkv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    q = q.float()
    k = k.float()
    v = v.float()
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if causal:
        i = torch.arange(M, device=scores.device).unsqueeze(1)
        j = torch.arange(M, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(j > i, float("-inf"))
    scores = scores - scores.amax(dim=-1, keepdim=True)
    p = torch.exp(scores)
    p = p / p.sum(dim=-1, keepdim=True)
    out = torch.einsum("bhqk,bhkd->bhqd", p, v)
    return out.to(out_dtype)
