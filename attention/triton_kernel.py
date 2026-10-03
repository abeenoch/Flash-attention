"""FlashAttention-2-style forward attention kernel in Triton.

Per-program: loads one BLOCK_M x D block of Q, walks K/V in BLOCK_N-wide
blocks, and does an online-softmax accumulation that never materialises the
full MxM matrix (activation memory O(BLOCK_M*D + M*D), not O(M^2)).

Recurrence (m=running row-max, l=running prob-sum, o=running value-sum):
  scores = Q @ K_j^T * sm_scale        (BLOCK_M x BLOCK_N)
  m_j    = rowmax(scores);  p_j = exp(scores - m_j)
  m_new  = max(m, m_j);   alpha=exp(m-m_new); beta=exp(m_j-m_new)
  o      = o*alpha + (p_j*beta) @ V_j
  l      = l*alpha + sum(p_j)*beta ;  m = m_new
  out    = o / l

Triton 3.0 notes baked into this file:
  * tl.isfinite does not exist -> fully-masked rows are detected with
    (m_j > NEG_INF), where NEG_INF is a `tl.constexpr` module global.
  * tl.arange bounds must be powers of two -> the head dim is padded to
    BLOCK_D = next_power_of_2(D) and all d-lanes are masked with (d < D).

Layout: Q (B,Hq,M,D), K/V (B,Hkv,M,D); GQA maps query head h -> KV head h//groups.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _get_configs():
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
    ]


@triton.autotune(configs=_get_configs(), key=["D", "causal"])
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, Hq, M, Hkv,
    sm_scale, causal,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    kv_h = off_h // (Hq // Hkv)
    q_off = off_b * stride_qb + off_h * stride_qh
    k_off = off_b * stride_kb + kv_h * stride_kh
    v_off = off_b * stride_vb + kv_h * stride_vh
    o_off = off_b * stride_ob + off_h * stride_oh

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)

    # lane masks: tail of the sequence and (for non-pow2 D) tail of the head dim
    m_valid = m_offs < M
    d_valid = d_offs < D

    q = tl.load(
        Q + q_off + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd,
                mask=m_valid[:, None] & d_valid[None, :], other=0.0)
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Causal block skipping: query block pid_m covers query rows
    # [pid_m*BLOCK_M, pid_m*BLOCK_M + BLOCK_M). A key block starting at
    # block_id*BLOCK_N can only affect it while block_id*BLOCK_N <=
    # (pid_m+1)*BLOCK_M - 1. Skipping blocks that lie entirely above the
    # diagonal halves the FLOPs on causal attention (this is the optimisation
    # that lets FA2's *credited* throughput exceed the nominal metric).
    if causal:
        max_needed = pid_m * BLOCK_M + BLOCK_M  # smallest block start that matters
        if M < max_needed:
            max_needed = M
        n_blocks = tl.cdiv(max_needed, BLOCK_N)
    else:
        n_blocks = tl.cdiv(M, BLOCK_N)
    block_id = 0
    while block_id < n_blocks:
        n_start = block_id * BLOCK_N
        abs_n = n_start + n_offs
        k_mask = abs_n < M
        k = tl.load(K + k_off + abs_n[:, None] * stride_km + d_offs[None, :] * stride_kd,
                    mask=k_mask[:, None] & d_valid[None, :], other=0.0)
        v = tl.load(V + v_off + abs_n[:, None] * stride_vm + d_offs[None, :] * stride_vd,
                                        mask=k_mask[:, None] & d_valid[None, :], other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
        if causal:
            scores = tl.where(m_offs[:, None] >= abs_n[None, :], scores, float("-inf"))
        else:
            # Ragged tail (M % BLOCK_N != 0): OOB key lanes load as 0 and
            # would otherwise enter the softmax with score exactly 0.0.
            scores = tl.where(k_mask[None, :], scores, float("-inf"))
        m_j = tl.max(scores, axis=1)
        p = tl.exp(scores - m_j[:, None])
        valid = m_j > float("-inf")
        p = tl.where(valid[:, None], p, 0.0)
        p_sum = tl.sum(p, axis=1)
        m_new = tl.maximum(m, m_j)
        alpha = tl.where(m > float("-inf"), tl.exp(m - m_new), 0.0)
        beta = tl.where(valid, tl.exp(m_j - m_new), 0.0)
        pv = tl.dot(p.to(v.dtype), v, input_precision="ieee")
        o = o * alpha[:, None] + pv * beta[:, None]
        l = l * alpha + p_sum * beta
        m = m_new
        block_id += 1

    out = o / tl.where(l == 0.0, 1.0, l)[:, None]
    tl.store(Out + o_off + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
             out, mask=m_valid[:, None] & d_valid[None, :])


def flash_attn_fwd_triton(q, k, v, causal=True, scale=None):
    """Launch the Triton FlashAttention forward kernel (CUDA only)."""
    if not torch.cuda.is_available():
        raise RuntimeError("Triton attention kernel requires a CUDA GPU")
    B, Hq, M, D = q.shape
    _, Hkv, _, _ = k.shape
    assert Hq % Hkv == 0, "GQA requires Hq % Hkv == 0"
    assert k.shape[0] == B and v.shape[0] == B
    assert k.shape[3] == D and v.shape[3] == D
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    block_d = triton.next_power_of_2(D)
    grid = lambda meta: (B, Hq, triton.cdiv(M, meta["BLOCK_M"]))
    _flash_attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, Hq, M, Hkv, scale, causal, D, block_d)
    return out
