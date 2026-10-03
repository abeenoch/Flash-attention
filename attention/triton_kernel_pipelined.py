"""Software-pipelined variant of the FlashAttention-2-style Triton kernel (v2).

Two structural changes vs attention/triton_kernel.py (v1):

1. `while` -> `for`. Triton's software pipeliner only rewrites structured
   `for` loops (scf.for): it reschedules the loop so the NEXT iteration's K/V
   global-memory loads are issued while the CURRENT iteration's matmuls are
   still executing, hiding HBM latency behind tensor-core work. A `while`
   loop (scf.while) is opaque to that pass, so in v1 `num_stages` was
   silently inert and every K/V load's full latency was exposed.

2. Two-phase KV walk (the canonical FA2 structure). For a causal query
   block, key blocks entirely below the diagonal need neither the causal
   mask nor the row-bounds mask, so phase 1 is a branch-free hot loop that
   pipelines cleanly. Phase 2 handles the 1-2 diagonal blocks (causal) or
   the ragged tail block (non-causal) with full masks.

Implementation note (Triton 3.0): the loop lives in a `@triton.jit` helper
called twice, and iterates `for start_n in range(lo, hi, BLOCK_N)` directly
over key offsets -- mirroring the official fused-attention tutorial. An
earlier inline two-loop version (`for blk in range(0, n)` with the block
index multiplied inside the body) crashed the Triton 3.0 pipeliner with an
LLVM SmallVector assertion (SIGABRT at compile time). The helper structure
is the heavily-tested path.

Also fixes a latent v1 bug: in non-causal mode with M % BLOCK_N != 0, the
out-of-bounds key lanes of the tail block loaded as 0 and entered the
softmax with score exactly 0.0, corrupting the denominator. The masked
phase masks those lanes to -inf here; v1 got the same one-line fix.

Public API identical to v1: flash_attn_fwd_triton(q, k, v, causal, scale).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _get_configs():
    # num_stages actually matters now: it is the pipeline depth, i.e. how
    # many iterations ahead the K/V loads are prefetched.
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        # Narrow-N configs for wide heads (D=256): the pipelined K/V tiles are
        # BLOCK_D x BLOCK_N, so halving BLOCK_N halves the staged working set.
        # At D=256 every 64/128-wide config overflows L4 shared memory, so
        # without these the whole space is infeasible and autotune has no
        # valid winner (found by the D-sweep: Required 106496 > 101376 limit).
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=3, num_warps=4),
    ]


@triton.jit
def _attn_fwd_inner(
    o, l, m, q,
    K, V, k_off, v_off,
    stride_km, stride_kd, stride_vm, stride_vd,
    m_offs, n_offs, d_offs, d_valid, limit,
    M, sm_scale, lo, hi,
    MASKED: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # The `for` loop is the software-pipelined hot path. It iterates over
    # key offsets directly, rather than over block indices, so the loop body can be a single, self-contained kernel that Triton can pipeline. The `MASKED` constexpr is specialized to True for the
    # second phase of the KV walk (the 1-2 diagonal blocks for causal, or the ragged tail block for non-causal), and False for the first phase (the unmasked hot loop). The `lo` and `hi` bounds are rounded out to whole BLOCK_N blocks, so the masked phase can handle the ragged tail without a runtime branch in the loop.
    for start_n in range(lo, hi, BLOCK_N):
        abs_n = start_n + n_offs
        if MASKED:
            k_mask = abs_n < M
            # K is loaded pre-transposed as (BLOCK_D, BLOCK_N) so the dot
            # needs no tl.trans: tl.trans(tl.load(...)) inside a pipelined
            # `for` loop crashes Triton 3.0's pipeliner (LLVM SmallVector
            # assertion, SIGABRT at compile time).
            k = tl.load(K + k_off + d_offs[:, None] * stride_kd + abs_n[None, :] * stride_km,
                        mask=d_valid[:, None] & k_mask[None, :], other=0.0)
            v = tl.load(V + v_off + abs_n[:, None] * stride_vm + d_offs[None, :] * stride_vd,
                        mask=k_mask[:, None] & d_valid[None, :], other=0.0)
            scores = tl.dot(q, k, input_precision="ieee") * sm_scale
            # One combined, branch-free mask: key must be in bounds AND at or
            # below this row's visibility limit (limit = row idx when causal,
            # M-1 when not).
            keep = (abs_n[None, :] <= limit[:, None]) & k_mask[None, :]
            scores = tl.where(keep, scores, float("-inf"))
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
        else:
            k = tl.load(K + k_off + d_offs[:, None] * stride_kd + abs_n[None, :] * stride_km,
                        mask=d_valid[:, None], other=0.0)
            v = tl.load(V + v_off + abs_n[:, None] * stride_vm + d_offs[None, :] * stride_vd,
                        mask=d_valid[None, :], other=0.0)
            scores = tl.dot(q, k, input_precision="ieee") * sm_scale
            m_new = tl.maximum(m, tl.max(scores, axis=1))
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new[:, None])
            pv = tl.dot(p.to(v.dtype), v, input_precision="ieee")
            o = o * alpha[:, None] + pv
            l = l * alpha + tl.sum(p, axis=1)
            m = m_new
    return o, l, m


@triton.autotune(configs=_get_configs(), key=["D", "causal"])
@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, Hq, M, Hkv,
    sm_scale,
    causal: tl.constexpr,
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
    # Per-row visibility limit: row index when causal, M-1 when not. Lets the
    # masked phase use one combined mask with no runtime branch in the loop.
    # causal is constexpr: as a *runtime* scalar, the scf.if it creates makes
    # the phase-2 loop bounds non-affine and crashes Triton 3.0's pipeliner
    # (LLVM SmallVector assertion, SIGABRT at compile time). Only the
    # causal=1 specialization survived, because Triton specializes int args
    # equal to 1 to a compile-time constant.
    if causal:
        limit = m_offs
    else:
        limit = m_offs * 0 + (M - 1)

    q = tl.load(
        Q + q_off + m_offs[:, None] * stride_qm + d_offs[None, :] * stride_qd,
        mask=m_valid[:, None] & d_valid[None, :], other=0.0)
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Phase split (key offsets, both rounded out to whole BLOCK_N blocks).
    # hi_full = keys needing NO row masks:
    #   causal:     blocks entirely below the diagonal -- every key index
    #               < pid_m*BLOCK_M is visible to every row of this block.
    #   non-causal: blocks entirely inside [0, M).
    # hi_total = furthest key block that can affect this query block (v1's
    # causal block-skipping bound) = phase 1 + the 1-2 diagonal/tail blocks.
    if causal:
        max_needed = pid_m * BLOCK_M + BLOCK_M
        if M < max_needed:
            max_needed = M
        hi_total = tl.cdiv(max_needed, BLOCK_N) * BLOCK_N
        hi_full = ((pid_m * BLOCK_M) // BLOCK_N) * BLOCK_N
    else:
        hi_total = tl.cdiv(M, BLOCK_N) * BLOCK_N
        # hi_full <= hi_total always, so this clamp is mathematically a no-op
        # -- but it makes the bound opaque to MLIR affine analysis. Triton
        # 3.0's pipeliner has a flaky LLVM SmallVector OOB crash (SIGABRT at
        # compile time) when analysing pipelined loops whose bounds are pure
        # affine functions of scalar args (this non-causal instantiation);
        # the causal bounds above depend on pid_m (non-affine) and never
        # crash. A plain `+ pid_m * 0` would be constant-folded; min() is not.
        hi_full = tl.minimum((M // BLOCK_N) * BLOCK_N, hi_total)

    # phase 1: unmasked hot loop (pipelines aggressively)
    o, l, m = _attn_fwd_inner(
        o, l, m, q, K, V, k_off, v_off,
        stride_km, stride_kd, stride_vm, stride_vd,
        m_offs, n_offs, d_offs, d_valid, limit,
        M, sm_scale, 0, hi_full,
        False, BLOCK_N)

    # phase 2: diagonal blocks (causal) / ragged tail (non-causal)
    o, l, m = _attn_fwd_inner(
        o, l, m, q, K, V, k_off, v_off,
        stride_km, stride_kd, stride_vm, stride_vd,
        m_offs, n_offs, d_offs, d_valid, limit,
        M, sm_scale, hi_full, hi_total,
        True, BLOCK_N)

    out = o / tl.where(l == 0.0, 1.0, l)[:, None]
    tl.store(Out + o_off + m_offs[:, None] * stride_om + d_offs[None, :] * stride_od,
             out, mask=m_valid[:, None] & d_valid[None, :])


def flash_attn_fwd_triton(q, k, v, causal=True, scale=None):
    """Launch the pipelined Triton FlashAttention forward kernel (CUDA only)."""
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

