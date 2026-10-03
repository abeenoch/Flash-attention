"""Correctness tests: triton/naive vs the PyTorch FlashAttention reference.

Runs on CPU (Triton isn't available on Windows here), exercising the naive
path and the dispatch layer. On a Linux GPU box, the triton path is tested too.

Tolerance note: on GPU the tensors are float16, so tests use fp16-friendly
tolerances (atol/rtol ~2e-2). CPU runs are float32 but keep the same thresholds.
"""
from __future__ import annotations

import pytest
import torch

from attention.kernel import flash_attn_fwd, kernel_available, triton_import_error
from attention.reference import naive_attention, reference_attention


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TRITON_READY = torch.cuda.is_available() and kernel_available()


def _make_tensors(B, Hq, Hkv, M, D, seed=0, causal=None):
    # `causal` is accepted (and ignored here) so configs can be splatted directly.
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # CPU float16 maths is flaky for softmax; promote to float32 on CPU.
    dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
    q = torch.randn(B, Hq, M, D, device=DEVICE, generator=g, dtype=dtype)
    k = torch.randn(B, Hkv, M, D, device=DEVICE, generator=g, dtype=dtype) * 0.5
    v = torch.randn(B, Hkv, M, D, device=DEVICE, generator=g, dtype=dtype) * 0.5
    return q, k, v


CONFIGS = [
    dict(B=1, Hq=8, Hkv=8, M=16, D=64, causal=True),
    dict(B=1, Hq=8, Hkv=8, M=32, D=64, causal=True),
    dict(B=2, Hq=16, Hkv=16, M=64, D=128, causal=False),
    dict(B=1, Hq=16, Hkv=8, M=48, D=64, causal=True),   # GQA groups=2
    dict(B=1, Hq=16, Hkv=4, M=96, D=128, causal=True),  # GQA groups=4
    dict(B=1, Hq=8, Hkv=8, M=256, D=64, causal=True),   # long-ish prompt (CPU ok)
    dict(B=2, Hq=8, Hkv=8, M=100, D=80, causal=True),   # non-power-of-2 M & D
    dict(B=1, Hq=8, Hkv=8, M=128, D=96, causal=False),  # non-power-of-2 D, non-causal
    dict(B=1, Hq=8, Hkv=8, M=100, D=64, causal=False),  # ragged tail (M % 64 != 0), non-causal
    dict(B=1, Hq=16, Hkv=4, M=100, D=64, causal=False),  # GQA + ragged tail, non-causal
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_triton_matches_reference(cfg):
    if not _TRITON_READY:
        err = triton_import_error()
        pytest.skip(f"Triton kernel not available: {err}")
    q, k, v = _make_tensors(**cfg)
    out_ref = reference_attention(q, k, v, causal=cfg["causal"])
    out_kernel = flash_attn_fwd(q, k, v, causal=cfg["causal"])
    torch.testing.assert_close(out_kernel, out_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_naive_matches_reference(cfg):
    q, k, v = _make_tensors(**cfg)
    out_ref = reference_attention(q, k, v, causal=cfg["causal"])
    out_naive = naive_attention(q, k, v, causal=cfg["causal"])
    torch.testing.assert_close(out_naive, out_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_dispatch_matches_reference(cfg):
    q, k, v = _make_tensors(**cfg)
    out_ref = reference_attention(q, k, v, causal=cfg["causal"])
    out = flash_attn_fwd(q, k, v, causal=cfg["causal"])
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=2e-2)


def test_triton_import_clean():
    """Fail loudly (never silently skip) when CUDA is present but the kernel failed to load."""
    if torch.cuda.is_available() and not _TRITON_READY:
        pytest.fail(f"Triton kernel failed to import on a CUDA host:\n{triton_import_error()}")


def test_causal_no_future_leak():
    """For causal attention, query i must not depend on value rows > i.

    Zeroing value-row ``r`` must leave outputs for every query < r unchanged,
    while rows >= r are allowed to change. This is the defining correctness
    property of the causal mask.
    """
    B, H, M, D = 1, 2, 8, 16
    g = torch.Generator(device=DEVICE).manual_seed(7)
    q = torch.randn(B, H, M, D, device=DEVICE, generator=g)
    k = torch.randn(B, H, M, D, device=DEVICE, generator=g)
    v = torch.randn(B, H, M, D, device=DEVICE, generator=g)
    out = flash_attn_fwd(q, k, v, causal=True)
    r = M // 2
    v_zeroed = v.clone()
    v_zeroed[:, :, r, :] = 0.0
    out2 = flash_attn_fwd(q, k, v_zeroed, causal=True)
    # queries strictly before r never saw value row r -> unchanged.
    torch.testing.assert_close(out2[:, :, :r, :], out[:, :, :r, :], atol=2e-2, rtol=2e-2)


def test_shape_and_dtype():
    q, k, v = _make_tensors(B=1, Hq=4, Hkv=4, M=16, D=64, seed=3)
    out = flash_attn_fwd(q, k, v, causal=True)
    assert out.shape == q.shape
    assert out.dtype == q.dtype
