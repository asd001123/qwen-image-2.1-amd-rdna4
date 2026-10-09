"""gfx1200 transposed-weight layout patch for ComfyUI.

WHY THIS EXISTS
===============

On AMD RDNA4 (gfx1200, e.g. RX 9060 XT) with ROCm 7.2 under WSL2, `hipBLASLt`
fails every GEMM where an operand is used **transposed** and silently falls back
to a path that is ~400x slower::

    bgemm_internal_cublaslt error: HIPBLAS_STATUS_INTERNAL_ERROR when calling
    hipblasLtMatmul with transpose_mat1 1 ... Will attempt to recover by
    calling cublas instead.

Measured on RX 9060 XT, 4096x12288x4096 fp16::

    F.linear(x, w)                    2392 ms
    x @ w.t()                         2383 ms
    torch.matmul(x, w.t())            2687 ms
    x @ w.t().contiguous()               6 ms   <-- only fast form

`torch.nn.Linear` / `F.linear` store weights as ``[out_features, in_features]``
and therefore hit the slow path for *every* layer of a transformer. A diffusion
sampler then takes hours per image instead of minutes.

THE FIX
=======

Hand `F.linear` an ``[in_features, out_features]`` **contiguous** weight and let
it use a plain `torch.matmul`. Since only the caller (the GGUF dequantizer)
knows when it has done the transpose, the weight is tagged via a tensor
attribute and `F.linear` checks that tag.

MEMORY NEUTRALITY
=================

The transposed buffer *replaces* the original rather than being added beside
it (the original is dropped as soon as the tagged copy is created), so the
steady-state footprint is unchanged. This matters: naive caching of
``weight.t().contiguous()`` per layer doubles weight memory and pushes a 13 GiB
model past a 16 GiB card's ceiling, producing ``hipBLASLt OOM (192 MiB)``.

Correctness: ``torch.allclose`` against the original path holds (max abs diff
~1.6e-2 on fp16, which is accumulated rounding, not a logic difference).

USAGE
=====

Import and call :func:`apply` before inference, then tag weights with
:func:`mark_transposed` as they are dequantized. The companion patch to
``ComfyUI-GGUF/ops.py`` does the tagging automatically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["apply", "revert", "mark_transposed", "is_transposed", "is_applied"]

_FLAG = "_gfx1200_layout_patch"
_ATTR = "_gfx1200_transposed_inout"

_orig_linear = F.linear

# Fallback registry for the rare case where attribute assignment on a tensor
# is refused (some wrapper/subclass tensors forbid new attributes).
_fallback_identity: list = []


def mark_transposed(weight: torch.Tensor) -> torch.Tensor:
    """Tag ``weight`` as stored ``[in_features, out_features]`` and contiguous.

    Returns the same tensor so it can be used inline.
    """
    try:
        setattr(weight, _ATTR, True)
    except Exception:
        _fallback_identity.append(weight)
    return weight


def is_transposed(weight) -> bool:
    """True if ``weight`` was produced by :func:`mark_transposed`."""
    if getattr(weight, _ATTR, False):
        return True
    return any(w is weight for w in _fallback_identity)


def _fast_linear(input, weight, bias=None):
    if (
        weight.dim() == 2
        and input.is_cuda
        and weight.is_cuda
        and getattr(weight, _ATTR, False)
    ):
        try:
            out = torch.matmul(input, weight)
            if bias is not None:
                out = out + bias
            return out
        except Exception:
            # Never let an optimisation break a run.
            pass
    return _orig_linear(input, weight, bias)


def is_applied() -> bool:
    return getattr(F, _FLAG, False)


def apply() -> bool:
    """Monkeypatch ``F.linear``. Returns True if newly applied."""
    if is_applied():
        return False
    F.linear = _fast_linear
    setattr(F, _FLAG, True)
    print("[gfx1200] transposed-weight layout patch applied")
    return True


def revert() -> None:
    """Restore the original ``F.linear``."""
    F.linear = _orig_linear
    setattr(F, _FLAG, False)
    _fallback_identity.clear()
    print("[gfx1200] transposed-weight layout patch reverted")


def _selftest() -> int:
    import time

    if not torch.cuda.is_available():
        print("SKIP: no CUDA/HIP device")
        return 0

    x = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    w = torch.randn(12288, 4096, dtype=torch.float16, device="cuda")

    def bench(fn, iters=10):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000

    print("  [1] slow path -- untagged [out, in] weight")
    slow = bench(lambda: _orig_linear(x, w))
    print(f"      F.linear(x, w)                : {slow:9.2f} ms")

    print("  [2] fast path -- tagged [in, out] contiguous weight")
    wt = mark_transposed(w.t().contiguous())
    apply()
    fast = bench(lambda: torch.matmul(x, wt))
    print(f"      matmul(x, wt)                 : {fast:9.2f} ms")

    print("  [2b] patched F.linear with the tagged weight")
    # The patched F.linear computes matmul(input, weight). Passing the tagged
    # [in, out] weight therefore matches what the GGUF patch does at runtime.
    fast_via_linear = bench(lambda: F.linear(x, wt))
    print(f"      F.linear(x, wt)               : {fast_via_linear:9.2f} ms")
    print(f"      tag recognised                : {is_transposed(wt)}")

    print("  [3] an untagged weight must still take the original path correctly")
    ref_out = _orig_linear(x, w)
    same_untagged = torch.allclose(F.linear(x, w), ref_out, atol=1e-2, rtol=1e-2)
    print(f"      untagged allclose vs original : {same_untagged}")

    print("  [4] numerics of the tagged (fast) path")
    # F.linear(x, w) == matmul(x, w.t()); the GGUF patch stores w.t() contiguous
    # and tags it, so this is exactly the arithmetic the model performs.
    good = torch.allclose(torch.matmul(x, wt), ref_out, atol=1e-2, rtol=1e-2)
    print(f"      tagged allclose vs original   : {good}")

    print()
    if not (same_untagged and good):
        print("RESULT: FAIL -- numerics differ; do not use this patch here.")
        return 1
    if slow < 100:
        print(f"RESULT: OK, but this GPU does not exhibit the bug "
              f"({slow:.1f} ms is already fast). Patch is unnecessary.")
        return 0
    print(f"RESULT: OK -- {slow / max(fast, 1e-6):.0f}x speedup on the affected shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
