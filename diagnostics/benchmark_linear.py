#!/usr/bin/env python3
"""Measure the RDNA4 (gfx1200) transposed-operand GEMM anomaly.

This is the fastest way to tell whether your card is affected. On an affected
card, a transposed *view* is roughly 400x slower than the same data laid out
contiguously::

    F.linear(x, w)                    2590 ms
    torch.matmul(x, w.t())            2687 ms
    torch.matmul(x, w.t().contiguous())  6 ms

The point is not the absolute numbers -- it is the ratio. If your ratio is
close to 1, you do not need the layout patch.

Usage:
    python3 diagnostics/benchmark_linear.py
    python3 diagnostics/benchmark_linear.py --shape 2048 8192 2048 --iters 5
"""

from __future__ import annotations

import argparse
import statistics
import sys

try:
    import torch
except ImportError:
    print("torch is not installed")
    raise SystemExit(1)


def bench(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = __import__("time").perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((__import__("time").perf_counter() - t0) * 1000)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"),
                    default=[4096, 12288, 4096])
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA/HIP device visible to PyTorch.")
        return 1

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    m, n, k = args.shape

    print("=" * 66)
    print("gfx1200 transposed-GEMM benchmark")
    print("=" * 66)
    print(f"  device      : {torch.cuda.get_device_name(0)}")
    print(f"  arch        : {getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '?')}")
    print(f"  torch       : {torch.__version__}")
    print(f"  hip         : {torch.version.hip}")
    print(f"  dtype       : {args.dtype}")
    print(f"  shape (m,n,k): {m} x {n} x {k}")
    print(f"  cuda.is_available(): {torch.cuda.is_available()}")
    print()

    x = torch.randn(m, k, dtype=dtype, device="cuda")
    w = torch.randn(n, k, dtype=dtype, device="cuda")          # [out, in]

    flops = 2 * m * n * k

    def report(label: str, ms: float) -> None:
        tf = flops / (ms / 1000) / 1e12
        print(f"  {label:44s} {ms:9.2f} ms  {tf:7.2f} TFLOPS")

    print("timings (median of %d):" % args.iters)
    t_linear = bench(lambda: torch.nn.functional.linear(x, w), args.iters)
    report("F.linear(x, w)               [transposed]", t_linear)

    t_view = bench(lambda: torch.matmul(x, w.t()), args.iters)
    report("matmul(x, w.t())             [transposed]", t_view)

    wt = w.t().contiguous()
    t_contig = bench(lambda: torch.matmul(x, wt), args.iters)
    report("matmul(x, w.t().contiguous())  [contiguous]", t_contig)

    # Also check bf16, which is slow on this class of GPU for a different reason.
    print()
    print("dtype cross-check (contiguous operands, no transpose):")
    for dt, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16"),
                     (torch.float32, "fp32")):
        a = torch.randn(m, k, dtype=dt, device="cuda")
        b = torch.randn(k, n, dtype=dt, device="cuda")
        ms = bench(lambda a=a, b=b: torch.matmul(a, b), max(3, args.iters // 2))
        report(f"matmul {name} (contiguous)", ms)
        del a, b

    print()
    ratio = t_linear / max(t_contig, 1e-9)
    print("=" * 66)
    print(f"  transposed / contiguous ratio : {ratio:.1f}x")

    if ratio > 20:
        print("  VERDICT: your GPU IS affected by the hipBLASLt transposed-GEMM bug.")
        print("           Apply patches/gfx1200_layout_patch.py (install.sh does this).")
        rc = 0
    elif ratio > 3:
        print("  VERDICT: mild anomaly; the patch may still help.")
        rc = 0
    else:
        print("  VERDICT: no meaningful anomaly -- the patch is unnecessary here.")
        rc = 0
    print("=" * 66)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
