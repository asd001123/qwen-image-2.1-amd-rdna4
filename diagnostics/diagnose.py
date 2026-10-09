#!/usr/bin/env python3
"""Diagnose the three gfx1200/WSL pitfalls and report which affect this machine.

Runs each check independently, prints a verdict per issue, and exits non-zero if
something that should be fixed is not.

Usage:
    python3 diagnostics/diagnose.py
    python3 diagnostics/diagnose.py --comfy-path /opt/ComfyUI
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

OK = "\033[32m  OK  \033[0m"
BAD = "\033[31m FAIL \033[0m"
WARN = "\033[33m WARN \033[0m"
INFO = "      "

results: list[tuple[str, str, str]] = []   # (name, status, detail)


def record(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))


def section(title: str) -> None:
    print()
    print("=" * 68)
    print(f" {title}")
    print("=" * 68)


# ---------------------------------------------------------------- issue 1
def check_torch() -> bool:
    section("Issue 1 -- is PyTorch talking to the GPU at all?")

    try:
        import torch
    except ImportError:
        print(f"{BAD} torch is not installed")
        record("torch import", "FAIL", "not installed")
        return False

    print(f"{INFO}torch          : {torch.__version__}")
    print(f"{INFO}hip            : {torch.version.hip}")
    print(f"{INFO}cuda.is_available: {torch.cuda.is_available()}")

    if torch.version.hip is None:
        print(f"{BAD} this is a CUDA build, not ROCm")
        print(f"{INFO} pip replaced the ROCm torch -- see docs/PITFALLS.md #1")
        record("ROCm torch build", "FAIL", "CUDA build installed")
        return False

    if not torch.cuda.is_available():
        print(f"{BAD} GPU not visible to PyTorch")
        record("GPU visibility", "FAIL", "cuda.is_available() is False")
        return False

    p = torch.cuda.get_device_properties(0)
    arch = getattr(p, "gcnArchName", "?")
    print(f"{INFO}device         : {p.name}")
    print(f"{INFO}arch           : {arch}")
    print(f"{INFO}total VRAM     : {p.total_memory / 2**30:.2f} GiB")

    try:
        x = torch.randn(256, 256, device="cuda")
        val = float((x @ x).sum())
        print(f"{OK} real GPU matmul succeeded ({val:.2f})")
        record("GPU compute", "OK", f"{p.name} / {arch}")
    except Exception as exc:
        print(f"{BAD} GPU matmul failed: {type(exc).__name__}: {exc}")
        record("GPU compute", "FAIL", str(exc)[:60])
        return False

    return True


# ---------------------------------------------------------------- issue 2
def check_gemm() -> None:
    section("Issue 2 -- hipBLASLt transposed-operand GEMM (~400x slowdown)")

    import time

    import torch

    if not torch.cuda.is_available():
        print(f"{WARN} skipped: no GPU")
        record("GEMM anomaly", "WARN", "skipped")
        return

    m, n, k = 4096, 12288, 4096
    x = torch.randn(m, k, dtype=torch.float16, device="cuda")
    w = torch.randn(n, k, dtype=torch.float16, device="cuda")
    wt = w.t().contiguous()

    def bench(fn, iters=5):
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    slow = bench(lambda: torch.nn.functional.linear(x, w))
    fast = bench(lambda: torch.matmul(x, wt))
    ratio = slow / max(fast, 1e-9)

    print(f"{INFO}F.linear(x, w)  [transposed]: {slow:9.2f} ms")
    print(f"{INFO}matmul(x, wt)   [contiguous]: {fast:9.2f} ms")
    print(f"{INFO}ratio                       : {ratio:9.1f}x")

    if ratio > 20:
        print(f"{BAD} affected -- apply patches/gfx1200_layout_patch.py")
        record("GEMM anomaly", "FAIL", f"{ratio:.0f}x slower when transposed")
    elif ratio > 3:
        print(f"{WARN} mild anomaly ({ratio:.1f}x); patch may help")
        record("GEMM anomaly", "WARN", f"{ratio:.1f}x")
    else:
        print(f"{OK} not affected; the layout patch is unnecessary here")
        record("GEMM anomaly", "OK", f"ratio {ratio:.1f}x")


# ---------------------------------------------------------------- issue 3
def check_vram_report() -> None:
    section("Issue 3 -- does mem_get_info() report nonsense?")

    import torch

    if not torch.cuda.is_available():
        print(f"{WARN} skipped: no GPU")
        record("VRAM reporting", "WARN", "skipped")
        return

    free, total = torch.cuda.mem_get_info()
    props_total = torch.cuda.get_device_properties(0).total_memory
    print(f"{INFO}mem_get_info free : {free} bytes ({free / 2**30:.4f} GiB)")
    print(f"{INFO}mem_get_info total: {total} bytes ({total / 2**30:.2f} GiB)")
    print(f"{INFO}props.total_memory: {props_total} bytes ({props_total / 2**30:.2f} GiB)")

    if props_total > 0 and free < props_total * 0.01:
        print(f"{BAD} free is bogus (<1% of device total) -- known WSL/ROCm issue")
        print(f"{INFO} apply patches/patch_vram_reporting.py")
        record("VRAM reporting", "FAIL", f"reports {free} B free")
    else:
        print(f"{OK} plausible")
        record("VRAM reporting", "OK", f"{free / 2**30:.2f} GiB free")


def check_allocatable() -> None:
    """How much can actually be allocated? Matters for GGUF dequant blow-up."""
    import torch

    if not torch.cuda.is_available():
        return

    section("VRAM capacity -- how much can actually be allocated?")
    held = []
    chunk = 256 * 1024 * 1024
    try:
        while True:
            t = torch.empty(chunk, dtype=torch.uint8, device="cuda")
            t.fill_(1)
            held.append(t)
    except Exception:
        pass
    gib = len(held) * chunk / 2**30
    print(f"{INFO}allocated until OOM: {gib:.2f} GiB")
    del held
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    needs = 13.25 + 5.88 + 0.63   # DiT(fp16) + text encoder + VAE
    print(f"{INFO}dequantized working set for Qwen-Image-2.1: ~{needs:.2f} GiB")
    if gib >= needs:
        print(f"{OK} fits fully resident")
        record("VRAM capacity", "OK", f"{gib:.2f} GiB available")
    else:
        print(f"{WARN} does not fit -- expect staged loading or OOM")
        print(f"{INFO} use --cpu-vae and a conservative GFX1200_USABLE_VRAM_MB")
        record("VRAM capacity", "WARN", f"{gib:.2f} GiB < {needs:.2f} GiB needed")


def check_patches(comfy: str | None) -> None:
    section("Patch status")

    if not comfy:
        for cand in ("/opt/ComfyUI", os.path.expanduser("~/ComfyUI")):
            if os.path.isdir(cand):
                comfy = cand
                break
    if not comfy or not os.path.isdir(comfy):
        print(f"{WARN} ComfyUI not found; skipping")
        record("ComfyUI patches", "WARN", "ComfyUI not found")
        return

    print(f"{INFO}ComfyUI: {comfy}")

    mm = os.path.join(comfy, "comfy", "model_management.py")
    if os.path.isfile(mm):
        src = open(mm, encoding="utf-8", errors="replace").read()
        if "ROCM_FREE_MEM_PATCH" in src:
            print(f"{OK} VRAM reporting patch present")
            record("VRAM patch", "OK", "applied")
        else:
            print(f"{BAD} VRAM reporting patch NOT applied")
            record("VRAM patch", "FAIL", "not applied")
    else:
        print(f"{WARN} {mm} not found")
        record("VRAM patch", "WARN", "model_management.py missing")

    ops = os.path.join(comfy, "custom_nodes", "ComfyUI-GGUF", "ops.py")
    if os.path.isfile(ops):
        src = open(ops, encoding="utf-8", errors="replace").read()
        if "gfx1200 layout fix" in src:
            print(f"{OK} GGUF ops patch present")
            record("GGUF ops patch", "OK", "applied")
        else:
            print(f"{BAD} GGUF ops patch NOT applied")
            record("GGUF ops patch", "FAIL", "not applied")
    else:
        print(f"{WARN} ComfyUI-GGUF not installed")
        record("GGUF ops patch", "WARN", "ComfyUI-GGUF missing")

    node = os.path.join(comfy, "custom_nodes", "gfx1200_layout")
    if os.path.isdir(node):
        print(f"{OK} layout custom node installed")
        record("Layout node", "OK", node)
    else:
        print(f"{BAD} layout custom node NOT installed")
        record("Layout node", "FAIL", "missing")

    # Models
    models = os.path.join(comfy, "models")
    expected = {
        os.path.join(models, "unet", "qwen_image_2.1-Q4_K_M.gguf"): 4335931552,
        os.path.join(models, "text_encoders", "qwen3vl_8b_w4a8.safetensors"): 6312105364,
        os.path.join(models, "vae", "qwen_image_2.1_vae_bf16.safetensors"): 675509688,
    }
    missing = []
    for path, want in expected.items():
        if not os.path.isfile(path):
            missing.append(os.path.basename(path))
        elif os.path.getsize(path) != want:
            missing.append(f"{os.path.basename(path)} (size mismatch)")
    if missing:
        print(f"{BAD} model files missing/incomplete: {', '.join(missing)}")
        record("Model files", "FAIL", ", ".join(missing))
    else:
        print(f"{OK} all three model files present and correct size")
        record("Model files", "OK", "3/3")


def check_stale_processes() -> None:
    section("Stale processes (a common source of phantom OOMs)")
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,rss,cmd", "--sort=-rss"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception as exc:
        print(f"{WARN} could not list processes: {exc}")
        return

    hits = [ln for ln in out.splitlines()
            if ("main.py --listen" in ln or "generate.py" in ln)]
    if not hits:
        print(f"{OK} none running")
        record("Stale processes", "OK", "none")
    else:
        print(f"{WARN} {len(hits)} process(es) running:")
        for ln in hits[:6]:
            print(f"{INFO}   {ln.strip()[:100]}")
        print(f"{INFO} a ComfyUI server holds ~11 GB RSS; kill stale ones:")
        print(f"{INFO}   pkill -9 -f 'main.py --listen'")
        record("Stale processes", "WARN", f"{len(hits)} running")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comfy-path")
    args = ap.parse_args()

    print("=" * 68)
    print(" gfx1200 / Qwen-Image-2.1 diagnostics")
    print("=" * 68)

    alive = check_torch()
    if alive:
        check_gemm()
        check_vram_report()
        check_allocatable()
    check_patches(args.comfy_path)
    check_stale_processes()

    section("Summary")
    for name, status, detail in results:
        tag = {"OK": OK, "FAIL": BAD, "WARN": WARN}[status]
        print(f"{tag} {name:22s} {detail}")

    fails = [r for r in results if r[1] == "FAIL"]
    print()
    if fails:
        print(f"{len(fails)} issue(s) need attention. See docs/PITFALLS.md.")
        return 1
    print("No blocking issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
