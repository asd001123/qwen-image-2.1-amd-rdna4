# Prerequisites — ROCm + PyTorch on WSL2

This repository assumes a working ROCm PyTorch install. If `torch.cuda.is_available()`
is `False` in WSL, fix that first — no amount of patching will help.

## 1. Confirm the hardware is right

```powershell
# From Windows
wsl -d Ubuntu-24.04 --exec /bin/bash -lc "ls -la /dev/dxg"
```

`/dev/dxg` must exist. That is WSL's GPU passthrough device. If it is missing,
your WSL kernel or GPU driver is too old.

```bash
# Inside WSL
/opt/rocm/bin/rocminfo | grep -iE 'gfx|Marketing'
```

Expected for an RX 9060 XT:

```
  Name:                    gfx1200
  Marketing Name:          AMD Radeon RX 9060 XT
```

If `rocminfo` reports no GPU agent, ROCm is not seeing the card.

## 2. Install ROCm + PyTorch

Two viable routes:

### A. Prebuilt ROCm image / wheel bundle (fastest)

If your distribution ships ROCm wheels locally (common on preconfigured images
and some cloud images), check for a wheel directory first:

```bash
ls /opt/wheels/
# torch-2.9.1+rocm7.2.0.lw...whl
# torchvision-0.24.0+rocm7.2.0...whl
# triton-3.5.1+rocm7.2.0...whl
```

Install with `--no-deps` so pip cannot swap in a CUDA build:

```bash
python3 -m pip install --break-system-packages --no-deps /opt/wheels/torch-*.whl
python3 -m pip install --break-system-packages --no-deps /opt/wheels/torchvision-*.whl
python3 -m pip install --break-system-packages --no-deps /opt/wheels/triton-*.whl
```

**Keep these wheels.** They are your recovery path if pip later replaces torch —
see [PITFALLS.md #1](PITFALLS.md).

### B. Official ROCm packages

Follow AMD's guide for your distribution, then install the matching PyTorch:

```bash
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

> Note the PyTorch index for ROCm may not carry a build for every ROCm version.
> If the install silently resolves to a CUDA wheel, you will get
> `torch.version.hip is None` — check it.

## 3. Verify

```bash
python3 - <<'EOF'
import torch
print("torch :", torch.__version__)
print("hip   :", torch.version.hip)
print("cuda  :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device:", p.name, getattr(p, "gcnArchName", ""))
    print("vram  : %.2f GiB" % (p.total_memory / 2**30))
    x = torch.randn(512, 512, device="cuda")
    print("matmul:", float((x @ x).sum()))
EOF
```

All three must look right:

- `hip` is a version string (not `None`)
- `cuda` is `True`
- the real matmul prints a number

A successful `import torch` is **not** sufficient — the failure mode we care
about is a torch that imports fine but cannot touch the GPU.

## 4. Known-good combination

| Component | Version |
|---|---|
| WSL | 3.0.1.0 |
| WSL kernel | 6.18.40.1-microsoft-standard-WSL2 |
| Ubuntu | 24.04 |
| Python | 3.12 |
| ROCm | 7.2.0 |
| PyTorch | 2.9.1+rocm7.2.0 |
| Driver (Adrenalin) | 26.7.1 |

## 5. Common failures

| Error | Meaning |
|---|---|
| `Can't initialize amdsmi - Error code: 34` | Expected under WSL (`AMDSMI_STATUS_DRIVER_NOT_LOADED`). Harmless, but it causes [PITFALLS #3](PITFALLS.md). |
| `torch.cuda.is_available() == False` with `hip=None` | pip installed the CUDA build. See [PITFALLS #1](PITFALLS.md). |
| `rocminfo` shows no GPU | Driver or WSL kernel too old; check `/dev/dxg`. |
| `HSA exception: MemoryRegion::BlockAllocator::alloc failed` | Out of GPU memory; another process may be holding it. Check `ps` for stale ComfyUI/python processes. |

## 6. Orphaned processes

Worth calling out explicitly, because it cost real debugging time here: a
ComfyUI server left running holds **~11 GB of RSS**, and several stale
generation clients can add tens of gigabytes more. The resulting memory
pressure produces confusing, seemingly unrelated OOM errors.

Before diagnosing anything, make sure nothing stale is running:

```bash
pkill -9 -f 'main.py --listen'
pkill -9 -f generate.py
ps -eo pid,rss,cmd --sort=-rss | head
free -m
```
