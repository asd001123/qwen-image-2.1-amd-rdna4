# Pitfalls — detailed analysis

Everything here was reproduced and measured on:

- AMD Radeon RX 9060 XT 16 GB, **gfx1200** (RDNA4)
- ROCm 7.2.0, PyTorch 2.9.1+rocm7.2.0
- WSL2 Ubuntu 24.04, kernel 6.18.40.1
- ComfyUI 0.38.0, ComfyUI-GGUF

---

## 1. pip replaces the ROCm PyTorch build with a CUDA one

### Symptom

After installing a ComfyUI dependency, the GPU vanishes:

```python
>>> import torch
>>> torch.cuda.is_available()
False
>>> torch.__version__
'2.14.1+cu130'        # was 2.9.1+rocm7.2.0
```

### Cause

`kornia` (a ComfyUI requirement) declares `torch>=2.0.0`. The ROCm build has a
local version identifier — `2.9.1+rocm7.2.0.git7e1940d4` — and pip's resolver
does not treat local-version strings as satisfying a plain `>=2.0.0` specifier.
So pip "helpfully" installs the newest matching wheel from PyPI, which is the
**CUDA** build. It also drags in ~2 GB of `nvidia-*` packages.

A constraints file does not help here, because pinning torch by its exact local
version reproduces the resolver conflict:

```
kornia 0.8.3 depends on torch>=2.0.0
The user requested (constraint) torch==2.9.1+rocm7.2.0.git7e1940d4
ERROR: ResolutionImpossible
```

### Fix

Install non-torch dependencies explicitly, never letting pip re-resolve torch:

```bash
python3 -m pip install --break-system-packages --no-deps <package>
```

or, when dependencies are genuinely needed, keep a constraint file **and**
verify afterwards:

```bash
python3 -c "import torch; assert torch.cuda.is_available()"
```

`scripts/install.sh` does both: it writes `rocm-constraints.txt` from the
current build and re-verifies `cuda.is_available()` after installing.

### Recovery

Keep the ROCm wheels — on many images they live in a local directory (here
`/opt/wheels/`):

```bash
python3 -m pip install --break-system-packages --no-deps --force-reinstall \
  /opt/wheels/torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl
```

Then remove the CUDA leftovers:

```bash
python3 -m pip uninstall -y nvidia-cublas nvidia-cudnn-cu13 cuda-toolkit triton
```

⚠️ Be careful with that uninstall list: removing `triton` breaks
`torch._dynamo`, and removing `comfy_aimdo` / `comfy_kitchen` breaks ComfyUI.
Reinstall them from `/opt/wheels/` if needed.

### Related trap: PEP 668

Ubuntu 24.04 refuses system-wide pip installs
(`externally-managed-environment`). `--break-system-packages` is required when
using the system interpreter that already holds the ROCm torch. A fresh venv is
*not* a safe alternative here unless you reinstall the ROCm wheels into it
(~2 GB) — otherwise you get CUDA torch again.

---

## 2. hipBLASLt is ~400× slower on transposed-operand GEMMs

### Symptom

Sampling never finishes. The log fills with:

```
bgemm_internal_cublaslt error: HIPBLAS_STATUS_INTERNAL_ERROR when calling
hipblasLtMatmul with transpose_mat1 1 ... Will attempt to recover by calling
cublas instead.
```

The "will attempt to recover" wording makes this look benign. It is not — the
recovery path is roughly 400× slower.

### Measurement

4096 × 12288 × 4096, fp16, RX 9060 XT:

| Expression | Time | Throughput |
|---|---|---|
| `F.linear(x, w)` | 2590 ms | 0.13 TFLOPS |
| `x @ w.t()` | 2383 ms | 0.14 TFLOPS |
| `torch.matmul(x, w.t())` | 2687 ms | 0.12 TFLOPS |
| **`torch.matmul(x, w.t().contiguous())`** | **6.0 ms** | **45 TFLOPS** |

The distinguishing factor is *not* the operation or the shape — it is whether
the operand is a **transposed view** or a **materialised contiguous buffer**.

Reproduce with `diagnostics/benchmark_linear.py`.

There is a second, orthogonal trap on this GPU: **bf16 is also slow** (842 ms
for a shape that takes 1.88 ms in fp16). ComfyUI defaults to bf16 when the
checkpoint permits, so run with `--fp16-unet` regardless.

### Why it matters so much

`torch.nn.Linear` stores weights as `[out_features, in_features]`, so
`F.linear` computes `x @ W.T` — the slow path. In a transformer *every* linear
layer does this, and a 7B DiT at 20 steps performs thousands of them.

### Fix

`patches/gfx1200_layout_patch.py`:

1. `ComfyUI-GGUF` transposes each weight to `[in, out]` once, at dequantization
   time, and tags it (`patches/patch_gguf_ops.py`).
2. A patched `F.linear` sees the tag and uses `torch.matmul(input, weight)`,
   which is the fast form.

Result: **418×** on the affected shape (2590 ms → 6.0 ms), numerics unchanged
(`torch.allclose` → `True`).

### Memory neutrality (important)

The naive version of this fix — cache `weight.t().contiguous()` per layer —
**doubles weight memory** and is worse than the disease, because it pushes a
13 GiB model past a 16 GiB ceiling and produces `hipBLASLt OOM`. An earlier
iteration of this project made exactly that mistake.

The correct approach replaces the original buffer rather than keeping both.
Verified: a 96 MiB weight occupies 96 MiB afterwards either way.

### Is the fix safe?

The self-test checks both branches:

```
untagged allclose vs original : True    # unpatched behaviour preserved
tagged allclose vs original   : True    # fast path is numerically equivalent
```

If your GPU is unaffected (fast path already < 100 ms), the self-test says so
and the patch is simply inert.

---

## 3. WSL reports 4 KiB of free VRAM on a 16 GiB card

### Symptom

ComfyUI's log says `0.00 MB usable` and drops into its lowest-memory mode.
Sampling is extremely slow even after fixing #2.

### Cause

Under WSL the GPU is reached through `/dev/dxg`; there is no `amdgpu` kernel
module, so `amdsmi` cannot initialise:

```
UserWarning: Can't initialize amdsmi - Error code: 34
(AMDSMI_STATUS_DRIVER_NOT_LOADED)
```

Consequently:

```python
torch.cuda.mem_get_info()                          # (4096, 16974905344)
torch.cuda.get_device_properties(0).total_memory   # 16974905344  (correct)
```

The **total** is right and allocation works fine (a 4 GiB tensor allocates
without trouble) — only the free-memory query is broken.

### Fix

`patches/patch_vram_reporting.py` rewrites ComfyUI's `get_free_memory()` so that
when the reported free value is absurdly small relative to the device
(< 1 % of total), it substitutes a usable ceiling:

```bash
export GFX1200_USABLE_VRAM_MB=15000    # default when unset: 4864
```

### The counter-intuitive part

**Over-reporting causes a different failure, not a fix.** If you tell ComfyUI
the full 15.8 GiB is free, it loads everything simultaneously:

```
HIP out of memory. Tried to allocate 192.00 MiB.
GPU 0 has a total capacity of 15.81 GiB of which 4.00 KiB is free.
Of the allocated memory 14.76 GiB is allocated by PyTorch
```

Note `4.00 KiB` in that message — hipBLASLt is reading the same broken
`mem_get_info` internally, and the Python-level patch cannot reach inside the
C++ library.

The root cause of the OOM is **dequantization blow-up**, described next.

---

## 4. GGUF dequantization inflates VRAM 3.3×

### The numbers

| | Footprint |
|---|---|
| DiT as stored (Q4_K + Q6_K) | 4.04 GiB |
| DiT after dequantization to fp16 | **13.25 GiB** |
| Text encoder | 5.88 GiB |
| VAE | 0.63 GiB |
| **All resident, dequantized** | **19.76 GiB** ❌ > 15.81 GiB |
| **Quantized-resident** | **10.55 GiB** ✅ |

`ComfyUI-GGUF`'s `GGMLOps` dequantizes weights and keeps the fp16 result, so a
4.34 GB file occupies 13.25 GiB once loaded.

### Consequences

- On a 16 GB card you **cannot** hold DiT + text encoder + VAE dequantized at
  once. Staged loading or CPU offload is mandatory, not optional.
- Any patch that *caches* dequantized weights makes this strictly worse. Do not
  write one. (See [#2 memory neutrality](#memory-neutrality-important).)

### Mitigations used here

```bash
--cpu-vae          # keeps the 0.63 GiB VAE off the GPU
--fp16-unet        # also fixes the bf16 slowness from #2
GFX1200_USABLE_VRAM_MB=<conservative>   # forces staged loading instead of OOM
```

---

## 5. Windows-native ROCm

AMD's Windows support matrix lists `gfx1200` under ROCm 7.1.1 with PyTorch 2.9
and Python 3.12, and FP8 on RDNA4. However:

> PyTorch on Windows includes ROCm 7.1.1 components; however, the entire ROCm
> stack is not yet supported on Windows.

The OS support matrix lists **Windows 11 only**. Windows 10 (build 19045) is not
listed. If you want the Windows-native route, prefer Windows 11, and expect the
same class of GEMM/VRAM issues — the patches here are Python-level and should
still apply, but that path is untested.

Known additional blockers on Windows-native ROCm for `gfx1200`:

- unsigned ROCm DLLs rejected by Smart App Control (Win11 feature)
- MIOpen errors specific to `gfx1200`

---

## 6. The text encoder must use the native `CLIPLoader`

### Symptom

```
mat1 and mat2 shapes cannot be multiplied (22x4096 and 2048x4096)
```

### Cause

`qwen3vl_8b_w4a8.safetensors` uses ComfyUI's own quantization format
(`asym_w4a8_int8` with `convrot`), and its tensors carry `comfy_quant` metadata.
Only the native `CLIPLoader` understands this.

Loading it through `CLIPLoaderGGUF` makes ComfyUI select the *older* Qwen-Image
text-encoder architecture (hidden size 2048) instead of Qwen3-VL-8B (4096),
hence the shape mismatch.

### Fix

```json
{"class_type": "CLIPLoader",
 "inputs": {"clip_name": "qwen3vl_8b_w4a8.safetensors", "type": "qwen_image"}}
```

The DiT still uses `UnetLoaderGGUF`. `scripts/generate.py` already does this.

---

## Diagnostic recipe

When something is wrong, isolate in this order:

```bash
# 1. is the GPU alive at all?
python3 -c "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available())"

# 2. is this the GEMM bug?
python3 diagnostics/benchmark_linear.py

# 3. what does ComfyUI think it has?
grep -iE 'vram|usable|device' /tmp/comfyui-gfx1200.log

# 4. where is time actually going?
py-spy dump --pid "$(pgrep -f 'main.py --listen' | head -1)"
```

A `py-spy` dump is what finally localised the bottleneck in this project — the
stack showed `dequantize_blocks_Q4_K` and made it obvious which layer to
optimise. Highly recommended over guessing.
