# Qwen-Image-2.1 on AMD RDNA4 (gfx1200) — working setup

Local, offline text-to-image generation with **Qwen-Image-2.1 (7B, quantized)**
on an **AMD Radeon RX 9060 XT** under **WSL2 + ROCm 7.2**.

This repository exists because getting this combination to actually produce an
image requires working around three independent, undocumented bugs. Each one on
its own is enough to make generation either crash or run ~400× too slowly. Two
of the three are invisible unless you measure.

Verified working — both images below were generated locally on the hardware
described, with no network access at inference time:

| Prompt | Output |
|---|---|
| *"A red apple resting on a weathered wooden table, soft window light, photorealistic"* | [`examples/apple.png`](examples/apple.png) |
| *"a Japanese ceramic tea bowl on a linen cloth, soft morning light, top-down view"* | [`examples/tea-bowl.png`](examples/tea-bowl.png) |

---

## TL;DR

```bash
# 1. install (env check, deps, ~11.3 GB models, patches)
bash scripts/install.sh

# 2. generate
bash scripts/run.sh "a red apple on a wooden table"
```

From Windows:

```powershell
wsl -d Ubuntu-24.04 --exec /bin/bash /path/to/repo/scripts/run.sh "a red apple on a wooden table"
```

---

## Is this for you?

| | |
|---|---|
| ✅ Use this if | You have an RX 9060 XT / RDNA4 GPU, WSL2, and want Qwen-Image-2.1 running offline |
| ⚠️ Probably unnecessary if | You have an NVIDIA GPU (the bugs are AMD-specific) |
| ❌ Not applicable if | You want to run this on Windows-native ROCm (see [docs/PITFALLS.md](docs/PITFALLS.md#4-windows-native-rocm)) |

Also worth saying plainly: **LM Studio cannot run this model.** LM Studio is an
LLM inference engine (llama.cpp); Qwen-Image-2.1 is a diffusion model. Both use
`.gguf` files, which makes this genuinely confusing, but they are different
model families and LM Studio has no diffusion sampler, VAE, or text-encoder
pipeline. You need ComfyUI (or another diffusion runtime) for this.

---

## Hardware and software tested

| Component | Version |
|---|---|
| GPU | AMD Radeon RX 9060 XT 16 GB (**gfx1200**, RDNA4) |
| Driver | Adrenalin 26.7.1 (`32.0.31035.1003`) |
| Host | Windows 10 22H2 (19045) |
| Runtime | WSL2, Ubuntu 24.04, kernel 6.18.40.1 |
| ROCm | 7.2.0 |
| PyTorch | 2.9.1+rocm7.2.0 |
| ComfyUI | 0.38.0 |
| Python | 3.12 |

Other RDNA4 cards (RX 9070 / 9070 XT) very likely hit the same bugs, since they
share the `gfx12xx` target. Reports welcome.

---

## Model files

Three files are required. Downloading only the DiT is the most common mistake —
the text encoder is the *larger* of the two.

| File | Size | Role |
|---|---|---|
| `unet/qwen_image_2.1-Q4_K_M.gguf` | 4.34 GB | diffusion transformer |
| `text_encoders/qwen3vl_8b_w4a8.safetensors` | 6.31 GB | Qwen3-VL-8B text encoder |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB | VAE |

Sources:

- DiT GGUF: [`pottokao/Qwen-Image-2.1-DiT-GGUF`](https://huggingface.co/pottokao/Qwen-Image-2.1-DiT-GGUF) (Q4_K_M / Q6_K / Q8_0)
- Text encoder + VAE: [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1)

`scripts/fetch_models.sh` downloads all three and verifies both the byte sizes
and the GGUF architecture string. Set `HF_ENDPOINT` if you are not in mainland
China (default is a mirror, because `huggingface.co` is often unreachable):

```bash
HF_ENDPOINT=https://huggingface.co bash scripts/fetch_models.sh
```

### Choosing a quantization

| Quant | Size | Notes |
|---|---|---|
| **Q4_K_M** | 4.34 GB | default here; best speed/quality trade-off for 16 GB |
| Q6_K | 6.00 GB | slower, slightly better |
| Q8_0 | 7.69 GB | fits in 16 GB, noticeably slower |

Important: the DiT is stored quantized but **dequantized to fp16 in memory**.
Q4_K_M becomes ~13.25 GiB of fp16 once loaded. See
[docs/PITFALLS.md #3](docs/PITFALLS.md) — this is why the text encoder and VAE
need care on a 16 GB card.

---

## The three bugs

Short version here; full analysis with measurements in
[docs/PITFALLS.md](docs/PITFALLS.md).

### 1. pip silently replaces the ROCm PyTorch build

Installing `kornia` pulls `torch>=2.0.0`, and pip resolves that to the **CUDA**
wheel from PyPI:

```
torch 2.9.1+rocm7.2.0   →   torch 2.14.1+cu130     cuda: False
```

Your GPU disappears. `scripts/install.sh` prevents this by writing a constraint
file pinning the installed torch build and passing `-c` to every pip call.

### 2. hipBLASLt is ~400× slower on transposed-operand GEMMs

On gfx1200 + ROCm 7.2, any GEMM with a transposed operand triggers

```
bgemm_internal_cublaslt error: HIPBLAS_STATUS_INTERNAL_ERROR ...
Will attempt to recover by calling cublas instead.
```

and the fallback is catastrophically slow. Measured, 4096×12288×4096 fp16:

| Expression | Time |
|---|---|
| `F.linear(x, w)` | 2590 ms |
| `x @ w.t()` | 2383 ms |
| **`torch.matmul(x, w.t().contiguous())`** | **6.0 ms** |

`nn.Linear` stores weights as `[out, in]`, so **every layer of the model hits
the slow path**. `patches/gfx1200_layout_patch.py` fixes this by transposing
weights once at dequantization time and using plain `matmul`. Result: **418×**.

### 3. WSL reports 4 KiB of free VRAM on a 16 GiB card

There is no `amdgpu` kernel module under WSL (the GPU is reached via `/dev/dxg`),
so `amdsmi` fails and:

```python
torch.cuda.mem_get_info()          # -> (4096, 16974905344)   <- bogus
torch.cuda.get_device_properties(0).total_memory   # -> 15.81 GiB  <- correct
```

ComfyUI believes there is no VRAM and drops to its lowest-memory mode
(`0.00 MB usable`), which makes sampling crawl. `patches/patch_vram_reporting.py`
substitutes a usable ceiling, tunable via `GFX1200_USABLE_VRAM_MB`.

A related trap: **over-reporting causes a different failure.** Claim too much
and ComfyUI loads everything, then dies with `hipBLASLt OOM (192.00 MiB)`.
The default here is conservative.

---

## Performance

Measured on the hardware above, 512×512, ComfyUI + Q4_K_M:

| Steps | Time | Per step |
|---|---|---|
| 6 | ~5 min | 21 s |
| 10 | ~4.5 min (warm cache) | 19.5 s |
| 12 | ~6.5 min | 20 s |

First run is slower (one-time MIOpen kernel compilation, cached afterwards in
`~/.cache/miopen`).

### Where the time actually goes

Per linear layer, measured:

| Step | Cost | × 160 layers |
|---|---|---|
| **Dequantize Q4_K → fp16** | **37.9 ms** | **6.1 s / step** |
| Transpose to `[in, out]` contiguous | 5.2 ms | 0.8 s |
| **The actual matmul** | **1.9 ms** | 0.3 s |

This is the honest headline: **the GPU spends ~20× longer reconstructing
weights than multiplying them.** `ComfyUI-GGUF` dequantizes on every forward
pass by design. On a card with enough VRAM you would cache the fp16 weights and
skip it entirely — see below.

### Why we do not cache the dequantized weights

Caching all 265 DiT tensors as fp16 needs **13.25 GiB**, and the text encoder
needs a further 5.88 GiB. On a 16 GB card that does not co-reside: the attempt
OOMs during sampling.

Measured A/B, 512×512 / 6 steps:

| Config | Wall time | Result |
|---|---|---|
| default (dequant per step) | 296 s | ✅ succeeds |
| `GFX1200_CACHE_DEQUANT=1` | 146 s | ❌ `HIP out of memory` |

The cache is therefore **opt-in and off by default**; it is shipped because it
is the right trade on a 24 GB+ card, where it roughly halves wall time. If you
have the VRAM:

```bash
GFX1200_CACHE_DEQUANT=1 bash scripts/run.sh "prompt"
```

### Why the layout patch matters

Before the fix, every timestep ran through a ~400× slower GEMM path. In practice
a 512×512 image did not finish in **20+ minutes** and frequently ended in OOM.
After: minutes.

---

## Repository layout

```
patches/
  gfx1200_layout_patch.py     fix #2 -- fast GEMM path (has a self-test)
  patch_gguf_ops.py           wires #2 into ComfyUI-GGUF's dequantizer
  patch_vram_reporting.py     fix #3 -- sane free-VRAM value
scripts/
  install.sh                  one-shot setup
  fetch_models.sh             download + verify model files
  run.sh                      start ComfyUI and generate
  generate.py                 workflow submission (API)
diagnostics/
  diagnose.py                 reproduce each bug and report timings
  benchmark_linear.py         measure the GEMM anomaly on your card
docs/
  PITFALLS.md                 detailed analysis of each bug
  PREREQUISITES.md            installing ROCm + PyTorch first
```

Both source patches are idempotent, syntax-check after editing, and keep a
timestamped `.bak`. To undo them:

```bash
python3 patches/patch_vram_reporting.py --unpatch
python3 patches/patch_gguf_ops.py --unpatch
# then remove the custom node directory
rm -rf /opt/ComfyUI/custom_nodes/gfx1200_layout
```

---

## Verifying the fix applies to your card

```bash
python3 patches/gfx1200_layout_patch.py
```

Expected on an affected card:

```
F.linear(x, w)                :   2590.54 ms
matmul(x, wt)                 :      6.20 ms
untagged allclose vs original : True
tagged allclose vs original   : True
RESULT: OK -- 418x speedup on the affected shape
```

If your GPU reports the slow path already fast (< 100 ms), the script tells you
the patch is unnecessary — useful for NVIDIA users or future ROCm releases.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cuda: False` after installing packages | pip swapped torch for CUDA | [PITFALLS #1](docs/PITFALLS.md) |
| `mat1 and mat2 shapes cannot be multiplied (…4096 and 2048…)` | text encoder loaded via `CLIPLoaderGGUF` | use native `CLIPLoader` (already done in `generate.py`) |
| `hipBLASLt ... OOM (192.00 MiB)` | VRAM ceiling too high | lower `GFX1200_USABLE_VRAM_MB` |
| `0.00 MB usable` in the log | fix #3 not applied | run `patch_vram_reporting.py` |
| Image takes >20 min | fix #2 not applied | run the self-test above |
| `huggingface.co` unreachable | network | `HF_ENDPOINT=https://hf-mirror.com` |

---

## License

Code: **MIT** — see [LICENSE](LICENSE).

**Model weights are not covered by MIT.** Qwen-Image-2.1 is released by Alibaba
under the [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE).
This repository contains no weights; it only downloads them. Check that license
before commercial use.

---

## Contributing

Reports from other `gfx12xx` cards (RX 9070, 9070 XT, Radeon AI PRO R9700) are
especially useful — please include `python3 diagnostics/benchmark_linear.py`
output and your ROCm/PyTorch versions.

## Acknowledgements

ComfyUI, ComfyUI-GGUF, the Qwen team, and the ROCm/HIP maintainers.
