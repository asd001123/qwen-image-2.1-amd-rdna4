# Licensing

## This repository's code

**MIT** — see [LICENSE](LICENSE).

## Model weights

**The MIT license does not cover the model weights.** This repository contains
no weights; it only downloads them.

Qwen-Image-2.1 is released by Alibaba under the **Qwen Research License**:

- https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE

Review that license before any commercial use of the model or its outputs.
Note that the quantized GGUF repacks used here are redistributions by third
parties (`pottokao/Qwen-Image-2.1-DiT-GGUF`, `Comfy-Org/Qwen-Image-2.1`); check
their terms as well.

## What this repository ships

| Included | Not included |
|---|---|
| Install, fetch, and run scripts | Model weights (`.gguf`, `.safetensors`) |
| Python source patches | ComfyUI itself |
| Diagnostics and documentation | ROCm / PyTorch |
| Two example PNGs generated locally | |

The example images in `examples/` were produced on the hardware described in
the README and are covered by the Qwen Research License as model output.
