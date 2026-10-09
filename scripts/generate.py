#!/usr/bin/env python3
"""Submit a Qwen-Image-2.1 text-to-image workflow to a running ComfyUI.

The node graph mirrors Comfy-Org's official
``image_qwen_image_2_1_t2i`` template, with one deliberate difference: the text
encoder is loaded through the **native** ``CLIPLoader``, not ``CLIPLoaderGGUF``.

That matters. The official ``qwen3vl_8b_w4a8.safetensors`` uses ComfyUI's own
quantization format (``asym_w4a8_int8`` with ``convrot``), which only the native
loader understands. Loading it through the GGUF loader yields:

    mat1 and mat2 shapes cannot be multiplied (22x4096 and 2048x4096)

because ComfyUI then selects the older Qwen-Image text-encoder architecture
(hidden size 2048) instead of Qwen3-VL-8B (hidden size 4096).

Environment: HOST, W, H, STEPS, CFG, SEED, PREFIX, TIMEOUT
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("HOST", "http://127.0.0.1:8188")
W = int(os.environ.get("W", "512"))
H = int(os.environ.get("H", "512"))
STEPS = int(os.environ.get("STEPS", "12"))
CFG = float(os.environ.get("CFG", "4.0"))
SEED = int(os.environ.get("SEED", "42"))
PREFIX = os.environ.get("PREFIX", "qwen")
TIMEOUT = float(os.environ.get("TIMEOUT", "7200"))

DIT = os.environ.get("DIT_NAME", "qwen_image_2.1-Q4_K_M.gguf")
TE = os.environ.get("TE_NAME", "qwen3vl_8b_w4a8.safetensors")
VAE = os.environ.get("VAE_NAME", "qwen_image_2.1_vae_bf16.safetensors")


def build(prompt: str) -> dict:
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": DIT}},
        # Native loader: required for the w4a8 quantization format.
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "", "clip": ["2", 0]}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": W, "height": H, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "seed": SEED, "steps": STEPS,
                         "cfg": CFG, "sampler_name": "euler",
                         "scheduler": "simple", "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0],
                         "denoise": 1.0}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["8", 0], "filename_prefix": PREFIX}},
    }


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path: str) -> dict:
    return json.loads(urllib.request.urlopen(HOST + path, timeout=60).read())


def main() -> int:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a red apple on a wooden table"

    print(f"submitting: {W}x{H}, {STEPS} steps, cfg={CFG}, seed={SEED}")
    t0 = time.time()
    try:
        pid = post("/prompt", {"prompt": build(prompt)})["prompt_id"]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"ComfyUI rejected the workflow (HTTP {exc.code}):")
        print(body[:1200])
        return 1

    print(f"prompt_id: {pid}")
    last = -1
    while time.time() - t0 < TIMEOUT:
        time.sleep(3)
        try:
            hist = get(f"/history/{pid}")
        except Exception:
            continue
        if pid not in hist:
            elapsed = int(time.time() - t0)
            if elapsed // 20 != last:
                last = elapsed // 20
                print(f"  ...{elapsed}s")
            continue

        entry = hist[pid]
        status = entry.get("status", {})
        elapsed = time.time() - t0
        print(f"\nfinished in {elapsed:.1f}s -- {status.get('status_str')}")

        for msg in status.get("messages", []):
            if msg[0] == "execution_error":
                info = msg[1]
                print("\nexception:", info.get("exception_type"))
                print(info.get("exception_message"))
                print("node:", info.get("node_type"), info.get("node_id"))
                return 1

        images = [im for out in entry.get("outputs", {}).values()
                  for im in out.get("images", [])]
        for im in images:
            print(f"  IMAGE -> {im['filename']}")
        return 0 if status.get("status_str") == "success" else 1

    print(f"timed out after {TIMEOUT}s")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
