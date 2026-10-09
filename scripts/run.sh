#!/usr/bin/env bash
#
# Start ComfyUI (if needed) and generate one image with Qwen-Image-2.1.
#
#   ./run.sh "a red apple on a wooden table"
#   W=1024 H=1024 STEPS=20 ./run.sh "a cat on a windowsill"
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"

COMFY="${COMFYUI_PATH:-/opt/ComfyUI}"
PROMPT="${1:-A red apple resting on a weathered wooden table, soft window light, photorealistic}"
W="${W:-512}"; H="${H:-512}"; STEPS="${STEPS:-12}"; CFG="${CFG:-4.0}"
PORT="${PORT:-8188}"
LOG="${LOG:-/tmp/comfyui-gfx1200.log}"

# The reported free-VRAM ceiling. A conservative default avoids hipBLASLt OOM;
# raise it only after confirming a full generation succeeds. See docs/PITFALLS.md.
export GFX1200_USABLE_VRAM_MB="${GFX1200_USABLE_VRAM_MB:-15000}"

cd "$COMFY"

if ! curl -s --max-time 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
  echo "Starting ComfyUI on port $PORT ..."
  # --cpu-vae keeps the VAE off the GPU (frees ~0.6 GiB; harmless for speed here)
  # --fp16-* avoids gfx1200's pathologically slow bf16 GEMM path
  nohup python3 main.py --listen 127.0.0.1 --port "$PORT" \
      --disable-smart-memory \
      --cpu-vae \
      --fp16-unet --fp16-text-enc \
      > "$LOG" 2>&1 &
  echo "  pid $!  log: $LOG"

  for _ in $(seq 1 60); do
    sleep 3
    curl -s --max-time 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break
  done
  curl -s --max-time 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 \
    || { echo "ComfyUI did not start; see $LOG" >&2; exit 1; }
  echo "  ready"
else
  echo "ComfyUI already running on port $PORT"
fi

echo
echo "Generating ${W}x${H}, ${STEPS} steps, cfg=${CFG}"
echo "  prompt: $PROMPT"
echo

HOST="http://127.0.0.1:$PORT" W="$W" H="$H" STEPS="$STEPS" CFG="$CFG" \
  python3 "$REPO_ROOT/scripts/generate.py" "$PROMPT" 2>&1 \
  | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|bgemm_internal|return torch'

echo
echo "Output images:"
ls -la "$COMFY/output"/*.png 2>/dev/null | tail -5 || echo "  (none)"
