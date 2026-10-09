#!/bin/bash
# A/B test: with and without the dequant cache, same seed and settings.
set -u
cd /opt/ComfyUI
R=/mnt/f/dpsk/qwen-image-2.1-amd-rdna4

run_case () {
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "# $label"
  echo "############################################################"
  pkill -9 -f 'main.py --listen' 2>/dev/null || true
  sleep 4
  rm -f /opt/ComfyUI/output/*.png

  env "$@" nohup python3 main.py --listen 127.0.0.1 --port 8188 \
      --disable-smart-memory --cpu-vae --fp16-unet --fp16-text-enc \
      > "/tmp/comfy_${label}.log" 2>&1 &
  for _ in $(seq 1 50); do
    sleep 3
    curl -s --max-time 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break
  done

  local t0; t0=$(date +%s)
  W=512 H=512 STEPS=6 CFG=4.0 SEED=42 PREFIX="$label" TIMEOUT=2400 \
    python3 "$R/scripts/generate.py" "a red apple on a wooden table" 2>&1 \
    | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|bgemm_internal|return torch' | tail -4
  local t1; t1=$(date +%s)

  echo "  wall: $((t1-t0)) s"
  echo "  per-step:"
  tail -c 4000 "/tmp/comfy_${label}.log" | tr '\r' '\n' \
    | grep -oE '[0-9]+/6 \[[^]]*\]' | tail -1
  echo "  cache report:"
  grep -a 'dequant cache' "/tmp/comfy_${label}.log" | tail -2
  ls -la /opt/ComfyUI/output/*.png 2>/dev/null | tail -2
}

run_case "nocache" GFX1200_CACHE_DEQUANT=0 GFX1200_USABLE_VRAM_MB=15000
run_case "cache"   GFX1200_CACHE_DEQUANT=1 GFX1200_USABLE_VRAM_MB=15000
