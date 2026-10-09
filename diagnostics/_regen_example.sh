#!/bin/bash
set -u
# Recover from the caching experiment: kill everything, ensure a clean GPU, then
# generate the example image with the default (non-caching) config.
pkill -9 -f 'main.py --listen' 2>/dev/null || true
pkill -9 -f generate.py 2>/dev/null || true
sleep 5

echo "=== residual processes ==="
ps -eo pid,rss,cmd | grep -E 'python3' | grep -vE 'grep|unattended' | head -5
echo "(end)"

echo
echo "=== GPU free now? ==="
cd /opt/ComfyUI
python3 - <<'PY' 2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|UserWarning|HSA exception|^\s*$'
import torch
held=[]
try:
    while True:
        t=torch.empty(256*1024*1024,dtype=torch.uint8,device="cuda"); t.fill_(1); held.append(t)
except Exception: pass
print(f"  allocatable: {len(held)*0.25:.2f} GiB")
del held
PY

echo
echo "=== start server and generate example ==="
GFX1200_USABLE_VRAM_MB=15000 nohup python3 main.py --listen 127.0.0.1 --port 8188 \
  --disable-smart-memory --cpu-vae --fp16-unet --fp16-text-enc > /tmp/comfy_example.log 2>&1 &
for _ in $(seq 1 50); do sleep 3; curl -s --max-time 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break; done

rm -f /opt/ComfyUI/output/*.png
W=512 H=512 STEPS=10 CFG=4.0 SEED=42 PREFIX=example TIMEOUT=2400 \
  python3 /mnt/f/dpsk/qwen-image-2.1-amd-rdna4/scripts/generate.py \
  'A red apple resting on a weathered wooden table, soft window light, photorealistic' 2>&1 \
  | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|bgemm_internal|return torch' | tail -5

echo
echo "=== copy to examples/ ==="
mkdir -p /mnt/f/dpsk/qwen-image-2.1-amd-rdna4/examples
cp -f /opt/ComfyUI/output/example_00001_.png \
      /mnt/f/dpsk/qwen-image-2.1-amd-rdna4/examples/apple.png 2>/dev/null \
  && echo "  examples/apple.png" || echo "  FAILED"
ls -la /mnt/f/dpsk/qwen-image-2.1-amd-rdna4/examples/
