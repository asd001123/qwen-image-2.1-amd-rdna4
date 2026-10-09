#!/bin/bash
# Why is generation still slow? Instrument the live server: count how many
# F.linear calls actually take the fast branch vs the slow one.
cd /opt/ComfyUI
pkill -9 -f 'main.py --listen' 2>/dev/null || true
sleep 3

# Add a temporary counter node that reports patch effectiveness at runtime.
DEST=/opt/ComfyUI/custom_nodes/zz_probe
mkdir -p "$DEST"
cat > "$DEST/__init__.py" <<'PY'
import torch, torch.nn.functional as F, os

STATS = {"fast": 0, "slow": 0, "tagged_seen": 0}

_orig = None
def install():
    global _orig
    if _orig is not None:
        return
    _orig = F.linear
    def counting(inp, w, bias=None):
        try:
            tagged = getattr(w, "_gfx1200_transposed_inout", False)
        except Exception:
            tagged = False
        if tagged:
            STATS["tagged_seen"] += 1
            STATS["fast"] += 1
        else:
            STATS["slow"] += 1
        return _orig(inp, w, bias)
    F.linear = counting

install()

class Probe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"default": "p"})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"
    CATEGORY = "zz"
    OUTPUT_NODE = True
    def go(self, text):
        msg = (f"fast={STATS['fast']} slow={STATS['slow']} "
               f"tagged={STATS['tagged_seen']} "
               f"F.linear={F.linear.__name__}")
        open("/tmp/probe_stats.txt", "w").write(msg + "\n")
        print("[ZZ PROBE] " + msg, flush=True)
        return (torch.zeros(1, 8, 8, 3),)

NODE_CLASS_MAPPINGS = {"Probe": Probe}
NODE_DISPLAY_NAME_MAPPINGS = {"Probe": "Probe"}
PY

nohup python3 main.py --listen 127.0.0.1 --port 8188 --disable-smart-memory \
    --cpu-vae --fp16-unet --fp16-text-enc > /tmp/comfy_instr.log 2>&1 &
for i in $(seq 1 50); do sleep 3; curl -s --max-time 3 http://127.0.0.1:8188/system_stats >/dev/null 2>&1 && break; done

echo "=== run a tiny generation, then read the counters ==="
W=256 H=256 STEPS=2 PREFIX=instr timeout 900 \
  python3 /mnt/f/dpsk/qwen-image-2.1-amd-rdna4/scripts/generate.py "a red apple" \
  2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|bgemm_internal|return torch' | tail -6

echo
echo "=== patch status in server ==="
grep -a 'gfx1200\]' /tmp/comfy_instr.log | head -3

echo
echo "=== do any weights get tagged? ==="
grep -a 'ZZ PROBE\|fast=\|slow=' /tmp/comfy_instr.log | tail -3
cat /tmp/probe_stats.txt 2>/dev/null || echo "(no probe stats yet)"

echo
echo "=== how long did 2 steps take? ==="
tail -c 2500 /tmp/comfy_instr.log | tr '\r' '\n' | grep -oE '[0-9]+/2 \[[^]]*\]' | tail -2
