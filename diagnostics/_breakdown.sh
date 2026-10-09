#!/bin/bash
# Where do the remaining ~24 s per step go? Break the step down.
pkill -9 -f 'main.py --listen' 2>/dev/null || true
sleep 2
cd /opt/ComfyUI

python3 - <<'PY' 2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|UserWarning|bgemm_internal|return torch|Triggered internally|^\s*$'
import sys, time, warnings
warnings.filterwarnings("ignore")
import torch
sys.path.insert(0, "/opt/ComfyUI"); sys.path.insert(0, "/opt/ComfyUI/custom_nodes/gfx1200_layout")
import gfx1200_layout_patch as P; P.apply()

print("=== attention cost at 512x512 latent (4096 tokens) ===")
# DiT attention: 4096 tokens, 24-ish heads, head_dim 128
for n_tok in (1024, 4096):          # 256x256 and 512x512 latents
    q = torch.randn(1, 24, n_tok, 128, dtype=torch.float16, device="cuda")
    def attn():
        return torch.nn.functional.scaled_dot_product_attention(q, q, q)
    for _ in range(2): attn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(5): attn()
    torch.cuda.synchronize()
    print(f"  SDPA {n_tok:5d} tokens: {(time.time()-t0)/5*1000:8.2f} ms")

print()
print("=== what does ComfyUI report during sampling? (check log) ===")
PY

echo
echo "=== per-step breakdown from the last run log ==="
grep -a -oE '[0-9]+\.[0-9]+s/it|[0-9]+%\|[^]]*\]' /tmp/comfyui-gfx1200.log 2>/dev/null | tail -5
echo
echo "=== was the text encoder reloaded each step? ==="
grep -acE 'Requested to load|loaded completely' /tmp/comfyui-gfx1200.log 2>/dev/null
grep -aE 'Requested to load|loaded completely|model_type|dtype' /tmp/comfyui-gfx1200.log 2>/dev/null | tail -8
