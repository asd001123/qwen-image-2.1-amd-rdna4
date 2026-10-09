#!/bin/bash
# The big question: 37.9 ms/layer is dequant. If we cache dequantized fp16
# weights we save 6.1 s/step. But caching costs 13.25 GiB of VRAM.
# Is that affordable? Measure the actual headroom.
cd /opt/ComfyUI
python3 - <<'PY' 2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|UserWarning|bgemm_internal|return torch|Triggered internally|^\s*$'
import sys, time, warnings, types, importlib.util
warnings.filterwarnings("ignore")
import torch
sys.path.insert(0, "/opt/ComfyUI")

sys.path.insert(0, "/opt/ComfyUI/custom_nodes/gfx1200_layout")
import gfx1200_layout_patch as P; P.apply()

pkg = types.ModuleType("cg"); pkg.__path__ = ["/opt/ComfyUI/custom_nodes/ComfyUI-GGUF"]
sys.modules["cg"] = pkg
for name in ("dequant", "loader", "ops"):
    s = importlib.util.spec_from_file_location(f"cg.{name}", f"/opt/ComfyUI/custom_nodes/ComfyUI-GGUF/{name}.py")
    m = importlib.util.module_from_spec(s); sys.modules[f"cg.{name}"] = m; s.loader.exec_module(m)
from cg.loader import gguf_sd_loader
from cg.ops import dequantize_tensor

sd, extra = gguf_sd_loader("/opt/ComfyUI/models/unet/qwen_image_2.1-Q4_K_M.gguf")

print("=== how much VRAM would caching ALL dequantized weights need? ===")
tot = 0
for k, v in sd.items():
    if hasattr(v, "tensor_shape"):
        n = 1
        for d in v.tensor_shape: n *= int(d)
        tot += n * 2
print(f"  DiT fp16 cache      : {tot/2**30:.2f} GiB")

# measure free VRAM
free, total = torch.cuda.mem_get_info()
print(f"  mem_get_info        : free={free}  total={total/2**30:.2f} GiB (free value is unreliable)")

# actually measure headroom by allocating until failure
held = []
chunk = 256*1024*1024
try:
    while True:
        t = torch.empty(chunk, dtype=torch.uint8, device="cuda"); t.fill_(1)
        held.append(t)
except Exception:
    pass
room = len(held)*chunk/2**30
del held
torch.cuda.empty_cache()
print(f"  allocatable (empty) : {room:.2f} GiB")
print()
print(f"  => caching needs {tot/2**30:.2f} GiB; we have {room:.2f} GiB")
if room >= tot/2**30 + 1.0:
    print("  => WOULD FIT (with ~1 GiB spare)")
else:
    print("  => DOES NOT FIT; caching dequantized weights is not viable.")
    print("     (this is exactly why the earlier caching attempt OOM'd)")
PY
