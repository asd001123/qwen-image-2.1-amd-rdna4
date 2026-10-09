#!/bin/bash
# The earlier caching attempt OOM'd because it kept BOTH the quantized weight
# and the fp16 copy, AND the text encoder was resident at the same time.
#
# Feasible design:
#   * dequantize ONCE per weight into fp16, transposed, and DROP the quantized
#     original (so total = 13.25 GiB, not 13.25 + 4.04)
#   * free the text encoder before sampling starts
#
# Simulate the memory profile of that design.
cd /opt/ComfyUI
python3 - <<'PY' 2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|UserWarning|bgemm_internal|return torch|Triggered internally|HSA exception|^\s*$'
import sys, warnings, types, importlib.util, gc
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

print("=== simulation: dequantize everything, dropping the originals ===")
sd, extra = gguf_sd_loader("/opt/ComfyUI/models/unet/qwen_image_2.1-Q4_K_M.gguf")

alloc0 = torch.cuda.memory_allocated()
keys = [k for k in sd if hasattr(sd[k], "tensor_shape")]
print(f"  tensors to convert: {len(keys)}")

converted = 0
fail = None
import time
t0 = time.time()
for k in keys:
    v = sd[k]
    try:
        d = dequantize_tensor(v, torch.float16, None).to("cuda")
        if d.dim() == 2:
            d = P.mark_transposed(d.t().contiguous())
        sd[k] = d          # replace, so the quantized original is freed
        converted += 1
    except Exception as e:
        fail = (k, type(e).__name__, str(e)[:60])
        break

dt = time.time() - t0
alloc = torch.cuda.memory_allocated()

print(f"  converted    : {converted}/{len(keys)}")
print(f"  GPU allocated: {alloc/2**30:.2f} GiB")
print(f"  wall time    : {dt:.1f} s  (one-off cost)")
if fail:
    print(f"  FAILED at    : {fail[0]}")
    print(f"                 {fail[1]}: {fail[2]}")
print()
if not fail:
    per_layer = 1.9 + 5.2   # matmul + nothing (already transposed)
    print(f"  if this were cached, per-step linear cost would drop from")
    print(f"  ~45 ms/layer to ~2 ms/layer => {160*2/1000:.1f} s saved per step")
else:
    print("  => full resident caching does NOT fit; staged loading is required.")
PY
