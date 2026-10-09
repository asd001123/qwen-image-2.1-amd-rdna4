#!/bin/bash
# 25 ms per linear call, yet 27 s per sampling step. Where does the rest go?
cd /opt/ComfyUI
pkill -9 -f 'main.py --listen' 2>/dev/null || true
sleep 2

python3 - <<'PY' 2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|UserWarning|bgemm_internal|return torch|Triggered internally|^\s*$'
import sys, time, warnings, types, importlib.util
warnings.filterwarnings("ignore")
import torch
sys.path.insert(0, "/opt/ComfyUI"); sys.path.insert(0, "/opt/ComfyUI/custom_nodes/gfx1200_layout")
import gfx1200_layout_patch as P; P.apply()

pkg = types.ModuleType("cg"); pkg.__path__ = ["/opt/ComfyUI/custom_nodes/ComfyUI-GGUF"]
sys.modules["cg"] = pkg
for name in ("dequant", "loader", "ops"):
    s = importlib.util.spec_from_file_location(f"cg.{name}", f"/opt/ComfyUI/custom_nodes/ComfyUI-GGUF/{name}.py")
    m = importlib.util.module_from_spec(s); sys.modules[f"cg.{name}"] = m; s.loader.exec_module(m)
cgops = sys.modules["cg.ops"]
from cg.loader import gguf_sd_loader

sd, extra = gguf_sd_loader("/opt/ComfyUI/models/unet/qwen_image_2.1-Q4_K_M.gguf")

# Count the Q4_K linears and estimate: dequant cost + matmul cost per pass
q4 = [k for k in sd if hasattr(sd[k], "tensor_type") and int(sd[k].tensor_type) == 12]
print(f"  Q4_K tensors: {len(q4)}")

# Time a full dequant of one tensor
from cg.ops import dequantize_tensor
k0 = [k for k in sd if "attn.to_k.weight" in k][0]
w = sd[k0]
for _ in range(1): dequantize_tensor(w, torch.float16, None)
t0=time.time()
for _ in range(5): dequantize_tensor(w, torch.float16, None)
t_deq = (time.time()-t0)/5
print(f"  dequant one 4096x4096 tensor: {t_deq*1000:.1f} ms")

# Estimate per-layer cost = dequant + transpose-copy + matmul
out_f, in_f = int(w.shape[0]), int(w.shape[1])
lin = cgops.GGMLOps.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.float16)
lin.weight = w
x = torch.randn(4096, in_f, dtype=torch.float16, device="cuda")   # 512x512 latent-ish
for _ in range(2): lin(x)
torch.cuda.synchronize(); t0=time.time()
for _ in range(5): lin(x)
torch.cuda.synchronize()
t_layer = (time.time()-t0)/5
print(f"  full layer (dequant+transpose+matmul): {t_layer*1000:.1f} ms")
print()
print(f"  => 160 Q4_K layers * {t_layer*1000:.0f} ms = {t_layer*160:.1f} s per forward pass")
print(f"  => at 4096 tokens this is dominated by DEQUANT, not matmul:")
print(f"     dequant alone: {t_deq*1000:.1f} ms  vs matmul ~25 ms")
print()
print("  *** The real fix: avoid re-dequantizing weights every step. ***")
PY
