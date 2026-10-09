#!/bin/bash
# Test whether a smarter patch (transpose cached across steps, dequant still
# per-call) reduces the per-step cost. Measures the ceiling of what's achievable.
cd /opt/ComfyUI
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
from cg.ops import dequantize_tensor

sd, extra = gguf_sd_loader("/opt/ComfyUI/models/unet/qwen_image_2.1-Q4_K_M.gguf")
k0 = [k for k in sd if "attn.to_k.weight" in k][0]
w = sd[k0]

print("=== cost split per layer ===")
t0=time.time()
for _ in range(5): d = dequantize_tensor(w, torch.float16, None)
t_deq = (time.time()-t0)/5

d = dequantize_tensor(w, torch.float16, None).to("cuda")
t0=time.time()
for _ in range(5): dt = d.t().contiguous()
t_tr = (time.time()-t0)/5

x = torch.randn(4096, d.shape[1], dtype=torch.float16, device="cuda")
dt = P.mark_transposed(d.t().contiguous())
for _ in range(2): torch.nn.functional.linear(x, dt)
torch.cuda.synchronize(); t0=time.time()
for _ in range(5): torch.nn.functional.linear(x, dt)
torch.cuda.synchronize()
t_mm = (time.time()-t0)/5

print(f"  dequant (CPU->GPU copy incl.) : {t_deq*1000:7.1f} ms")
print(f"  transpose+contiguous on GPU   : {t_tr*1000:7.1f} ms")
print(f"  matmul                        : {t_mm*1000:7.1f} ms")
print(f"  TOTAL per layer               : {(t_deq+t_tr+t_mm)*1000:7.1f} ms")
print()
print("=== if we cached the transposed weight (transpose once, reuse) ===")
print(f"  per layer would be            : {(t_deq+t_mm)*1000:7.1f} ms")
saved = t_tr*160
print(f"  saved per forward pass        : {saved:.1f} s per step")
print()
print("  NOTE: dequant is unavoidable with ComfyUI-GGUF's design;")
print(f"  it alone costs {t_deq*160:.1f} s per step across 160 layers.")
PY
