#!/bin/bash
# Definitive check: simulate exactly what ComfyUI-GGUF does per call, with the
# installed ops.py, and see whether the weight ends up tagged.
cd /opt/ComfyUI
python3 - <<'PY' 2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|UserWarning|bgemm_internal|return torch|Triggered internally|^\s*$'
import sys, types, importlib.util, time, warnings
warnings.filterwarnings("ignore")
import torch

sys.path.insert(0, "/opt/ComfyUI")
sys.path.insert(0, "/opt/ComfyUI/custom_nodes/gfx1200_layout")

# load the repo's patch the way the custom node does
import gfx1200_layout_patch as P
P.apply()
print("patch module   :", P.__file__)
print("F.linear now   :", torch.nn.functional.linear.__name__)
print("is_applied     :", P.is_applied())
print()

# load ComfyUI-GGUF exactly as ComfyUI does
pkg = types.ModuleType("cg"); pkg.__path__ = ["/opt/ComfyUI/custom_nodes/ComfyUI-GGUF"]
sys.modules["cg"] = pkg
for name in ("dequant", "loader", "ops"):
    s = importlib.util.spec_from_file_location(
        f"cg.{name}", f"/opt/ComfyUI/custom_nodes/ComfyUI-GGUF/{name}.py")
    m = importlib.util.module_from_spec(s); sys.modules[f"cg.{name}"] = m; s.loader.exec_module(m)
cgops = sys.modules["cg.ops"]

from cg.loader import gguf_sd_loader
sd, extra = gguf_sd_loader("/opt/ComfyUI/models/unet/qwen_image_2.1-Q4_K_M.gguf")
k0 = [k for k in sd if "attn.to_k.weight" in k][0]
w = sd[k0]
out_f, in_f = int(w.shape[0]), int(w.shape[1])

lin = cgops.GGMLOps.Linear(in_f, out_f, bias=False, device="cuda", dtype=torch.float16)
lin.weight = w
x = torch.randn(4, in_f, dtype=torch.float16, device="cuda")

print("=== does the weight arriving at F.linear carry the tag? ===")
seen = {}
_real = torch.nn.functional.linear
def spy(inp, wt, bias=None):
    seen["tagged"] = getattr(wt, "_gfx1200_transposed_inout", False)
    seen["shape"] = tuple(wt.shape)
    seen["contig"] = wt.is_contiguous()
    return _real(inp, wt, bias)
torch.nn.functional.linear = spy
lin(x)
torch.cuda.synchronize()
print("  weight shape :", seen.get("shape"))
print("  contiguous   :", seen.get("contig"))
print("  TAGGED       :", seen.get("tagged"))
print()

# restore and time the real path
torch.nn.functional.linear = _real
print("=== real timing (patched F.linear, no spy) ===")
for i in range(5):
    t0 = time.time(); lin(x); torch.cuda.synchronize()
    print(f"  call {i}: {(time.time()-t0)*1000:8.2f} ms")
PY
