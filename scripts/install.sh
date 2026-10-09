#!/usr/bin/env bash
#
# One-shot installer for Qwen-Image-2.1 on AMD RDNA4 (gfx1200) under WSL2.
#
# What it does, in order:
#   1. sanity-check the ROCm/PyTorch/WSL environment
#   2. install ComfyUI Python dependencies WITHOUT letting pip replace the
#      ROCm torch build (see docs/PITFALLS.md #1 -- this is the dangerous one)
#   3. download the quantized model files (~11.3 GB) via a HuggingFace mirror
#   4. apply the two gfx1200 source patches
#   5. verify everything
#
# Usage:
#   ./install.sh                 # full install
#   ./install.sh --skip-models   # environment + patches only
#   ./install.sh --dry-run       # report what would happen
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$HERE")"

COMFY="${COMFYUI_PATH:-/opt/ComfyUI}"
MODELS="${MODELS_DIR:-/opt/ComfyUI/models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

SKIP_MODELS=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --dry-run)     DRY_RUN=1 ;;
    -h|--help)     sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ helpers
c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_rst=$'\033[0m'
ok()   { echo "${c_grn}  OK${c_rst}   $*"; }
warn() { echo "${c_ylw}  WARN${c_rst} $*"; }
die()  { echo "${c_red}  FAIL${c_rst} $*" >&2; exit 1; }
step() { echo; echo "=== $* ==="; }

run() {
  if [ "$DRY_RUN" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi
}

# ------------------------------------------------------- 1. environment
step "1/5  Environment check"

[ -e /dev/dxg ] || warn "/dev/dxg not present - is this really WSL2 with GPU passthrough?"

python3 - <<'PY' || die "PyTorch/ROCm check failed. See docs/PREREQUISITES.md"
import sys
try:
    import torch
except Exception as e:
    print(f"  torch import failed: {e}"); sys.exit(1)
print(f"  torch      : {torch.__version__}")
print(f"  hip        : {torch.version.hip}")
print(f"  cuda avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  device     : {p.name}  ({getattr(p, 'gcnArchName', '?')})")
    print(f"  total VRAM : {p.total_memory / 2**30:.2f} GiB")
    x = torch.randn(256, 256, device="cuda")
    print(f"  matmul     : OK ({(x @ x).sum().item():.3f})")
else:
    print("  GPU not visible to PyTorch"); sys.exit(1)
PY
ok "ROCm + PyTorch operational"

# ------------------------------------------------------------- 2. deps
step "2/5  ComfyUI Python dependencies"

[ -d "$COMFY" ] || die "ComfyUI not found at $COMFY (set COMFYUI_PATH)"

# Constraint file freezes the ROCm torch build so pip cannot swap in a CUDA one.
CONSTRAINTS="$COMFY/rocm-constraints.txt"
if [ -f "$CONSTRAINTS" ]; then
  ok "constraints present: $CONSTRAINTS"
else
  warn "no constraints file; generating one from the current torch build"
  if [ "$DRY_RUN" = 0 ]; then
    python3 - "$CONSTRAINTS" <<'PY'
import sys, torch, importlib.util
out = []
out.append(f"torch=={torch.__version__}")
for name in ("torchvision", "torchaudio"):
    spec = importlib.util.find_spec(name)
    if spec:
        mod = __import__(name)
        out.append(f"{name}=={mod.__version__}")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
print("  wrote", sys.argv[1])
PY
  fi
fi

# Packages ComfyUI needs at runtime, minus the torch stack.
PKGS=(numpy einops transformers tokenizers sentencepiece safetensors
      aiohttp yarl pyyaml Pillow scipy tqdm psutil alembic
      "SQLAlchemy>=2.0.0" filelock "av>=17.0.0" requests
      "simpleeval>=1.0.0" blake3 "kornia>=0.7.1" spandrel
      "pydantic~=2.0" "pydantic-settings~=2.0")

echo "  installing runtime deps (torch held back by constraints)..."
run python3 -m pip install --no-input --break-system-packages \
    -c "$CONSTRAINTS" "${PKGS[@]}" 2>&1 | tail -3

# ComfyUI-GGUF needs the gguf reader; torchsde is required by comfy.samplers.
run python3 -m pip install --no-input --break-system-packages \
    -c "$CONSTRAINTS" "gguf>=0.13.0" torchsde 2>&1 | tail -3

echo "  verifying torch survived..."
python3 -c "import torch;print('  torch is now:', torch.__version__, '| cuda:', torch.cuda.is_available())"
python3 -c "import torch;assert torch.cuda.is_available(),'GPU LOST'" \
  || die "pip replaced the ROCm torch build. Restore from /opt/wheels or reinstall ROCm torch."
ok "dependencies installed, torch intact"

# ----------------------------------------------------------- 3. models
step "3/5  Model files"

if [ "$SKIP_MODELS" = 1 ]; then
  warn "skipped (--skip-models)"
else
  echo "  mirror: $HF_ENDPOINT"
  bash "$HERE/fetch_models.sh" || die "model download failed"
fi

# ----------------------------------------------------------- 4. patches
step "4/5  gfx1200 source patches"

run python3 "$REPO_ROOT/patches/patch_vram_reporting.py" --comfy-path "$COMFY"
run python3 "$REPO_ROOT/patches/patch_gguf_ops.py"

# The F.linear patch is applied at runtime by a tiny custom node.
NODE_DIR="$COMFY/custom_nodes/gfx1200_layout"
if [ "$DRY_RUN" = 0 ]; then
  mkdir -p "$NODE_DIR"
  cp "$REPO_ROOT/patches/gfx1200_layout_patch.py" "$NODE_DIR/"
  cat > "$NODE_DIR/__init__.py" <<'PY'
"""Apply the gfx1200 transposed-weight layout patch at ComfyUI startup."""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import gfx1200_layout_patch as _p
    _p.apply()
except Exception as e:
    print(f"[gfx1200] layout patch failed to load: {type(e).__name__}: {e}")
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
PY
  ok "custom node installed: $NODE_DIR"
fi

# ---------------------------------------------------------- 5. verify
step "5/5  Verification"

echo "  running the layout-patch self-test..."
if [ "$DRY_RUN" = 0 ]; then
  python3 "$REPO_ROOT/patches/gfx1200_layout_patch.py" 2>&1 \
    | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|UserWarning|bgemm_internal|Triggered internally' \
    | sed 's/^/  /'
fi

echo
echo "  model files:"
for f in "$MODELS/unet/qwen_image_2.1-Q4_K_M.gguf" \
         "$MODELS/text_encoders/qwen3vl_8b_w4a8.safetensors" \
         "$MODELS/vae/qwen_image_2.1_vae_bf16.safetensors"; do
  if [ -f "$f" ]; then ok "$(stat -c %s "$f") bytes  $(basename "$f")"
  else warn "missing: $f"; fi
done

cat <<EOF

${c_grn}Done.${c_rst}

Start ComfyUI and generate:

  wsl -d Ubuntu-24.04 --exec /bin/bash $HERE/run.sh "a red apple on a wooden table"

If generation is slow or OOMs, read docs/PITFALLS.md -- the VRAM ceiling is
tunable:

  export GFX1200_USABLE_VRAM_MB=15000

EOF
