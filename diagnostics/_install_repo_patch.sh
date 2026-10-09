#!/bin/bash
# Install the repo's patch exactly as scripts/install.sh does, replacing the
# older hand-written node, then run a generation to prove the packaged version.
set -u
R=/mnt/f/dpsk/qwen-image-2.1-amd-rdna4
COMFY=/opt/ComfyUI

echo "=== remove the old hand-written node ==="
rm -rf "$COMFY/custom_nodes/dspk_gfx1200_layout"
rm -rf "$COMFY/custom_nodes/dspk_gfx1200_patch"

echo "=== install the repo node (same logic as install.sh step 4) ==="
NODE_DIR="$COMFY/custom_nodes/gfx1200_layout"
mkdir -p "$NODE_DIR"
cp "$R/patches/gfx1200_layout_patch.py" "$NODE_DIR/"
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
ls -la "$NODE_DIR"

echo
echo "=== verify the GGUF ops patch points at a resolvable module ==="
grep -n "gfx1200_layout_patch" "$COMFY/custom_nodes/ComfyUI-GGUF/ops.py"

echo
echo "=== restart and generate ==="
pkill -9 -f 'main.py --listen' || true
sleep 3
rm -f "$COMFY"/output/*.png

W=512 H=512 STEPS=10 bash "$R/scripts/run.sh" \
  'a brass alarm clock on a marble surface, studio lighting' 2>&1 | tail -14

echo
echo "=== confirm the repo patch was the one loaded ==="
grep -a 'gfx1200\]' /tmp/comfyui-gfx1200.log | head -3
ls -la "$COMFY"/output/*.png 2>/dev/null
