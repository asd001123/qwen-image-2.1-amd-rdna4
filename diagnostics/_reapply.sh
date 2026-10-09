#!/bin/bash
# Restore ops.py from the pre-patch backup, then apply the new patch once.
set -u
OPS=/opt/ComfyUI/custom_nodes/ComfyUI-GGUF/ops.py

echo "=== available backups ==="
ls -la "$OPS".gfx1200-bak-* "$OPS".dspk-bak 2>/dev/null | tail -5

# prefer the oldest (truly pristine) backup
PRISTINE=$(ls -tr "$OPS".gfx1200-bak-* "$OPS".dspk-bak 2>/dev/null | head -1)
echo "using pristine: $PRISTINE"

if [ -n "$PRISTINE" ]; then
  cp -f "$PRISTINE" "$OPS"
  echo "restored"
fi

echo
echo "=== confirm the original block is present ==="
grep -n -A3 'def forward_ggml_cast_weights' "$OPS" | head -12

echo
echo "=== apply the patch ==="
python3 /mnt/f/dpsk/qwen-image-2.1-amd-rdna4/patches/patch_gguf_ops.py 2>&1 | tail -8

echo
echo "=== resulting block ==="
grep -n -A22 'def forward_ggml_cast_weights' "$OPS" | head -28
python3 -m py_compile "$OPS" && echo "syntax OK"
