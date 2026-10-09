#!/bin/bash
# Verify the extracted patches are syntactically valid and self-test correctly.
set -u
P=/mnt/f/dpsk/qwen-image-2.1-amd-rdna4/patches

echo "=== 1) syntax check ==="
for f in "$P"/*.py; do
  python3 -m py_compile "$f" && echo "  OK  $(basename "$f")" || echo "  FAIL $(basename "$f")"
done

echo
echo "=== 2) dry-run the source patches against the live install ==="
python3 "$P/patch_vram_reporting.py" --dry-run 2>&1 | tail -3
echo "---"
python3 "$P/patch_gguf_ops.py" --dry-run 2>&1 | tail -3

echo
echo "=== 3) run the layout patch self-test ==="
cd /opt/ComfyUI
python3 "$P/gfx1200_layout_patch.py" 2>&1 | grep -vE 'amdsmi|GetSegmentId|SharedSignalPool|sdp_utils|warnings.warn|sdp_utils'

echo
echo "=== 4) check the live install is already patched (should say so) ==="
python3 "$P/patch_vram_reporting.py" 2>&1 | tail -2
python3 "$P/patch_gguf_ops.py" 2>&1 | tail -2
