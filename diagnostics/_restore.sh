#!/bin/bash
# The dry-run test accidentally applied the new patch on top of the old one.
# Restore the known-good single-patch version, then verify.
set -u
cd /opt/ComfyUI/comfy

echo "=== current marker counts ==="
grep -c 'DSPK_ROCM_FREE_MEM_PATCH' model_management.py
grep -c 'GFX1200_ROCM_FREE_MEM_PATCH' model_management.py

echo
echo "=== restoring from dspk-bak3 (the last known-good single-patch state) ==="
cp -f model_management.py.dspk-bak3 model_management.py
rm -f model_management.py.gfx1200-bak-*
python3 -m py_compile model_management.py && echo "  syntax OK"

echo
echo "=== marker counts after restore ==="
echo -n "  DSPK (old): "; grep -c 'DSPK_ROCM_FREE_MEM_PATCH' model_management.py
echo -n "  GFX1200 (new): "; grep -c 'GFX1200_ROCM_FREE_MEM_PATCH' model_management.py || true

echo
echo "=== verify the patch still guards on the bogus value ==="
grep -n 'usable VRAM probed\|mem_free_cuda < (' model_management.py | head -3
