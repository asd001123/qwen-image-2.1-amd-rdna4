#!/bin/bash
# Validate the repo's scripts end to end (syntax, dry-run install, real generate).
set -u
R=/mnt/f/dpsk/qwen-image-2.1-amd-rdna4

echo "=== bash syntax ==="
for f in "$R"/scripts/*.sh; do
  bash -n "$f" && echo "  OK  $(basename "$f")" || echo "  FAIL $(basename "$f")"
done

echo
echo "=== python syntax ==="
for f in "$R"/scripts/*.py "$R"/patches/*.py; do
  python3 -m py_compile "$f" && echo "  OK  $(basename "$f")" || echo "  FAIL $(basename "$f")"
done

echo
echo "=== install.sh --dry-run ==="
COMFYUI_PATH=/opt/ComfyUI bash "$R/scripts/install.sh" --dry-run --skip-models 2>&1 | tail -25
