#!/bin/bash
set -u
R=/mnt/f/dpsk/qwen-image-2.1-amd-rdna4

echo "=== remove __pycache__ ==="
find "$R" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
find "$R" -name '*.pyc' -delete 2>/dev/null
echo "  done"

echo
echo "=== regenerate a clean example image (apple) ==="
ls /opt/ComfyUI/output/*.png 2>/dev/null

echo
echo "=== repo tree (final) ==="
cd "$R" && find . -type f | sort | sed 's/^\.\///'

echo
echo "=== size ==="
du -sh "$R"
