#!/bin/bash
pkill -9 -f 'main.py --listen' 2>/dev/null || true
pkill -9 -f generate.py 2>/dev/null || true
sleep 3
echo "killed stale processes"
echo
echo "=== 1) what marker is in the live ops.py? ==="
grep -n 'gfx1200 layout fix\|_mark\|mark_transposed' /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/ops.py | head -10
echo
echo "=== 2) files in the node dir ==="
ls -la /opt/ComfyUI/custom_nodes/gfx1200_layout/
echo
echo "=== 3) what does the node __init__ import? ==="
cat /opt/ComfyUI/custom_nodes/gfx1200_layout/__init__.py
echo
echo "=== 4) does the module name the ops patch imports exist? ==="
grep -n 'import' /opt/ComfyUI/custom_nodes/ComfyUI-GGUF/ops.py | grep -i gfx1200
echo
echo "=== 5) can that module be imported from the node dir? ==="
ls /opt/ComfyUI/custom_nodes/gfx1200_layout/*.py
