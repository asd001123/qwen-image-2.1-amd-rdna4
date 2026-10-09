#!/bin/bash
# Debug why scripts/run.sh fails when invoked from WSL.
R=/mnt/f/dpsk/qwen-image-2.1-amd-rdna4

echo "=== run.sh with tracing ==="
export W=256 H=256 STEPS=2
bash -x "$R/scripts/run.sh" 'test' 2>&1 | tail -40
echo
echo "exit code: $?"
