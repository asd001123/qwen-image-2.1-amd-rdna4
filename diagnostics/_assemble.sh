#!/bin/bash
# Assemble the final repository: copy examples, verify contents, check for
# anything that must not be published.
set -u
R=/mnt/f/dpsk/qwen-image-2.1-amd-rdna4

echo "=== copy example images ==="
mkdir -p "$R/examples"
cp -f /opt/ComfyUI/output/nocache_00001_.png "$R/examples/apple.png" 2>/dev/null && echo "  apple.png"
cp -f /mnt/f/dpsk/qwen-result-2.png "$R/examples/tea-bowl.png" 2>/dev/null && echo "  tea-bowl.png"

echo
echo "=== final tree ==="
cd "$R"
find . -type f -not -path './.git/*' | sort | sed 's/^\.\///'

echo
echo "=== sizes ==="
du -sh "$R"
echo
echo "=== SAFETY: anything with secrets or absolute user paths? ==="
grep -rniE 'password|passwd|token|api[_-]?key|secret|2162602236|Asd001123' \
  --include='*.md' --include='*.py' --include='*.sh' . 2>/dev/null \
  | grep -viE 'password authentication|PAT|personal access token|api key|credential' || echo "  none found"

echo
echo "=== SAFETY: hardcoded /mnt/f/dpsk paths? ==="
grep -rn '/mnt/f/dpsk' --include='*.py' --include='*.sh' --include='*.md' . 2>/dev/null | head -5 || echo "  none"
