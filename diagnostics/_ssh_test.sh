#!/bin/bash
# Verify SSH to GitHub actually completes a handshake (port open != working).
echo "=== ssh handshake test (no auth needed) ==="
timeout 20 ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
  -o ConnectTimeout=15 -T git@github.com 2>&1 | head -5
echo "  exit: $?"

echo
echo "=== same via ssh.github.com:443 (fallback for blocked port 22) ==="
timeout 20 ssh -o StrictHostKeyChecking=no -o BatchMode=yes \
  -o ConnectTimeout=15 -p 443 -T git@ssh.github.com 2>&1 | head -5
echo "  exit: $?"

echo
echo "=== do we already have a key? ==="
ls -la ~/.ssh/ 2>/dev/null || echo "  no ~/.ssh"

echo
echo "=== can we create one? ==="
command -v ssh-keygen && echo "  ssh-keygen available"
