#!/bin/bash
# Generate an SSH key for GitHub push (port 22 works, HTTPS is blocked).
set -u
KEY=~/.ssh/id_ed25519_github

if [ -f "$KEY" ]; then
  echo "key already exists: $KEY"
else
  ssh-keygen -t ed25519 -f "$KEY" -N "" -C "gfx1200-toolkit@wsl" >/dev/null 2>&1
  echo "generated: $KEY"
fi

mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat > ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
  ServerAliveInterval 30

# Fallback for networks that block port 22
Host github-ssh443
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
echo "wrote ~/.ssh/config"

echo
echo "=== PUBLIC KEY (this is the one to register) ==="
cat "$KEY.pub"
echo
echo "=== fingerprint ==="
ssh-keygen -lf "$KEY.pub"

echo
echo "=== test auth (should still fail until the key is registered) ==="
timeout 20 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -T git@github.com 2>&1 | head -2
