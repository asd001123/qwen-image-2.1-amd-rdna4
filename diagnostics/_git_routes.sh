#!/bin/bash
# Can we do everything through api.github.com, bypassing the blocked web UI?
echo "=== 1) api.github.com reachable and responding? ==="
curl -s --max-time 15 https://api.github.com/rate_limit -o /tmp/rl.json -w "  HTTP %{http_code}  in %{time_total}s\n"
head -c 200 /tmp/rl.json 2>/dev/null; echo

echo
echo "=== 2) can we reach the git endpoint (push target)? ==="
GIT_HOST="https://github.com/USER/REPO.git/info/refs?service=git-upload-pack"
curl -s -o /dev/null -w "  github.com git endpoint: HTTP %{http_code} in %{time_total}s\n" \
     --max-time 15 "$GIT_HOST"

echo
echo "=== 3) git protocol over https -- dry probe ==="
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/octocat/Hello-World.git HEAD 2>&1 | head -3

echo
echo "=== 4) is ssh (port 22) open? ==="
timeout 8 bash -c 'cat < /dev/null > /dev/tcp/github.com/22' 2>/dev/null && echo "  port 22 OPEN" || echo "  port 22 blocked"
timeout 8 bash -c 'cat < /dev/null > /dev/tcp/ssh.github.com/443' 2>/dev/null && echo "  ssh.github.com:443 OPEN" || echo "  ssh.github.com:443 blocked"
