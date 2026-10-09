#!/bin/bash
# Which GitHub web pages are reachable? The API is fine; the web UI is not.
# Test the specific URLs the user would need.
echo "=== GitHub 网页可达性测试（每个 12 秒超时）==="
for u in \
  "https://github.com/new" \
  "https://github.com/settings/tokens" \
  "https://github.com/settings/keys" \
  "https://github.com/login" \
  "https://github.com/signup" \
  "https://github.com" \
  "https://api.github.com" \
  "https://gist.github.com" \
  "https://gitee.com" \
  "https://gitcode.com" \
  "https://gitlab.com" ; do
  out=$(curl -s -o /dev/null -w "%{http_code}|%{time_total}" --max-time 12 "$u" 2>/dev/null)
  code="${out%%|*}"; t="${out##*|}"
  case "$code" in
    200|301|302) mark="OK  " ;;
    000)         mark="DEAD" ;;
    *)           mark="?$code" ;;
  esac
  printf "  %-6s %-42s %ss\n" "$mark" "$u" "$t"
done
