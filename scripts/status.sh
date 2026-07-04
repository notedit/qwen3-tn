#!/usr/bin/env bash
# 查看所有 launch.sh 起的任务状态与日志尾部。用法: scripts/status.sh [name]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/logs" 2>/dev/null || { echo "no logs/"; exit 0; }
for f in ${1:+$1.pid} $( [ -z "${1:-}" ] && ls *.pid 2>/dev/null ); do
  [ -f "$f" ] || continue
  name="${f%.pid}"; pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then st="RUNNING"; else st="DEAD"; fi
  echo "== $name (pid $pid) [$st] =="
  tail -n 5 "$name.log" 2>/dev/null | sed 's/^/   /'
done
