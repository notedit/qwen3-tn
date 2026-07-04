#!/usr/bin/env bash
# 后台起长任务,terminal 断开不影响。
# 用法: scripts/launch.sh <name> -- <cmd...>
# 日志: logs/<name>_<ts>.log(logs/<name>.log 软链到最新),PID: logs/<name>.pid
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="$1"; shift
[ "${1:-}" = "--" ] && shift
mkdir -p "$ROOT/logs"

PIDFILE="$ROOT/logs/$NAME.pid"
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: $NAME already running (pid $(cat "$PIDFILE"))" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/${NAME}_${TS}.log"
ln -sf "$(basename "$LOG")" "$ROOT/logs/$NAME.log"

# .env 注入(API key 等)
if [ -f "$ROOT/.env" ]; then set -a; . "$ROOT/.env"; set +a; fi

cd "$ROOT"
setsid nohup "$@" >"$LOG" 2>&1 </dev/null &
PID=$!
echo "$PID" >"$PIDFILE"
echo "launched $NAME pid=$PID log=$LOG"
