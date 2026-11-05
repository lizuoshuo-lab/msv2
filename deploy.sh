#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash deploy.sh [--no-start]'
  echo 'Installs the pinned runtime, verifies it, and watches $MSV2_ROOT/videos.'
  echo 'Default data root: $HOME/msv2. Use MSV2_PYTHON to verify an existing environment without changing it.'
  exit 0
fi
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --no-start ) ]] || { echo 'Usage: bash deploy.sh [--no-start]' >&2; exit 2; }
export MSV2_ROOT="${MSV2_ROOT:-$HOME/msv2}"
bash "$APP_DIR/scripts/bootstrap.sh"
bash "$APP_DIR/msv2.sh" doctor
if [[ "${1:-}" == --no-start ]]; then
  echo 'MSV2_READY'
  exit 0
fi
mkdir -p -- "$MSV2_ROOT/logs" "$MSV2_ROOT/queue" "$MSV2_ROOT/videos"
command -v flock >/dev/null || { echo 'flock is required (util-linux).' >&2; exit 1; }
if ! flock -n "$MSV2_ROOT/queue/watcher.lock" true; then
  echo "MSV2_ALREADY_WATCHING $MSV2_ROOT/videos"
  exit 0
fi
nohup bash "$APP_DIR/msv2.sh" watch \
  </dev/null >>"$MSV2_ROOT/logs/watcher.log" 2>&1 &
watch_pid=$!
printf '%s\n' "$watch_pid" > "$MSV2_ROOT/queue/launcher.pid"
# watch owns its advisory lock; polling it here would race its initial acquire.
# Require the same Python PID to report successful initialization.
for _ in {1..50}; do
  if ! kill -0 "$watch_pid" 2>/dev/null; then
    echo "Watcher failed; see $MSV2_ROOT/logs/watcher.log" >&2
    exit 1
  fi
  if python3 - "$MSV2_ROOT/queue/watcher.json" "$watch_pid" <<'PY'
import json,os,sys
from pathlib import Path
try:
    state=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    pid=int(sys.argv[2])
    assert state['state']=='watching' and state['pid']==pid
    os.kill(pid,0)
except (OSError,ValueError,KeyError,AssertionError):
    raise SystemExit(1)
PY
  then
    echo "MSV2_DEPLOYED videos=$MSV2_ROOT/videos logs=$MSV2_ROOT/logs/watcher.log"
    exit 0
  fi
  sleep 0.2
done
echo "Watcher did not report readiness; see $MSV2_ROOT/logs/watcher.log" >&2
exit 1
