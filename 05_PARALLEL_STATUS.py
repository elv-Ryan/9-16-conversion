#!/usr/bin/env python3
from __future__ import annotations

import subprocess

REMOTE = "mltrain@63.247.64.18"
PORT = "5055"
D = "/home/mltrain/elv-ryan/projects/nba-yolo-podman-joe518-direct-v3_1"
GAME = "iq__2q6ZyYAmFfKDBJeWMGsWLP549twd"

script = f'''set -u
D={D!r}
GAME={GAME!r}
BASE="$D/test-runs/$GAME"
RUN=$(cat "$BASE/LATEST_PARALLEL_RUN" 2>/dev/null || true)
echo "===== PARALLEL JOE518 ====="
echo "RUN=$RUN"
if [ -n "$RUN" ]; then
  echo "--- master ---"
  PID=$(cat "$RUN/MASTER_PID" 2>/dev/null || true)
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    echo "MASTER_PID=$PID RUNNING=True"
  else
    echo "MASTER_PID=${{PID:-none}} RUNNING=False"
  fi
  echo
  echo "--- status ---"
  cat "$RUN/STATUS.json" 2>/dev/null || echo NOT_YET
  echo
  echo "--- result ---"
  cat "$RUN/PARALLEL_RESULT.json" 2>/dev/null || echo NOT_YET
  echo
  echo "--- log tail ---"
  tail -80 "$RUN/master.log" 2>/dev/null || true
fi
'''

subprocess.run(
    ["ssh", "-p", PORT, REMOTE, "bash", "-s"],
    input=script,
    text=True,
    check=False,
)
