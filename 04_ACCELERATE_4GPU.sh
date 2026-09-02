#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE=${REMOTE:-mltrain@63.247.64.18}
PORT=${PORT:-5055}
REMOTE_REPO=${REMOTE_REPO:-/home/mltrain/elv-ryan/projects/nba-yolo-podman-joe518-direct-v3_1}
GAME=${GAME:-iq__2q6ZyYAmFfKDBJeWMGsWLP549twd}
GPUS=${GPUS:-2,3,4,5}
IMAGE=${IMAGE:-nba-yolo-shot-tagger:joe518-v3.1}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$REMOTE_REPO/test-runs/$GAME/parallel_4gpu_$STAMP"

scp -P "$PORT" "$ROOT/scripts/parallel_joe518_qualify.py" "$REMOTE:$REMOTE_REPO/scripts/parallel_joe518_qualify.py" >/dev/null
ssh -p "$PORT" "$REMOTE" "mkdir -p '$OUT'; nohup python3 -u '$REMOTE_REPO/scripts/parallel_joe518_qualify.py' --repo '$REMOTE_REPO' --shot-dir '/home/mltrain/elv-joe/shots/$GAME' --image '$IMAGE' --output-root '$OUT' --expected-count 518 --gpu-candidates '$GPUS' --min-gpus 2 --batch-size 8 --inference-fps 10 > '$OUT/master.log' 2>&1 < /dev/null & echo \$! > '$OUT/MASTER_PID'; echo '$OUT' > '$REMOTE_REPO/test-runs/$GAME/LATEST_PARALLEL_RUN'; echo RUNNING=True; echo RUN_ROOT='$OUT'; echo MASTER_PID=\$(cat '$OUT/MASTER_PID')"

echo "Monitor: python3 05_PARALLEL_STATUS.py"
