#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DEST="$ROOT/models/nba_yolo_student/best.pt"
MANIFEST="$ROOT/models/nba_yolo_student/model_manifest.json"

REMOTE=${REMOTE:-mltrain@63.247.64.18}
SSH_PORT=${SSH_PORT:-5055}

MODEL_SOURCE=${MODEL_SOURCE:-/home/mltrain/elv-ryan/projects/9-16-conversion-ryan-v2.1-student-mvp/output/generic_basketball_v11_student_round00_v1/yolo_fp32_6gpu_round00_v7/long_run/focus_target_round00_fp32/weights/best.pt}

EXPECTED_SHA=e1f498b0447f77c30d2b82210367b556d5f649caeee370f5fd3d20f7a0fec770

mkdir -p "$(dirname "$DEST")"

if [[ -f "$MODEL_SOURCE" ]]; then
    cp -f "$MODEL_SOURCE" "$DEST.tmp"
else
    scp -P "$SSH_PORT" "$REMOTE:$MODEL_SOURCE" "$DEST.tmp"
fi

ACTUAL_SHA=$(sha256sum "$DEST.tmp" | awk '{print $1}')

if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
    rm -f "$DEST.tmp"
    echo "ERROR: model SHA mismatch" >&2
    echo "expected=$EXPECTED_SHA" >&2
    echo "actual=$ACTUAL_SHA" >&2
    exit 1
fi

mv "$DEST.tmp" "$DEST"

python3 "$ROOT/scripts/verify_model_artifact.py" \
    --model "$DEST" \
    --manifest "$MANIFEST"

echo "MODEL_ARTIFACT=PASS"
