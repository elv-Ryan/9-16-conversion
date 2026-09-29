#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST="$ROOT/models/nba_yolo_student/best.pt"
MANIFEST="$ROOT/models/nba_yolo_student/model_manifest.json"
REMOTE=${REMOTE:-mltrain@63.247.64.18}; SSH_PORT=${SSH_PORT:-5055}
MODEL_SOURCE=${MODEL_SOURCE:-/home/mltrain/elv-ryan/projects/9-16-conversion-ryan-v2.1-student-mvp/output/generic_basketball_v11_student_round00_v1/round01_round02_autopilot_v1/runs/round01_schemafix_v3/weights/best.pt}
EXPECTED_SHA=0921e4843864557bc1e5141f655a461af24c758d7390f2ea0553091df743865a
mkdir -p "$(dirname "$DEST")"
if [[ -f "$MODEL_SOURCE" ]]; then cp -f "$MODEL_SOURCE" "$DEST.tmp"; else scp -P "$SSH_PORT" "$REMOTE:$MODEL_SOURCE" "$DEST.tmp"; fi
ACTUAL_SHA=$(sha256sum "$DEST.tmp" | awk '{print $1}')
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || { rm -f "$DEST.tmp"; echo "model SHA mismatch $ACTUAL_SHA" >&2; exit 1; }
mv "$DEST.tmp" "$DEST"
python3 "$ROOT/scripts/verify_model_artifact.py" --model "$DEST" --manifest "$MANIFEST"
echo MODEL_ARTIFACT=PASS
