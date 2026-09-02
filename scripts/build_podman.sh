#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-nba-yolo-shot-tagger:mvp}
python3 "$ROOT/scripts/verify_model_artifact.py" \
  --model "$ROOT/models/nba_yolo_student/best.pt" \
  --manifest "$ROOT/models/nba_yolo_student/model_manifest.json"
podman build --format docker -t "$IMAGE" -f "$ROOT/Containerfile" "$ROOT"
echo "Built $IMAGE"
