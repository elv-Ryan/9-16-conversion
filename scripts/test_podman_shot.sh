#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-nba-yolo-shot-tagger:mvp}
SHOT=${1:?Usage: scripts/test_podman_shot.sh /absolute/path/to/shot.mp4}
DEVICE=${DEVICE:-0}
if [[ ! -f "$SHOT" ]]; then echo "Shot not found: $SHOT" >&2; exit 2; fi

SHOT_DIR=$(cd "$(dirname "$SHOT")" && pwd)
SHOT_NAME=$(basename "$SHOT")
OUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/nba-yolo-shot-test.XXXXXX")
chmod 777 "$OUT_DIR"
OUT_JSONL="$OUT_DIR/out.jsonl"

CONTAINER_DEVICE="$DEVICE"
if [[ "$DEVICE" != "cpu" ]]; then CONTAINER_DEVICE=0; fi

PARAMS=$(python3 - "$CONTAINER_DEVICE" <<'PY'
import json,sys
device=sys.argv[1]
print(json.dumps({
  "device":device,
  "input_mode":"shot_file",
  "inference_fps":10.0,
  "batch_size":2 if device == "cpu" else 8,
  "use_fp16":device != "cpu",
  "continue_on_error":True,
  "emit_focus_track":False,
  "include_focus_samples":True,
  "emit_progress_ratio":False
}))
PY
)

GPU_ARGS=()
if [[ "$DEVICE" != "cpu" ]]; then
  GPU_ARGS+=(--device "nvidia.com/gpu=$DEVICE")
fi

printf '/elv/input/%s\n' "$SHOT_NAME" | podman run --rm -i \
  "${GPU_ARGS[@]}" \
  -v "$SHOT_DIR:/elv/input:ro,Z" \
  -v "$OUT_DIR:/elv/output:Z" \
  "$IMAGE" \
  --output-path /elv/output/out.jsonl \
  --params "$PARAMS"

python3 "$ROOT/scripts/validate_tagger_jsonl.py" "$OUT_JSONL" \
  --expected-source "/elv/input/$SHOT_NAME"
echo "Output: $OUT_JSONL"
