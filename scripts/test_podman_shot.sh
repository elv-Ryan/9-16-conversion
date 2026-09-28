#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd);IMAGE=${IMAGE:-nba-yolo-shot-tagger:mvp};SHOT=${1:?shot required};DEVICE=${DEVICE:-0}
SHOT_DIR=$(cd "$(dirname "$SHOT")" && pwd);SHOT_NAME=$(basename "$SHOT");OUT_DIR=$(mktemp -d);chmod 777 "$OUT_DIR";OUT="$OUT_DIR/out.jsonl"
SOURCE_IQ=${SOURCE_IQ:-$(python3 - "$SHOT" <<'PY'
import re,sys
m=re.search(r'iq__[A-Za-z0-9]+',sys.argv[1]);print(m.group(0) if m else '')
PY
)}
CONTAINER_DEVICE="$DEVICE";[[ "$DEVICE" == cpu ]] || CONTAINER_DEVICE=0
PARAMS=$(python3 - "$CONTAINER_DEVICE" "$SOURCE_IQ" <<'PY'
import json,sys
print(json.dumps({'device':sys.argv[1],'source_iq':sys.argv[2],'input_mode':'shot_file','inference_fps':10.0,'batch_size':8,'use_fp16':False,'continue_on_error':True}))
PY
)
GPU_ARGS=();[[ "$DEVICE" == cpu ]] || GPU_ARGS+=(--device "nvidia.com/gpu=$DEVICE")
printf '/elv/input/%s\n' "$SHOT_NAME" | podman run --rm -i "${GPU_ARGS[@]}" -v "$SHOT_DIR:/elv/input:ro,Z" -v "$OUT_DIR:/elv/output:Z" "$IMAGE" --output-path /elv/output/out.jsonl --params "$PARAMS"
python3 "$ROOT/scripts/validate_tagger_jsonl.py" "$OUT" --expected-source "/elv/input/$SHOT_NAME"
echo "Output: $OUT"
