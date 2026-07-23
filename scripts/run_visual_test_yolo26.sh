#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"
VENV="${YOLO_VENV:-.venv-yolo26}"
SHORTS="${SHORTS_DIR:-/home/elv-ryan/projects/9-16-conversion/test-files/short}"
OUT="${OUT_DIR:-test-output/visual-v5.3-yolo26}"
PROFILE="${YOLO_QUALITY_PROFILE:-balanced}"
MODEL_DIR="$ROOT/models/yolo26"
PYTHON="$VENV/bin/python"
RENDERER="$ROOT/scripts/render_focus_previews.py"
ANALYZER="$ROOT/scripts/analyze_focus_run.py"

case "$PROFILE" in
  high)
    DEFAULT_SPORTS_DETECT="$MODEL_DIR/yolo26m.pt"
    DEFAULT_MOVIE_DETECT="$MODEL_DIR/yolo26s.pt"
    DEFAULT_SPORTS_POSE="$MODEL_DIR/yolo26s-pose.pt"
    DEFAULT_MOVIE_POSE="$MODEL_DIR/yolo26s-pose.pt"
    DEFAULT_SPORTS_IMGSZ=1600
    DEFAULT_MOVIE_IMGSZ=1280
    DEFAULT_SPORTS_POSE_IMGSZ=1280
    DEFAULT_MOVIE_POSE_IMGSZ=1280
    DEFAULT_SPORTS_DETECTION_FPS=20
    DEFAULT_MOVIE_DETECTION_FPS=15
    DEFAULT_SPORTS_MAX_MISSES=5
    DEFAULT_MOVIE_MAX_MISSES=5
    DEFAULT_SPORTS_BALL_MAX_MISSES=4
    ;;
  balanced)
    DEFAULT_SPORTS_DETECT="$MODEL_DIR/yolo26s.pt"
    DEFAULT_MOVIE_DETECT="$MODEL_DIR/yolo26s.pt"
    DEFAULT_SPORTS_POSE="$MODEL_DIR/yolo26n-pose.pt"
    DEFAULT_MOVIE_POSE="$MODEL_DIR/yolo26n-pose.pt"
    DEFAULT_SPORTS_IMGSZ=1280
    DEFAULT_MOVIE_IMGSZ=960
    DEFAULT_SPORTS_POSE_IMGSZ=960
    DEFAULT_MOVIE_POSE_IMGSZ=960
    DEFAULT_SPORTS_DETECTION_FPS=10
    DEFAULT_MOVIE_DETECTION_FPS=10
    DEFAULT_SPORTS_MAX_MISSES=5
    DEFAULT_MOVIE_MAX_MISSES=5
    DEFAULT_SPORTS_BALL_MAX_MISSES=4
    ;;
  *)
    echo "YOLO_QUALITY_PROFILE must be 'high' or 'balanced'" >&2
    exit 2
    ;;
esac

SPORTS_DETECT_MODEL="${SPORTS_YOLO_DETECT_MODEL:-${YOLO_DETECT_MODEL:-$DEFAULT_SPORTS_DETECT}}"
MOVIE_DETECT_MODEL="${MOVIE_YOLO_DETECT_MODEL:-${YOLO_DETECT_MODEL:-$DEFAULT_MOVIE_DETECT}}"
SPORTS_POSE_MODEL="${SPORTS_YOLO_POSE_MODEL:-${YOLO_POSE_MODEL:-$DEFAULT_SPORTS_POSE}}"
MOVIE_POSE_MODEL="${MOVIE_YOLO_POSE_MODEL:-${YOLO_POSE_MODEL:-$DEFAULT_MOVIE_POSE}}"
SPORTS_IMGSZ="${SPORTS_YOLO_IMGSZ:-$DEFAULT_SPORTS_IMGSZ}"
MOVIE_IMGSZ="${MOVIE_YOLO_IMGSZ:-$DEFAULT_MOVIE_IMGSZ}"
SPORTS_POSE_IMGSZ="${SPORTS_YOLO_POSE_IMGSZ:-$DEFAULT_SPORTS_POSE_IMGSZ}"
MOVIE_POSE_IMGSZ="${MOVIE_YOLO_POSE_IMGSZ:-$DEFAULT_MOVIE_POSE_IMGSZ}"
SPORTS_DETECTION_FPS="${SPORTS_DETECTION_FPS:-$DEFAULT_SPORTS_DETECTION_FPS}"
MOVIE_DETECTION_FPS="${MOVIE_DETECTION_FPS:-$DEFAULT_MOVIE_DETECTION_FPS}"
SPORTS_MAX_MISSES="${SPORTS_MAX_MISSED_UPDATES:-$DEFAULT_SPORTS_MAX_MISSES}"
MOVIE_MAX_MISSES="${MOVIE_MAX_MISSED_UPDATES:-$DEFAULT_MOVIE_MAX_MISSES}"
SPORTS_BALL_MAX_MISSES="${SPORTS_BALL_MAX_MISSES:-$DEFAULT_SPORTS_BALL_MAX_MISSES}"

[[ -x "$PYTHON" ]] || { echo "Missing $ROOT/$VENV; run setup_yolo26_experiment.sh" >&2; exit 2; }
[[ -d "$SHORTS" ]] || { echo "Missing shorts: $SHORTS" >&2; exit 2; }

if "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  DEVICE="${YOLO_DEVICE:-0}"
  HALF=true
else
  DEVICE="cpu"
  HALF=false
fi

rm -rf "$OUT"
mkdir -p "$OUT/previews" "$OUT/debug" "$OUT/metrics"

for mode in sports movie; do
  mapfile -t clips < <(find "$SHORTS" -maxdepth 1 -type f -name "${mode}_*.mp4" -print | sort)
  if (( ${#clips[@]} == 0 )); then
    echo "No ${mode}_*.mp4 clips found in $SHORTS" >&2
    exit 2
  fi

  if [[ "$mode" == sports ]]; then
    DETECT_MODEL="$SPORTS_DETECT_MODEL"
    POSE_MODEL="$SPORTS_POSE_MODEL"
    IMGSZ="$SPORTS_IMGSZ"
    POSE_IMGSZ="$SPORTS_POSE_IMGSZ"
    DETECTION_FPS="$SPORTS_DETECTION_FPS"
    MAX_MISSES="$SPORTS_MAX_MISSES"
    POLICY_OVERRIDES="{\"detection\":{\"detection_fps\":${DETECTION_FPS}},\"yolo26\":{\"pose_imgsz\":${POSE_IMGSZ}},\"tracking\":{\"max_missed_updates\":${MAX_MISSES}},\"selection\":{\"ball_max_misses\":${SPORTS_BALL_MAX_MISSES}}}"
  else
    DETECT_MODEL="$MOVIE_DETECT_MODEL"
    POSE_MODEL="$MOVIE_POSE_MODEL"
    IMGSZ="$MOVIE_IMGSZ"
    POSE_IMGSZ="$MOVIE_POSE_IMGSZ"
    DETECTION_FPS="$MOVIE_DETECTION_FPS"
    MAX_MISSES="$MOVIE_MAX_MISSES"
    POLICY_OVERRIDES="{\"detection\":{\"detection_fps\":${DETECTION_FPS}},\"yolo26\":{\"pose_imgsz\":${POSE_IMGSZ}},\"tracking\":{\"max_missed_updates\":${MAX_MISSES}}}"
  fi
  [[ -f "$DETECT_MODEL" ]] || { echo "Missing $DETECT_MODEL" >&2; exit 2; }
  [[ -f "$POSE_MODEL" ]] || { echo "Missing $POSE_MODEL" >&2; exit 2; }

  jsonl="$OUT/${mode}.jsonl"
  log="$OUT/${mode}.log"
  debug="$OUT/debug/${mode}_frames.jsonl"
  : > "$jsonl"
  : > "$debug"

  echo "=== YOLO26 v5.3: ${mode}; profile=${PROFILE}; clips=${#clips[@]}; device=${DEVICE}; detect_imgsz=${IMGSZ}; pose_imgsz=${POSE_IMGSZ}; detection_fps=${DETECTION_FPS}; detect=$(basename "$DETECT_MODEL"); pose=$(basename "$POSE_MODEL") ==="
  printf '%s\n' "${clips[@]}" \
    | PYTHONPATH=src "$PYTHON" -u run.py \
        --output-path "$jsonl" \
        --params "{\"mode\":\"${mode}\",\"detector_backend\":\"yolo26\",\"yolo_detect_model\":\"${DETECT_MODEL}\",\"yolo_pose_model\":\"${POSE_MODEL}\",\"yolo_device\":\"${DEVICE}\",\"yolo_imgsz\":${IMGSZ},\"yolo_half\":${HALF},\"yolo_end2end\":false,\"policy_overrides\":${POLICY_OVERRIDES},\"debug_jsonl_path\":\"${debug}\",\"progress_log_interval_seconds\":5}" \
        2>&1 | tee "$log"

  PYTHONPATH=src "$PYTHON" scripts/validate_output.py "$jsonl" "${#clips[@]}"
  for file in "${clips[@]}"; do
    name=$(basename "${file%.*}")
    side_output="$OUT/previews/${name}_side_by_side.mp4"
    vertical_output="$OUT/previews/${name}_vertical.mp4"
    "$PYTHON" "$RENDERER" \
      --input "$file" \
      --jsonl "$jsonl" \
      --debug-jsonl "$debug" \
      --side-by-side-output "$side_output" \
      --vertical-output "$vertical_output"
    "$PYTHON" "$ROOT/scripts/validate_preview_outputs.py" \
      --source "$file" \
      --side-by-side "$side_output" \
      --vertical "$vertical_output"
  done
done

"$PYTHON" "$ANALYZER" \
  --sports-jsonl "$OUT/sports.jsonl" \
  --movie-jsonl "$OUT/movie.jsonl" \
  --sports-debug "$OUT/debug/sports_frames.jsonl" \
  --movie-debug "$OUT/debug/movie_frames.jsonl" \
  --output-dir "$OUT/metrics"

echo "YOLO26 v5.3 outputs: $ROOT/$OUT"
