#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/elv-ryan/projects/9-16-conversion}"
cd "$ROOT"

VENV=".venv"
SHORTS="test-files/short"
OUT="test-output/visual-v3"
OBJECT_MODEL="$ROOT/models/mp_tasks/object_detector/efficientdet_lite0.tflite"
FACE_MODEL="$ROOT/models/mp_tasks/face_detector/blaze_face_short_range.tflite"
RENDERER="$ROOT/scripts/render_focus_previews.py"

[[ -x "$VENV/bin/python" ]] || { echo "Missing $ROOT/$VENV" >&2; exit 2; }
[[ -d "$SHORTS" ]] || { echo "Missing $ROOT/$SHORTS" >&2; exit 2; }
[[ -f "$OBJECT_MODEL" ]] || { echo "Missing object model: $OBJECT_MODEL" >&2; exit 2; }
[[ -f "$FACE_MODEL" ]] || { echo "Missing face model: $FACE_MODEL" >&2; exit 2; }
[[ -f "$RENDERER" ]] || { echo "Missing renderer: $RENDERER" >&2; exit 2; }

for mode in sports movie; do
  count=$(find "$SHORTS" -maxdepth 1 -type f -name "${mode}_*.mp4" | wc -l)
  [[ "$count" -eq 4 ]] || { echo "Expected 4 ${mode} clips under $SHORTS; found $count" >&2; exit 2; }
done

rm -rf "$OUT"
mkdir -p "$OUT/previews"

for mode in sports movie; do
  jsonl="$OUT/${mode}.jsonl"
  log="$OUT/${mode}.log"
  : > "$jsonl"

  echo "=== Running ${mode} focus on four short clips ==="
  find "$SHORTS" -maxdepth 1 -type f -name "${mode}_*.mp4" -print \
    | sort \
    | PYTHONPATH=src "$VENV/bin/python" -u run.py \
        --output-path "$jsonl" \
        --params "{\"mode\":\"${mode}\",\"delegate\":\"cpu\",\"object_model\":\"${OBJECT_MODEL}\",\"face_model\":\"${FACE_MODEL}\",\"progress_log_interval_seconds\":5}" \
        2>&1 | tee "$log"

  PYTHONPATH=src "$VENV/bin/python" scripts/validate_output.py "$jsonl" 4

  while IFS= read -r file; do
    name=$(basename "${file%.*}")
    "$VENV/bin/python" "$RENDERER" \
      --input "$file" \
      --jsonl "$jsonl" \
      --side-by-side-output "$OUT/previews/${name}_side_by_side.mp4" \
      --vertical-output "$OUT/previews/${name}_vertical.mp4"
  done < <(find "$SHORTS" -maxdepth 1 -type f -name "${mode}_*.mp4" -print | sort)
done

echo
echo "Preview outputs:"
find "$OUT/previews" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | sort

echo
echo "Audio verification:"
for file in "$OUT"/previews/*.mp4; do
  audio=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$file" | head -n 1)
  printf '%s audio=%s\n' "$(basename "$file")" "${audio:-none}"
done
