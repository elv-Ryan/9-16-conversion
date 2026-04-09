#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/projects/916_conversion}"
IMG="${IMG:-mediapipe-autoflip:cpu}"
PLATFORM="${PLATFORM:-linux/arm64}"
GRAPH_REL="${GRAPH_REL:-graphs/autoflip_graph_faceprimary_target406x720_RAW.pbtxt}"

IN_REL="${1:?Usage: scripts/run_baseline.sh <data/in/*.avi> [tag]}"
TAG="${2:-}"

cd "$ROOT"
mkdir -p data/out data/out/logs

if [[ ! -f "$IN_REL" ]]; then
  echo "ERROR: input not found: $IN_REL" >&2
  exit 1
fi

STEM="$(basename "$IN_REL")"
STEM="${STEM%.*}"
if [[ -n "$TAG" ]]; then
  STEM="$TAG"
fi

OUT_REL="data/out/${STEM}_9x16_RAW.mp4"
LOG_REL="data/out/logs/${STEM}.log"

rm -f "$OUT_REL" "$LOG_REL"

echo "== Locate run_autoflip inside image =="
BIN_PATH="$(docker run --rm --platform="$PLATFORM" -v "$ROOT:/work" -w /work "$IMG" \
  bash -lc "find / -name run_autoflip -type f 2>/dev/null | head -n 1")"
if [[ -z "${BIN_PATH}" ]]; then
  echo "ERROR: run_autoflip not found in image: $IMG" >&2
  exit 1
fi
echo "BIN_PATH=${BIN_PATH}"

echo
echo "== Run baseline: ${IN_REL} -> ${OUT_REL} =="
docker run --rm --platform="$PLATFORM" \
  --cpus="8" --memory="40g" \
  -v "$ROOT:/work" -w /work \
  "$IMG" \
  bash -lc "
set -euo pipefail
'${BIN_PATH}' \
  --resource_root_dir=/work/models \
  --calculator_graph_config_file=/work/${GRAPH_REL} \
  --input_side_packets=\"input_video_path=/work/${IN_REL},output_video_path=/work/${OUT_REL},aspect_ratio=9:16\"
" 2>&1 | tee "$LOG_REL"

echo
echo "== Verify output file =="
ls -lah "$OUT_REL"

echo
echo "== ffprobe summary =="
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,duration \
  -of default=noprint_wrappers=1:nokey=0 \
  "$OUT_REL"

echo
echo "== sha256 (repro check) =="
shasum -a 256 "$OUT_REL"

echo
echo "== shot changes (from log) =="
grep -E "Shot change at:" "$LOG_REL" || echo "(no shot change lines found)"
SHOT_COUNT="$(grep -cE 'Shot change at:' "$LOG_REL" || true)"
echo "shot_change_count=${SHOT_COUNT}"
