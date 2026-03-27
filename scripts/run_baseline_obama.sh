#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/projects/916_conversion}"
IN_REL="data/in/obama_full__mjpeg.avi"
OUT_REL="data/out/obama_full_9x16_RAW.mp4"
GRAPH_REL="graphs/autoflip_graph_faceprimary_target406x720_RAW.pbtxt"
IMG="${IMG:-mediapipe-autoflip:cpu}"
PLATFORM="${PLATFORM:-linux/arm64}"

cd "$ROOT"
mkdir -p data/out
rm -f "$OUT_REL"

echo "== Locate run_autoflip inside image =="
BIN_PATH="$(docker run --rm --platform="$PLATFORM" -v "$ROOT:/work" -w /work "$IMG" \
  bash -lc "find / -name run_autoflip -type f 2>/dev/null | head -n 1")"
if [[ -z "${BIN_PATH}" ]]; then
  echo "ERROR: run_autoflip not found in image: $IMG" >&2
  exit 1
fi
echo "BIN_PATH=${BIN_PATH}"

echo
echo "== Run baseline =="
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
"

echo
echo "== Verify output =="
ls -lah "$OUT_REL"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,duration \
  -of default=noprint_wrappers=1:nokey=0 \
  "$OUT_REL"
