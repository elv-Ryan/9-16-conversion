#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/ryan-renslow/projects/916_conversion"
IN_DIR="$ROOT/data/in"
OUT_DIR="$ROOT/data/out"
GRAPH_HOST="$ROOT/graphs/autoflip_graph_noaudio_faceonly.pbtxt"
HOST_MEDIAPIPE="$ROOT/vendor/mediapipe"

mkdir -p "$OUT_DIR"

if ! ls -1 "$IN_DIR"/*.avi >/dev/null 2>&1; then
  echo "ERROR: expected an .avi in $IN_DIR (MJPEG AVI works best with OpenCV decoder)"
  echo "Tip: ffmpeg -y -i input.mp4 -an -c:v mjpeg -q:v 3 $IN_DIR/input_mjpeg.avi"
  exit 1
fi

IN_FILE="$(ls -1 "$IN_DIR"/*.avi | head -n 1)"
OUT_FILE="$OUT_DIR/$(basename "${IN_FILE%.avi}")__autoflip_faceonly_9x16_noaudio.mp4"

IN_BASE="$(basename "$IN_FILE")"
OUT_BASE="$(basename "$OUT_FILE")"

echo "IN : $IN_FILE"
echo "OUT: $OUT_FILE"

docker run --rm --platform=linux/arm64 \
  -v "$IN_DIR":/io/in \
  -v "$OUT_DIR":/io/out \
  -v "$GRAPH_HOST":/io/graph.pbtxt:ro \
  -v "$HOST_MEDIAPIPE":/host_mediapipe:ro \
  -e IN_BASE="$IN_BASE" \
  -e OUT_BASE="$OUT_BASE" \
  mediapipe-autoflip:cpu \
  bash -lc '
    set -euo pipefail
    /work/bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip \
      --resource_root_dir=/host_mediapipe \
      --calculator_graph_config_file=/io/graph.pbtxt \
      --input_side_packets="input_video_path=/io/in/${IN_BASE},output_video_path=/io/out/${OUT_BASE},aspect_ratio=9:16"
  '

echo
echo "Done. Inspect:"
echo "  $OUT_FILE"
echo
echo "ffprobe:"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height \
  -of default=noprint_wrappers=1:nokey=0 \
  "$OUT_FILE"
