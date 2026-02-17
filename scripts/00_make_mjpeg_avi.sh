#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/input.mp4"
  exit 1
fi

IN="$1"
ROOT="/Users/ryan-renslow/projects/916_conversion"
OUT="$ROOT/data/in/$(basename "${IN%.*}")__mjpeg.avi"

mkdir -p "$ROOT/data/in"

ffmpeg -y -i "$IN" -an -c:v mjpeg -q:v 3 "$OUT"
echo "Wrote: $OUT"
