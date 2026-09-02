#!/usr/bin/env bash
set -Eeuo pipefail
SOURCE=${1:?Usage: scripts/create_test_shot.sh source.mp4 [output.mp4] [start_seconds] [duration_seconds]}
OUTPUT=${2:-/tmp/nba-yolo-test-shot.mp4}
START=${3:-120}
DURATION=${4:-4}
ffmpeg -hide_banner -loglevel error -y -ss "$START" -i "$SOURCE" -t "$DURATION" \
  -map 0:v:0 -an -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p "$OUTPUT"
echo "$OUTPUT"
