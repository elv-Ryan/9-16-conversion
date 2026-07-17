#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT/test-files/short}"
PREVIEW_DIR="${PREVIEW_DIR:-$ROOT/test-output/visual/previews}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/test-output/visual/previews-audio}"

mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
previews=("$PREVIEW_DIR"/*_side_by_side.mp4)
if [ "${#previews[@]}" -eq 0 ]; then
  echo "No side-by-side previews found under $PREVIEW_DIR" >&2
  exit 2
fi

for preview in "${previews[@]}"; do
  base=$(basename "$preview" _side_by_side.mp4)
  source="$SOURCE_DIR/${base}.mp4"
  output="$OUTPUT_DIR/${base}_side_by_side_audio.mp4"
  if [ ! -f "$source" ]; then
    echo "Skipping $base: source clip missing at $source" >&2
    continue
  fi

  if ffprobe -v error -select_streams a:0 -show_entries stream=index -of csv=p=0 "$source" | grep -q .; then
    ffmpeg -nostdin -hide_banner -loglevel error -stats -y \
      -i "$preview" -i "$source" \
      -map 0:v:0 -map 1:a:0 \
      -c:v copy -c:a aac -b:a 192k -shortest \
      -movflags +faststart "$output"
  else
    cp "$preview" "$output"
    echo "No audio stream in $source; copied silent preview" >&2
  fi
  echo "Created $output"
done
