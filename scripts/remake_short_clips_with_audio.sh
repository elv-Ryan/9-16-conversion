#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
TEST_ROOT="$ROOT/test-files"
OUT="$TEST_ROOT/short"
mkdir -p "$OUT"

find_one() {
  local first_pattern="$1"
  local second_pattern="$2"
  local found
  found=$(find "$TEST_ROOT" -maxdepth 1 -type f \( -iname "$first_pattern" -o -iname "$second_pattern" \) -print | head -n 1)
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
}

SPORTS=$(find_one '*NBA*Summer*League*.mp4' '*PORTLAND*DENVER*.mp4') || {
  echo "Could not find the full sports source directly under $TEST_ROOT" >&2
  exit 2
}
MOVIE=$(find_one '*SPIDER*VERSE*.mp4' '*11022640*.mp4') || {
  echo "Could not find the full movie source directly under $TEST_ROOT" >&2
  exit 2
}

for source in "$SPORTS" "$MOVIE"; do
  if ! ffprobe -v error -select_streams a:0 -show_entries stream=index -of csv=p=0 "$source" | grep -q .; then
    echo "Source has no audio stream: $source" >&2
    exit 2
  fi
done

make_clip() {
  local source="$1"
  local start="$2"
  local output="$3"
  echo "Creating $(basename "$output") at $start from $(basename "$source")"
  ffmpeg -nostdin -hide_banner -loglevel error -stats -y \
    -ss "$start" -i "$source" -t 90 \
    -map '0:v:0' -map '0:a:0?' \
    -c:v libx264 -preset veryfast -crf 21 -pix_fmt yuv420p \
    -c:a aac -b:a 192k \
    -movflags +faststart "$output"
}

# Four separate basketball gameplay samples distributed through the game.
make_clip "$SPORTS" 681.698  "$OUT/sports_01.mp4"
make_clip "$SPORTS" 2498.442 "$OUT/sports_02.mp4"
make_clip "$SPORTS" 4315.186 "$OUT/sports_03.mp4"
make_clip "$SPORTS" 6131.930 "$OUT/sports_04.mp4"

# Main-character/action/dialogue samples from Spider-Verse.
make_clip "$MOVIE" 885  "$OUT/movie_01.mp4"
make_clip "$MOVIE" 1810 "$OUT/movie_02.mp4"
make_clip "$MOVIE" 4700 "$OUT/movie_03.mp4"
make_clip "$MOVIE" 7160 "$OUT/movie_04.mp4"

echo
echo "Created audio-bearing short clips:"
for file in "$OUT"/*.mp4; do
  audio=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$file" | head -n 1)
  duration=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$file")
  printf '%-20s duration=%7.2fs audio=%s\n' "$(basename "$file")" "$duration" "${audio:-none}"
done
