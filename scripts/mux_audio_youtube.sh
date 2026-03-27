#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/mux_audio_youtube.sh <yt_id> <video_no_audio.mp4> <out_with_audio.mp4> [start_sec]
#
# Notes:
# - Assumes your cropped video corresponds to the YouTube timeline starting at start_sec (default 0).
# - Downloads bestaudio once, then ffmpeg trims to match the cropped video duration.
# - Deterministic mux: video stream copy, AAC audio encode, -shortest, faststart.

YT_ID="${1:?yt_id required (e.g. NAhJor6C4ik)}"
VID_IN="${2:?video_no_audio.mp4 required}"
OUT="${3:?out_with_audio.mp4 required}"
START_SEC="${4:-0}"

test -f "$VID_IN"

# Probe output video duration (seconds)
DUR="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$VID_IN" | awk '{printf "%.3f\n",$1}')"
echo "VID_IN=$VID_IN"
echo "START_SEC=$START_SEC"
echo "DUR_SEC=$DUR"

# Download audio (cache by yt_id)
AUDIO_DIR="data/audio"
AUDIO_SRC="${AUDIO_DIR}/${YT_ID}.m4a"
mkdir -p "$AUDIO_DIR"

if ! test -f "$AUDIO_SRC"; then
  echo "Downloading bestaudio for YT_ID=$YT_ID -> $AUDIO_SRC"
  yt-dlp -f "bestaudio[ext=m4a]/bestaudio" -o "$AUDIO_SRC" "https://www.youtube.com/watch?v=${YT_ID}"
fi

# Mux: copy video, encode audio, trim audio to [START_SEC, START_SEC + DUR]
ffmpeg -hide_banner -loglevel error -y \
  -i "$VID_IN" \
  -ss "$START_SEC" -t "$DUR" -i "$AUDIO_SRC" \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy \
  -c:a aac -b:a 192k \
  -shortest -movflags +faststart \
  "$OUT"

echo "Wrote: $OUT"
ffprobe -v error -show_streams -select_streams a "$OUT" | grep -E "codec_name=|codec_type=|sample_rate=|channels=" || true
