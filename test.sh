#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-sports}"
IMAGE_NAME="${IMAGE_NAME:-verticalvideo}"
INPUT_DIR="test-files/short"
OUTPUT_DIR="test-output/container-${MODE}"
OUTPUT_FILE="${OUTPUT_DIR}/out.jsonl"

case "$MODE" in
  sports)
    PATTERN="sports_*.mp4"
    ;;
  movie)
    PATTERN="movie_*.mp4"
    ;;
  *)
    echo "MODE must be sports or movie"
    exit 1
    ;;
esac

mapfile -t FILES < <(
  find "$INPUT_DIR" -maxdepth 1 -type f -name "$PATTERN" | sort
)

FILE_COUNT="${#FILES[@]}"

if [ "$FILE_COUNT" -eq 0 ]; then
  echo "No test files found for $MODE in $INPUT_DIR"
  exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

INPUT=""
for file in "${FILES[@]}"; do
  name=$(basename "$file")
  INPUT+="/elv/test/${name}"$'\n'
done

printf '%s' "$INPUT" | podman run --rm -i \
  --volume="$PWD/$INPUT_DIR:/elv/test:ro" \
  --volume="$PWD/$OUTPUT_DIR:/elv/tags" \
  --network=host \
  "$IMAGE_NAME:latest" \
  --output-path /elv/tags/out.jsonl \
  --params "{\"continue_on_error\":true,\"detector_backend\":\"yolo26\",\"mode\":\"${MODE}\"}"

if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Container test failed: $OUTPUT_FILE was not created"
  exit 1
fi

python3 scripts/validate_output.py "$OUTPUT_FILE"

ERROR_COUNT=$(jq -s '[.[] | select(.type == "error")] | length' "$OUTPUT_FILE")
TAG_COUNT=$(jq -s '[.[] | select(.type == "tag")] | length' "$OUTPUT_FILE")
PROGRESS_COUNT=$(jq -s '[.[] | select(.type == "progress")] | length' "$OUTPUT_FILE")

if [ "$ERROR_COUNT" -ne 0 ]; then
  echo "Container test failed: $ERROR_COUNT error messages"
  jq -c 'select(.type == "error")' "$OUTPUT_FILE"
  exit 1
fi

if [ "$TAG_COUNT" -eq 0 ]; then
  echo "Container test failed: no tags produced"
  exit 1
fi

if [ "$PROGRESS_COUNT" -ne "$FILE_COUNT" ]; then
  echo "Container test failed: expected $FILE_COUNT progress messages, got $PROGRESS_COUNT"
  exit 1
fi

echo "Container test passed:"
echo "  mode: $MODE"
echo "  files: $FILE_COUNT"
echo "  tags: $TAG_COUNT"
echo "  progress: $PROGRESS_COUNT"
echo "  errors: $ERROR_COUNT"

jq -s '{
  total_messages: length,
  tags: [.[] | select(.type == "tag")] | length,
  progress: [.[] | select(.type == "progress")] | length,
  errors: [.[] | select(.type == "error")] | length,
  tracks: (
    [.[] | select(.type == "tag") | .data.track]
    | sort
    | group_by(.)
    | map({track: .[0], count: length})
  )
}' "$OUTPUT_FILE"
