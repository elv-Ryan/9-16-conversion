#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUTPUT="$ROOT/models/mp_tasks/face_detector/blaze_face_short_range.tflite"
URL="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

mkdir -p "$(dirname "$OUTPUT")"
if [[ -s "$OUTPUT" ]] && [[ $(stat -c '%s' "$OUTPUT") -gt 100000 ]]; then
  echo "Face model already present: $OUTPUT"
  exit 0
fi

python3 - "$URL" "$OUTPUT" <<'PY'
from pathlib import Path
import sys
import urllib.request

url = sys.argv[1]
output = Path(sys.argv[2])
temporary = output.with_suffix(output.suffix + ".tmp")
urllib.request.urlretrieve(url, temporary)
if temporary.stat().st_size <= 100_000:
    temporary.unlink(missing_ok=True)
    raise SystemExit("Downloaded face model is unexpectedly small")
temporary.replace(output)
print(f"Downloaded {output} ({output.stat().st_size} bytes)")
PY
