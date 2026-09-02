#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m unittest discover -s "$ROOT/tests" -v
if [[ -n "${TEST_SHOT:-}" ]]; then
  "$ROOT/scripts/test_podman_shot.sh" "$TEST_SHOT"
else
  echo "TEST_SHOT is not set; protocol/container test skipped."
fi
