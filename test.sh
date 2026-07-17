#!/usr/bin/env bash
set -euo pipefail

CONTAINER_SYSTEM="${CONTAINER_SYSTEM:-podman}"
MODE="${MODE:-movie}"
IMAGE="${IMAGE_NAME:-verticalvideo}:${IMAGE_TAG:-latest}"
TEST_DIR="${TEST_DIR:-test-files/short}"

mapfile -t HOST_FILES < <(find "$TEST_DIR" -maxdepth 1 -type f \
  \( -iname "${MODE}_*.mp4" -o -iname "${MODE}_*.mov" -o -iname "${MODE}_*.mkv" -o -iname "${MODE}_*.avi" \) | sort)
if [ "${#HOST_FILES[@]}" -eq 0 ]; then
  echo "No ${MODE} test media found under ${TEST_DIR}/" >&2
  exit 2
fi

rm -rf test-output/container-${MODE}
mkdir -p test-output/container-${MODE}
INPUT=""
for file in "${HOST_FILES[@]}"; do
  INPUT+="/elv/test/$(basename "$file")"$'\n'
done

RUN_ARGS=(--rm -i --network host \
  --volume "$(pwd)/${TEST_DIR}:/elv/test:ro" \
  --volume "$(pwd)/test-output/container-${MODE}:/elv/tags")
if [ "${USE_GPU:-0}" = "1" ]; then
  RUN_ARGS+=(--device nvidia.com/gpu=0)
fi
if [ -n "${ELV_CONTENT:-}" ]; then RUN_ARGS+=(--env ELV_CONTENT); fi
if [ -n "${ELV_TOKEN:-}" ]; then RUN_ARGS+=(--env ELV_TOKEN); fi

printf '%s' "$INPUT" | "$CONTAINER_SYSTEM" run "${RUN_ARGS[@]}" "$IMAGE" \
  --output-path /elv/tags/out.jsonl \
  --params "{\"mode\":\"$MODE\",\"delegate\":\"${DELEGATE:-cpu}\"}"

PYTHON_BIN=python3
if [ -x .venv/bin/python ]; then PYTHON_BIN=.venv/bin/python; fi
"$PYTHON_BIN" scripts/validate_output.py "test-output/container-${MODE}/out.jsonl" "${#HOST_FILES[@]}"
echo "Container test passed: mode=$MODE files=${#HOST_FILES[@]}"
