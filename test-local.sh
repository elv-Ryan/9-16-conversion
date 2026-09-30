#!/bin/bash
set -euo pipefail

set -x

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Run against the checkout containing this script's caller.
# The harness is invoked while cwd is either the old or new repo.
export PYTHONPATH="${PYTHONPATH:-src}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
RESET='\033[0m'

step() { echo -e "${MAGENTA}  ──▶${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo -e "${GREEN}  ✔  $*${RESET}"; }
fail() { echo -e "${RED}  ✘  $*${RESET}"; }

MODE="${1:-default}"

/bin/rm -f out.jsonl test-local.wall_seconds

if [[ -n "${TEST_INPUT_FILE:-}" ]]; then
    INPUT="$TEST_INPUT_FILE"
elif [[ -n "${TEST_INPUT_DIR:-}" ]]; then
    INPUT="$(
        /usr/bin/find "$TEST_INPUT_DIR" \
          -maxdepth 1 \
          -type f \
          | /usr/bin/sort
    )"
else
    INPUT="$(
        /usr/bin/find test-files/ \
          -maxdepth 1 \
          -type f \
          | /usr/bin/sort \
          | /usr/bin/head -10
    )"
fi

params='{"input_mode":"segment_file"}'

case "$MODE" in
    shot)
        params='{"input_mode":"shot_file"}'
        ;;

    preshot)
        if [[ -z "${TEST_INPUT_FILE:-}" && -z "${TEST_INPUT_DIR:-}" ]]; then
            INPUT="$(
                /usr/bin/find test-files/nba \
                  -maxdepth 1 \
                  -type f \
                  | /usr/bin/sort \
                  | /usr/bin/tail -n +100 \
                  | /usr/bin/head -10
            )"
        fi

        params='{"input_mode":"shot_file","family_determination_max_seconds":99999}'
        ;;

    preshotsegs)
        if [[ -z "${TEST_INPUT_FILE:-}" && -z "${TEST_INPUT_DIR:-}" ]]; then
            INPUT="$(
                /usr/bin/find test-files/nba \
                  -maxdepth 1 \
                  -type f \
                  | /usr/bin/sort \
                  | /usr/bin/tail -n +100 \
                  | /usr/bin/head -10
            )"
        fi

        params='{"input_mode":"segment_file","family_determination_max_seconds":99999}'
        ;;

    default)
        ;;

    *)
        fail "Unknown mode: $MODE"
        exit 2
        ;;
esac

FILE_COUNT="$(
    printf '%s\n' "$INPUT" |
    /usr/bin/grep -c . || true
)"

if [[ "$FILE_COUNT" -lt 1 ]]; then
    fail "No input files"
    exit 1
fi

step "Mode: $MODE"
step "Python: $PYTHON_BIN"
step "Found $FILE_COUNT input file(s):"

printf '%s\n' "$INPUT" |
    /usr/bin/sed 's/^/       ▸ /'

START_NS="$(/usr/bin/date +%s%N)"

printf '%s\n' "$INPUT" |
"$PYTHON_BIN" run.py \
    --output-path out.jsonl \
    --params "$params"

RC=$?

END_NS="$(/usr/bin/date +%s%N)"

if [[ "$RC" -ne 0 ]]; then
    fail "run.py exited with code $RC"
    exit "$RC"
fi

if [[ ! -s out.jsonl ]]; then
    fail "out.jsonl missing or empty"
    exit 1
fi

JQ_BIN="$(command -v jq)"

if [[ -z "$JQ_BIN" ]]; then
    fail "jq is not installed/on PATH"
    exit 1
fi

RESULT_COUNT="$(
"$JQ_BIN" -s \
  '[.[] | select(.type == "progress" or .type == "error")] | length' \
  out.jsonl
)"

ERROR_COUNT="$(
"$JQ_BIN" -s \
  '[.[] | select(.type == "error")] | length' \
  out.jsonl
)"

PROGRESS_COUNT="$(
"$JQ_BIN" -s \
  '[.[] | select(.type == "progress")] | length' \
  out.jsonl
)"

VERTICAL_COUNT="$(
"$JQ_BIN" -s \
  '[.[]
    | select(
        .type == "tag"
        and .data.track == "vertical_video"
      )
   ] | length' \
  out.jsonl
)"

if [[ "$RESULT_COUNT" -ne "$FILE_COUNT" ]]; then
    fail "Expected $FILE_COUNT terminal rows, found $RESULT_COUNT"
    exit 1
fi

if [[ "$ERROR_COUNT" -ne 0 ]]; then
    fail "Found $ERROR_COUNT model/error row(s)"
    "$JQ_BIN" -c 'select(.type == "error")' out.jsonl
    exit 1
fi

if [[ "$PROGRESS_COUNT" -ne "$FILE_COUNT" ]]; then
    fail "Expected $FILE_COUNT progress rows, found $PROGRESS_COUNT"
    exit 1
fi

if [[ "$MODE" == "shot" || "$MODE" == "preshot" ]]; then
    if [[ "$VERTICAL_COUNT" -ne "$FILE_COUNT" ]]; then
        fail "Expected $FILE_COUNT vertical_video tags, found $VERTICAL_COUNT"
        exit 1
    fi
fi

ELAPSED_SEC="$(
"$PYTHON_BIN" - "$START_NS" "$END_NS" <<'PY'
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])

print(f"{(end-start)/1e9:.6f}")
PY
)"

ok "progress=$PROGRESS_COUNT errors=$ERROR_COUNT vertical=$VERTICAL_COUNT"
ok "wall_seconds=$ELAPSED_SEC"

printf '%s\n' "$ELAPSED_SEC" > test-local.wall_seconds

echo "TEST_LOCAL_STRICT_PASS"
