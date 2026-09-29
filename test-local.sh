#!/bin/bash

## colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${CYAN}${BOLD}"
echo "  ╔═══════════════════════════════════╗"
echo "  ║          TEST TIME BABY.          ║"
echo "  ╚═══════════════════════════════════╝"
echo -e "${RESET}"

step() { echo -e "${MAGENTA}  ──▶${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo -e "${GREEN}  ✔  $*${RESET}"; }
fail() { echo -e "${RED}  ✘  $*${RESET}"; }

step "Cleaning up previous run..."
rm -rf out.jsonl

echo ""
step "Launching it or something ${YELLOW}${IMAGE_NAME}${RESET}..."
echo ""

# Build newline-separated list of test files as container-side paths
INPUT=$(find test-files/ -maxdepth 1 -type f | sort | head -10)

params='{"input_mode": "segment_file"}'
if [ "$1" = "preshot" ]; then
    INPUT=$(find test-files/nba -maxdepth 1 -type f | sort | tail -n +100 | head -10)
    params='{"family_determination_max_seconds": 99999}'
elif [ "$1" = "preshotsegs" ]; then
    INPUT=$(find test-files/nba -maxdepth 1 -type f | sort | tail -n +100 | head -10)
    params='{"input_mode":"segment_file","family_determination_max_seconds": 99999}'
fi


FILE_COUNT=$(echo "$INPUT" | grep -c .)
step "Found ${YELLOW}${FILE_COUNT}${RESET} file(s) to process:"
echo "$INPUT" | sed "s/^/       ${CYAN}▸${RESET} /"

##--volume=$(pwd)/models:/elv/models:ro
echo "$INPUT" | python run.py --output-path out.jsonl --params "$params"


ex=$?
echo ""
if [ $ex -ne 0 ]; then
    fail "Container exited with error code ${ex}"
    echo -e "${RED}${BOLD}"
    echo "  ╔═══════════════════════════════════╗"
    echo "  ║            TEST FAILED            ║"
    echo "  ╚═══════════════════════════════════╝"
    echo -e "${RESET}"
    exit $ex
fi

step "Checking output..."
if [ ! -f out.jsonl ]; then
    fail "test-output/out.jsonl is missing"
    echo -e "${RED}${BOLD}"
    echo "  ╔═══════════════════════════════════╗"
    echo "  ║            TEST FAILED            ║"
    echo "  ╚═══════════════════════════════════╝"
    echo -e "${RESET}"
    exit 1
fi

RESULT_COUNT=$(jq -s '[.[] | select(.type == "progress" or .type == "error")] | length' out.jsonl)
if [ "$RESULT_COUNT" -ne "$FILE_COUNT" ]; then
    fail "Expected ${YELLOW}${FILE_COUNT}${RED} progress/error rows but found ${YELLOW}${RESULT_COUNT}${RED} in out.jsonl"
    echo -e "${RED}${BOLD}"
    echo "  ╔═══════════════════════════════════╗"
    echo "  ║            TEST FAILED            ║"
    echo "  ╚═══════════════════════════════════╝"
    echo -e "${RESET}"
    exit 1
fi
ok "out.jsonl: ${YELLOW}${RESULT_COUNT}/${FILE_COUNT}${RESET} file(s) accounted for (progress or error)"

echo ""
echo -e "${GREEN}${BOLD}"
echo "  ╔═══════════════════════════════════╗"
echo "  ║           TEST PASSED  🎉         ║"
echo "  ╚═══════════════════════════════════╝"
echo -e "${RESET}"

cd test-output
find
