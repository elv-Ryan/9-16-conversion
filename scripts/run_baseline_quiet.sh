#!/usr/bin/env bash
set -euo pipefail

# Wrapper: keep full logs from run_baseline.sh, but suppress the giant graph pbtxt dump in terminal.
scripts/run_baseline.sh "$@" | awk '
/Get calculator graph config contents:/{skip=1;next}
/Initialize the calculator graph/{skip=0}
!skip{print}
'
