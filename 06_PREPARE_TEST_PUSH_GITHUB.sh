#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REMOTE=${REMOTE:-mltrain@63.247.64.18}
PORT=${PORT:-5055}
REPO_URL=${REPO_URL:-git@github.com:elv-Ryan/9-16-conversion.git}
BASE_REF=${BASE_REF:-ryan-v2.1}
BRANCH=${BRANCH:-nba-yolo-shot-tagger-v3.3}
REMOTE_WORKDIR=${REMOTE_WORKDIR:-/home/mltrain/elv-ryan/projects/nba-yolo-shot-tagger-github-v3_3}
TEST_SHOT=${TEST_SHOT:-/home/mltrain/elv-joe/shots/iq__2q6ZyYAmFfKDBJeWMGsWLP549twd/000-16_1818.mp4}
MODEL_SOURCE=${MODEL_SOURCE:-/home/mltrain/elv-ryan/projects/9-16-conversion-ryan-v2.1-student-mvp/output/generic_basketball_v11_student_round00_v1/yolo_fp32_6gpu_round00_v7/long_run/focus_target_round00_fp32/weights/best.pt}
PY=${PY:-/home/mltrain/elv-ryan/projects/9-16-conversion-ryan-v2.1-student-mvp/.venv-yolo26/bin/python}
GPU0_WAIT_SECONDS=${GPU0_WAIT_SECONDS:-7200}
TMP=$(mktemp /tmp/nba-yolo-handoff-v3.3.XXXXXX.tar.gz)
trap 'rm -f "$TMP"' EXIT

# Production/handoff source only. Do not publish test outputs or model weights.
tar -czf "$TMP" \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='test-runs' --exclude='models/nba_yolo_student/best.pt' \
  -C "$ROOT" \
  .dockerignore .gitignore .gitmodules Containerfile Makefile build.sh README.md \
  EXPERIMENTAL_LICENSE_NOTICE.md pyproject.toml requirements.txt run.py setup.py user_config.yaml \
  src docs tests scripts models

scp -P "$PORT" "$TMP" "$REMOTE:/tmp/nba-yolo-handoff-v3.3.tar.gz"

ssh -p "$PORT" "$REMOTE" 'bash -s' -- \
  "$REPO_URL" "$BASE_REF" "$BRANCH" "$REMOTE_WORKDIR" "$TEST_SHOT" "$MODEL_SOURCE" "$PY" "$GPU0_WAIT_SECONDS" <<'REMOTE_SCRIPT'
set -Eeuo pipefail
REPO_URL=$1; BASE_REF=$2; BRANCH=$3; WORK=$4; TEST_SHOT=$5; MODEL_SOURCE=$6; PY=$7; GPU0_WAIT_SECONDS=$8

[[ -f "$MODEL_SOURCE" ]] || { echo "MODEL_SOURCE missing: $MODEL_SOURCE" >&2; exit 2; }
[[ -f "$TEST_SHOT" ]] || { echo "TEST_SHOT missing: $TEST_SHOT" >&2; exit 2; }
command -v podman >/dev/null || { echo "podman missing" >&2; exit 2; }
command -v jq >/dev/null || { echo "jq missing (required by official buildscripts tester)" >&2; exit 2; }

if [[ ! -d "$WORK/.git" ]]; then
  rm -rf "$WORK"
  git clone "$REPO_URL" "$WORK"
fi
cd "$WORK"
git fetch origin --prune
git reset --hard
git clean -fdx -e buildscripts
git checkout "$BASE_REF"
git reset --hard "origin/$BASE_REF"
git switch -C "$BRANCH"

# Remove obsolete runtime trees on this dedicated handoff branch.
git rm -r --ignore-unmatch src tests scripts docs configs models >/dev/null 2>&1 || true
git rm --ignore-unmatch requirements-yolo26.txt >/dev/null 2>&1 || true
rm -rf src tests scripts docs configs models

tar -xzf /tmp/nba-yolo-handoff-v3.3.tar.gz -C "$WORK"
rm -f /tmp/nba-yolo-handoff-v3.3.tar.gz
chmod +x build.sh test.sh scripts/*.sh scripts/*.py 2>/dev/null || true

git submodule sync --recursive
git submodule update --init --recursive

MODEL_SOURCE="$MODEL_SOURCE" ./scripts/fetch_current_model.sh

# Ensure binary model is never committed while the SHA manifest is publishable.
if git check-ignore -q models/nba_yolo_student/best.pt; then
  echo "MODEL_WEIGHT_GITIGNORE=PASS"
else
  echo "MODEL_WEIGHT_GITIGNORE=FAIL" >&2
  exit 3
fi

git add -A
if git diff --cached --quiet; then
  echo "No source changes to commit."
else
  git commit -m "Add Eluvio NBA YOLO shot-to-X tagger v3.3"
fi
COMMIT=$(git rev-parse HEAD)
echo "COMMIT=$COMMIT"

echo "===== MAKE UNIT-TEST ====="
make unit-test

echo "===== OFFICIAL ELUVIO BUILDSCRIPTS BUILD ====="
make build

echo "===== WAIT FOR GPU0 FOR UNMODIFIED OFFICIAL MAKE TEST ====="
# qluvio/buildscripts/testers/test-model.sh currently hardcodes physical GPU0.
# Do not terminate or reset any process; wait until GPU0 is genuinely free.
START=$(date +%s)
while true; do
  read -r USED FREE UTIL < <(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i 0 | tr -d ' ' | tr ',' ' ')
  UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | tr -d ' ')
  ACTIVE=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | grep -Fx "$UUID" | wc -l || true)
  if [[ "$ACTIVE" -eq 0 && "$USED" -le 1024 && "$FREE" -ge 20000 && "$UTIL" -le 10 ]]; then
    echo "GPU0_FREE=PASS used=$USED free=$FREE util=$UTIL"
    break
  fi
  NOW=$(date +%s)
  if (( NOW - START >= GPU0_WAIT_SECONDS )); then
    echo "Timed out waiting for GPU0; official make test not run." >&2
    exit 4
  fi
  echo "GPU0 busy; waiting without interrupting it: used=$USED free=$FREE util=$UTIL active_compute=$ACTIVE"
  sleep 30
done

echo "===== OFFICIAL ELUVIO MAKE TEST ====="
rm -rf test-files test-output .cache
mkdir -p test-files
cp "$TEST_SHOT" test-files/joe-shot-000.mp4
make test

echo "===== STRICT X-SEMANTICS PODMAN TEST ====="
DEVICE=0 IMAGE=nba-yolo-shot-tagger:latest ./scripts/test_podman_shot.sh "$TEST_SHOT"

# Remove test/build scratch before the clean-tree release gate.
rm -rf test-files test-output .cache version

echo "===== GIT CLEAN CHECK ====="
git status --short
if ! git diff-index --quiet HEAD --; then
  echo "Tracked tree changed during tests/build; refusing push." >&2
  exit 5
fi
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "Untracked release artifacts remain; refusing push." >&2
  git status --short >&2
  exit 5
fi

echo "===== PUSH ====="
git push -u origin "$BRANCH"

echo "GITHUB_PUSH=PASS"
echo "BRANCH=$BRANCH"
echo "COMMIT=$COMMIT"
REMOTE_SCRIPT
