#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
cd "$ROOT"
VENV="${YOLO_VENV:-.venv-yolo26}"
MODEL_DIR="$ROOT/models/yolo26"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install \
  "common-ml @ git+https://github.com/eluv-io/common-ml.git@5e112f615acb0a13fa044ea4dacc94ab1612d9ef" \
  "loguru>=0.7,<1" "numpy>=1.24,<2" "opencv-contrib-python>=4.9,<5" \
  "PyYAML>=6,<7" "requests>=2.31,<3"
"$VENV/bin/python" -m pip install -r requirements-yolo26.txt
"$VENV/bin/python" -m pip install -e . --no-deps

mkdir -p "$MODEL_DIR"
(
  cd "$MODEL_DIR"
  "$ROOT/$VENV/bin/python" - <<'PY'
from ultralytics import YOLO
for model in (
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
):
    print(f"Loading/downloading {model}")
    YOLO(model)
PY
)

"$VENV/bin/python" - <<'PY'
import torch
import ultralytics
print("ultralytics", ultralytics.__version__)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

test -f "$MODEL_DIR/yolo26s.pt"
test -f "$MODEL_DIR/yolo26m.pt"
test -f "$MODEL_DIR/yolo26n-pose.pt"
test -f "$MODEL_DIR/yolo26s-pose.pt"
echo "YOLO26 experiment ready at $ROOT"
