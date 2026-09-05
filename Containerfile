ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/elv/model/.torch \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

WORKDIR /elv/model

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       ffmpeg \
       git \
       libglib2.0-0 \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml setup.py ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY models ./models
COPY model-shot ./model-shot

COPY src ./src
COPY run.py ./run.py

COPY scripts/verify_model_artifact.py ./scripts/verify_model_artifact.py

RUN python -m pip install --no-deps . \
    && python scripts/verify_model_artifact.py \
       --model models/nba_yolo_student/best.pt \
       --manifest models/nba_yolo_student/model_manifest.json \
    && python - <<'PY'
from common_ml.tagging.messages import Tag, Progress, Error, FrameInfo
from common_ml.tagging.producer import TagMessageProducer
from common_ml.tagging.run_helpers import run_default, get_params, catch_errors
from nba_yolo_shot_tagger.producer import NbaShotFocusProducer
print("Eluvio NBA shot-tagger imports: PASS")
PY

ENTRYPOINT ["python", "-u", "run.py"]
