FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VERTICAL_FOCUS_CONFIG=/elv/model/configs/policies.yml

WORKDIR /elv/model

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       ffmpeg \
       git \
       openssh-client \
       libegl1 \
       libgl1 \
       libgles2 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-yolo26.txt pyproject.toml setup.py README.md ./
COPY configs ./configs
COPY models ./models
COPY src ./src
COPY run.py ./run.py

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install . \
    && python -c "import common_ml, vertical_focus; from common_ml.tagging.run_helpers import catch_errors, get_params, run_default; print('vertical-focus imports OK')"

##ENV CUDA_VISIBLE_DEVICES=0


ENTRYPOINT ["python", "-u", "run.py"]
