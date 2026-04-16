FROM continuumio/miniconda3:latest
WORKDIR /elv

RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda create -n mlpod python=3.12 -y

## blah, everything in the UNIVERSE needs ffmpeg
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked \
    apt-get update && apt-get install -y ffmpeg

ENV PATH="/opt/conda/envs/mlpod/bin:$PATH"

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=ubu22-aptlists \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked,id=ubu22-aptarchives \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    wget \
    ffmpeg \
    git \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    unzip \
    zip \
    lld \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 

# Create the SSH directory and set correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add GitHub to known_hosts to bypass host verification
RUN ssh-keyscan -t rsa github.com >> /root/.ssh/known_hosts

ARG SSH_AUTH_SOCK
ENV SSH_AUTH_SOCK ${SSH_AUTH_SOCK}

COPY requirements.txt .

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/envs/mlpod/bin/pip install -r requirements.txt 

WORKDIR /elv

COPY models ./models
RUN mkdir -p /app

RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=ubu22-aptlists \
    --mount=type=cache,target=/var/cache/apt/archives,sharing=locked,id=ubu22-aptarchives \
    apt-get update && apt-get install -y --no-install-recommends libgles2

COPY run.py /app

ENTRYPOINT ["/opt/conda/envs/mlpod/bin/python", "-u", "/app/run.py"]
