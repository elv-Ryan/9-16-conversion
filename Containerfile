FROM ubuntu:22.04 AS commonbase

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
    openjdk-17-jdk \
    clang \
    lld \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 


FROM commonbase AS builder

# Bazelisk (correct binary for container arch)
RUN --mount=type=cache,target=/cache/bazel-dl,id=bazel-dl \
    arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
      amd64)  url="https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-amd64" ;; \
      arm64)  url="https://github.com/bazelbuild/bazelisk/releases/download/v1.20.0/bazelisk-linux-arm64" ;; \
      *) echo "Unsupported arch: $arch" && exit 1 ;; \
    esac && \
    wget -O /cache/bazel-dl/bazel "$url" && cp /cache/bazel-dl/bazel /usr/local/bin && chmod +x /usr/local/bin/bazel

WORKDIR /work/vendor/mediapipe

COPY graphs /work/graphs
COPY models /work/models
COPY vendor /work/vendor

ENV CC=clang
ENV CXX=clang++
ENV USE_BAZEL_VERSION=7.4.1

# Build CPU-only AutoFlip runner (explicit OpenCV4 include path)
#RUN --mount=type=cache,target=/root/.cache/bazelisk,id=bazel741 \
#    --mount=type=cache,target=/root/bazelbuild,id=bazelbuild \
#    bazel help build

RUN --mount=type=cache,target=/root/.cache/bazelisk,id=bazel741 \
    --mount=type=cache,target=/root/.cache/bazel,id=bazelbuild \
    bazel build --disk_cache=/root/.cache/bazel/WHEE -c opt \
        --define MEDIAPIPE_DISABLE_GPU=1 \
        --repo_env=CC=clang --repo_env=CXX=clang++ \
        --action_env=CC=clang --action_env=CXX=clang++ \
        --copt=-I/usr/include/opencv4 --cxxopt=-I/usr/include/opencv4 \
        --copt="-march=native" --define xnn_enable_avxvnni=false --define xnn_enable_avxvnniint8=false \
        mediapipe/examples/desktop/autoflip:run_autoflip

RUN mkdir /unbazelify
COPY unbazelify/copy_rundir.bash /unbazelify

RUN --mount=type=cache,target=/root/.cache/bazelisk,id=bazel741 \
    --mount=type=cache,target=/root/.cache/bazel,id=bazelbuild \
    /unbazelify/copy_rundir.bash

##ENTRYPOINT ["tail", "-n", "+1", "find_output.txt", "binary_location.txt" ]



FROM commonbase AS target

COPY --from=builder /runfiles /runfiles

COPY graphs /work/graphs
COPY models /work/models
COPY vendor /work/vendor

WORKDIR /work

RUN ln -s /dev/null /null.mp4

COPY container/run_tagger.py /app/run_tagger.py

RUN chmod +x /app/run_tagger.py
RUN ls -l /app

ENTRYPOINT ["python3", "/app/run_tagger.py"]
