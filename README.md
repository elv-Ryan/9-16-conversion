# 9:16 conversion using MediaPipe AutoFlip (Apple Silicon + Docker)

A minimal, reproducible pipeline to reframe 16:9 videos into **presentation-grade 9:16**
using **MediaPipe AutoFlip** on **macOS Apple Silicon (arm64)** via Docker.

Key properties of this repo’s “known-good” setup:

- Stable on arm64 (Docker Desktop)
- Uses **face-primary** reframing (object detection disabled for stability)
- **Preserves full input height** (no zoom) by computing crops in the **raw** frame stream
- Outputs MP4 from AutoFlip, then you can re-encode with ffmpeg if desired

---

## Background

AutoFlip is an intelligent video reframing framework from Google Research. It uses detection and scene
signals (faces, objects, shot boundaries, borders, motion) to determine how to crop a video to a target
aspect ratio.

References:

- Google Research announcement:
  https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/
- MediaPipe AutoFlip documentation (legacy solution):
  https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md

---

## Current Working Configuration

This repository builds and runs a stable AutoFlip pipeline on Apple Silicon via Docker.

What works:

- Builds (or uses a prebuilt) `run_autoflip` binary inside an Ubuntu 22.04 arm64 container
- Runs a stable AutoFlip graph configuration:
  - OpenCV decode
  - Shot boundary detection
  - Face detection (enabled)
  - SceneCroppingCalculator with `aspect_ratio=9:16`
  - OpenCV encode
- Produces clean 9:16 output at full input height (no zoom)

Typical example (720p input):

- Input: 1280x720
- Output: 406x720 (9:16)

---

## What Was Fixed (Important)

### 1) Missing face detection model on host-mounted graphs

On some builds, AutoFlip expects to load face detection TFLite resources via a resource root.
We copy the required file into the repo and pass `--resource_root_dir=/work/models`.

Required file:

- `models/mediapipe/modules/face_detection/face_detection_full_range_sparse.tflite`

### 2) “Zoom” on longer videos (full height not preserved)

The critical fix is to compute cropping decisions using **raw frames**:

- Use `KEY_FRAMES: video_raw` (not the scaled/downsampled stream) in `SceneCroppingCalculator`.

This prevents the internal crop-size heuristics from drifting into a “zoom-like” behavior on longer videos.

---

## Repository Structure

    .
    ├── Dockerfile.autoflip
    ├── graphs/
    │   ├── autoflip_graph_faceprimary_target406x720_RAW.pbtxt
    │   └── (other graphs)
    ├── models/
    │   └── mediapipe/modules/face_detection/face_detection_full_range_sparse.tflite
    ├── scripts/
    │   ├── 00_make_mjpeg_avi.sh
    │   └── 01_run_autoflip_faceonly_noaudio.sh
    ├── vendor/
    │   └── mediapipe/   (git submodule)
    └── data/
        ├── in/          (ignored)
        └── out/         (ignored)

---

## Build Instructions

### Requirements

- Docker Desktop
- git
- ffmpeg (host machine)

### Build the Docker Image

From the repo root:

```bash
cd vendor/mediapipe
docker build --platform=linux/arm64 -t mediapipe-autoflip:cpu -f Dockerfile.autoflip .
```

---

## Run Instructions (Step by Step)

### Step 1: Convert input to MJPEG AVI (recommended for decode stability)

OpenCV decode inside the container is most reliable with MJPEG AVI on arm64.

Example:

```bash
./scripts/00_make_mjpeg_avi.sh "/path/to/input.mp4"
```

This writes an AVI file into `data/in/`.

---

### Step 2: Ensure face detection model exists under `models/`

Verify:

```bash
ls -lh models/mediapipe/modules/face_detection/face_detection_full_range_sparse.tflite
```

If it is missing, AutoFlip face detection will fail with “Failed to load resource”.

---

### Step 3: Run AutoFlip using the RAW graph (no-zoom, full-height preserved)

**Important:** `aspect_ratio` must be passed as a string like `9:16` (not a float).

**Important:** on zsh, wrap the entire `--input_side_packets=...` argument in quotes because `[...]`
in filenames triggers glob expansion.

Example (Sunflower):

```bash
docker run --rm \
  --cpus="8" --memory="40g" \
  -v "/Users/ryan-renslow/projects/916_conversion:/work" \
  -w /work \
  mediapipe-autoflip:cpu \
  /root/.cache/bazel/_bazel_root/1e0bb3bee2d09d2e4ad3523530d3b40c/execroot/mediapipe/bazel-out/aarch64-opt/bin/mediapipe/examples/desktop/autoflip/run_autoflip \
    --resource_root_dir=/work/models \
    --calculator_graph_config_file=/work/graphs/autoflip_graph_faceprimary_target406x720_RAW.pbtxt \
    "--input_side_packets=input_video_path=/work/data/in/sunflower_full__mjpeg.avi,output_video_path=/work/data/out/sunflower_full_9x16_RAW.mp4,aspect_ratio=9:16"
```

Outputs go to `data/out/`.

---

### Step 4: Verify output dimensions

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 data/out/*_9x16_RAW.mp4
```

Expected (for 720p inputs):

```text
406,720
```

---

## Graph Notes (RAW graph)

In `graphs/autoflip_graph_faceprimary_target406x720_RAW.pbtxt`, the critical section is:

- `VIDEO_FRAMES:video_raw`
- `KEY_FRAMES:video_raw`
- `target_width: 406`
- `target_height: 720`
- `target_size_type: USE_TARGET_DIMENSION`

This is the combination that preserves full height and avoids zoom.

---

## Known Issues / Next Improvements

1) Object detection on arm64
- The full face + object graph may crash on some arm64 builds.
- This repo prioritizes stability (face-primary) first.

2) Audio
- Current pipeline is video-only in AutoFlip.
- If you need audio, mux it back with ffmpeg after reframing.

3) Decode robustness
- If OpenCV decode fails on certain inputs, MJPEG AVI is the pragmatic workaround.

---

## Notes on Aspect Ratio

`SceneCroppingCalculator` expects aspect ratio as:

```text
9:16
```

Not a float like `0.5625`.

---

## Status

Verified working on Apple Silicon (M-series):

- Docker image builds
- Face-aware vertical reframing succeeds
- Full-height preserved (no zoom) using RAW graph
- Output confirmed to be 9:16
