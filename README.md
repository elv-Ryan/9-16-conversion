# 9:16 conversion using MediaPipe AutoFlip

A minimal, reproducible example that converts a \~16:9 input into 9:16
vertical format using **MediaPipe AutoFlip**.

This repository is optimized for: - Apple Silicon (arm64) - Docker-based
build and run - A lowest-friction path that is verified to work
end-to-end

------------------------------------------------------------------------

## Background

AutoFlip is an intelligent video reframing framework from Google
Research. It uses detection and scene signals (faces, objects, shot
boundaries, borders, motion) to determine how to crop a video to a
target aspect ratio.

References:

-   Google Research announcement:\
    https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/

-   MediaPipe AutoFlip documentation (legacy solution):\
    https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/autoflip.md

------------------------------------------------------------------------

## Current Working Configuration

This repository builds and runs a stable AutoFlip pipeline on Apple
Silicon via Docker.

What works:

-   Builds `run_autoflip` from MediaPipe using Bazel inside an Ubuntu
    22.04 arm64 container
-   Runs a stable AutoFlip graph configuration:
    -   OpenCV video decode
    -   Frame scaling and shot boundary detection
    -   Face detection (enabled)
    -   SceneCroppingCalculator with `aspect_ratio=9:16`
    -   OpenCV video encode
-   Produces correct 9:16 output

Example:

Input: 832x480\
Output: 270x480 (9:16)

------------------------------------------------------------------------

## What Is Disabled (and Why)

### Object Detection

The full AutoFlip graph (face + object detection) segfaults on arm64 in
this Docker build.\
This repository uses a **face-only configuration** to ensure a stable
minimal demo.

### Audio Passthrough

The stock AutoFlip graph attempts audio extraction and remuxing via
OpenCV decoder side packets.\
This fails in the current container configuration, so audio is
intentionally removed.\
Output is video-only.

### MP4 / H.264 Decode Reliability

OpenCV decoding inside the container is unreliable for certain MP4/H.264
inputs on arm64.\
The stable workaround is transcoding to MJPEG AVI before running
AutoFlip.

------------------------------------------------------------------------

## Repository Structure

    .
    ├── Dockerfile.autoflip
    ├── graphs/
    │   └── autoflip_graph_noaudio_faceonly.pbtxt
    ├── scripts/
    │   ├── 00_make_mjpeg_avi.sh
    │   └── 01_run_autoflip_faceonly_noaudio.sh
    ├── vendor/
    │   └── mediapipe/   (git submodule)
    └── data/
        ├── in/          (ignored)
        └── out/         (ignored)

------------------------------------------------------------------------

## Build Instructions

### Requirements

-   Docker Desktop
-   git
-   ffmpeg (host machine)

### Build the Docker Image

    cd vendor/mediapipe
    docker build --platform=linux/arm64 -t mediapipe-autoflip:cpu -f Dockerfile.autoflip .

------------------------------------------------------------------------

## Run Instructions

### Step 1: Convert Input to MJPEG AVI

OpenCV decode is most reliable with MJPEG AVI in this setup:

    ./scripts/00_make_mjpeg_avi.sh /path/to/input.mp4

This writes an AVI file into `data/in/`.

------------------------------------------------------------------------

### Step 2: Run AutoFlip (Face-Only, No Audio)

    ./scripts/01_run_autoflip_faceonly_noaudio.sh

Output is written to `data/out/`.

------------------------------------------------------------------------

### Step 3: Verify Output Aspect Ratio

    ffprobe -v error -select_streams v:0 -show_entries stream=width,height   -of default=noprint_wrappers=1:nokey=0 data/out/*.mp4

Expected output ratio ≈ 9:16.

------------------------------------------------------------------------

## Known Issues

1.  Re-enable object detection safely on arm64\
    Likely requires modifying TFLite inference configuration (delegates,
    XNNPACK, threading) or replacing the object detection subgraph.

2.  Restore audio\
    Practical solution: run AutoFlip video-only, then mux original audio
    back using ffmpeg.

3.  Improve decode pipeline\
    Replace OpenCV decoder with FFmpeg-based decode stage or feed frame
    sequences into MediaPipe.

------------------------------------------------------------------------

## Notes on Aspect Ratio

`SceneCroppingCalculator` expects aspect ratio as a string in the form:

    width:height

Example:

    9:16

Not a float value.

------------------------------------------------------------------------

## Status

Verified working on Apple Silicon (M-series):

-   Docker build succeeds
-   Face-aware vertical reframing succeeds
-   Output confirmed to be 9:16
