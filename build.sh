#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p model-shot
rsync --progress --update --times --recursive --links --delete /ml/models/shot model-shot

export TITLE="NBA YOLO Shot Focus"
export DESCRIPTION="Eluvio shot-in NBA YOLO family/focus tagger emitting normalized horizontal X trajectories."


exec buildscripts/build_container.bash \
  -t "verticalvideo-v2:${IMAGE_TAG:-latest}" \
  -f Containerfile \
  .
