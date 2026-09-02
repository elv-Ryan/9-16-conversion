#!/usr/bin/env bash
set -Eeuo pipefail
export TITLE="NBA YOLO Shot Focus"
export DESCRIPTION="Eluvio shot-in NBA YOLO family/focus tagger emitting normalized horizontal X trajectories."
exec buildscripts/build_container.bash \
  -t "nba-yolo-shot-tagger:${IMAGE_TAG:-latest}" \
  -f Containerfile \
  .
