#!/usr/bin/env bash
set -euo pipefail

git submodule update --init --recursive
export TITLE="vertical-focus"
export DESCRIPTION="Eluvio shot-aware sports/movie 9:16 focus tracker"

exec buildscripts/build_container.bash -t "verticalvideo:${IMAGE_TAG:-latest}" -f Containerfile . "$@"
