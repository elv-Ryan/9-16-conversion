#!/bin/bash

set -e

git submodule update --init --recursive

exec buildscripts/build_container.bash -t "verticalvideo:${IMAGE_TAG:-latest}" . -f Containerfile "$@"
