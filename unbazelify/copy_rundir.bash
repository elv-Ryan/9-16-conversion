#!/bin/bash

set -x -v

count="$(find /root/.cache/bazel -name run_autoflip.runfiles | tee /root/.cache/bazel/find_runfiles | wc -l)"

if [ "$count" != 1 ]; then
    echo 1>&2 there are "$count" rundirs and only expecting one
    echo 1>&2 'maybe try removing the bazel build cache and rebuilding??'
    exit -1
fi

cp -vLr "$(cat /root/.cache/bazel/find_runfiles)" /runfiles
