# Strict Eluvio tagger audit — v3.2

## Current conclusion

The runtime architecture now follows the current Eluvio tagger pattern:
`common-ml` owns the daemon and JSONL protocol, while a `TagMessageProducer`
wraps the NBA YOLO shot service.

The real v3.1 image has successfully built on AI-03, verified the pinned NBA
checkpoint SHA-256, loaded the custom seven-class YOLO Student on an L40S, and
completed a 3/3 real-shot Podman smoke with zero errors and legal source-frame X
trajectories.

## Issues corrected before handoff

### Corrected — NumPy trajectory expansion

The first real smoke showed that inference itself was successful but
postprocessing used truth-value testing on a NumPy array. v3.1 changed that to
an explicit length check and added a regression test.

### Corrected — Joe shot ordinal parsing

Shot names such as `000-16_1818.mp4` are keyed by the leading ordinal. The
corpus preflight now requires exact 0..517 coverage, no gaps, and no duplicates.

### Corrected — Eluvio buildscripts integration

The experimental bundle's `build.sh` was not invoking
`buildscripts/build_container.bash` correctly. v3.2 uses the standard submodule,
Makefile include, and executable build script so Git revision/source/version
annotations come from the normal Eluvio build tooling.

### Corrected — undocumented progress extension in strict mode

Pinned `common-ml` implements `ProgressRatio`, but current `elv-ml` protocol
documentation explicitly describes `tag`, `progress`, and `error`. v3.2 keeps
`ProgressRatio` available as an opt-in diagnostic but disables it by default and
in strict Podman protocol tests.

### Preserved — exact source media contract

Each stdin path is an independent already-segmented shot. `source_media` is
preserved exactly, timing is relative to that source, and there is no cumulative
cross-file offset.

### Preserved — one YOLO stream

The deployed Student is one custom seven-family detector-style YOLO checkpoint.
It is not the earlier generic COCO + pose two-pass implementation.

## Remaining non-code gates

- The full 518-shot qualification must finish with 518 successful shots and zero
  protocol/model errors.
- Editorial crop quality still requires targeted human review; structural JSONL
  correctness is not sufficient.
- The current model is still the Round00 trusted checkpoint; later Student
  training can replace it only after passing the same artifact/protocol gates.
- Licensing and approved registry/deployment authorization remain organizational
  release requirements.
