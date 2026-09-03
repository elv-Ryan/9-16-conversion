# Eluvio tagger compliance checklist

Authority order used for this checklist:

1. `qluvio/elv-ml` current tag-container implementation guide.
2. `qluvio/elv-ml` current tag-container protocol.
3. `eluv-io/common-ml` pinned source used by this image.
4. `qluvio/buildscripts` current build/test conventions.
5. This repository.

## Container protocol

| Requirement | Status | Implementation |
|---|---|---|
| OCI-compatible image | PASS | `Containerfile`, Podman build/run |
| Long-lived process | PASS | `common_ml.tagging.run_helpers.run_default` |
| Newline-delimited input paths on stdin | PASS | `run_default` stdin reader |
| Required `--output-path` | PASS | parsed and enforced by `run_default` |
| Optional runtime `--params` JSON | PASS | `get_params()` + strict `RuntimeConfig` validation |
| Output is newline-delimited JSON | PASS | `common-ml` `write_message` |
| `tag` messages | PASS | `Tag` on `vertical_video` |
| terminal per-source `progress` | PASS | `Progress(source_media=...)` after tags |
| source-scoped `error` | PASS | `Error(message=..., source_media=...)` |
| `source_media` equals the exact input path | PASS | producer propagates stdin path unchanged |
| times are integer milliseconds | PASS | shot/source-relative `start_time` / `end_time` |
| optional `additional_info` | PASS | X trajectory + family/model metadata |
| optional `frame_info` | PASS | `{"frame_idx": ...}` on every tag; adds `box` on the optional focus track when a representative detection exists |
| uncaught setup errors reach Tagger | PASS | `catch_errors()` installed before config/model init |
| per-file inference errors can continue | PASS | source-scoped `Error`; `continue_on_error` defaults true |

## `common-ml` usage

- No custom stdin daemon.
- No custom Fabric JSONL serializer in production.
- `TagMessageProducer` is the model/runtime adapter.
- `run_default` owns argument parsing, stdin queueing, output-path handling, and
  JSONL writes.
- `get_params` is used for runtime configuration.
- `catch_errors` is used for initialization failures.
- `Tag`, `Progress`, `Error`, and `FrameInfo` are imported from current
  `common_ml.tagging.messages`.

The pinned `common-ml` also implements `ProgressRatio`, but the current
`elv-ml` protocol document does not list it among the documented wire message
types. Therefore `emit_progress_ratio` is **false by default** and all strict
Podman qualification tests leave it disabled.

## Shot / model contract

| Requirement | Status |
|---|---|
| input is already one shot | PASS |
| no shot detection in default runtime | PASS |
| backup external shot manifest | PASS |
| one custom NBA YOLO Student stream | PASS |
| no Qwen runtime | PASS |
| no second YOLO detector/pose pass | PASS |
| family and semantic focus from same YOLO result stream | PASS |
| normalized X output | PASS |
| exactly one X per source frame | PASS |
| legal full-height 9:16 center bounds | PASS |
| no Y trajectory / zoom | PASS |
| no vertical MP4 render/encode | PASS |

## Artifact integrity

- Model class order is locked to seven runtime families.
- Model SHA-256 is checked before image build and again during image build.
- `best.pt` is excluded from Git; `model_manifest.json` is tracked.
- Image import test validates `common-ml` and tagger imports.

## Build/test conventions

- `.gitmodules` points at `qluvio/buildscripts`.
- `Makefile` includes `buildscripts/Makefile.tagger-model`.
- `build.sh` invokes `buildscripts/build_container.bash` and therefore receives
  Eluvio image annotations/labels/version metadata.
- `make unit-test` runs repository tests.
- the release helper creates `test-files/` and runs the unmodified standard
  `make test` from `buildscripts/Makefile.tagger-model`.
- `scripts/test_podman_shot.sh` then adds a stricter X-semantic GPU black-box
  contract check (one X per frame, finite/legal X, terminal progress).

## Release gates

Before calling a revision deployable:

1. unit tests PASS;
2. model SHA gate PASS;
3. standard buildscripts build PASS;
4. real Podman one-shot/three-shot contract smoke PASS;
5. full 518-shot corpus protocol gate PASS;
6. human review of selected crops remains required for editorial quality;
7. applicable Ultralytics/model licensing must be cleared;
8. image registry push/deploy must use the approved Eluvio registry workflow.
