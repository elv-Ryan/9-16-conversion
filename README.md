# Eluvio NBA YOLO shot-to-X tagger — reviewed Round01 botics v4

Qwen-free Eluvio runtime for NBA horizontal reframing. The deployed tagger outputs normalized horizontal crop-center X coordinates. Vertical rendering and video encoding are handled downstream.

The expensive Qwen teacher is an **offline training/review system only**. Production uses the reviewed seven-family YOLO student, the reviewed causal controller, and the reviewed One Euro filter. In continuous `segment_file` mode, TransNetV2 provides shot boundaries.

## Current branch and validated checkpoint

Branch:

```text
nba-live-botics-v4
```

Current validated functional checkpoint:

```text
7df1652cdd30699d360e5271ea31962156637dd0
```

Migration checkpoints:

| Stage | Commit | Purpose |
| --- | --- | --- |
| Stage 1 | `a16b97b51a8be74c2b0ba5d049b21b25bd9bc7d2` | Replace the old Round00 model with the reviewed Round01 checkpoint and matching model manifest |
| Stage 2C | `fd2f1f013c94783b12b58e39eecb190e590bdfb9` | Activate the reviewed Round01 runtime, causal controller, One Euro filtering, reviewed inference contract, jump-aware reconstruction, integration tests, and hardened `test-local.sh` |
| Stage 2D | `7df1652cdd30699d360e5271ea31962156637dd0` | Tolerate microscopic FPS metadata jitter between adjacent stream segments while preserving strict rejection of real FPS/geometry changes |

Botics remains the host application. The migration deliberately preserved the Common-ML protocol, shot/segment ingestion, TransNet shot detection, metadata, tag format, source-frame X output, live fixed-point encoding, and container architecture while replacing the old steering path with the reviewed v4 runtime.

---

## Production contract

### Already-segmented VOD shots

```text
one shot file on stdin
  -> reviewed seven-family NBA YOLO student
  -> all retained detector candidates
  -> reviewed causal controller
  -> One Euro filter
  -> normalized source-frame X trajectory
  -> Eluvio tag JSONL + terminal progress
```

In `shot_file` mode the caller already supplies true shot boundaries. TransNetV2 is not used.

### Continuous segment stream

```text
consecutive segment files on stdin
  -> rolling TransNetV2 shot detector across file joins
  -> reviewed seven-family NBA YOLO student
  -> persistent per-shot reviewed controller state
  -> persistent per-shot One Euro state
  -> normalized source-frame X trajectory
  -> Eluvio tag JSONL + terminal progress
```

A file boundary is **not** a shot boundary. Controller/filter state persists through segment joins. State resets when botics closes a real shot or the existing maximum-shot safety valve closes it.

The primary output track remains:

```text
vertical_video
```

The tag field:

```text
additional_info["x-coordinates"]
```

contains exactly one legal normalized horizontal crop center per source frame.

---

## Runtime families

The deployed student has exactly seven classes:

- `active_speaker`
- `gameplay_follow`
- `graphic_text_lock`
- `person_subject`
- `safe_center`
- `split_screen`
- `static_composition`

The finer Qwen Category-Gold taxonomy is not deployed as a separate runtime head. `category` remains the coarse policy mapping of the runtime family.

---

## Reviewed model currently pinned

Artifact version:

```text
round01_schemafix_v3_reviewed_oneeuro
```

Trusted source checkpoint on AI-03:

```text
/home/mltrain/elv-ryan/projects/9-16-conversion-ryan-v2.1-student-mvp/output/
generic_basketball_v11_student_round00_v1/round01_round02_autopilot_v1/
runs/round01_schemafix_v3/weights/best.pt
```

Verified SHA-256:

```text
0921e4843864557bc1e5141f655a461af24c758d7390f2ea0553091df743865a
```

Model properties:

- FP32
- `imgsz=1280`
- reviewed evidence cadence: 10 Hz
- seven classes listed above
- model artifact size: 20,380,165 bytes

The current branch carries the reviewed `best.pt` artifact and matching `model_manifest.json`.

`scripts/fetch_current_model.sh` was also updated so it restores this reviewed Round01 checkpoint rather than the old Round00 model.

---

## Reviewed inference contract

Current detector settings:

```json
{
  "device": "0",
  "imgsz": 1280,
  "inference_fps": 10.0,
  "batch_size": 8,
  "min_detection_confidence": 0.001,
  "iou": 0.7,
  "max_detections": 100,
  "top_k": 20,
  "use_fp16": false
}
```

The detector intentionally runs with a low confidence floor and a larger detector cap. Candidates are sorted by confidence and the strongest 20 are retained for the reviewed runtime.

This supersedes the older botics inference settings of approximately `conf=0.05`, `max_detections=20`, and FP16.

---

## Reviewed controller and One Euro filter

Runtime manifest:

```text
models/nba_yolo_student/runtime_manifest.json
```

Selected filter:

```json
{
  "kind": "one_euro",
  "min_cutoff": 0.4,
  "beta": 1.0
}
```

Active steering path:

```text
retained YOLO candidates
  -> ReviewedRound01Runtime
  -> causal policy controller
  -> One Euro 0.4 / 1.0
  -> committed X samples
```

### Global controller parameters

```json
{
  "center_fallback_timeout_seconds": 1.0,
  "max_missing_target_hold_seconds": 0.5
}
```

### Gameplay

```json
{
  "acquire_confidence": 0.04,
  "maintain_confidence": 0.0,
  "occlusion_hold_seconds": 2.0,
  "pan_time_constant_seconds": 0.65
}
```

### Person subject

```json
{
  "acquire_confidence": 0.08,
  "maintain_confidence": 0.02,
  "identity_hold_seconds": 2.0,
  "pan_time_constant_seconds": 0.8
}
```

### Active speaker

```json
{
  "acquire_confidence": 0.05,
  "maintain_confidence": 0.02,
  "speaker_change_confirmation_observations": 5,
  "speaker_jump_distance_x": 0.12,
  "speaker_jump_cooldown_seconds": 0.5
}
```

The reviewed runtime receives **all retained candidates**. The locked botics family remains output/routing metadata but does not gate the reviewed steering evidence.

The old `_select_focus_series` and legacy `smooth_samples` helpers remain in the tree for compatibility/reference, but are not used by active Stage 2C+ steering.

### Causality

The reviewed controller and One Euro filter are causal.

They use no future frames and no hidden future-lookahead buffer.

Any segment-mode latency from shot detection or the botics commit horizon is therefore separate from the reviewed controller/filter.

---

## Current runtime defaults

Representative current defaults:

```json
{
  "device": "0",
  "imgsz": 1280,
  "inference_fps": 10.0,
  "batch_size": 8,
  "min_detection_confidence": 0.001,
  "iou": 0.7,
  "max_detections": 100,
  "top_k": 20,
  "use_fp16": false,
  "input_mode": "segment_file",
  "max_shot_seconds": 900.0,
  "min_shot_seconds": 0.75,
  "family_determination_max_seconds": 5.0,
  "trajectory_commit_lag_frames": 300,
  "hold_and_cut_families": "active_speaker",
  "min_hold_seconds": 1.0,
  "output_track": "vertical_video",
  "include_focus_samples": false,
  "emit_focus_track": false,
  "emit_progress_ratio": false
}
```

`hold_and_cut_families` and `min_hold_seconds` still parse for interface compatibility, but they no longer override the active reviewed controller.

---

## Input modes and latency semantics

### `shot_file`

Each stdin path is already one complete shot.

- TransNetV2 is not used.
- Every file closes one shot.
- The segment-mode trajectory commit lag is effectively disabled.
- Family determination defaults to the whole supplied shot.

This is the current VOD/back-up path when upstream already supplies shots.

### `segment_file`

Each stdin path is a consecutive slice of one continuous stream.

- TransNetV2 runs over a rolling buffer across file joins.
- A shot may span many files.
- One file may contain multiple shots.
- A segment boundary must not reset reviewed state.
- Default family determination is 5 seconds.
- `min_shot_seconds=0.75`.
- TransNet shot-detection lookahead is 25 source frames.
- `trajectory_commit_lag_frames=300`.

At approximately 60 fps:

```text
25-frame TransNet lookahead ≈ 0.42 s
300-frame commit lag        ≈ 5.0 s
```

Those are botics/streaming latencies, **not One Euro lookahead**.

A shot is emitted against the file being processed when the shot closes. Therefore a shot that began in an earlier segment may have a negative `start_time` and `frame_info.frame_idx` relative to the current file.

---

## Segment FPS metadata handling — Stage 2D

Real adjacent ~59.94-fps stream chunks were observed to expose tiny container/OpenCV FPS differences, for example:

```text
59.94005994005994
59.93956043956044
```

That difference is roughly 8.3 ppm and is not a real frame-rate switch.

The initial Stage 2C implementation compared the full geometry/FPS tuple exactly, so a long shot could fail when an otherwise identical later segment reported the alternate representation.

Stage 2D changed the open-shot guard so that:

- width changes are still rejected
- height changes are still rejected
- target-aspect changes are still rejected
- material FPS changes such as ~59.94 -> 30 fps are still rejected
- microscopic adjacent-segment FPS metadata jitter is accepted
- the reviewed controller clock is frozen to the FPS observed at the start of the open shot

Tolerance:

```python
math.isclose(current_fps, shot_fps, rel_tol=1e-4, abs_tol=1e-3)
```

This prevents harmless container metadata jitter from breaking a continuous shot while preserving fail-closed behavior for real stream incompatibilities.

---

## Source-frame reconstruction and speaker jumps

Final JSON tags still contain one normalized X per source frame.

Ordinary reviewed samples are interpolated to the source-frame grid.

Confirmed active-speaker jump events are handled specially: reconstruction holds the previous X up to the jump frame and switches at the jump instead of interpolating a fake pan through the cut.

Botics legal crop geometry remains authoritative.

---

## Live X sink contract

The live sink receives committed inference-sample X values, not the expanded per-source-frame JSON tag trajectory.

Each X is encoded as a four-byte little-endian fixed-point integer:

```text
encoded = round(x * 10000)
```

The live encoding and Fabric data-stream protocol were not changed by the reviewed runtime migration.

---

## Eluvio protocol implementation

The entrypoint continues to delegate protocol/daemon behavior to pinned `common-ml`:

- `catch_errors()`
- `get_params()`
- `TagMessageProducer`
- `Tag`
- `Progress`
- `Error`
- `FrameInfo`
- `run_default(...)`

`ProgressRatio` remains supported but disabled by default.

See:

- `docs/ELUVIO_COMPLIANCE_CHECKLIST.md`
- `docs/ELUVIO_TAGGER_AUDIT.md`
- `docs/OUTPUT_CONTRACT.md`
- `docs/REVIEWED_BOTICS_STAGE02C.md`

---

## Hardened `test-local.sh`

The September 29 migration converted `test-local.sh` into a strict acceptance harness.

Current behavior includes:

- `set -euo pipefail`
- explicit system PATH
- configurable `PYTHON_BIN`
- explicit `PYTHONPATH`
- `TEST_INPUT_FILE` override
- `TEST_INPUT_DIR` override
- `shot` explicitly uses `shot_file`
- default mode explicitly uses `segment_file`
- zero error rows required
- one progress row required per input file
- `shot`/`preshot` require one `vertical_video` tag per input
- wall timing written to `test-local.wall_seconds`
- `TEST_LOCAL_STRICT_PASS` only emitted after strict validation

Shot example:

```bash
TEST_INPUT_FILE=/absolute/path/to/shot.mp4 ./test-local.sh shot
```

Segment example:

```bash
TEST_INPUT_DIR=/absolute/path/to/ordered/segments ./test-local.sh default
```

---

## Clone / model / build

```bash
git clone --recurse-submodules git@github.com:elv-Ryan/9-16-conversion.git
cd 9-16-conversion

git checkout nba-live-botics-v4
git submodule update --init --recursive

./scripts/fetch_current_model.sh
./build.sh
```

### TransNet model asset

`segment_file` mode requires:

```text
models/shot/transnetv2/torch_transnetv2.pth
```

The local `models/shot` path resolves into the `model-shot` runtime asset tree.

`build.sh` populates it from the AI-03 shared model store:

```bash
mkdir -p model-shot
rsync --progress --update --times --recursive --links --delete \
  /ml/models/shot model-shot
```

`model-shot/**` is gitignored.

This means the TransNet checkpoint is available to local segment testing and the container without committing that shared model asset to Git.

---

## Standard Eluvio buildscripts

The repository still uses the standard `qluvio/buildscripts` integration.

`build.sh` invokes:

```text
buildscripts/build_container.bash
```

with the image:

```text
verticalvideo-v2.5:${IMAGE_TAG:-latest}
```

Typical targets remain:

```bash
make build
make test
make deploy
```

Deploy only after model/runtime assets, tests, container checks, and registry authentication are all confirmed.

---

# September 29, 2026 migration and validation

## 1. Reviewed Round01 model promoted into botics

Old botics checkpoint:

```text
artifact: round00_fp32_6gpu_v7
SHA-256: e1f498b0447f77c30d2b82210367b556d5f649caeee370f5fd3d20f7a0fec770
```

Current reviewed checkpoint:

```text
artifact: round01_schemafix_v3_reviewed_oneeuro
SHA-256: 0921e4843864557bc1e5141f655a461af24c758d7390f2ea0553091df743865a
```

The reviewed checkpoint, matching model manifest, reviewed runtime manifest, and runtime code are now on `nba-live-botics-v4`.

## 2. Reviewed runtime integration — Stage 2C

Stage 2C added or activated:

- reviewed `runtime_manifest.json`
- `reviewed_core.py`
- `reviewed_runtime.py`
- reviewed detector defaults in `config.py`
- `iou` and retained `top_k`
- FP32 reviewed inference
- active reviewed steering in `service.py`
- persistent reviewed runtime per shot
- persistent One Euro state per shot
- all retained candidate families provided to the reviewed controller
- jump-aware source-frame reconstruction
- reviewed runtime contract tests
- botics integration tests
- model-fetch protection against restoring Round00
- hardened `test-local.sh`

No ball model, 150 ms future buffer, or retraining was added.

## 3. Automated tests

After the Stage 2D regression tests were added:

```text
reviewed botics integration tests: 65 / 65 PASS
frozen reviewed-runtime tests:      2 / 2 PASS
```

The integration tests cover:

- all seven families
- reviewed runtime parity
- controller + One Euro behavior
- active-speaker jumps
- source-frame expansion
- segment persistence
- no replay of committed observations
- future pending evidence
- deferred cuts
- final flush
- negative metadata offsets
- live sample cadence
- live fixed-point encoding
- geometry safety
- old model-constructor compatibility
- microscopic segment FPS jitter
- rejection of real/material FPS changes

The original legacy suite remains at the same known baseline observed before the Stage 2C migration: 56 tests with 9 known stale-default/timing failures and no new failures attributable to the migration.

## 4. Representative shot-file A/B validation

Real-video A/B review against original botics covered:

- gameplay
- active speaker
- person subject
- static composition
- graphic/text
- split screen

The Stage 2C output passed visual review.

In the gameplay example that previously produced a large wrong-way excursion, the reviewed controller visibly improved the camera behavior.

A four-gameplay-shot `test-local.sh shot` benchmark measured:

```text
total source video:       23.273 s
original botics wall:     22.818 s  ≈ 1.02x realtime
reviewed V4 wall:         18.398 s  ≈ 1.27x realtime
```

Longest tested shot:

```text
source duration: 11.63 s

original botics:
  6.59 s wall
  ≈1.77x realtime

reviewed V4:
  5.28 s wall
  ≈2.20x realtime
```

These are cold per-shot measurements, not a steady-state production benchmark. Original botics was always run first and V4 second, so OS/library cache ordering may contribute to the measured difference.

Do not interpret the aggregate ~19% wall-time reduction as a production-speed guarantee.

## 5. Real segment corpus inventory

Corpus:

```text
/home/mltrain/elv-joe/segments/test-files
```

September 29 inventory:

```text
MP4 files:       3443
stream groups:    230
total video:     6891.09 s
                 114.85 min

median segment:  2.002 s
segment range:   0.200 - 2.002 s
typical format:  1920x1080 @ ~59.94 fps
```

Typical groups contain 15 consecutive ~2.002-second segments:

```text
15 segments ≈ 30.03 s continuous source
```

## 6. First real segment-mode regression

Real group `0114` initially exposed the FPS metadata-jitter issue described above.

After Stage 2D, the exact previously failing stream was rerun through the real committed `test-local.sh default` / `segment_file` path.

Result:

```text
group:                    0114
segments:                    15
source frames:              1800
source duration:          ~30.03 s

progress rows:                15
errors:                        0
vertical tags:                 1

family:             split_screen
shot frame count:           1800
emitted frame_idx:         -1680

wall time:             11.262143 s
throughput:             ~2.67x realtime
```

The single shot remained open across all 15 input files.

That means the reviewed runtime and its One Euro/controller state persisted through **14 segment-file joins** without treating those joins as shot resets.

The negative `frame_idx=-1680` is expected: the shot began in segment 1 and was finally emitted while segment 15 was the current source file.

This is a real segment-mode test, not a synthetic fixture.

---

## Joe 518-shot qualification

Shot corpus on AI-03:

```text
/home/mltrain/elv-joe/shots/iq__2q6ZyYAmFfKDBJeWMGsWLP549twd
```

Historical v3.1 work already included real Podman smoke testing and a 518-shot qualification path.

The reviewed V4 branch still needs to pass the corresponding container/corpus gates before production deployment.

For faster corpus-only checks, `scripts/parallel_joe518_qualify.py` can shard independent shots over genuinely free GPUs.

Parallel corpus testing is **not** a substitute for single-container steady-state latency/throughput qualification.

---

## What the September 29 migration deliberately did not add

This migration does **not** include:

- ball tracking
- ball-aware steering
- V6/V7 ball-model work
- new student training
- a 150 ms future-evidence buffer
- next-generation future-lookahead policies
- vertical MP4 rendering inside the production runtime
- new LIVE-specific policy tuning beyond the existing botics segment architecture

Those remain separate future experiments so this reviewed Round01 migration stays auditable.

---

## Output consumed by the renderer

The Fabric-facing result remains JSONL.

For each shot, the renderer consumes:

```text
data.additional_info["x-coordinates"]
```

This is one normalized source-width horizontal crop center per source frame.

Other fields are metadata/diagnostics and do not alter the X trajectory contract.

Local preview MP4s are QA artifacts only and are not part of the production runtime.

---

## Next validation steps

After Stage 2D:

1. broader multi-stream `segment_file` acceptance and continuous visual QA
2. instrument actual live publication timing:
   - time-to-first-X
   - steady-state publication delay
   - shot-cut/reset delay
   - effect of the current 300-frame commit horizon
3. Podman/container qualification using the reviewed V4 branch
4. modest multi-shot and larger 518-shot smoke tests
5. only then evaluate ball-aware steering or future-evidence policies

The immediate live-latency question is no longer One Euro: the reviewed controller/filter is causal.

The next measurement should isolate latency introduced by the surrounding stream architecture, especially:

```text
TransNet lookahead:             25 frames ≈ 0.42 s @ 60 fps
family determination horizon:   5 s
trajectory commit lag:         300 frames ≈ 5.0 s @ 60 fps
```

---

# Historical README before reviewed-Round01 migration

The material below is retained for historical context. Where values conflict with the current sections above, the current reviewed-v4 sections are authoritative.

# Eluvio NBA YOLO shot-to-X tagger — v3.3 handoff

YOLO-only Eluvio tagger MVP for already-segmented NBA shots.

## Production contract

```text
one shot file on stdin
  -> one custom seven-family NBA YOLO Student
  -> shot family + semantic focus bbox
  -> full-shot horizontal trajectory controller
  -> normalized source-frame X trajectory
  -> Eluvio tag JSONL + terminal progress
```

Production runtime does **not** use Qwen, shot detection, a second YOLO pass,
vertical rendering, or video encoding.

The primary output track is `vertical_video`. The tag's
`additional_info["x-coordinates"]` contains exactly one legal normalized crop
center per source frame.

## Runtime families

- `active_speaker`
- `gameplay_follow`
- `graphic_text_lock`
- `person_subject`
- `safe_center`
- `split_screen`
- `static_composition`

The current checkpoint does not expose the finer Category-Gold taxonomy as a
separate head. `category` is therefore the coarse policy mapping of the runtime
family.

## Eluvio protocol implementation

The entrypoint delegates the daemon/protocol machinery to current pinned
`common-ml`:

- `catch_errors()`
- `get_params()`
- `TagMessageProducer`
- `Tag`, `Progress`, `Error`, `FrameInfo`
- `run_default(...)`

`ProgressRatio` is supported by the pinned `common-ml`, but it is disabled by
default because the current `elv-ml` protocol document explicitly documents
`tag`, `progress`, and `error` as the protocol message types. Runtime progress
is available in logs/test-controller status without requiring the extension.

See:

- `docs/ELUVIO_COMPLIANCE_CHECKLIST.md`
- `docs/ELUVIO_TAGGER_AUDIT.md`
- `docs/OUTPUT_CONTRACT.md`

## Runtime parameters

```json
{
  "device": "0",
  "imgsz": 1280,
  "inference_fps": 10.0,
  "batch_size": 8,
  "min_detection_confidence": 0.05,
  "max_detections": 20,
  "use_fp16": true,
  "input_mode": "shot_file",
  "min_shot_seconds": 0.25,
  "hold_and_cut_families": "active_speaker",
  "min_hold_seconds": 0.7,
  "output_track": "vertical_video",
  "include_focus_samples": false,
  "trajectory_commit_lag_frames": 120,
  "emit_focus_track": false,
  "emit_progress_ratio": false,
  "continue_on_error": true
}
```

### Input modes

Both modes run the same pipeline; they differ only in where shot boundaries
come from.

- `shot_file` (default): every stdin path is already one whole shot, so a
  boundary is placed at the end of each file and one tag is emitted per file.
- `segment_file`: stdin paths are consecutive slices of one continuous stream
  (roughly 2s–60s each). TransNetV2 (`shot_model_path`) detects the cuts, so a
  shot may span several files and one file may contain several shots. Frames
  are carried across the joins between files so the detector never sees a
  segment edge as a cut, which means a cut near the end of a file is only
  reported once the next file supplies the frames after it.

`family_determination_max_seconds` caps how much of a shot feeds the runtime
family vote; the rest of the shot reuses that family. When it is not set
explicitly it defaults to `999999` in `shot_file` mode (vote on the whole
shot) and `3` in `segment_file` mode. A shot shorter than that is voted on
with whatever it has.

Once the family is settled the X trajectory is worked out as the segments
arrive — catching up whatever the vote was waiting on, then keeping pace —
rather than in one pass when the shot ends. `trajectory_commit_lag_frames`
(default 120, ~2s) is how far behind the decoded frames it commits. It must
clear shot detection's 25-frame lookahead so a late cut can never land in
committed trajectory; past that it buys right-context for the bidirectional
smoothing passes. Committed X is final and is never revised.

A shot is emitted against the file being processed when it is cut, so a shot
that began earlier carries a negative `start_time` and `frame_info.frame_idx`
measured back from that file's frame 0. Because a cut near a file's end is
deferred until the next file, `end_time` can be negative too. See
`docs/OUTPUT_CONTRACT.md`.

## Clone / model / build

```bash
git clone --recurse-submodules git@github.com:elv-Ryan/9-16-conversion.git
cd 9-16-conversion
git checkout nba-yolo-shot-tagger-v3.3
git submodule update --init --recursive

./scripts/fetch_current_model.sh
make unit-test
make build
```

`best.pt` is intentionally not committed. `model_manifest.json` is committed and
pins its SHA-256. On AI-03, `fetch_current_model.sh` copies the trusted checkpoint
locally. Elsewhere it can fetch it over SSH using `REMOTE`, `SSH_PORT`, and
`MODEL_SOURCE`.

## Black-box Podman test

```bash
DEVICE=2 IMAGE=nba-yolo-shot-tagger:latest \
  ./scripts/test_podman_shot.sh /absolute/path/to/one-shot.mp4
```

The test requires a successful `vertical_video` tag, one X per source frame,
legal normalized crop centers, and exactly one terminal `Progress` message.

## Standard Eluvio buildscripts

The repository includes the normal `qluvio/buildscripts` submodule and
`Makefile.tagger-model` integration. `build.sh` calls
`buildscripts/build_container.bash`, so normal build metadata/annotations are
added by the Eluvio build tooling. The release helper also prepares `test-files/`
and runs the unmodified official `make test` before the stricter X-semantic
black-box test and GitHub push.

Typical targets:

```bash
make build
make test
make deploy
```

`make deploy` should only be used after the repository is clean, the model
artifact is present, the Podman test passes, and registry authentication is
configured.

## Joe 518-shot qualification

Source corpus on AI-03:

```text
/home/mltrain/elv-joe/shots/iq__2q6ZyYAmFfKDBJeWMGsWLP549twd
```

The real v3.1 image has already passed a three-shot, zero-error Podman smoke on
this corpus. A full single-container 518-shot run is used to qualify actual
Eluvio-like long-lived behavior and measure single-container throughput.

For faster corpus-only qualification, `scripts/parallel_joe518_qualify.py` may
run independent shot shards on multiple genuinely free GPUs. That result is
**not** a substitute for the single-container latency/throughput measurement;
it is only a faster corpus correctness gate.

## Model currently pinned

Trusted checkpoint source on AI-03:

```text
/home/mltrain/elv-ryan/projects/9-16-conversion-ryan-v2.1-student-mvp/output/
generic_basketball_v11_student_round00_v1/yolo_fp32_6gpu_round00_v7/long_run/
focus_target_round00_fp32/weights/best.pt
```

Current verified SHA-256 from the v3.1 image build:

```text
e1f498b0447f77c30d2b82210367b556d5f649caeee370f5fd3d20f7a0fec770
```

Replace the checkpoint later only after a newer Student champion passes the same
seven-class artifact, protocol, Podman, and corpus gates.

## Output consumed by the renderer

The Fabric-facing file is JSONL, as required by the Tagger protocol. For each
shot, the renderer consumes `data.additional_info["x-coordinates"]`: one
normalized source-width horizontal crop center per source frame. Other fields
are diagnostic/routing metadata and do not change the X contract. The local
`x-output.json` helper is a convenience export, not the Fabric wire format.
