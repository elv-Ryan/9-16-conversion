# Eluvio YOLO26 Vertical Focus v5

This repository emits a shot-aware horizontal center trajectory for a fixed
9:16 crop. The runtime is YOLO26-only and preserves the existing Eluvio
`common-ml` protocol.

## Runtime

The detector components are:

- `yolo26s.pt` for COCO `person` and `sports ball` detections.
- `yolo26n-pose.pt` for person detections and pose-derived face/head evidence.

There is no MediaPipe inference, import, dependency, or fallback. YOLO model
loading remains lazy so deterministic tests can inject a fake detector.

The default model paths are:

```text
/elv/model/models/yolo26/yolo26s.pt
/elv/model/models/yolo26/yolo26n-pose.pt
```

## Policy separation

`SportsFocusPolicy` is a gameplay-state controller. It prioritizes a trusted
observed ball, then a bounded predicted ball during a short miss, then
ball-anchored local action. Non-gameplay states use separate close-up,
announcer/studio, timeout/bench, crowd/idle, graphic/static, and weak-evidence
behavior. A person or group cannot override trusted ball evidence.

`MovieFocusPolicy` is a persistent-character and composition controller. It
pairs frontal face/head evidence to an enclosing person, preserves character
identity during temporary face loss, suppresses small background evidence,
anchors interaction-group identity to the dominant character, and holds exact
composition when evidence does not justify movement.

Neither policy contains source names, clip timestamps, venue/title rules, or
fixed coordinates from an evaluation clip. Configuration is in
`configs/policies.yml`.

## Output contract

The model emits only these tracks:

```text
vertical_video
focus
focus_bbox
```

The `vertical_video` tag contains one normalized horizontal center per output
frame:

```json
{
  "frame_info": {"frame_idx": 0},
  "additional_info": {"x-coordinates": [0.5, 0.5, 0.51]}
}
```

The implementation does not emit Y movement, zoom, or image-compression
instructions. Shot-level JSONL remains compact. A terminal `Progress` message
is emitted after all tags for each successful input; errors continue through
`common-ml`.

## Local setup

YOLO26 use is license-gated. Read `EXPERIMENTAL_LICENSE_NOTICE.md` before
building or distributing anything.

```bash
./scripts/setup_yolo26_experiment.sh /path/to/checkout
```

That script creates `.venv-yolo26`, installs the pinned dependencies, and
places the two YOLO weights under `models/yolo26/`.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The suite covers protocol serialization, YOLO-only configuration, pose-derived
frontal/nonfrontal evidence, predicted-position tracking, sports ball trust and
scene state, movie character persistence/background suppression, shot cues,
and smoothing holds.

## Visual evaluation

Place any number of clips matching `sports_*.mp4` and `movie_*.mp4` in the
short-clips directory, then run:

```bash
SHORTS_DIR=/path/to/short \
OUT_DIR=test-output/visual-v5-yolo26 \
./scripts/run_visual_test_yolo26.sh
```

The runner produces:

```text
sports.jsonl
movie.jsonl
sports.log
movie.log
previews/*_side_by_side.mp4
previews/*_vertical.mp4
debug/sports_frames.jsonl
debug/movie_frames.jsonl
metrics/trajectory_by_clip.csv
metrics/trajectory_by_shot.csv
metrics/detection_summary.json
metrics/track_continuity.json
metrics/focus_label_summary.json
metrics/acceptance_report.md
```

The debug sidecars are QA-only and do not alter the common-ml output. The
side-by-side renderer uses frame-level sidecar labels when present and marks the
fallback overlay as shot-dominant when not present. Both preview variants map
source audio into the output. The runner verifies audio streams, dimensions,
frame counts, and duration within one source frame for every preview pair.

The existing clips are regression examples rather than the sole tuning set.
Use the source-level development/holdout procedure in
`docs/V5_GENERALIZATION_TEST_PLAN.md` when adding evaluation videos.

## Main runtime parameters

| Parameter | Default | Meaning |
|---|---|---|
| `mode` | `movie` | Exactly `sports` or `movie`. |
| `detector_backend` | `yolo26` | Only accepted value. |
| `yolo_detect_model` | `.../yolo26s.pt` | Object model path. |
| `yolo_pose_model` | `.../yolo26n-pose.pt` | Pose model path. |
| `yolo_device` | `0` | CUDA index or `cpu`. |
| `yolo_imgsz` | policy value | Optional object-model image-size override. |
| `yolo_half` | `true` | FP16 on non-CPU devices. |
| `detection_fps` | policy value | Optional cadence override. |
| `debug_jsonl_path` | empty | Optional QA frame-sidecar path. |

Legacy `object_model`, `face_model`, and `delegate` fields are accepted only to
avoid breaking existing request envelopes. They are inert and cannot enable an
alternate detector.

## Deployment boundary

Do not commit, push, build/publish a container, or deploy this branch until the
code diff, tests, real-video previews, metrics, and the YOLO/Ultralytics license
have been reviewed explicitly.
