# Eluvio 9:16 Vertical Focus

This repository is an Eluvio tagger container that analyzes each video frame and emits the horizontal center point needed for a fixed-height 9:16 crop.

It does not render the production video. It does not move vertically, zoom, stretch, or compress the picture. It only decides the normalized X center for each frame and writes compact JSONL tags for the Fabric.

Supported policies:

- `sports`
- `movie`

## What changed from v1

| v1 | v2 (`ryan-v2`) |
|---|---|
| AutoFlip-oriented local reframing demo | Deployable Eluvio tagger container |
| Primarily face-based crop selection | Separate sports and movie policies |
| Rendered a cropped video | Emits frame-level normalized X coordinates |
| Custom/legacy execution paths | Uses `common-ml` stdin, JSONL, error, and progress handling |
| One implicit focus result | Three output tracks per shot |
| Weak shot awareness | Tagstore `shot_detection` integration with local cut fallback |
| Pixel/frame smoothing | Time-normalized causal speed and acceleration limits |
| No stable focus identity | Deterministic person, face, and ball tracking IDs |
| Could chase a small ball | Ball normally selects a nearby player or action group |
| Mixed local artifacts and old MediaPipe source | Clean Python package, OCI Containerfile, buildscripts workflow |

The current version establishes the deployment foundation. The exact sports/movie focus selection and smoothing values still require refinement from visual QA.

## Models and algorithms

### Object model

The object detector is **MediaPipe Tasks ObjectDetector** using:

```text
models/mp_tasks/object_detector/efficientdet_lite0.tflite
```

This is **EfficientDet-Lite0**, run through TensorFlow Lite by MediaPipe. The code keeps only:

- `person`
- `sports ball` / `ball`

The default delegate is CPU. GPU can be requested with `"delegate":"gpu"` only on a host where the MediaPipe GPU delegate works.

### Face model

Movie and announcer evidence uses MediaPipe's full-range face detector through:

```python
mp.solutions.face_detection.FaceDetection(model_selection=1)
```

If face inference is unavailable, the model continues with person, motion, and safe-center evidence rather than terminating.

### Tracking

Detections are associated frame to frame using a deterministic lightweight tracker. Association uses:

- matching label
- bounding-box intersection-over-union
- normalized center distance
- exponentially blended boxes
- a maximum missed-update lifetime

The resulting IDs look like `person:3`, `face:7`, or `ball:2`.

### Shot detection

When `ELV_CONTENT` and `ELV_TOKEN` are available, shot ranges are loaded from the Fabric `shot_detection` track. Requests are paginated.

Without Fabric credentials, a local HSV histogram cut detector resets the focus tracker and camera smoothing at detected cuts.

### Motion

Motion is image-derived evidence used only as a fallback or supporting signal. It is not a separate neural network.

### Smoothing

The output trajectory is a causal, critically damped second-order camera controller. It limits normalized speed and acceleration, applies a deadband, prevents overshoot, and clamps the crop window inside the frame.

## Input and runtime protocol

The container follows the current Eluvio tag-container protocol:

- OCI-compatible image
- stays alive while stdin remains open
- accepts newline-separated input paths through stdin
- accepts `--output-path`
- accepts JSON runtime parameters through `--params`
- writes newline-separated JSON messages
- writes all tags for a source before its terminal `progress` message
- uses `common-ml` for serialization, flushing, standard errors, and the stdin loop

Example:

```bash
printf '/elv/test/sports_01.mp4\n' | podman run --rm -i \
  -v "$PWD/test-files/short:/elv/test:ro" \
  -v "$PWD/test-output:/elv/tags" \
  verticalvideo:latest \
  --output-path /elv/tags/out.jsonl \
  --params '{"mode":"sports","delegate":"cpu"}'
```

## Output tracks

Each shot emits three logical outputs.

### `vertical_video`

One tag contains:

- shot start and end time in milliseconds
- first frame index for the shot
- `additional_info.x-coordinates`
- one normalized X center for every decoded frame in that shot

`0.0` is the left edge, `0.5` is center, and `1.0` is the right edge.

### `focus`

One tag summarizes why the shot was focused where it was:

- dominant focus label
- selected focus IDs
- labels encountered in the shot
- average confidence
- policy name
- policy version
- policy schema

### `focus_bbox`

Up to `max_boxes_per_shot` representative normalized bounding boxes are emitted for the selected focus IDs. If there is no detected subject, a full-height safe-center 9:16 crop box is emitted.

## `configs/policies.yml`, line by line

### Top-level fields

| Field | Meaning and use |
|---|---|
| `version` | Human-readable policy revision. Written into `focus.additional_info.policy_version`. |
| `schema_version` | Name of the policy-file structure. Written into `focus.additional_info.policy_schema`. |
| `policies` | Contains the two permitted runtime modes: `sports` and `movie`. |
| `output` | Controls output track names, numeric precision, and bbox count. |

### Fields shared by both policies

#### `detection`

| Field | Meaning and use |
|---|---|
| `detection_fps` | Frequency at which neural detection is run. Frames between detections use tracker state. Higher values react faster but cost more compute. |
| `person_min_score` | Minimum EfficientDet confidence accepted for a person. Lower values increase recall and false positives. |
| `ball_min_score` | Minimum EfficientDet confidence accepted for a sports ball. |
| `face_min_score` | Minimum MediaPipe face confidence accepted. |
| `max_results` | Maximum object detections returned by MediaPipe per detection frame. |

#### `tracking`

| Field | Meaning and use |
|---|---|
| `max_missed_updates` | Number of detector updates a track may survive without a matching detection. |
| `match_center_distance` | Maximum normalized center distance used to match a detection to an existing track when overlap is weak. |
| `min_iou` | Minimum intersection-over-union that can qualify a detection-track match. A match is allowed when IoU is sufficient or center distance is sufficiently small. |
| `box_alpha` | Weight assigned to the newest detected box when smoothing tracker boxes. Larger values follow the newest box more quickly. |

#### `temporal`

| Field | Meaning and use |
|---|---|
| `min_hold_seconds` | Minimum time the current selected focus should be retained before switching to another valid candidate. |
| `lost_hold_seconds` | Time to preserve the previous focus after its evidence temporarily disappears. |
| `switch_margin` | Required score advantage before another candidate replaces the current focus. |
| `deadband` | Normalized X difference ignored by the camera controller. This removes tiny left/right jitter. |
| `response_time_seconds` | Approximate time for the camera trajectory to respond to a new target. Larger values move more slowly. |
| `max_speed_normalized_per_second` | Hard limit on horizontal crop-center speed as a fraction of the source width per second. |
| `max_acceleration_normalized_per_second2` | Hard limit on how quickly horizontal crop-center speed may change. |

#### `local_shots`

| Field | Meaning and use |
|---|---|
| `histogram_threshold` | Bhattacharyya-distance threshold for declaring a local hard cut when Fabric shot tags are unavailable. Lower values create more cuts. |
| `minimum_shot_seconds` | Minimum time allowed between local cuts, preventing rapid duplicate cut detections. |

### Sports-only `selection` fields

| Field | Meaning and use |
|---|---|
| `min_track_hits` | Minimum matched detector updates required before a tracked candidate is trusted. |
| `min_person_area` | Minimum normalized person-box area accepted as a meaningful player/person candidate. |
| `direct_ball_min_area` | Minimum normalized ball area required before the ball itself may become the crop target. Tiny balls remain supporting evidence only. |
| `ball_player_radius` | Maximum normalized distance used to associate a ball with a nearby player. |
| `min_ball_player_score` | Minimum score required for a ball-associated player candidate. |
| `group_radius` | Maximum normalized separation for combining nearby players into one action group. |
| `max_group_size` | Maximum number of players included in the selected action group. |
| `announcer_face_min_area` | Minimum normalized face area considered large enough for an announcer/interview-style shot. |
| `min_motion_score` | Minimum motion evidence needed before motion can be used as a focus fallback. |
| `crowd_min_people` | Minimum number of detected people needed before crowd logic is considered. |
| `crowd_max_median_area` | Maximum median person area for treating many detections as a distant crowd rather than nearby players. |
| `crowd_max_motion_score` | Maximum motion allowed for a likely crowd/static audience classification. |
| `crowd_group_score` | Confidence assigned to the crowd-group fallback candidate. |

### Movie-only `selection` fields

| Field | Meaning and use |
|---|---|
| `min_track_hits` | Minimum matched detector updates required before a tracked face/person is trusted. |
| `min_face_area` | Minimum normalized area for a face to qualify as a meaningful movie-focus candidate. |
| `group_score_ratio` | A secondary face/person must score at least this fraction of the primary candidate to form an interaction group. |
| `group_crop_width_factor` | Multiplier used when deciding whether multiple characters fit meaningfully within the fixed 9:16 composition. It affects group selection, not zoom. |
| `min_motion_score` | Minimum motion evidence required before action/motion can be used as a fallback. |

### Output fields

| Field | Meaning and use |
|---|---|
| `vertical_track` | Fabric track name for frame-level X arrays. Default: `vertical_video`. |
| `focus_track` | Fabric track name for focus labels, IDs, confidence, and policy metadata. Default: `focus`. |
| `bbox_track` | Fabric track name for representative focus boxes. Default: `focus_bbox`. |
| `coordinate_decimals` | Number of decimal places retained for normalized X and bbox coordinates. |
| `max_boxes_per_shot` | Maximum representative focus boxes emitted for one shot. |

## Runtime parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `mode` | `movie` | Must be `sports` or `movie`. Aliases `policy` and `profile` are accepted and converted to `mode`. |
| `delegate` | `cpu` | MediaPipe delegate: `cpu` or `gpu`. |
| `config_path` | container policy path | Alternate policy YAML. |
| `object_model` | EfficientDet-Lite0 path | Alternate TFLite object detector. |
| `detection_fps` | policy value | Overrides the selected policy's detection frequency. |
| `shot_track` | `shot_detection` | Fabric track used for shot ranges. |
| `tagstore_url` | `https://ai.contentfabric.io` | Tagstore base URL. |
| `request_timeout_seconds` | `30` | Tagstore HTTP timeout. |
| `initial_content_offset_ms` | `0` | Global content-time offset for the first stdin fragment. |
| `progress_log_interval_seconds` | `2` | Interval between stderr progress log lines. |
| `vertical_track` | YAML value | Runtime output-track override. |
| `focus_track` | YAML value | Runtime output-track override. |
| `bbox_track` | YAML value | Runtime output-track override. |
| `max_boxes_per_shot` | YAML value | Runtime bbox-count override. |
| `policy_overrides` | none | Nested dictionary merged into the selected policy for controlled experiments. |

## Local validation

Unit tests:

```bash
make unit-test
```

Build:

```bash
make
```

Container tests use only the short clips and never the full-length files directly under `test-files`:

```bash
MODE=sports make test
MODE=movie make test
```

Expected test inputs:

```text
test-files/short/sports_01.mp4 ... sports_04.mp4
test-files/short/movie_01.mp4  ... movie_04.mp4
```

## Side-by-side preview audio

The visual renderer creates silent side-by-side previews. To copy audio from each original short clip into its matching preview:

```bash
./scripts/remux_preview_audio.sh
```

Audio previews are written to:

```text
test-output/visual/previews-audio/
```

This remux step does not alter the model output or production JSONL.

## Build and deployment

The repository uses the current `qluvio/buildscripts` tagger-model workflow.

```bash
git submodule update --init --recursive
make unit-test
make
MODE=sports make test
MODE=movie make test
make push-latest
```

`make push-latest` publishes:

```text
us-docker.pkg.dev/github-qluvio/ml/verticalvideo:latest
```

It does not change Fabric configuration by itself. The deployed Tagger must then be pointed at that image for protocol and JSON evaluation.
