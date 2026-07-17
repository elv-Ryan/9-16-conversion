# Eluvio 9:16 Vertical Focus

This repository is an Eluvio tagger container that analyzes video and emits the horizontal crop center required for a fixed-height 9:16 version.

It does not render the production video. It does not move vertically, zoom, stretch, or compress the image. It writes one normalized X center per decoded frame.

Supported policies:

- `sports`
- `movie`

## What changed from the first version

| First version | Current `ryan-v2`, policy v3 |
|---|---|
| Local AutoFlip reframing demo | Deployable Eluvio `common-ml` tagger container |
| Rendered a cropped video | Emits compact shot tags containing frame-level X centers |
| Primarily face-based behavior | Separate sports and movie policies |
| Custom stdin and JSON handling | Uses `common-ml` stdin, JSONL, progress, and error handling |
| No stable subject identity | Deterministic ball, person, and face tracking IDs |
| Weak shot reset behavior | Fabric shot tags with local cut fallback |
| Ball was mostly supporting evidence | Ball is the primary sports target when it is credible, with the nearest player used as context |
| Camera could drift after evidence disappeared | The crop holds its last valid composition and zeroes residual velocity |
| Large focus changes were slow | A bounded fast-catchup controller activates for large errors and priority switches |
| Legacy MediaPipe face API | MediaPipe Tasks FaceDetector with BlazeFace, plus an OpenCV fallback |
| Silent visual checks | Audio-bearing side-by-side and pure vertical previews |

The container and output contract remain unchanged. Policy v3 changes focus selection, tracking, smoothing, face detection, and visual QA only.

## Models used

### Object detector

The object detector is MediaPipe Tasks `ObjectDetector` using EfficientDet-Lite0:

```text
models/mp_tasks/object_detector/efficientdet_lite0.tflite
```

Only these object classes are retained:

- `person`
- `sports ball`

### Face detector

The primary face detector is MediaPipe Tasks `FaceDetector` using BlazeFace short range:

```text
models/mp_tasks/face_detector/blaze_face_short_range.tflite
```

If the MediaPipe Tasks face model cannot be created, the code falls back to OpenCV's bundled `haarcascade_frontalface_alt2.xml`. The fallback prevents a face-model problem from terminating the tagger, but BlazeFace is the intended production path.

### Motion

Motion is a lightweight frame-difference signal, not a neural model. Global flashes, cuts, broad pans, and full-frame changes are rejected. Motion is used only when the policy does not have a credible ball, face, or person candidate.

### Tracking

The tracker associates detections by label, bounding-box overlap, and normalized center distance. Ball tracks use a larger match radius, a faster box update, and a longer prediction horizon than person or face tracks.

## Output tracks

Each shot emits three logical tracks.

### `vertical_video`

Contains:

- shot start and end time in milliseconds
- first frame index in the shot
- `additional_info.x-coordinates`
- one normalized X center for every decoded frame in the shot

Coordinate meaning:

- `0.0`: far left
- `0.5`: image center
- `1.0`: far right

### `focus`

Summarizes the selected focus behavior for the shot:

- dominant focus label
- focus IDs
- labels encountered in the shot
- average confidence
- policy name
- policy version
- policy schema

### `focus_bbox`

Contains representative normalized bounding boxes for the selected focus identities. A sports ball focus normally includes the ball box first and a nearby player box second.

## Runtime protocol

The container:

- stays alive while stdin is open
- accepts newline-separated video paths through stdin
- requires `--output-path`
- accepts JSON parameters through `--params`
- writes JSONL tag, progress, and error messages
- writes all tags for one source before its terminal progress message
- uses `common-ml` for serialization and the stdin loop

Example:

```bash
printf '/elv/test/sports_01.mp4\n' | podman run --rm -i \
  -v "$PWD/test-files/short:/elv/test:ro" \
  -v "$PWD/test-output:/elv/tags" \
  verticalvideo:latest \
  --output-path /elv/tags/out.jsonl \
  --params '{"mode":"sports","delegate":"cpu"}'
```

## `configs/policies.yml`, field by field

### Top-level fields

| Field | How it is used |
|---|---|
| `version` | Policy revision written to `focus.additional_info.policy_version`. |
| `schema_version` | Policy schema name written to `focus.additional_info.policy_schema`. |
| `policies` | Contains the supported `sports` and `movie` configurations. |
| `output` | Controls output track names, coordinate precision, and bbox count. |

### Detection fields

These fields exist under each policy's `detection` section.

| Field | How it is used |
|---|---|
| `detection_fps` | Number of frames per second sent through the neural detectors. Frames between detector calls use tracker prediction. Sports uses a higher rate to follow a fast ball. |
| `person_min_score` | Minimum EfficientDet confidence accepted for a person. |
| `ball_min_score` | Minimum EfficientDet confidence accepted for a sports ball. |
| `face_min_score` | Minimum BlazeFace confidence accepted for a face. |
| `face_min_suppression` | Non-maximum-suppression threshold used by MediaPipe FaceDetector to merge overlapping face detections. |
| `face_min_size_fraction` | Minimum face size used by the OpenCV cascade fallback, expressed as a fraction of the shorter frame dimension. |
| `max_results` | Maximum object detections returned by EfficientDet for one detector frame. |

### Tracking fields

These fields exist under each policy's `tracking` section.

| Field | How it is used |
|---|---|
| `max_missed_updates` | Number of detector updates a track may survive without a match. |
| `match_center_distance` | Maximum normalized center distance for matching person and face detections when overlap is weak. |
| `ball_match_center_distance` | Larger normalized match distance allowed for fast-moving ball detections. |
| `min_iou` | Minimum intersection-over-union that can qualify a detection-track match. A match may also qualify by center distance. |
| `box_alpha` | Weight assigned to a new person or face box when blending with the existing track. |
| `ball_box_alpha` | Higher new-box weight for ball tracks so they respond faster and lag less. |
| `prediction_max_frames` | Maximum number of decoded frames used when extrapolating person or face motion between detections. |
| `ball_prediction_max_frames` | Longer extrapolation horizon for a temporarily missed ball. |
| `prediction_max_distance` | Maximum normalized distance any track may be extrapolated away from its last detected box. |

### Sports selection fields

| Field | How it is used |
|---|---|
| `min_track_hits` | Minimum matched detector updates before a person track is treated as stable. |
| `min_person_area` | Minimum normalized person-box area accepted as a player or person candidate. |
| `ball_min_track_hits` | Minimum detector hits before a ball may become the primary focus. A nearby player can make a one-hit ball credible. |
| `ball_max_misses` | Maximum missed detector updates for retaining a ball as focus evidence. |
| `ball_min_area` | Minimum normalized ball-box area. Rejects degenerate tiny boxes. |
| `ball_max_area` | Maximum normalized ball-box area. Rejects obviously incorrect large ball detections. |
| `ball_context_radius` | Maximum normalized distance from a person that makes a new ball detection credible. A stable ball track can remain valid without nearby context. |
| `ball_stable_hits_without_context` | Detector-hit count after which the ball can remain credible while airborne or temporarily away from a player. |
| `ball_focus_weight` | Horizontal composition weight assigned to the ball when combining ball and player centers. The remaining weight is assigned to the nearby player. |
| `ball_priority_score` | Base candidate score that makes a credible ball outrank normal player, group, crowd, and motion candidates. |
| `ball_player_radius` | Distance scale used to score which player is associated with the ball. |
| `min_ball_player_score` | Minimum association score for the player-near-ball fallback when the ball is not yet trusted as a primary track. |
| `group_radius` | Maximum horizontal distance from the action anchor for including players in an action group. |
| `max_group_size` | Maximum number of players included in an action or crowd group. |
| `announcer_face_min_area` | Minimum normalized face area for announcer or interview-style sports shots. |
| `min_motion_score` | Minimum motion confidence for using motion as a fallback target. |
| `motion_only_without_subjects` | When true, motion is considered only when no credible ball, person, or face exists. |
| `crowd_min_people` | Minimum number of people needed before crowd logic is considered. |
| `crowd_max_median_area` | Maximum median person area for classifying many detections as a distant crowd. |
| `crowd_max_motion_score` | Maximum motion allowed for a likely static crowd shot. |
| `crowd_group_score` | Candidate score assigned to the crowd-group fallback. |

### Movie selection fields

| Field | How it is used |
|---|---|
| `min_track_hits` | Minimum matched detector updates before a face or person track is stable. |
| `min_face_area` | Minimum normalized face area for a movie focus candidate. |
| `group_score_ratio` | A secondary face must score at least this fraction of the best face to form an interaction group. |
| `group_crop_width_factor` | Controls whether multiple important faces fit inside the fixed 9:16 composition. It does not change zoom. |
| `min_motion_score` | Minimum motion confidence for the action fallback. |
| `motion_only_without_subjects` | When true, motion cannot override a stable face or person. |

### Temporal and smoothing fields

These fields exist under each policy's `temporal` section.

| Field | How it is used |
|---|---|
| `min_hold_seconds` | Minimum normal hold time before switching from the current valid focus. |
| `lost_hold_seconds` | Time to preserve a focus identity through brief missed detections. |
| `switch_margin` | Score advantage required for a normal candidate switch. |
| `hold_last_on_empty` | When true, no-evidence intervals freeze at the last valid composition instead of drifting to center. |
| `fast_switch_labels` | Focus labels allowed to interrupt normal hold behavior. Sports includes `ball_focus`; movie includes `action_region`. |
| `fast_switch_min_hold_seconds` | Reduced hold time before switching to a fast-priority label. |
| `fast_switch_margin` | Score margin used for a fast-priority switch. A negative sports value allows a credible ball to interrupt a player quickly. |
| `deadband` | X error ignored by the camera controller. Velocity is immediately set to zero inside this band. |
| `response_time_seconds` | Normal response time used to convert target error into desired camera speed. |
| `max_speed_normalized_per_second` | Normal horizontal speed limit as a fraction of source width per second. |
| `max_acceleration_normalized_per_second2` | Normal acceleration limit. |
| `fast_response_time_seconds` | Faster response time used during a large error or focus switch. |
| `fast_max_speed_normalized_per_second` | Speed limit during fast catch-up. |
| `fast_max_acceleration_normalized_per_second2` | Acceleration limit during fast catch-up. |
| `fast_error_threshold` | X error large enough to activate fast catch-up automatically. |
| `fast_boost_seconds` | Time that fast catch-up remains active after it is triggered. |

### Local shot fields

| Field | How it is used |
|---|---|
| `histogram_threshold` | HSV histogram distance required to declare a local cut when Fabric shot tags are unavailable. |
| `minimum_shot_seconds` | Minimum interval between local cuts. |

### Output fields

| Field | How it is used |
|---|---|
| `vertical_track` | Track name for frame-level X arrays. |
| `focus_track` | Track name for focus labels, IDs, confidence, and policy metadata. |
| `bbox_track` | Track name for representative focus boxes. |
| `coordinate_decimals` | Decimal precision retained for normalized X and bbox coordinates. |
| `max_boxes_per_shot` | Maximum representative focus boxes emitted for one shot. |

## Runtime parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `mode` | `movie` | Must be `sports` or `movie`. |
| `delegate` | `cpu` | MediaPipe delegate: `cpu` or `gpu`. |
| `config_path` | policy path | Alternate policy YAML. |
| `object_model` | EfficientDet path | Alternate object detector model. |
| `face_model` | BlazeFace path | Alternate face detector model. |
| `detection_fps` | policy value | Runtime override for detector frequency. |
| `shot_track` | `shot_detection` | Fabric track used for shot ranges. |
| `tagstore_url` | `https://ai.contentfabric.io` | Tagstore base URL. |
| `request_timeout_seconds` | `30` | Tagstore request timeout. |
| `initial_content_offset_ms` | `0` | Content-time offset for the first stdin fragment. |
| `progress_log_interval_seconds` | `2` | Interval between stderr progress logs. |
| `vertical_track` | YAML value | Runtime vertical track override. |
| `focus_track` | YAML value | Runtime focus track override. |
| `bbox_track` | YAML value | Runtime bbox track override. |
| `max_boxes_per_shot` | YAML value | Runtime bbox-count override. |
| `policy_overrides` | none | Nested policy values merged for controlled experiments. |

## Fast local QA workflow

### 1. Ensure the BlazeFace model is present

```bash
./scripts/fetch_face_model.sh
```

### 2. Recreate the eight 90-second clips with audio

```bash
./scripts/remake_short_clips_with_audio.sh
```

This reads only selected 90-second ranges from the two full videos directly under `test-files`. It does not run focus analysis on the full videos.

### 3. Run unit tests

```bash
make unit-test
```

### 4. Generate JSONL and both preview formats

```bash
./scripts/run_visual_test.sh
```

Outputs:

```text
test-output/visual-v3/sports.jsonl
test-output/visual-v3/movie.jsonl
test-output/visual-v3/previews/*_side_by_side.mp4
test-output/visual-v3/previews/*_vertical.mp4
```

Both preview types include the source audio when the short clip has audio. The pure vertical preview contains no labels or overlays.

### 5. Build and test the container

```bash
make
MODE=sports make test
MODE=movie make test
```

The container tests use only `test-files/short`.

## Deployment

```bash
git add -A
git commit -m "Refine sports movie focus policy v3"
git push origin ryan-v2
make deploy
```

The deployment repository is:

```text
us-docker.pkg.dev/github-qluvio/ml/verticalvideo
```
