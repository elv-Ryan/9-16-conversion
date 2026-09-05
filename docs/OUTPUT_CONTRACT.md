# NBA shot-to-X output contract

The container accepts newline-delimited media paths on stdin. In the default
`shot_file` mode, each path is already one shot; no shot detection runs. In
`segment_file` mode the paths are consecutive slices of one continuous stream
and TransNetV2 detects the shot boundaries, so a shot may span several files
and one file may contain several shots.

The production tag is written on the `vertical_video` track. Its `tag` field is
the detected runtime family, and `additional_info.x-coordinates` contains one
legal normalized horizontal crop center for every source frame in the shot.

```json
{
  "type": "tag",
  "data": {
    "tag": "gameplay_follow",
    "track": "vertical_video",
    "start_time": 0,
    "end_time": 4000,
    "source_media": "/elv/input/shot.mp4",
    "frame_info": {"frame_idx": 0},
    "additional_info": {
      "schema_version": "eluvio.nba-yolo-shot-x.v1",
      "shot_id": "shot_000000",
      "family": "gameplay_follow",
      "category": "gameplay",
      "category_level": "runtime_policy_family",
      "family_confidence": 0.82,
      "source_fps": 59.94,
      "source_frame_start": 0,
      "frame_count": 240,
      "x_coordinate_alignment": "source_frame",
      "x_coordinate_units": "normalized_source_width",
      "crop_width_norm": 0.316406,
      "legal_x_center_min": 0.158203,
      "legal_x_center_max": 0.841797,
      "x-coordinates": [0.5, 0.501, 0.503],
      "focus_sample_fps": 10.0
    }
  }
}
```

Semantics:

- `0.0` is the source-frame left edge.
- `0.5` is source-frame center.
- `1.0` is the source-frame right edge.
- Every emitted value is clamped to the legal center range for a full-height
  9:16 crop.
- `x-coordinates[i]` corresponds to source frame
  `source_frame_start + i`.
- `start_time` and `end_time` are milliseconds relative to this exact
  `source_media` input. There is no cross-file cumulative offset.
- `frame_info.frame_idx` is `source_frame_start`, present on every tag.
- A shot is emitted against the file being processed when it is cut. In
  `segment_file` mode a shot that began in an earlier file therefore carries a
  negative `start_time`/`source_frame_start`/`frame_idx`, measured back from
  frame 0 of that file. `shot_file` mode always starts at 0 because each file
  is exactly one shot.
- `end_time` can also be negative, for the same reason. Shot detection will
  not commit a cut until it has ~25 frames of real video after it, so a cut in
  the last ~0.4s of a file is reported while the next file is being processed,
  and the shot it closes lies entirely before that file. `end_time` is always
  greater than `start_time`, and `frame_count` is always the shot's true
  length, so resolving a shot is the same arithmetic either way.
- Negative offsets are by design, not a degenerate case. Every offset a tag
  carries is relative to the `source_media` it names, and the caller knows
  where each shot/segment it fed in sits in the wider video, so it rebases
  these onto absolute positions before storing the JSONL. The tagger's job is
  only to keep offsets consistent with the file it names; it never needs to
  know the absolute position itself.
- In `segment_file` mode the trajectory is committed as segments arrive, a
  `trajectory_commit_lag_frames` margin behind the decoded frames, instead of
  in one pass at the end of the shot. Committed values are final. Because the
  smoothing passes are bidirectional, a shot long enough to be committed in
  several batches gets slightly less right-context than a single whole-shot
  pass would (measured: ~1-2px mean, ~60px worst of 1920, at the default
  margin). `shot_file` mode is unaffected: a shot there is always finished in
  one pass.
- `additional_info.focus_sample_fps` is always emitted. The per-frame
  `additional_info.focus_samples` array (one entry per sampled focus-model
  frame, at `focus_sample_fps`) is large and is only emitted when the runtime
  parameter `include_focus_samples` is `true`; it defaults to `false` to keep
  tag output small. Sampling density is unaffected by this flag — only
  whether the samples are included in the tag.

The current model has a seven-family runtime head. It does not yet expose the
fine eleven-way Category-Gold subtype taxonomy as a separately trained head.

## Compact local X JSON

The Fabric-facing output remains JSONL. For local integration tests only:

```bash
python3 scripts/extract_x_json.py out.jsonl -o x.json
```

This writes `eluvio.nba-yolo-x-json.v1` with each shot's family/category and
`x_center_norm` array. The extractor is not part of the production protocol loop.

## Wire-message strictness

Production/default output uses documented Eluvio wire types only:

```text
tag
progress
error
```

`progress_ratio` is an opt-in diagnostic extension exposed by the pinned
`common-ml`; it is disabled by default and disabled for strict qualification.
