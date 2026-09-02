# NBA shot-to-X output contract

The container accepts newline-delimited media paths on stdin. In the default
`shot_file` mode, each path is already one shot; no shot detection runs.

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
      "x-coordinates": [0.5, 0.501, 0.503]
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
