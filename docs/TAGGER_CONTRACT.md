# Vertical focus tagger contract

## Input

The OCI container remains alive and accepts newline-separated, container-local video paths over stdin. It accepts:

- `--output-path /path/out.jsonl` — required by `common-ml`
- `--params '{...}'` — optional runtime JSON

## Required output ordering

For each input source:

1. Emit all `tag` messages attributable to that source.
2. Emit one terminal `progress` message.
3. Flush every JSONL line as it is written.

An `error` message replaces terminal progress for a failed source. The process continues with the next source.

## Vertical trajectory

```json
{
  "type": "tag",
  "data": {
    "tag": "primary_face",
    "start_time": 0,
    "end_time": 4004,
    "source_media": "/elv/test/shot.mp4",
    "track": "vertical_video",
    "frame_info": {"frame_idx": 0},
    "additional_info": {
      "x-coordinates": [0.5, 0.5, 0.4982]
    }
  }
}
```

`x-coordinates` contains one finite normalized center X value for every decoded frame represented by the tag. No Y value, scale, zoom, or output-video path is emitted.

## Focus label

```json
{
  "type": "tag",
  "data": {
    "tag": "player_near_ball",
    "start_time": 0,
    "end_time": 4004,
    "source_media": "/elv/test/shot.mp4",
    "track": "focus",
    "frame_info": {"frame_idx": 0},
    "additional_info": {
      "focus_ids": ["person:4", "ball:2"],
      "labels": ["player_near_ball"],
      "confidence": 0.87,
      "policy": "sports",
      "policy_version": 2,
      "policy_schema": "vertical_focus_policy_v2"
    }
  }
}
```

## Focus box

Each selected focus identity may produce one representative box on `focus_bbox`:

```json
{
  "type": "tag",
  "data": {
    "tag": "person",
    "start_time": 0,
    "end_time": 4004,
    "source_media": "/elv/test/shot.mp4",
    "track": "focus_bbox",
    "frame_info": {
      "frame_idx": 12,
      "box": {"x1": 0.31, "y1": 0.18, "x2": 0.46, "y2": 0.92}
    },
    "additional_info": {
      "focus_id": "person:4",
      "confidence": 0.91
    }
  }
}
```

All times are integer milliseconds. All coordinates are normalized to `[0, 1]`. `frame_idx` is relative to `source_media`.
