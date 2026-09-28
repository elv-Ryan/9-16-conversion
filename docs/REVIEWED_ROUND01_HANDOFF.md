# Reviewed Round01 + One Euro handoff

This branch freezes the reviewed seven-family Round01 `schemafix_v3` YOLO26s checkpoint and its exact causal controller, followed by the selected One Euro filter (`min_cutoff=0.4`, `beta=1.0`). It performs no Qwen inference, shot detection, ball-specialist inference, or vertical-video rendering.

The production result is a `vertical_video` tag stream containing normalized horizontal crop centers, one per source frame. Tags are split into 240-frame chunks to match the supplied Spider-Verse renderer interchange convention.

## Important protocol difference from the supplied recovered JSONL

The supplied `spiderverse_v3_gold_vertical_video.commonml.jsonl` is a recovered/offline artifact: its `progress` record appears first and it omits `source_media`. Current Eluvio Tagger protocol requires `source_media` on tags and treats `progress` as a terminal "file processed" message. The production container therefore preserves the attached tag payload fields but adds required `source_media` and emits terminal progress. `scripts/export_spiderverse_format.py` creates the literal attached/offline shape when needed by a renderer or regression test.

## Frozen runtime

- model SHA-256: `0921e4843864557bc1e5141f655a461af24c758d7390f2ea0553091df743865a`
- `imgsz=1280`
- FP32
- evidence cadence: 10 Hz
- detector confidence: 0.001
- IoU: 0.7
- max detections: 100; retained candidates: 20
- One Euro: 0.4 / 1.0
- crop X output: source-frame aligned, normalized to source width

Round06 remains research and is not imported into this release branch.
