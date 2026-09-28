# NBA reviewed YOLO X-coordinate tagger

Eluvio/common-ml deployment branch for the completed reviewed NBA baseline.

```text
already-segmented shot
  -> reviewed Round01 seven-family YOLO26s
  -> reviewed causal controller
  -> One Euro (0.4 / 1.0)
  -> one X per source frame
  -> 240-frame vertical_video JSONL chunks
```

No vertical video is reconstructed in this container.

## Repository / branch

Repository: `elv-Ryan/9-16-conversion`

This release branch is based on `nba-yolo-shot-tagger-v3.3` and replaces its older Round00 artifact/runtime with the reviewed Round01 + One Euro handoff.

## Build

```bash
./scripts/fetch_current_model.sh
make unit-test
make build
```

The committed model artifact is also verified by SHA-256 during image build.

## Protocol

The container remains alive, reads newline-delimited input paths on stdin, writes JSONL to `--output-path`, and accepts `--params`. Production output is strict Eluvio/common-ml: `tag`, terminal `progress`, and `error` messages. The `vertical_video` tag payload uses the same `frame_info.frame_idx`, `additional_info["x-coordinates"]`, `source_iq`, `fps`, and `provenance` structure as the supplied Spider-Verse reference. `source_media` is additionally retained because current Eluvio protocol requires it.

To produce a byte-structure-compatible offline renderer file matching the supplied recovered reference (progress first, no `source_media`):

```bash
python scripts/export_spiderverse_format.py out.jsonl -o renderer.jsonl \
  --source-iq iq__... --title "..."
python scripts/validate_spiderverse_jsonl.py renderer.jsonl
```

See `docs/REVIEWED_ROUND01_HANDOFF.md`.
