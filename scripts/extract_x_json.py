#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract compact normalized-X JSON from Eluvio tagger JSONL."
    )
    parser.add_argument("jsonl")
    parser.add_argument("--output", "-o", default="-")
    args = parser.parse_args()

    shots = []
    for line in Path(args.jsonl).read_text().splitlines():
        if not line.strip():
            continue
        message = json.loads(line)
        data = message.get("data") or {}
        if message.get("type") != "tag" or data.get("track") != "vertical_video":
            continue
        info = data.get("additional_info") or {}
        shots.append(
            {
                "source_media": data.get("source_media"),
                "shot_id": info.get("shot_id"),
                "start_time_ms": data.get("start_time"),
                "end_time_ms": data.get("end_time"),
                "family": info.get("family", data.get("tag")),
                "category": info.get("category"),
                "source_fps": info.get("source_fps"),
                "source_frame_start": info.get("source_frame_start"),
                "x_center_norm": info.get("x-coordinates"),
            }
        )
    if not shots:
        raise SystemExit("No vertical_video tags found")
    payload = {
        "schema_version": "eluvio.nba-yolo-x-json.v1",
        "shots": shots,
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output == "-":
        print(encoded, end="")
    else:
        Path(args.output).write_text(encoded)
        print(args.output)


if __name__ == "__main__":
    main()
