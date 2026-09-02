#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    parser.add_argument("--expected-source")
    args = parser.parse_args()
    path = Path(args.jsonl)
    if not path.is_file():
        raise SystemExit(f"missing JSONL output: {path}")

    messages: List[Dict] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"line {line_number}: invalid JSON: {error}")
        if message.get("type") not in {"tag", "progress", "progress_ratio", "error"}:
            raise SystemExit(f"line {line_number}: invalid message type: {message.get('type')}")
        if not isinstance(message.get("data"), dict):
            raise SystemExit(f"line {line_number}: data must be an object")
        messages.append(message)

    vertical = [
        message for message in messages
        if message["type"] == "tag" and message["data"].get("track") == "vertical_video"
    ]
    if not vertical:
        errors = [m for m in messages if m["type"] == "error"]
        raise SystemExit(f"no vertical_video tag found; errors={errors}")

    for message in vertical:
        data = message["data"]
        if args.expected_source and data.get("source_media") != args.expected_source:
            raise SystemExit(
                f"source_media mismatch: {data.get('source_media')!r} != {args.expected_source!r}"
            )
        info = data.get("additional_info") or {}
        values = info.get("x-coordinates")
        if not isinstance(values, list) or not values:
            raise SystemExit("vertical_video tag has no non-empty x-coordinates list")
        if len(values) != info.get("frame_count"):
            raise SystemExit(
                f"x-coordinate/frame count mismatch: {len(values)} != {info.get('frame_count')}"
            )
        lower = float(info.get("legal_x_center_min"))
        upper = float(info.get("legal_x_center_max"))
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SystemExit(f"x-coordinates[{index}] is not a finite number: {value!r}")
            if value < lower - 1e-9 or value > upper + 1e-9:
                raise SystemExit(
                    f"x-coordinates[{index}]={value} is outside [{lower}, {upper}]"
                )

    progress_sources = {
        message["data"].get("source_media")
        for message in messages if message["type"] == "progress"
    }
    for message in vertical:
        source = message["data"].get("source_media")
        if source not in progress_sources:
            raise SystemExit(f"missing terminal progress for {source}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "messages": len(messages),
                "vertical_tags": len(vertical),
                "progress_sources": sorted(source for source in progress_sources if source),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
