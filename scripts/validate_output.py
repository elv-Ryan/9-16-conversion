#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
import sys


def fail(message: str) -> None:
    raise SystemExit(f"OUTPUT VALIDATION FAILED: {message}")


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: validate_output.py OUT.jsonl [EXPECTED_INPUTS]")
    path = Path(sys.argv[1])
    expected_inputs = int(sys.argv[2]) if len(sys.argv) == 3 else None
    if not path.is_file():
        fail(f"missing {path}")

    terminal = Counter()
    tracks = Counter()
    lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            lines += 1
            try:
                message = json.loads(raw)
            except json.JSONDecodeError as exc:
                fail(f"line {line_number} is not JSON: {exc}")
            if set(message) != {"type", "data"}:
                fail(f"line {line_number} must contain type and data")
            message_type = message["type"]
            data = message["data"]
            if message_type in {"progress", "error"}:
                source = data.get("source_media")
                if source:
                    terminal[source] += 1
                continue
            if message_type != "tag":
                fail(f"line {line_number} has unsupported type {message_type!r}")
            required = {"tag", "start_time", "end_time", "source_media", "track", "frame_info", "additional_info"}
            missing = required - set(data)
            if missing:
                fail(f"line {line_number} missing {sorted(missing)}")
            if not isinstance(data["start_time"], int) or not isinstance(data["end_time"], int):
                fail(f"line {line_number} time values must be integers")
            if data["end_time"] <= data["start_time"]:
                fail(f"line {line_number} end_time must exceed start_time")
            tracks[data["track"]] += 1
            frame_idx = data["frame_info"].get("frame_idx")
            if not isinstance(frame_idx, int) or frame_idx < 0:
                fail(f"line {line_number} has invalid frame_idx")
            if data["track"] == "vertical_video":
                xs = data["additional_info"].get("x-coordinates")
                if not isinstance(xs, list) or not xs:
                    fail(f"line {line_number} has no x-coordinates")
                if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in xs):
                    fail(f"line {line_number} has invalid normalized X")
            if data["track"] == "focus_bbox":
                box = data["frame_info"].get("box")
                if not isinstance(box, dict) or set(box) != {"x1", "y1", "x2", "y2"}:
                    fail(f"line {line_number} has invalid focus box")
                if not (0 <= box["x1"] < box["x2"] <= 1 and 0 <= box["y1"] < box["y2"] <= 1):
                    fail(f"line {line_number} has out-of-range focus box")

    if lines == 0:
        fail("output is empty")
    if expected_inputs is not None and len(terminal) != expected_inputs:
        fail(f"expected {expected_inputs} terminal progress/error sources, found {len(terminal)}")
    missing_tracks = {"vertical_video", "focus", "focus_bbox"} - set(tracks)
    if missing_tracks and not any(terminal.values()):
        fail(f"missing tracks: {sorted(missing_tracks)}")
    print(json.dumps({"lines": lines, "terminal_sources": len(terminal), "tracks": tracks}, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
