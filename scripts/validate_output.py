#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

try:
    import cv2
except ImportError:
    cv2 = None


EXPECTED_TRACKS = {"vertical_video", "focus", "focus_bbox"}


def fail(message: str) -> None:
    raise SystemExit(f"OUTPUT VALIDATION FAILED: {message}")


def media_frame_count(source: str) -> int | None:
    if cv2 is None or not Path(source).is_file():
        return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return count if count > 0 else None


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: validate_output.py OUT.jsonl [EXPECTED_INPUTS]")
    path = Path(sys.argv[1])
    expected_inputs = int(sys.argv[2]) if len(sys.argv) == 3 else None
    if not path.is_file():
        fail(f"missing {path}")

    terminal = Counter()
    terminal_kind = {}
    terminal_seen = set()
    tracks = Counter()
    tracks_by_source = defaultdict(Counter)
    x_counts = Counter()
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
            if not isinstance(data, dict):
                fail(f"line {line_number} data must be an object")
            if message_type in {"progress", "error"}:
                source = str(data.get("source_media", ""))
                if not source:
                    fail(f"line {line_number} terminal message has no source_media")
                terminal[source] += 1
                terminal_kind[source] = message_type
                terminal_seen.add(source)
                continue
            if message_type != "tag":
                fail(f"line {line_number} has unsupported type {message_type!r}")
            required = {
                "tag", "start_time", "end_time", "source_media", "track",
                "frame_info", "additional_info",
            }
            missing = required - set(data)
            if missing:
                fail(f"line {line_number} missing {sorted(missing)}")
            source = str(data["source_media"])
            if source in terminal_seen:
                fail(f"line {line_number} emits a tag after terminal message for {source}")
            if not isinstance(data["start_time"], int) or not isinstance(data["end_time"], int):
                fail(f"line {line_number} time values must be integers")
            if data["end_time"] <= data["start_time"]:
                fail(f"line {line_number} end_time must exceed start_time")
            track = str(data["track"])
            if track not in EXPECTED_TRACKS:
                fail(f"line {line_number} has unexpected track {track!r}")
            tracks[track] += 1
            tracks_by_source[source][track] += 1
            frame_info = data["frame_info"]
            additional_info = data["additional_info"]
            if not isinstance(frame_info, dict) or not isinstance(additional_info, dict):
                fail(f"line {line_number} frame_info/additional_info must be objects")
            frame_idx = frame_info.get("frame_idx")
            if not isinstance(frame_idx, int) or frame_idx < 0:
                fail(f"line {line_number} has invalid frame_idx")
            if track == "vertical_video":
                xs = additional_info.get("x-coordinates")
                if not isinstance(xs, list) or not xs:
                    fail(f"line {line_number} has no x-coordinates")
                if any(
                    not isinstance(x, (int, float))
                    or not math.isfinite(float(x))
                    or not 0.0 <= float(x) <= 1.0
                    for x in xs
                ):
                    fail(f"line {line_number} has invalid normalized X")
                x_counts[source] += len(xs)
            if track == "focus_bbox":
                box = frame_info.get("box")
                if not isinstance(box, dict) or set(box) != {"x1", "y1", "x2", "y2"}:
                    fail(f"line {line_number} has invalid focus box")
                if not (
                    0 <= box["x1"] < box["x2"] <= 1
                    and 0 <= box["y1"] < box["y2"] <= 1
                ):
                    fail(f"line {line_number} has out-of-range focus box")

    if lines == 0:
        fail("output is empty")
    if expected_inputs is not None and len(terminal) != expected_inputs:
        fail(f"expected {expected_inputs} terminal sources, found {len(terminal)}")
    repeated = {source: count for source, count in terminal.items() if count != 1}
    if repeated:
        fail(f"each source must have exactly one terminal message: {repeated}")
    for source, kind in terminal_kind.items():
        if kind == "progress" and set(tracks_by_source[source]) != EXPECTED_TRACKS:
            fail(
                f"successful source {source} must contain exactly {sorted(EXPECTED_TRACKS)}, "
                f"found {sorted(tracks_by_source[source])}"
            )
        expected_frames = media_frame_count(source)
        if kind == "progress" and expected_frames is not None and x_counts[source] != expected_frames:
            fail(
                f"source {source} has {x_counts[source]} X values for {expected_frames} frames"
            )
    missing_tracks = EXPECTED_TRACKS - set(tracks)
    successful = any(kind == "progress" for kind in terminal_kind.values())
    if missing_tracks and successful:
        fail(f"missing tracks: {sorted(missing_tracks)}")

    print(
        json.dumps(
            {
                "lines": lines,
                "terminal_sources": len(terminal),
                "tracks": dict(tracks),
                "x_counts": dict(x_counts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
