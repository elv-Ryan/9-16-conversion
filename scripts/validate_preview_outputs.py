#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MediaProbe:
    path: Path
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: Optional[int]
    audio_streams: int


def _positive_float(*values: Any) -> float:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            return parsed
    return 0.0


def _fps(value: Any) -> float:
    if value in (None, "", "0/0"):
        return 0.0
    try:
        parsed = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0.0 else 0.0


def _frame_count(stream: Dict[str, Any]) -> Optional[int]:
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value in (None, "", "N/A"):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def probe(path: Path) -> MediaProbe:
    if not path.is_file():
        raise RuntimeError(f"missing media: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(videos)}")
    video = videos[0]
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"invalid video dimensions in {path}: {width}x{height}")
    fps = _fps(video.get("avg_frame_rate")) or _fps(video.get("r_frame_rate"))
    if fps <= 0.0:
        raise RuntimeError(f"invalid frame rate in {path}")
    duration_s = _positive_float(video.get("duration"), (payload.get("format") or {}).get("duration"))
    if duration_s <= 0.0:
        frame_count = _frame_count(video)
        if frame_count is None:
            raise RuntimeError(f"missing duration and frame count in {path}")
        duration_s = frame_count / fps
    return MediaProbe(
        path=path,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        frame_count=_frame_count(video),
        audio_streams=sum(stream.get("codec_type") == "audio" for stream in streams),
    )


def even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def validate(source: MediaProbe, side: MediaProbe, vertical: MediaProbe) -> Dict[str, Any]:
    crop_width = min(source.width, even(int(round(source.height * 9.0 / 16.0))))
    expected_side = (even(source.width + crop_width), even(source.height))
    expected_vertical = (crop_width, even(source.height))
    failures = []

    if source.audio_streams < 1:
        failures.append("source has no audio stream; QA acceptance requires audio")
    if side.audio_streams < 1:
        failures.append("side-by-side output has no audio stream")
    if vertical.audio_streams < 1:
        failures.append("pure-vertical output has no audio stream")
    if (side.width, side.height) != expected_side:
        failures.append(
            f"side-by-side dimensions {(side.width, side.height)} != {expected_side}"
        )
    if (vertical.width, vertical.height) != expected_vertical:
        failures.append(
            f"vertical dimensions {(vertical.width, vertical.height)} != {expected_vertical}"
        )

    one_frame_s = 1.0 / source.fps
    duration_tolerance_s = one_frame_s + 0.005
    for label, output in (("side-by-side", side), ("vertical", vertical)):
        delta = abs(output.duration_s - source.duration_s)
        if delta > duration_tolerance_s:
            failures.append(
                f"{label} duration differs by {delta:.6f}s; tolerance={duration_tolerance_s:.6f}s"
            )
        if source.frame_count is not None and output.frame_count is not None:
            if output.frame_count != source.frame_count:
                failures.append(
                    f"{label} frame count {output.frame_count} != source {source.frame_count}"
                )

    summary = {
        "source": {
            "path": str(source.path),
            "dimensions": [source.width, source.height],
            "fps": source.fps,
            "duration_s": source.duration_s,
            "frame_count": source.frame_count,
            "audio_streams": source.audio_streams,
        },
        "side_by_side": {
            "path": str(side.path),
            "dimensions": [side.width, side.height],
            "duration_s": side.duration_s,
            "frame_count": side.frame_count,
            "audio_streams": side.audio_streams,
        },
        "vertical": {
            "path": str(vertical.path),
            "dimensions": [vertical.width, vertical.height],
            "duration_s": vertical.duration_s,
            "frame_count": vertical.frame_count,
            "audio_streams": vertical.audio_streams,
        },
        "duration_tolerance_s": duration_tolerance_s,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    if failures:
        raise RuntimeError(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--side-by-side", type=Path, required=True)
    parser.add_argument("--vertical", type=Path, required=True)
    args = parser.parse_args()
    summary = validate(probe(args.source), probe(args.side_by_side), probe(args.vertical))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
