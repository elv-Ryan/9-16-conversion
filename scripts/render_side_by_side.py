#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def load_trajectory(jsonl_path: Path, source: Path) -> Tuple[Dict[int, float], Dict[int, str]]:
    trajectories: Dict[int, float] = {}
    labels: Dict[int, str] = {}
    source_text = str(source)
    source_name = source.name

    vertical_rows = []
    focus_by_start: Dict[int, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            message = json.loads(raw)
            if message.get("type") != "tag":
                continue
            data = message.get("data", {})
            row_source = str(data.get("source_media", ""))
            if row_source != source_text and Path(row_source).name != source_name:
                continue
            frame_info = data.get("frame_info") or {}
            start = int(frame_info.get("frame_idx", 0))
            if data.get("track") == "focus":
                focus_by_start[start] = str(data.get("tag", "focus"))
            elif data.get("track") == "vertical_video":
                xs = (data.get("additional_info") or {}).get("x-coordinates") or []
                vertical_rows.append((start, [float(x) for x in xs]))

    for start, xs in sorted(vertical_rows):
        label = focus_by_start.get(start, "focus")
        for offset, x in enumerate(xs):
            frame_idx = start + offset
            trajectories[frame_idx] = min(1.0, max(0.0, x))
            labels[frame_idx] = label

    if not trajectories:
        raise RuntimeError(f"No vertical_video trajectory found for {source} in {jsonl_path}")
    return trajectories, labels


def even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    xs, labels = load_trajectory(args.jsonl, args.input)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.input}")

    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    crop_width = min(width, even(int(round(height * 9.0 / 16.0))))
    output_width = even(width + crop_width)
    output_height = even(height)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}",
        "-r", f"{fps:.8f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None

    frame_idx = 0
    last_x = 0.5
    last_label = "safe_center"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x = xs.get(frame_idx, last_x)
            label = labels.get(frame_idx, last_label)
            last_x = x
            last_label = label

            left = int(round(x * width - crop_width / 2.0))
            left = max(0, min(width - crop_width, left))
            right = left + crop_width

            annotated = frame.copy()
            cv2.rectangle(annotated, (left, 0), (right - 1, height - 1), (0, 255, 255), 4)
            cv2.line(annotated, (int(round(x * width)), 0), (int(round(x * width)), height - 1), (0, 255, 255), 2)
            cv2.putText(annotated, "ORIGINAL + 9:16 WINDOW", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"focus={label}  x={x:.3f}", (24, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

            cropped = frame[:, left:right]
            cv2.putText(cropped, "VERTICAL OUTPUT", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            canvas[:height, :width] = annotated
            canvas[:height, width:width + crop_width] = cropped
            encoder.stdin.write(canvas.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        encoder.stdin.close()
        return_code = encoder.wait()

    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {return_code}")
    print(f"Rendered {frame_idx} frames: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
