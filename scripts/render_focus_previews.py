#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameDebug:
    label: str
    state: str
    evidence: str
    regime: str
    selected_boxes: Tuple[Tuple[str, Tuple[float, float, float, float]], ...]


def _same_source(row_source: str, source: Path) -> bool:
    return row_source == str(source) or Path(row_source).name == source.name


def load_trajectory(jsonl_path: Path, source: Path) -> Tuple[Dict[int, float], Dict[int, str]]:
    trajectories: Dict[int, float] = {}
    labels: Dict[int, str] = {}
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
            if not _same_source(str(data.get("source_media", "")), source):
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


def load_debug(debug_path: Optional[Path], source: Path) -> Dict[int, FrameDebug]:
    if debug_path is None or not debug_path.is_file():
        return {}
    result: Dict[int, FrameDebug] = {}
    with debug_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not _same_source(str(row.get("source", "")), source):
                continue
            selected = row.get("selected") or {}
            boxes = []
            # V5 sidecars serialize selected boxes directly so predicted or held
            # evidence remains visible even when its numeric track has expired.
            for selected_box in selected.get("boxes") or []:
                track_id = str(selected_box.get("id", ""))
                box = tuple(float(value) for value in selected_box.get("box") or [])
                if track_id and len(box) == 4:
                    boxes.append((track_id, box))
            if not boxes:
                # Backward-compatible fallback for older sidecars.
                selected_ids = set(
                    str(item) for item in selected.get("focus_ids") or []
                )
                for track in row.get("tracks") or []:
                    track_id = str(track.get("id", ""))
                    if track_id not in selected_ids:
                        continue
                    box = tuple(float(value) for value in track.get("box") or [])
                    if len(box) == 4:
                        boxes.append((track_id, box))
            result[int(row["frame"])] = FrameDebug(
                label=str(selected.get("label", "focus")),
                state=str(selected.get("scene_state", "unknown")),
                evidence=str(selected.get("evidence_kind", "unknown")),
                regime=str(selected.get("smoothing_regime", "unknown")),
                selected_boxes=tuple(boxes),
            )
    return result


def even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def open_encoder(
    *, output: Path, width: int, height: int, fps: float, source: Path
) -> subprocess.Popen:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-thread_queue_size", "512", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "pipe:0",
        "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe was not created")
    return process


def draw_debug_boxes(frame: np.ndarray, debug: FrameDebug) -> None:
    height, width = frame.shape[:2]
    for track_id, box in debug.selected_boxes:
        x1, y1, x2, y2 = box
        p1 = (int(round(x1 * width)), int(round(y1 * height)))
        p2 = (int(round(x2 * width)), int(round(y2 * height)))
        cv2.rectangle(frame, p1, p2, (255, 255, 0), 2)
        cv2.putText(
            frame,
            track_id,
            (p1[0], max(18, p1[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--debug-jsonl", type=Path)
    parser.add_argument("--side-by-side-output", required=True, type=Path)
    parser.add_argument("--vertical-output", required=True, type=Path)
    args = parser.parse_args()

    xs, shot_labels = load_trajectory(args.jsonl, args.input)
    debug_by_frame = load_debug(args.debug_jsonl, args.input)
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

    side_encoder = open_encoder(
        output=args.side_by_side_output,
        width=output_width,
        height=output_height,
        fps=fps,
        source=args.input,
    )
    vertical_encoder = open_encoder(
        output=args.vertical_output,
        width=crop_width,
        height=output_height,
        fps=fps,
        source=args.input,
    )
    assert side_encoder.stdin is not None
    assert vertical_encoder.stdin is not None

    frame_idx = 0
    last_x = 0.5
    last_shot_label = "safe_center"
    last_debug: Optional[FrameDebug] = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            x = xs.get(frame_idx, last_x)
            shot_label = shot_labels.get(frame_idx, last_shot_label)
            debug = debug_by_frame.get(frame_idx, last_debug)
            last_x = x
            last_shot_label = shot_label
            last_debug = debug

            left = int(round(x * width - crop_width / 2.0))
            left = max(0, min(width - crop_width, left))
            right = left + crop_width

            annotated = frame.copy()
            cv2.rectangle(annotated, (left, 0), (right - 1, height - 1), (0, 255, 255), 4)
            cv2.line(
                annotated,
                (int(round(x * width)), 0),
                (int(round(x * width)), height - 1),
                (0, 255, 255),
                2,
            )
            if debug is not None:
                draw_debug_boxes(annotated, debug)
                label = debug.label
                detail = f"state={debug.state}  evidence={debug.evidence}  regime={debug.regime}"
                basis = "FRAME-LEVEL QA"
            else:
                label = shot_label
                detail = "debug sidecar unavailable; label is shot-dominant"
                basis = "SHOT-DOMINANT QA"

            cv2.putText(
                annotated,
                f"ORIGINAL + 9:16 WINDOW | {basis}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"focus={label}  x={x:.3f}",
                (24, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                detail,
                (24, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            vertical = frame[:, left:right].copy()
            side_vertical = vertical.copy()
            cv2.putText(
                side_vertical,
                "VERTICAL OUTPUT",
                (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            canvas[:height, :width] = annotated
            canvas[:height, width : width + crop_width] = side_vertical
            side_encoder.stdin.write(canvas.tobytes())
            vertical_encoder.stdin.write(vertical.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        side_encoder.stdin.close()
        vertical_encoder.stdin.close()
        side_code = side_encoder.wait()
        vertical_code = vertical_encoder.wait()

    if side_code != 0 or vertical_code != 0:
        raise RuntimeError(
            f"ffmpeg failed: side_by_side={side_code}, vertical={vertical_code}"
        )
    print(
        f"Rendered {frame_idx} frames: {args.side_by_side_output} and {args.vertical_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
