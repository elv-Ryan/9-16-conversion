#!/usr/bin/env python3
import argparse, json, math, subprocess
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

def clamp(x, lo, hi): return max(lo, min(hi, x))

def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    radius = int(max(1, round(3.0 * sigma)))
    kx = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(kx * kx) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    pad = radius
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    y = np.convolve(xp, kernel, mode="valid")
    return y

def vel_clamp_forward(x: np.ndarray, max_dx: float) -> np.ndarray:
    y = x.copy().astype(np.float32)
    for i in range(1, len(y)):
        y[i] = clamp(y[i], y[i-1] - max_dx, y[i-1] + max_dx)
    return y

def vel_clamp_fb(x: np.ndarray, max_dx: float) -> np.ndarray:
    y = vel_clamp_forward(x, max_dx)
    y = vel_clamp_forward(y[::-1], max_dx)[::-1]
    y = vel_clamp_forward(y, max_dx)
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_video", required=True)
    ap.add_argument("--shots_json", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--model", default="models/mp_tasks/object_detector/efficientdet_lite0_int8_1.tflite")
    ap.add_argument("--crop_w", type=int, default=406)
    ap.add_argument("--crop_h", type=int, default=720)
    ap.add_argument("--score_person", type=float, default=0.35)
    ap.add_argument("--score_ball", type=float, default=0.25)
    ap.add_argument("--ball_boost", type=float, default=10.0)
    ap.add_argument("--sigma_frames", type=float, default=24.0)  # heavy smoothing
    ap.add_argument("--max_dx", type=float, default=3.0)         # strong velocity clamp
    args = ap.parse_args()

    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)

    shots = json.load(open(args.shots_json))["shots"]
    shot_edges = [(float(s["start_sec"]), float(s["end_sec"])) for s in shots]

    cap = cv2.VideoCapture(args.in_video)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {args.in_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.crop_h != H:
        raise SystemExit(f"ERROR: expects crop_h==input_h. crop_h={args.crop_h} input_h={H}")
    if args.crop_w > W:
        raise SystemExit(f"ERROR: crop_w {args.crop_w} > input_w {W}")

    # Mediapipe detector
    BaseOptions = mp_python.BaseOptions
    ObjectDetector = vision.ObjectDetector
    ObjectDetectorOptions = vision.ObjectDetectorOptions
    RunningMode = vision.RunningMode
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        max_results=50,
        score_threshold=min(args.score_ball, args.score_person),
        running_mode=RunningMode.VIDEO,
    )
    detector = ObjectDetector.create_from_options(options)

    # Pass A: compute raw target_x per frame
    raw_x = []
    raw_reason = []
    raw_shot = []
    raw_ball_present = []

    shot_idx = 0
    last_ball_cx = None
    last_ball_frame = -10_000

    frame_idx = 0
    total_frames = 0
    ball_det_frames = 0
    ball_inside_raw = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        t_sec = frame_idx / fps
        while shot_idx < len(shot_edges) and t_sec >= shot_edges[shot_idx][1] - 1e-6:
            shot_idx += 1
            last_ball_cx = None
            last_ball_frame = -10_000

        if shot_idx >= len(shot_edges):
            shot_idx = len(shot_edges) - 1

        ts_ms = int(round(t_sec * 1000.0))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = detector.detect_for_video(mp_image, ts_ms)

        best_ball = None
        persons = []

        for d in res.detections:
            if not d.categories:
                continue
            cat = d.categories[0]
            name = (getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").lower()
            score = float(cat.score)
            bb = d.bounding_box
            area = float(bb.width * bb.height)
            cx = float(bb.origin_x + bb.width * 0.5)

            if name == "sports ball" and score >= args.score_ball:
                cand = (score, area, cx, bb)
                if (best_ball is None) or (score * area) > (best_ball[0] * best_ball[1]):
                    best_ball = cand

            elif name == "person" and score >= args.score_person:
                persons.append((score, area, cx, bb))

        reason = "hold"
        target_cx = None
        ball_present = False

        if best_ball is not None:
            ball_present = True
            ball_det_frames += 1
            _, _, cx, bb = best_ball
            last_ball_cx = cx
            last_ball_frame = frame_idx
            target_cx = cx
            reason = "ball"

            # raw ball-inside metric
            x0 = clamp(int(round(cx - args.crop_w / 2)), 0, W - args.crop_w)
            bx0 = bb.origin_x
            bx1 = bb.origin_x + bb.width
            if (bx0 >= x0) and (bx1 <= x0 + args.crop_w):
                ball_inside_raw += 1

        elif persons:
            # If we recently saw ball in this shot, lock onto nearest person to last_ball_cx
            if last_ball_cx is not None and (frame_idx - last_ball_frame) <= int(round(1.0 * fps)):
                best = min(persons, key=lambda p: abs(p[2] - last_ball_cx))
                target_cx = best[2]
                reason = "person_near_ball"
            else:
                # else pick largest person
                best = max(persons, key=lambda p: p[1] * p[0])
                target_cx = best[2]
                reason = "person_big"

        if target_cx is None:
            target_cx = W / 2.0
            reason = "center"

        tx = int(round(target_cx - args.crop_w / 2))
        tx = clamp(tx, 0, W - args.crop_w)

        raw_x.append(tx)
        raw_reason.append(reason)
        raw_shot.append(shot_idx)
        raw_ball_present.append(1 if ball_present else 0)

        frame_idx += 1
        total_frames += 1

    cap.release()
    detector.close()

    raw_x = np.asarray(raw_x, dtype=np.float32)
    raw_shot = np.asarray(raw_shot, dtype=np.int32)
    raw_ball_present = np.asarray(raw_ball_present, dtype=np.int32)

    # Pass B: smooth per shot (gaussian + forward/backward velocity clamp)
    smooth_x = raw_x.copy()
    for sid in range(raw_shot.min(), raw_shot.max() + 1):
        idx = np.where(raw_shot == sid)[0]
        if len(idx) <= 2:
            continue
        xs = raw_x[idx]
        xs = gaussian_smooth(xs, sigma=args.sigma_frames)
        xs = vel_clamp_fb(xs, max_dx=args.max_dx)
        xs = np.clip(xs, 0, W - args.crop_w)
        smooth_x[idx] = xs

    # Jitter proxy
    raw_dx = np.abs(np.diff(raw_x)).mean() if len(raw_x) > 1 else 0.0
    sm_dx = np.abs(np.diff(smooth_x)).mean() if len(smooth_x) > 1 else 0.0

    # Render via ffmpeg pipe
    ff = subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{args.crop_w}x{H}",
            "-r", f"{fps}",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            args.out_video,
        ],
        stdin=subprocess.PIPE,
    )
    assert ff.stdin is not None

    cap2 = cv2.VideoCapture(args.in_video)
    i = 0
    while True:
        ok, frame_bgr = cap2.read()
        if not ok or i >= len(smooth_x):
            break
        x0 = int(smooth_x[i])
        crop = frame_bgr[:, x0:x0 + args.crop_w]
        ff.stdin.write(crop.tobytes())
        i += 1
    cap2.release()
    ff.stdin.close()
    ff.wait()

    # Metrics
    det_frames = total_frames
    print(f"Wrote: {args.out_video}")
    print(f"input={W}x{H} fps={fps:.3f} frames={total_frames} shots={len(shots)}")
    print(f"ball_det_frames={ball_det_frames} ball_inside_raw_pct={100.0*ball_inside_raw/max(1,ball_det_frames):.1f}%")
    print(f"mean_abs_dx_raw={raw_dx:.3f} mean_abs_dx_smooth={sm_dx:.3f}")

    # Shot-level ball presence (for policy)
    for sid in range(raw_shot.min(), raw_shot.max() + 1):
        idx = np.where(raw_shot == sid)[0]
        if len(idx) == 0:
            continue
        bp = raw_ball_present[idx].mean()
        print(f"shot_{sid:02d}_ball_presence_pct={100.0*bp:.1f}%")

if __name__ == "__main__":
    main()
