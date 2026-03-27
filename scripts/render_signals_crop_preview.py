#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

def clamp(x, lo, hi): return max(lo, min(hi, x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_video", required=True)
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--model", default="models/mp_tasks/object_detector/efficientdet_lite0_int8_1.tflite")
    ap.add_argument("--crop_w", type=int, default=406)
    ap.add_argument("--crop_h", type=int, default=720)
    ap.add_argument("--stride", type=int, default=2, help="run detector every N frames; hold last target in between")
    ap.add_argument("--score_person", type=float, default=0.35)
    ap.add_argument("--score_ball", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=0.85, help="EMA smoothing for crop_x (higher=more smoothing)")
    args = ap.parse_args()

    in_path = args.in_video
    out_path = args.out_video
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Mediapipe object detector
    BaseOptions = mp_python.BaseOptions
    ObjectDetector = vision.ObjectDetector
    ObjectDetectorOptions = vision.ObjectDetectorOptions
    RunningMode = vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        max_results=20,
        score_threshold=min(args.score_ball, args.score_person),
        running_mode=RunningMode.VIDEO,
    )
    detector = ObjectDetector.create_from_options(options)

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.crop_h != H:
        raise SystemExit(f"ERROR: this preview expects full-height crop_h==input_h. Got crop_h={args.crop_h} input_h={H}")

    crop_w = args.crop_w
    if crop_w > W:
        raise SystemExit(f"ERROR: crop_w {crop_w} > input_w {W}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (crop_w, H))

    last_target_cx = None
    x_smooth = None

    frames = 0
    det_frames = 0
    ball_dets = 0
    ball_inside = 0
    jerk_sum = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Run detector every stride frames
        if frames % args.stride == 0:
            det_frames += 1
            ts_ms = int(round((frames / fps) * 1000.0))

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = detector.detect_for_video(mp_image, ts_ms)

            best_ball = None
            best_person = None

            for d in res.detections:
                if not d.categories:
                    continue
                cat = d.categories[0]
                name = (getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").lower()
                score = float(cat.score)
                bb = d.bounding_box

                if name == "sports ball" and score >= args.score_ball:
                    area = bb.width * bb.height
                    cand = (score, area, bb)
                    if (best_ball is None) or (cand > best_ball):
                        best_ball = cand

                if name == "person" and score >= args.score_person:
                    area = bb.width * bb.height
                    cand = (score, area, bb)
                    if (best_person is None) or (cand > best_person):
                        best_person = cand

            # Priority: ball if present, else biggest/best person, else keep last
            if best_ball is not None:
                ball_dets += 1
                bb = best_ball[2]
                last_target_cx = bb.origin_x + bb.width * 0.5

                # ball-inside check (on detection frames)
                crop_x_raw = clamp(int(round(last_target_cx - crop_w / 2)), 0, W - crop_w)
                bx0 = bb.origin_x
                bx1 = bb.origin_x + bb.width
                if (bx0 >= crop_x_raw) and (bx1 <= crop_x_raw + crop_w):
                    ball_inside += 1

            elif best_person is not None:
                bb = best_person[2]
                last_target_cx = bb.origin_x + bb.width * 0.5

        # Compute crop x
        if last_target_cx is None:
            target_x = int(round(W/2 - crop_w/2))
        else:
            target_x = int(round(last_target_cx - crop_w/2))
        target_x = clamp(target_x, 0, W - crop_w)

        # Smooth
        if x_smooth is None:
            x_smooth = target_x
        else:
            x_prev = x_smooth
            x_smooth = int(round(args.alpha * x_smooth + (1.0 - args.alpha) * target_x))
            jerk_sum += abs(x_smooth - x_prev)

        crop = frame_bgr[:, x_smooth:x_smooth+crop_w]
        writer.write(crop)
        frames += 1

    cap.release()
    writer.release()
    detector.close()

    det_frames = max(1, det_frames)
    print(f"Wrote: {out_path}")
    print(f"input={W}x{H} fps={fps:.3f} frames={frames} det_stride={args.stride} det_frames={det_frames}")
    print(f"ball_dets={ball_dets} ball_inside_on_det_frames={ball_inside} ball_inside_pct={100.0*ball_inside/det_frames:.1f}%")
    print(f"jerk_sum_px={jerk_sum:.1f} mean_abs_dx_per_frame={jerk_sum/max(1,frames-1):.3f}")
