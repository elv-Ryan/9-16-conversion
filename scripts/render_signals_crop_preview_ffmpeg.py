#!/usr/bin/env python3
import argparse, subprocess
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
    ap.add_argument("--ball_boost", type=float, default=6.0, help="ball weight multiplier in centroid fusion")
    ap.add_argument("--alpha", type=float, default=0.92, help="EMA smoothing for target_x")
    ap.add_argument("--max_dx", type=int, default=6, help="max pan pixels per frame (velocity clamp)")
    args = ap.parse_args()

    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.in_video)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: cannot open {args.in_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.crop_h != H:
        raise SystemExit(f"ERROR: preview expects crop_h==input_h. crop_h={args.crop_h} input_h={H}")
    if args.crop_w > W:
        raise SystemExit(f"ERROR: crop_w {args.crop_w} > input_w {W}")

    # MediaPipe object detector (VIDEO mode)
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

    # ffmpeg encoder via stdin (bgr24 raw frames)
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

    last_target_cx = None
    x_smooth = None

    frames = 0
    det_frames = 0
    ball_det_frames = 0
    ball_inside = 0
    jerk_sum = 0.0

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            best_ball_bb = None

            if frames % args.stride == 0:
                det_frames += 1
                ts_ms = int(round((frames / fps) * 1000.0))

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                res = detector.detect_for_video(mp_image, ts_ms)

                # centroid fusion
                sum_w = 0.0
                sum_cx = 0.0

                for d in res.detections:
                    if not d.categories:
                        continue
                    cat = d.categories[0]
                    name = (getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").lower()
                    score = float(cat.score)
                    bb = d.bounding_box
                    area = float(bb.width * bb.height)
                    cx = float(bb.origin_x + bb.width * 0.5)

                    if name == "person" and score >= args.score_person:
                        w = score * area
                        sum_w += w
                        sum_cx += w * cx

                    elif name == "sports ball" and score >= args.score_ball:
                        w = args.ball_boost * score * area
                        sum_w += w
                        sum_cx += w * cx
                        # keep best ball for metric
                        if best_ball_bb is None or (score * area) > (best_ball_bb[0] * best_ball_bb[1]):
                            best_ball_bb = (score, area, bb)

                if sum_w > 0.0:
                    last_target_cx = sum_cx / sum_w

                if best_ball_bb is not None:
                    ball_det_frames += 1
                    bb = best_ball_bb[2]
                    crop_x_raw = clamp(int(round(last_target_cx - args.crop_w / 2)), 0, W - args.crop_w)
                    bx0 = bb.origin_x
                    bx1 = bb.origin_x + bb.width
                    if (bx0 >= crop_x_raw) and (bx1 <= crop_x_raw + args.crop_w):
                        ball_inside += 1

            # compute target_x
            if last_target_cx is None:
                target_x = int(round(W/2 - args.crop_w/2))
            else:
                target_x = int(round(last_target_cx - args.crop_w/2))
            target_x = clamp(target_x, 0, W - args.crop_w)

            # smooth + velocity clamp
            if x_smooth is None:
                x_smooth = target_x
            else:
                desired = int(round(args.alpha * x_smooth + (1.0 - args.alpha) * target_x))
                dx = desired - x_smooth
                dx = clamp(dx, -args.max_dx, args.max_dx)
                x_smooth += dx
                jerk_sum += abs(dx)

            crop = frame_bgr[:, x_smooth:x_smooth + args.crop_w]
            ff.stdin.write(crop.tobytes())
            frames += 1

    finally:
        cap.release()
        detector.close()
        ff.stdin.close()
        ff.wait()

    det_frames = max(1, det_frames)
    print(f"Wrote: {args.out_video}")
    print(f"input={W}x{H} fps={fps:.3f} frames={frames} det_stride={args.stride} det_frames={det_frames}")
    print(f"ball_det_frames={ball_det_frames} ball_inside_on_ball_frames={ball_inside} ball_inside_pct={100.0*ball_inside/max(1,ball_det_frames):.1f}%")
    print(f"jerk_sum_px={jerk_sum:.1f} mean_abs_dx_per_frame={jerk_sum/max(1,frames-1):.3f}")

if __name__ == "__main__":
    main()
