#!/usr/bin/env python3

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

import sys
import os
import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from dataclasses import dataclass
from typing import List, Optional
from dacite import from_dict
import setproctitle
import requests

@dataclass
class RuntimeConfig:
    model: str = "/elv/models/mp_tasks/object_detector/efficientdet_lite0.tflite"
    score_person: float = 0.35
    score_ball: float = 0.25
    sigma_frames: float = 24.0
    max_dx: float = 3
    action_hold_sec: float = 0.75
    mode: str = "movie"
    output_video: bool = False
    output_overlay: bool = False



def get_shot_detection_tags(iq, token):
    url = f"https://ai.contentfabric.io/tagstore/{iq}/tags"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    
    params = {
        "track": "shot_detection"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)        
        response.raise_for_status()        
        return response.json()    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tags: {e}")
        return None
    
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    radius = int(max(1, round(3.0 * sigma)))
    kx = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(kx * kx) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    xp = np.pad(x.astype(np.float32), (radius, radius), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def vel_clamp_forward(x: np.ndarray, max_dx: float) -> np.ndarray:
    y = x.copy().astype(np.float32)
    for i in range(1, len(y)):
        y[i] = clamp(y[i], y[i - 1] - max_dx, y[i - 1] + max_dx)
    return y


def vel_clamp_fb(x: np.ndarray, max_dx: float) -> np.ndarray:
    y = vel_clamp_forward(x, max_dx)
    y = vel_clamp_forward(y[::-1], max_dx)[::-1]
    y = vel_clamp_forward(y, max_dx)
    return y


def motion_centroid(prev_gray, gray):
    if prev_gray is None:
        return None
    diff = cv2.absdiff(gray, prev_gray)
    _, th = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    m = cv2.moments(th)
    if m["m00"] <= 0:
        return None
    return float(m["m10"] / m["m00"])


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def bbox_from_objdet(bb, W, H):
    x0 = int(round(bb.origin_x))
    y0 = int(round(bb.origin_y))
    x1 = int(round(bb.origin_x + bb.width))
    y1 = int(round(bb.origin_y + bb.height))
    x0 = clamp(x0, 0, W - 1)
    y0 = clamp(y0, 0, H - 1)
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    return [x0, y0, x1, y1]


def bbox_center_x(bb):
    return 0.5 * (bb[0] + bb[2])


def bbox_area(bb):
    return max(0, bb[2] - bb[0]) * max(0, bb[3] - bb[1])


def focus_color_bgr(reason):
    colors = {
        "ball": (0, 255, 0),
        "person_near_ball": (0, 255, 255),
        "person_big": (255, 255, 0),
        "action": (0, 165, 255),
        "center": (255, 255, 255),
    }
    return colors.get(reason, (200, 200, 200))


def main():

    params = get_params()

    params = from_dict(RuntimeConfig, data=params)
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_video", required=False)
    ap.add_argument("--out_video", default="/dev/null")
    ap.add_argument("--overlay_video", default=None)
    
    ap.add_argument("--mode", default=None)
    ap.add_argument("--params", default=None)

    ap.add_argument("--model", default=None)
    ap.add_argument("--score_person", type=float, default=None)
    ap.add_argument("--score_ball", type=float, default=None)
    ap.add_argument("--sigma_frames", type=float, default=None)
    ap.add_argument("--max_dx", type=float, default=None)
    ap.add_argument("--action_hold_sec", type=float, default=None)
    
    ap.add_argument("--output-path", default="out.jsonl")
    args = ap.parse_args()

    output_file = open(args.output_path, "w")

    ### xxx add unhandled exception handler
    
    out_video = Path(args.out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    if args.overlay_video is None:
        args.overlay_video = str(out_video.with_suffix("")) + ".overlay.mp4"

    shots_doc = get_shot_detection_tags(os.environ["ELV_CONTENT"], os.environ["ELV_TOKEN"])
    shots = shots_doc["tags"]
    print(f"loaded {len(shots)} tags from tagstore")
    
    shot_edges = [(float(s["start_time"]) / 1000 , float(s["end_time"] / 1000)) for s in shots]
  
    for i, shot in enumerate(shots):
      shot["shot_id"] = i
      
    base_options = mp_python.BaseOptions(
        model_asset_path=params.model,
        delegate=mp_python.BaseOptions.Delegate.GPU
    )

    ObjectDetector = vision.ObjectDetector
    ObjectDetectorOptions = vision.ObjectDetectorOptions
    RunningMode = vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=base_options,
        max_results=50,
        score_threshold=min(params.score_ball, params.score_person),
        running_mode=RunningMode.VIDEO,
    )
    detector = ObjectDetector.create_from_options(options)

    raw_x = []
    raw_reason = []
    raw_shot = []
    raw_ball_present = []
    raw_target_cx = []
    raw_t_sec = []
    raw_focus_bbox = []

    shot_idx = 0
    last_ball_cx = None
    last_ball_frame = -10_000

    prev_gray = None
    last_action_cx = None
    last_action_frame = -10_000

    frame_idx = 0
    total_frames = 0
    ball_det_frames = 0
    ball_inside_raw = 0

    fps = None

    input_files = []
    
    for input_filename in sys.stdin:        
        input_filename = input_filename.strip()
        print("reading " + input_filename)
        if input_filename == "": continue
        
        cap = cv2.VideoCapture(input_filename)
        if not cap.isOpened():
            raise SystemExit(f"ERROR: cannot open {input_filename}")

        input_files.append(input_filename)
        
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None:
                raise Exception("could not determine framerate")

        if fps != cap.get(cv2.CAP_PROP_FPS):
            raise Exception("Variable FPS, panic")
        
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        crop_h = H
        r = round(crop_h * .5625000)
        if r % 2 != 0: r = r + 1
        crop_w = r
        
        if crop_h != H:
            raise SystemExit(f"ERROR: expects crop_h==input_h. crop_h={crop_h} input_h={H}")
        if crop_w > W:
            raise SystemExit(f"ERROR: crop_w {crop_w} > input_w {W}")

        if crop_w % 2 != 0:
            raise SystemExit(f"ERROR: crop_w must be even for libx264. Got crop_w={crop_w}")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
    
            t_sec = frame_idx / fps
            while shot_idx < len(shot_edges) and t_sec >= shot_edges[shot_idx][1] - 1e-6:
                shot_idx += 1
                last_ball_cx = None
                last_ball_frame = -10_000
                last_action_cx = None
                last_action_frame = -10_000
    
            if shot_idx >= len(shot_edges):
                shot_idx = len(shot_edges) - 1
    
            ts_ms = int(round(t_sec * 1000.0))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
            frame_resized = frame_rgb ##cv2.resize(frame_rgb, (320, 320))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_resized)
    
            res = detector.detect_for_video(mp_image, ts_ms)
            best_ball = None
            persons = []
    
            for d in res.detections:
                if not d.categories:
                    continue
                cat = d.categories[0]
                name = (getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").lower()
                score = float(cat.score)
                bb = bbox_from_objdet(d.bounding_box, W, H)
                area = float(bbox_area(bb))
                cx = float(bbox_center_x(bb))
    
                if name == "sports ball" and score >= params.score_ball:
                    cand = (score, area, cx, bb)
                    if (best_ball is None) or (score * area) > (best_ball[0] * best_ball[1]):
                        best_ball = cand
                elif name == "person" and score >= params.score_person:
                    persons.append((score, area, cx, bb))
    
            motion_cx = motion_centroid(prev_gray, frame_gray)
            prev_gray = frame_gray
            if motion_cx is not None:
                last_action_cx = motion_cx
                last_action_frame = frame_idx
    
            reason = "hold"
            target_cx = None
            ball_present = False
            focus_bbox = None
    
            if params.mode == "sports":
                if best_ball is not None:
                    ball_present = True
                    ball_det_frames += 1
                    _, _, cx, bb = best_ball
                    last_ball_cx = cx
                    last_ball_frame = frame_idx
                    target_cx = cx
                    focus_bbox = bb
                    reason = "ball"
    
                    x0 = clamp(int(round(cx - crop_w / 2)), 0, W - crop_w)
                    bx0 = bb[0]
                    bx1 = bb[2]
                    if (bx0 >= x0) and (bx1 <= x0 + crop_w):
                        ball_inside_raw += 1
    
                elif persons:
                    if last_ball_cx is not None and (frame_idx - last_ball_frame) <= int(round(1.0 * fps)):
                        best = min(persons, key=lambda p: abs(p[2] - last_ball_cx))
                        target_cx = best[2]
                        focus_bbox = best[3]
                        reason = "person_near_ball"
                    else:
                        best = max(persons, key=lambda p: p[1] * p[0])
                        target_cx = best[2]
                        focus_bbox = best[3]
                        reason = "person_big"
    
                elif last_action_cx is not None and (frame_idx - last_action_frame) <= int(round(params.action_hold_sec * fps)):
                    target_cx = last_action_cx
                    reason = "action"
    
            elif params.mode == "movie":
                if persons:
                    best = max(persons, key=lambda p: p[1] * p[0])
                    target_cx = best[2]
                    focus_bbox = best[3]
                    reason = "person_big"
                elif last_action_cx is not None and (frame_idx - last_action_frame) <= int(round(params.action_hold_sec * fps)):
                    target_cx = last_action_cx
                    reason = "action"
    
            if target_cx is None:
                target_cx = W / 2.0
                reason = "center"
    
            tx = int(round(target_cx - crop_w / 2))
            tx = clamp(tx, 0, W - crop_w)
    
            raw_x.append(tx)
            raw_reason.append(reason)
            raw_shot.append(shot_idx)
            raw_ball_present.append(1 if ball_present else 0)
            raw_target_cx.append(float(target_cx))
            raw_t_sec.append(float(t_sec))
            raw_focus_bbox.append(focus_bbox)

            frame_idx += 1
            total_frames += 1

        cap.release()

        ## this is a lie but we want to report some progress
        print(json.dumps({"type": "progress", "data" : { "source_media": input_filename }}) + "\n", file=output_file)

    detector.close()

    raw_x = np.asarray(raw_x, dtype=np.float32)
    raw_shot = np.asarray(raw_shot, dtype=np.int32)
    raw_ball_present = np.asarray(raw_ball_present, dtype=np.int32)

    smooth_x = raw_x.copy()
    for sid in range(raw_shot.min(), raw_shot.max() + 1):
        idx = np.where(raw_shot == sid)[0]
        if len(idx) <= 2:
            continue
        xs = raw_x[idx]
        xs = gaussian_smooth(xs, sigma=params.sigma_frames)
        xs = vel_clamp_fb(xs, max_dx=params.max_dx)
        xs = np.clip(xs, 0, W - crop_w)
        smooth_x[idx] = xs

    raw_dx = np.abs(np.diff(raw_x)).mean() if len(raw_x) > 1 else 0.0
    sm_dx = np.abs(np.diff(smooth_x)).mean() if len(smooth_x) > 1 else 0.0

    if False: ## render video
      ff = subprocess.Popen(
          [
              "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              "-f", "rawvideo",
              "-pix_fmt", "bgr24",
              "-s", f"{crop_w}x{H}",
              "-r", f"{fps}",
              "-i", "-",
              "-c:v", "libx264",
              "-pix_fmt", "yuv420p",
              args.out_video,
          ],
          stdin=subprocess.PIPE,
      )
      assert ff.stdin is not None

      cap2 = cv2.VideoCapture("TODO NEED ALL VIDEOS, SHOULD KEEP")
      i = 0
      while True:
          ok, frame_bgr = cap2.read()
          if not ok or i >= len(smooth_x):
              break
          x0 = int(round(float(smooth_x[i])))
          crop = frame_bgr[:, x0:x0 + crop_w]
          ff.stdin.write(crop.tobytes())
          i += 1
      cap2.release()
      ff.stdin.close()
      ff.wait()

    if False:
      ff2 = subprocess.Popen(
          [
              "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              "-f", "rawvideo",
              "-pix_fmt", "bgr24",
              "-s", f"{W}x{H}",
              "-r", f"{fps}",
              "-i", "-",
              "-c:v", "libx264",
              "-pix_fmt", "yuv420p",
              args.overlay_video,
          ],
          stdin=subprocess.PIPE,
      )
      assert ff2.stdin is not None

      cap3 = cv2.VideoCapture("SHOULD KEEP TRACK OF INPUT FILES")
      i = 0
      while True:
          ok, frame_bgr = cap3.read()
          if not ok or i >= len(smooth_x):
              break

          reason = raw_reason[i]
          color = focus_color_bgr(reason)
          x0 = int(round(float(smooth_x[i])))
          x1 = min(W - 1, x0 + crop_w - 1)

          cv2.rectangle(frame_bgr, (x0, 0), (x1, H - 1), (180, 180, 180), 2)

          bb = raw_focus_bbox[i]
          if bb is not None:
              cv2.rectangle(frame_bgr, (bb[0], bb[1]), (bb[2], bb[3]), color, 3)

          label = f"{reason}  frame={i}  t={raw_t_sec[i]:.3f}s  shot={int(raw_shot[i])}"
          cv2.rectangle(frame_bgr, (12, 12), (min(W - 12, 820), 56), (0, 0, 0), -1)
          cv2.putText(frame_bgr, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

          ff2.stdin.write(frame_bgr.tobytes())
          i += 1

      cap3.release()
      ff2.stdin.close()
      ff2.wait()

    shot_reason_counts = defaultdict(Counter)
    shot_reason_order = defaultdict(list)
    shot_focus_rows = defaultdict(list)
    x_rows = []

    for i, sid in enumerate(raw_shot.tolist()):
        reason = raw_reason[i]
        shot_reason_counts[int(sid)][reason] += 1
        if not shot_reason_order[int(sid)] or shot_reason_order[int(sid)][-1] != reason:
            shot_reason_order[int(sid)].append(reason)
    
    for s in shots:
        sid = int(s["shot_id"])
        idx = np.where(raw_shot == sid)[0]
        if len(idx) == 0:
            continue

        start_ms = s["start_time"]
        end_ms = s["end_time"]
        start_frame_idx = int(idx[0])

        x_coords = []

        focus_labels = []
        for j in idx.tolist():
            center_x_norm = float((smooth_x[j] + (crop_w / 2.0)) / float(W))
            center_x_norm = max(0.0, min(1.0, center_x_norm))
            x_coords.append(round(center_x_norm, 6))
            focus_labels.append(raw_reason[j])

        #reasons = shot_reason_order.get(sid, [])
        #tag_name = params.mode + "__" + "__".join(reasons) if reasons else params.mode + "__none"

        reason_counts = shot_reason_counts.get(sid, Counter())
        tag_name = reason_counts.most_common(1)[0][0]
        
        record = {
            "type": "tag",
            "data": {
                "tag": tag_name,
                "start_time": start_ms,
                "end_time": end_ms,
                "track": "vertical_video",
                "frame_info": {
                    "frame_idx": start_frame_idx
                },
                "additional_info": {
                    "x-coordinates": x_coords,
                    "reasons": dict(reason_counts)
                    ##"focus-labels": focus_labels                
                },
                "source_media": input_files[0]
            }
        }
        print(json.dumps(record), file = output_file)
        
        

        last_video_tag = None
        for j in idx.tolist():
            start_time = start_ms + round(1000 * float((j - start_frame_idx)) / fps)
            end_time = start_ms + round(1000 * float((1 + j - start_frame_idx)) / fps)
            if last_video_tag is not None:
                if raw_reason[j] != last_video_tag["data"]["tag"]:
                    print(json.dumps(last_video_tag), file = output_file)
                    last_video_tag = None
                else:
                    last_video_tag["data"]["end_time"] = end_time
                    
            if last_video_tag is None:
                last_video_tag = {
                    "type": "tag",
                    "data": {
                        "tag": raw_reason[j],
                        "start_time": start_time,
                        "end_time": end_time,
                        "track": "focus",
                        "source_media": input_files[0]
                    },
                }
                                        
            ## this is not fps, but still better than every frame
            if j % 4 != 0: continue

            bb = raw_focus_bbox[j]
            if bb is None:
                bbox_norm = None
            else:
                x0, y0, x1, y1 = bb
                x0 = round(float(x0) / float(W), 6)
                y0 = round(float(y0) / float(H), 6)
                x1 = round(float(x1) / float(W), 6)
                y1 = round(float(y1) / float(H), 6)
                
                record = {
                    "type": "tag",
                    "data": {
                        "tag": raw_reason[j],
                        "start_time": start_time,
                        "end_time": end_time,
                        "track": "focus",
                        "frame_info": {
                            "frame_idx": j,
                            "box": {
                                "x0": x0,
                                "y0": y0,
                                "x1": x1,
                                "y1": y1,
                            }
                        }
                    },
                    "source_media": input_files[0]
                }
                print(json.dumps(record), file = output_file)

        if last_video_tag is not None:
            print(json.dumps(last_video_tag), file = output_file)
            last_video_tag = None

    print(f"?Wrote: {args.out_video}")
    print(f"?Wrote: {args.overlay_video}")
    print(f"Wrote: {args.output_path}")
    print(f"input={W}x{H} fps={fps:.3f} frames={total_frames} shots={len(shots)} mode={params.mode}")
    print(f"ball_det_frames={ball_det_frames} ball_inside_raw_pct={100.0 * ball_inside_raw / max(1, ball_det_frames):.1f}%")
    print(f"mean_abs_dx_raw={raw_dx:.3f} mean_abs_dx_smooth={sm_dx:.3f}")

    for sid in range(raw_shot.min(), raw_shot.max() + 1):
        idx = np.where(raw_shot == sid)[0]
        if len(idx) == 0:
            continue
        bp = raw_ball_present[idx].mean()
        print(f"shot_{sid:02d}_ball_presence_pct={100.0 * bp:.1f}% focus={shot_reason_counts[sid].most_common(1)[0][0]}")


if __name__ == "__main__":
    main()
