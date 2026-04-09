#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

SIGNALS = {}

def register(name):
    def deco(fn):
        SIGNALS[name] = fn
        return fn
    return deco

def _bbox_norm_from_abs(origin_x, origin_y, width, height, img_w, img_h):
    x0 = max(0.0, float(origin_x) / float(img_w))
    y0 = max(0.0, float(origin_y) / float(img_h))
    x1 = min(1.0, float(origin_x + width) / float(img_w))
    y1 = min(1.0, float(origin_y + height) / float(img_h))
    return [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)]

@register("noop")
def sig_noop(img_path, **kwargs):
    return []

def _mp_object_detector(model_path: str, score_threshold: float, max_results: int):
    # Lazy import: allows noop to run without mediapipe installed.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    BaseOptions = mp_python.BaseOptions
    ObjectDetector = vision.ObjectDetector
    ObjectDetectorOptions = vision.ObjectDetectorOptions
    RunningMode = vision.RunningMode

    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=max_results,
        score_threshold=score_threshold,
        running_mode=RunningMode.IMAGE,
    )
    detector = ObjectDetector.create_from_options(options)

    def detect(img_path: str):
        mp_image = mp.Image.create_from_file(img_path)
        res = detector.detect(mp_image)
        return mp_image.width, mp_image.height, res.detections

    return detector, detect

def _objdet_regions(img_path: str, detect_fn, model_name: str, allow_names: set):
    w, h, dets = detect_fn(img_path)
    regs = []
    for d in dets:
        if not d.categories:
            continue
        cat = d.categories[0]
        name = (getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").lower()
        if name not in allow_names:
            continue
        bb = d.bounding_box
        regs.append({
            "bbox_norm": _bbox_norm_from_abs(bb.origin_x, bb.origin_y, bb.width, bb.height, w, h),
            "score": float(cat.score),
            "tags": {"class_name": name, "model": model_name},
        })
    return regs

@register("person")
def sig_person(img_path, **kwargs):
    return _objdet_regions(
        img_path,
        kwargs["detect_fn"],
        kwargs["model_name"],
        allow_names={"person"},
    )

@register("ball")
def sig_ball(img_path, **kwargs):
    # COCO-style label is typically "sports ball"
    return _objdet_regions(
        img_path,
        kwargs["detect_fn"],
        kwargs["model_name"],
        allow_names={"sports ball"},
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--shots_json", required=True)
    ap.add_argument("--frames_index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--signals", default="noop", help="comma-separated: noop,person,ball")
    ap.add_argument("--model", default="models/mp_tasks/object_detector/efficientdet_lite0_int8_1.tflite")
    ap.add_argument("--score_threshold", type=float, default=0.2)
    ap.add_argument("--max_results", type=int, default=10)
    args = ap.parse_args()

    frames = json.load(open(args.frames_index))["frames"]
    sig_names = [s.strip() for s in args.signals.split(",") if s.strip()]

    out = {
        "version": "signals.v1",
        "video": args.video,
        "shots_json": args.shots_json,
        "frames_index_json": args.frames_index,
        "signals": {},
    }

    needs_objdet = any(s in ("person", "ball") for s in sig_names)
    detector = None
    detect_fn = None
    model_name = "none"

    if needs_objdet:
        if not os.path.exists(args.model):
            raise SystemExit("Missing model: {}".format(args.model))
        detector, detect_fn = _mp_object_detector(args.model, args.score_threshold, args.max_results)
        model_name = Path(args.model).name

    try:
        for sig in sig_names:
            if sig not in SIGNALS:
                raise SystemExit("Unknown signal: {}. Known: {}".format(sig, sorted(SIGNALS)))
            out["signals"][sig] = {"model_id": model_name if sig in ("person","ball") else "none", "regions": []}

        for fr in frames:
            shot_id = int(fr["shot_id"])
            t_sec = float(fr["t_sec"])
            img_path = fr["path"]
            if not os.path.exists(img_path):
                continue

            for sig in sig_names:
                if sig in ("person", "ball"):
                    regs = SIGNALS[sig](img_path, detect_fn=detect_fn, model_name=model_name)
                else:
                    regs = SIGNALS[sig](img_path)

                for r in regs:
                    out["signals"][sig]["regions"].append({
                        "t_sec": t_sec,
                        "shot_id": shot_id,
                        "signal": sig,
                        "score": float(r.get("score", 0.0)),
                        "bbox_norm": r.get("bbox_norm", [0, 0, 0, 0]),
                        "tags": r.get("tags", {}),
                    })
    finally:
        if detector is not None:
            detector.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Wrote: {}".format(args.out))
    for sig in sig_names:
        print("{}: regions={}".format(sig, len(out["signals"][sig]["regions"])))

if __name__ == "__main__":
    main()
