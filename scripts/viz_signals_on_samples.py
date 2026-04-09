#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COLORS = {
  "person": (0, 255, 0),   # green
  "ball": (255, 0, 0),     # red
  "text": (0, 200, 255),   # cyan (future)
}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_index", required=True)
    ap.add_argument("--signals_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_score", type=float, default=0.25)
    ap.add_argument("--signals", default="person,ball")
    args = ap.parse_args()

    frames = json.load(open(args.frames_index))["frames"]
    sig_root = json.load(open(args.signals_json))["signals"]
    want = [s.strip() for s in args.signals.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Key regions by (shot_id, t_ms)
    regmap = {}
    for sname in want:
        regs = sig_root.get(sname, {}).get("regions", [])
        for r in regs:
            if float(r.get("score", 0.0)) < args.min_score:
                continue
            shot_id = int(r["shot_id"])
            t_ms = int(round(float(r["t_sec"]) * 1000.0))
            regmap.setdefault((shot_id, t_ms), []).append(r)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    overlays = 0
    frames_with_ball = []

    for fr in frames:
        shot_id = int(fr["shot_id"])
        t_sec = float(fr["t_sec"])
        t_ms = int(round(t_sec * 1000.0))
        img_path = fr["path"]
        if not os.path.exists(img_path):
            continue

        regs = regmap.get((shot_id, t_ms), [])

        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im)

        has_ball = False
        for r in regs:
            sname = r.get("signal", "")
            score = float(r.get("score", 0.0))
            x0, y0, x1, y1 = r["bbox_norm"]

            px0 = int(clamp(x0, 0, 1) * W)
            py0 = int(clamp(y0, 0, 1) * H)
            px1 = int(clamp(x1, 0, 1) * W)
            py1 = int(clamp(y1, 0, 1) * H)

            color = COLORS.get(sname, (255, 255, 0))
            draw.rectangle([px0, py0, px1, py1], outline=color, width=3)

            label = f"{sname} {score:.2f}"
            if font is not None:
                draw.text((px0 + 3, max(0, py0 - 12)), label, fill=color, font=font)

            if sname == "ball":
                has_ball = True

        # stamp
        stamp = f"shot={shot_id} t={t_sec:.3f}s regs={len(regs)}"
        draw.rectangle([0, 0, W, 20], fill=(0, 0, 0))
        if font is not None:
            draw.text((6, 3), stamp, fill=(255, 255, 255), font=font)

        base = Path(img_path).name.replace(".jpg", "")
        out_path = out_dir / f"{base}__overlay.jpg"
        im.save(out_path, quality=92)

        overlays += 1
        if has_ball:
            frames_with_ball.append(str(out_path))

    (out_dir / "ball_frames.txt").write_text("\n".join(frames_with_ball) + ("\n" if frames_with_ball else ""))

    print(f"wrote_overlays={overlays}")
    print(f"frames_with_ball={len(frames_with_ball)}")
    if frames_with_ball:
        print("ball_examples:")
        for p in frames_with_ball[:5]:
            print("  " + p)

if __name__ == "__main__":
    main()
