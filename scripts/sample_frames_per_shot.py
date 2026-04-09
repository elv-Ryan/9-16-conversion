#!/usr/bin/env python3
import argparse, json, math, subprocess
from pathlib import Path

def run(cmd):
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--shots_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frames_per_shot", type=int, default=3)
    args = ap.parse_args()

    video = Path(args.video)
    shots_path = Path(args.shots_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = json.loads(shots_path.read_text())
    shots = d["shots"]

    # Evenly spaced fractions inside the shot, avoiding exact endpoints.
    n = max(1, args.frames_per_shot)
    fracs = [ (i+1)/(n+1) for i in range(n) ]  # e.g., n=3 => 0.25,0.5,0.75

    index = {
        "video": str(video),
        "shots_json": str(shots_path),
        "frames_per_shot": n,
        "frames": []  # list of {shot_id, t_sec, path}
    }

    for s in shots:
        sid = int(s["shot_id"])
        start = float(s["start_sec"])
        end = float(s["end_sec"])
        dur = max(0.0, end - start)

        for j, frac in enumerate(fracs):
            t = start + frac * dur
            # Clamp to valid interior point
            t = max(start, min(t, max(start, end - 1e-3)))
            t_str = f"{t:.3f}"

            jpg = out_dir / f"shot_{sid:03d}_f{j:02d}_t{t_str}.jpg"

            # Accurate seek (slower but deterministic): -ss after -i
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y",
                "-i", str(video),
                "-ss", t_str,
                "-frames:v", "1",
                "-q:v", "2",
                str(jpg),
            ]
            run(cmd)

            index["frames"].append({
                "shot_id": sid,
                "frame_id": j,
                "t_sec": float(t_str),
                "path": str(jpg),
            })

    (out_dir / "frames_index.json").write_text(json.dumps(index, indent=2))
    print(f"Wrote {len(index['frames'])} frames -> {out_dir}")
    print(f"Index: {out_dir/'frames_index.json'}")

if __name__ == "__main__":
    main()
