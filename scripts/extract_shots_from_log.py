#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

SHOT_RE = re.compile(r"Shot change at:\s*([0-9.]+)\s*seconds")

def ffprobe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def parse_shot_times(log_path: Path) -> list[float]:
    times: list[float] = []
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m = SHOT_RE.search(line)
            if m:
                try:
                    times.append(float(m.group(1)))
                except ValueError:
                    pass
    return sorted(set(times))

def build_shots(boundary_times: list[float], duration: float, min_shot_sec: float = 0.0) -> list[dict]:
    boundaries = [0.0] + boundary_times + [duration]
    shots: list[dict] = []
    shot_id = 0
    for i in range(len(boundaries) - 1):
        s = float(boundaries[i])
        e = float(boundaries[i + 1])
        if e <= s:
            continue
        if (e - s) < min_shot_sec:
            if shots:
                shots[-1]["end_sec"] = round(e, 3)
                shots[-1]["duration_sec"] = round(shots[-1]["end_sec"] - shots[-1]["start_sec"], 3)
            continue
        shots.append({
            "shot_id": shot_id,
            "start_sec": round(s, 3),
            "end_sec": round(e, 3),
            "duration_sec": round(e - s, 3),
        })
        shot_id += 1
    return shots

def main():
    ap = argparse.ArgumentParser(description="Extract shot boundaries from AutoFlip log and build a shot manifest JSON.")
    ap.add_argument("--video", required=True, help="Path to input video (used for duration via ffprobe).")
    ap.add_argument("--log", required=True, help="Path to AutoFlip log containing 'Shot change at: X seconds.' lines.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--min_shot_sec", type=float, default=0.0, help="Merge shots shorter than this into previous shot.")
    args = ap.parse_args()

    video_path = str(Path(args.video))
    log_path = Path(args.log)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    duration = ffprobe_duration(video_path)
    boundary_times = parse_shot_times(log_path)
    shots = build_shots(boundary_times, duration, min_shot_sec=float(args.min_shot_sec))

    payload = {
        "video_path": video_path,
        "log_path": str(log_path),
        "duration_sec": round(duration, 3),
        "shot_change_times_sec": [round(t, 3) for t in boundary_times],
        "shots": shots,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/extract_shots_from_log.py",
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"duration_sec={payload['duration_sec']}")
    print(f"shot_change_count={len(boundary_times)}")
    print(f"shot_count={len(shots)}")
    if boundary_times:
        print("shot_change_times_sec=" + ", ".join(f"{t:.3f}" for t in boundary_times))

if __name__ == "__main__":
    main()
