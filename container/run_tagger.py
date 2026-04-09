#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_run_autoflip() -> str:
    candidates = [
        "/work/vendor/mediapipe/bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip",
        "/work/bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    probe = subprocess.run(
        "find /work -name run_autoflip -type f 2>/dev/null | head -n 1",
        shell=True,
        text=True,
        capture_output=True,
    )
    path = (probe.stdout or "").strip()
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path

    raise FileNotFoundError("run_autoflip not found inside container")


def ffprobe_value(path: str, field: str) -> str:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", field,
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def emit_jsonl(fp, obj):
    fp.write(json.dumps(obj, separators=(",", ":")) + "\n")
    fp.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--graph", default="/work/graphs/autoflip_graph_faceprimary_target406x720_RAW.pbtxt")
    ap.add_argument("--resource-root-dir", default="/work/models")
    ap.add_argument("--aspect-ratio", default="9:16")
    ap.add_argument("--output-suffix", default="_9x16_RAW.mp4")
    ap.add_argument("--track", default="vertical_conversion")
    ap.add_argument("--tag", default="vertical_conversion_completed")
    ap.add_argument("--artifacts-dir", default="")
    args = ap.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (output_path.parent / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_autoflip = find_run_autoflip()
    eprint(f"RUN_AUTOFLOP_BIN={run_autoflip}")
    eprint(f"GRAPH={args.graph}")
    eprint(f"RESOURCE_ROOT={args.resource_root_dir}")
    eprint(f"ARTIFACTS_DIR={artifacts_dir}")

    with output_path.open("a", encoding="utf-8") as fp:
        for raw in sys.stdin:
            source_media = raw.strip()
            if not source_media:
                continue

            if not os.path.exists(source_media):
                emit_jsonl(fp, {
                    "type": "error",
                    "data": {
                        "message": f"input file does not exist: {source_media}",
                        "source_media": source_media,
                    }
                })
                continue

            stem = Path(source_media).stem
            output_media = str(artifacts_dir / f"{stem}{args.output_suffix}")

            cmd = [
                run_autoflip,
                f"--resource_root_dir={args.resource_root_dir}",
                f"--calculator_graph_config_file={args.graph}",
                f'--input_side_packets=input_video_path={source_media},output_video_path={output_media},aspect_ratio={args.aspect_ratio}',
            ]

            eprint("")
            eprint(f"== PROCESS {source_media} -> {output_media} ==")
            proc = subprocess.run(cmd, text=True, capture_output=True)

            if proc.stdout:
                eprint(proc.stdout.rstrip())
            if proc.stderr:
                eprint(proc.stderr.rstrip())

            if proc.returncode != 0:
                emit_jsonl(fp, {
                    "type": "error",
                    "data": {
                        "message": f"run_autoflip failed with exit code {proc.returncode}",
                        "source_media": source_media,
                    }
                })
                continue

            duration_s = ffprobe_value(output_media, "format=duration") or "0"
            width_s = ffprobe_value(output_media, "stream=width") or ""
            height_s = ffprobe_value(output_media, "stream=height") or ""
            codec_s = ffprobe_value(output_media, "stream=codec_name") or ""

            try:
                duration = float(duration_s.splitlines()[0].strip())
            except Exception:
                duration = 0.0

            emit_jsonl(fp, {
                "type": "tag",
                "data": {
                    "start_time": 0.0,
                    "end_time": duration,
                    "source_media": source_media,
                    "tag": args.tag,
                    "track": args.track,
                    "additional_info": {
                        "output_media": output_media,
                        "width": width_s.splitlines()[0].strip() if width_s else "",
                        "height": height_s.splitlines()[0].strip() if height_s else "",
                        "codec_name": codec_s.splitlines()[0].strip() if codec_s else "",
                    },
                }
            })

            emit_jsonl(fp, {
                "type": "progress",
                "data": {
                    "source_media": source_media,
                }
            })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
