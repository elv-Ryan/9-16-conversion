#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import glob
from pathlib import Path
from multiprocessing import Process
import time


SHOT_RE = re.compile(r"Shot change at:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds\.")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_run_autoflip() -> str:
    candidates = [
        "/runfiles/mediapipe/mediapipe/examples/desktop/autoflip/run_autoflip", 
        "/work/vendor/mediapipe/bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip",
        "/work/bazel-bin/mediapipe/examples/desktop/autoflip/run_autoflip",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    probe = subprocess.run(
        "find / -name run_autoflip -type f 2>/dev/null | head -n 1",
        shell=True,
        text=True,
        capture_output=True,
    )
    path = (probe.stdout or "").strip()
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path

    raise FileNotFoundError("run_autoflip not found inside container")


def ffprobe_value(path: str, field: str, count_frames: bool = False) -> str:
    cmd = ["ffprobe"]
    if count_frames:
        cmd.append("-count_frames")
    cmd += [
        "-v",
        "error",
        "-show_entries",
        field,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def load_x_coordinates(path: str):
    xs = []
    if not os.path.exists(path):
        return xs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            x = obj.get("x_center_norm")
            if isinstance(x, (int, float)):
                xs.append(float(x))
    return xs


def parse_shot_change_seconds(text: str):
    vals = []
    for match in SHOT_RE.finditer(text or ""):
        try:
            vals.append(float(match.group(1)))
        except Exception:
            pass
    return sorted(set(vals))


def sec_to_frame_idx(sec: float, total_frames: int, duration_sec: float) -> int:
    if total_frames <= 0 or duration_sec <= 0:
        return 0
    fps = total_frames / duration_sec
    idx = int(round(sec * fps))
    return max(0, min(idx, total_frames))


def build_shot_ranges(total_frames: int, duration_sec: float, shot_change_seconds):
    boundaries = [0]
    for sec in shot_change_seconds:
        idx = sec_to_frame_idx(sec, total_frames, duration_sec)
        if 0 < idx < total_frames:
            boundaries.append(idx)
    boundaries.append(total_frames)
    boundaries = sorted(set(boundaries))

    ranges = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        if end_idx <= start_idx:
            continue
        if total_frames > 0 and duration_sec > 0:
            start_ms = int(round((start_idx / total_frames) * duration_sec * 1000.0))
            end_ms = int(round((end_idx / total_frames) * duration_sec * 1000.0))
        else:
            start_ms = 0
            end_ms = 0
        ranges.append((start_idx, end_idx, start_ms, end_ms))
    return ranges


def derive_focus_tag(graph_path: str) -> str:
    name = Path(graph_path).stem.lower()
    name = re.sub(r"^autoflip_graph_", "", name)
    name = re.sub(r"_with_x$", "", name)
    name = re.sub(r"_raw$", "", name)
    name = re.sub(r"_target\d+x\d+$", "", name)

    tokens = [t for t in name.split("_") if t and t not in {"noaudio", "panonly", "nozoom", "keeph"}]
    cleaned = "_".join(tokens) if tokens else "autoflip"

    mapping = {
        "faceprimary": "faceprimary_model",
        "faceonly": "faceonly_model",
    }
    return mapping.get(cleaned, f"{cleaned}_model")


def write_pretty_json(output_path: Path, records):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:            
            json.dump(record, f)
            f.write("\n")

def combine_to_pipe_no_temp(video_files, pipe_name="video_stream.mp4"):
    video_files = sorted(video_files) ##glob.glob(os.path.join(os.path.abspath(directory), "*.mp4")))
    
    if not video_files:
        raise Exception("No MP4 files specified.")
        return


    # make concat spec
    ## TODO handle relative paths for local testing
    ##concat_content = "\n".join([f"file 'f' for f in video_files])

    concat_content = "".join([f"file 'file:{os.path.abspath(f)}'\n" for f in video_files])
    
    ## prep & run ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        '-protocol_whitelist', 'pipe,file', '-i', 'pipe:0',
        "-c", "copy",
        "-f", "mp4",
##        "-movflags", "frag_keyframe+empty_moov",
        pipe_name
    ]

    print(f"Streaming to {pipe_name}... (Awaiting reader)", file = sys.stderr)
    
    try:
        # 5. Use Popen and communicate the string to stdin
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout = sys.stdout, stderr = sys.stdout, text=True)

        print(f"subprocess.Popen done", file = sys.stderr)

        # This sends the string to FFmpeg's stdin and then closes it
        process.communicate(input=concat_content)

        print(f"process.communicate done", file = sys.stderr)

    except KeyboardInterrupt as ki:
        process.terminate()
        raise ki
    finally:
        print("Done.", file = sys.stderr)

        
    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-path", required=True)
    ap.add_argument(
        "--graph",
        default="/work/graphs/autoflip_graph_faceprimary_target406x720_RAW_with_x.pbtxt",
    )
    ap.add_argument("--resource-root-dir", default="/work/models")
    ap.add_argument("--aspect-ratio", default="9:16")
    ap.add_argument("--output-suffix", default="_9x16_RAW.mp4")
    ap.add_argument("--track", default="vertical_video")
    ap.add_argument("--tag", default="")
    ap.add_argument("--artifacts-dir", default="")
    ap.add_argument("--x-output-stream", default="external_render_json")
    args = ap.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (output_path.parent / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    run_autoflip = find_run_autoflip()
    resolved_tag = args.tag.strip() if args.tag and args.tag.strip() else derive_focus_tag(args.graph)

    eprint(f"RUN_AUTOFLOP_BIN={run_autoflip}")
    eprint(f"GRAPH={args.graph}")
    eprint(f"RESOURCE_ROOT={args.resource_root_dir}")
    eprint(f"ARTIFACTS_DIR={artifacts_dir}")
    eprint(f"FOCUS_TAG={resolved_tag}")

    all_records = []

    file_list = []
    for raw in sys.stdin:
        a_source_file = raw.strip()
        if not a_source_file:
            continue

        if not os.path.exists(a_source_file):
            raise FileNotFoundError(f"input file does not exist: {a_source_file}")

        file_list += [ a_source_file ]


    print("The file list")
    print(file_list)
    
    source_media = "video_stream.mp4"
    concatter = Process(target = combine_to_pipe_no_temp, args = [file_list])

    print("calling concatter.start")
    concatter.start()
    concatter.join()
        
    if True:        
        stem = Path(source_media).stem
        output_media = str(artifacts_dir / f"{stem}{args.output_suffix}")
        x_jsonl_path = str(artifacts_dir / f"{stem}.x.jsonl")

        if os.path.exists(x_jsonl_path):
            os.remove(x_jsonl_path)

        cmd = [
            run_autoflip,
            f"--resource_root_dir={args.resource_root_dir}",
            f"--calculator_graph_config_file={args.graph}",
            f"--input_side_packets=input_video_path={source_media},output_video_path={output_media},aspect_ratio={args.aspect_ratio}",
            f"--output_stream={args.x_output_stream}",
            f"--output_stream_file={x_jsonl_path}",
            "--strip_timestamps",
        ]

        eprint("")
        eprint(f"== PROCESS {source_media} -> {output_media} ==")
        proc = subprocess.run(cmd, text=True, capture_output=True)

        if proc.stdout:
            eprint(proc.stdout.rstrip())
        if proc.stderr:
            eprint(proc.stderr.rstrip())

        if proc.returncode != 0:
            raise RuntimeError(f"run_autoflip failed with exit code {proc.returncode}")

        x_coordinates = load_x_coordinates(x_jsonl_path)
        shot_change_seconds = parse_shot_change_seconds((proc.stdout or "") + "\n" + (proc.stderr or ""))

        duration_s = ffprobe_value(output_media, "format=duration") or "0"
        frame_count_s = ffprobe_value(output_media, "stream=nb_read_frames", count_frames=True) or ""

        try:
            duration = float(duration_s.splitlines()[0].strip())
        except Exception:
            duration = 0.0

        try:
            frame_count = int(frame_count_s.splitlines()[0].strip())
        except Exception:
            frame_count = len(x_coordinates)

        total_frames = frame_count if frame_count > 0 else len(x_coordinates)
        shot_ranges = build_shot_ranges(total_frames, duration, shot_change_seconds)

        if not shot_ranges and x_coordinates:
            shot_ranges = [(0, len(x_coordinates), 0, int(round(duration * 1000.0)))]

        for start_idx, end_idx, start_ms, end_ms in shot_ranges:
            shot_xs = x_coordinates[start_idx:end_idx]
            all_records.append(
                {
                    "type": "tag",
                    "data": {
                        "tag": resolved_tag,
                        "start_time": start_ms,
                        "end_time": end_ms,
                        "track": args.track,
                        "frame_info": {
                            "frame_idx": start_idx
                        },
                        "additional_info": {
                            "x-coordinates": shot_xs
                        },
                        "source_media": source_media
                    }
                }
            )

        if os.path.exists(x_jsonl_path):
            os.remove(x_jsonl_path)

        ## append progress dump
        all_records.append({"type": "progress", "data" : { "source_media": source_media }})

    concatter.join()
    write_pretty_json(output_path, all_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
