#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
FORBIDDEN_GPUS = set()
DEFAULT_GPU_CANDIDATES = [7, 0, 1, 2, 3, 4, 5, 6]


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def natural_key(path: Path) -> Tuple:
    pieces = re.split(r"(\d+)", path.as_posix().lower())
    return tuple(int(piece) if piece.isdigit() else piece for piece in pieces)


def numeric_index(path: Path) -> Optional[int]:
    # Joe's shot files are named like 000-16_1818.mp4.  The shot ordinal is
    # the leading integer, not the trailing source timestamp.
    match = re.match(r"^(\d+)(?:[-_]|$)", path.stem)
    return int(match.group(1)) if match else None


def discover_videos(root: Path, expected_count: int) -> Tuple[List[Path], Dict]:
    videos = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VIDEO_SUFFIXES
            and not path.name.startswith(".")
        ),
        key=natural_key,
    )
    index_map: Dict[int, Path] = {}
    duplicate_indices: Dict[int, List[str]] = {}
    for path in videos:
        index = numeric_index(path)
        if index is None:
            continue
        if index in index_map:
            duplicate_indices.setdefault(index, [str(index_map[index])]).append(str(path))
        else:
            index_map[index] = path

    expected_indices = set(range(expected_count))
    exact_coverage = not duplicate_indices and expected_indices.issubset(index_map)
    if exact_coverage:
        selected = [index_map[index] for index in range(expected_count)]
    elif len(videos) == expected_count:
        selected = videos
    else:
        selected = []

    details = {
        "discovered_video_files": len(videos),
        "selected_video_files": len(selected),
        "numeric_indices_parsed": len(index_map),
        "exact_numeric_coverage_0_through_n_minus_1": exact_coverage,
        "missing_numeric_indices": sorted(expected_indices - set(index_map))[:100],
        "duplicate_numeric_indices": duplicate_indices,
        "first_files": [str(path) for path in videos[:5]],
        "last_files": [str(path) for path in videos[-5:]],
    }
    return selected, details


def query_gpus() -> List[Dict]:
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    uuid_to_processes: Dict[str, List[Dict]] = {}
    try:
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        processes = ""
    for line in processes.splitlines():
        if not line.strip():
            continue
        parts = [piece.strip() for piece in line.split(",", 3)]
        if len(parts) != 4:
            continue
        uuid, pid, name, memory = parts
        uuid_to_processes.setdefault(uuid, []).append(
            {"pid": int(pid), "name": name, "memory_mib": int(memory.split()[0])}
        )

    rows: List[Dict] = []
    for line in inventory.splitlines():
        parts = [piece.strip() for piece in line.split(",", 4)]
        if len(parts) != 5:
            continue
        index, uuid, used, free, utilization = parts
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(used.split()[0]),
                "memory_free_mib": int(free.split()[0]),
                "utilization_percent": int(utilization.split()[0]),
                "compute_processes": uuid_to_processes.get(uuid, []),
            }
        )
    return rows


def gpu_is_free(row: Dict) -> bool:
    return (
        not row["compute_processes"]
        and row["memory_used_mib"] <= 1024
        and row["memory_free_mib"] >= 20000
        and row["utilization_percent"] <= 10
    )


def select_gpu(requested: str, candidates: Sequence[int]) -> Tuple[Optional[int], List[Dict]]:
    rows = query_gpus()
    by_index = {row["index"]: row for row in rows}
    if requested != "auto":
        index = int(requested)
        row = by_index.get(index)
        return (index if row and gpu_is_free(row) else None), rows
    for index in candidates:
        row = by_index.get(index)
        if row and gpu_is_free(row):
            return index, rows
    return None, rows


def count_terminal_messages(jsonl: Path) -> Tuple[int, int, int]:
    if not jsonl.exists():
        return 0, 0, 0
    progress = 0
    errors = 0
    tags = 0
    try:
        with jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                message = json.loads(line)
                kind = message.get("type")
                if kind == "progress":
                    progress += 1
                elif kind == "error":
                    errors += 1
                elif kind == "tag":
                    data = message.get("data") or {}
                    if data.get("track") == "vertical_video":
                        tags += 1
    except (OSError, json.JSONDecodeError):
        pass
    return progress, errors, tags


@dataclass
class Controller:
    repo: Path
    shot_dir: Path
    run_root: Path
    image: str
    expected_count: int
    requested_gpu: str
    candidates: List[int]
    poll_seconds: float
    max_file_wait_seconds: float
    max_gpu_wait_seconds: float
    inference_fps: float
    batch_size: int

    def __post_init__(self) -> None:
        self.state_path = self.run_root / "STATUS.json"
        self.log_dir = self.run_root / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)

    def state(self, stage: str, **values) -> None:
        previous: Dict = {}
        if self.state_path.exists():
            try:
                previous = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        previous.update(values)
        previous.update(
            {
                "stage": stage,
                "updated_utc": utcnow(),
                "shot_dir": str(self.shot_dir),
                "run_root": str(self.run_root),
                "image": self.image,
                "expected_shots": self.expected_count,
                "forbidden_gpu": None,
            }
        )
        atomic_json(self.state_path, previous)

    def wait_for_files(self) -> List[Path]:
        started = time.monotonic()
        stable_signature: Optional[Tuple[Tuple[str, int, int], ...]] = None
        stable_checks = 0
        while True:
            selected, details = discover_videos(self.shot_dir, self.expected_count)
            signature = tuple(
                (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in selected
            )
            if selected and signature == stable_signature:
                stable_checks += 1
            else:
                stable_checks = 0
            stable_signature = signature if selected else None
            self.state(
                "WAITING_FOR_518_SHOTS",
                file_discovery=details,
                stable_checks=stable_checks,
                elapsed_wait_seconds=time.monotonic() - started,
            )
            if selected and len(selected) == self.expected_count and stable_checks >= 2:
                return selected
            if time.monotonic() - started > self.max_file_wait_seconds:
                raise TimeoutError(
                    f"shot directory did not reach a stable {self.expected_count} files: {details}"
                )
            time.sleep(self.poll_seconds)

    def write_input_list(self, shots: Sequence[Path], destination: Path) -> List[str]:
        inputs: List[str] = []
        for shot in shots:
            relative = shot.relative_to(self.shot_dir)
            if "\n" in relative.as_posix():
                raise ValueError(f"newline in shot filename is unsupported: {shot}")
            inputs.append(f"/elv/input/{relative.as_posix()}")
        destination.write_text("\n".join(inputs) + "\n", encoding="utf-8")
        return inputs

    def wait_for_gpu(self) -> Tuple[int, object]:
        started = time.monotonic()
        while True:
            selected, rows = select_gpu(self.requested_gpu, self.candidates)
            self.state(
                "WAITING_FOR_FREE_GPU",
                requested_gpu=self.requested_gpu,
                gpu_candidates=self.candidates,
                gpu_inventory=rows,
                elapsed_gpu_wait_seconds=time.monotonic() - started,
            )
            if selected is not None:
                lock_path = Path(f"/tmp/nba-yolo-shot-tagger-gpu-{selected}.lock")
                lock_handle = lock_path.open("w")
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock_handle.close()
                    time.sleep(self.poll_seconds)
                    continue
                reselected, rows_after_lock = select_gpu(str(selected), self.candidates)
                if reselected == selected:
                    self.state(
                        "GPU_RESERVED",
                        selected_gpu=selected,
                        gpu_inventory=rows_after_lock,
                    )
                    return selected, lock_handle
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            if time.monotonic() - started > self.max_gpu_wait_seconds:
                raise TimeoutError("no allowed GPU became free before timeout")
            time.sleep(self.poll_seconds)

    def params(self) -> str:
        return json.dumps(
            {
                "device": "0",
                "input_mode": "shot_file",
                "imgsz": 1280,
                "inference_fps": self.inference_fps,
                "batch_size": self.batch_size,
                "use_fp16": True,
                "continue_on_error": True,
                "emit_progress_ratio": False,
                "emit_focus_track": False,
                "include_focus_samples": True,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def run_container(
        self,
        *,
        gpu: int,
        input_file: Path,
        output_dir: Path,
        stage: str,
        total_inputs: int,
    ) -> Tuple[int, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(output_dir, 0o777)
        output_jsonl = output_dir / "out.jsonl"
        output_jsonl.unlink(missing_ok=True)
        container_log = self.log_dir / f"{stage.lower()}.log"
        command = [
            "podman",
            "run",
            "--rm",
            "-i",
            "--device",
            f"nvidia.com/gpu={gpu}",
            "-v",
            f"{self.shot_dir}:/elv/input:ro",
            "-v",
            f"{output_dir}:/elv/output",
            self.image,
            "--output-path",
            "/elv/output/out.jsonl",
            "--params",
            self.params(),
        ]
        self.state(
            stage,
            selected_gpu=gpu,
            total_inputs=total_inputs,
            command=command,
            output_jsonl=str(output_jsonl),
        )
        started = time.monotonic()
        with input_file.open("rb") as stdin_handle, container_log.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                stdin=stdin_handle,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            (self.run_root / f"{stage}.pid").write_text(str(process.pid), encoding="utf-8")
            while process.poll() is None:
                completed, errors, tags = count_terminal_messages(output_jsonl)
                elapsed = time.monotonic() - started
                self.state(
                    stage,
                    selected_gpu=gpu,
                    container_pid=process.pid,
                    total_inputs=total_inputs,
                    progress_messages=completed,
                    error_messages=errors,
                    vertical_tags=tags,
                    completed_terminal_messages=completed + errors,
                    percent_complete=100.0 * (completed + errors) / max(1, total_inputs),
                    elapsed_seconds=elapsed,
                )
                time.sleep(self.poll_seconds)
            elapsed = time.monotonic() - started
            return process.returncode, elapsed

    def validate_run(
        self,
        *,
        output_dir: Path,
        expected_input_file: Path,
        count: int,
        elapsed: float,
    ) -> None:
        command = [
            sys.executable,
            str(self.repo / "scripts" / "validate_shot_directory_jsonl.py"),
            str(output_dir / "out.jsonl"),
            "--expected-inputs",
            str(expected_input_file),
            "--output-dir",
            str(output_dir),
            "--expected-count",
            str(count),
            "--wall-seconds",
            str(elapsed),
        ]
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output_dir / "validator.log").write_text(completed.stdout or "", encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"JSONL validation failed rc={completed.returncode}:\n{completed.stdout[-8000:]}"
            )

    def run(self) -> None:
        if not self.shot_dir.is_dir():
            raise FileNotFoundError(f"shot directory is missing: {self.shot_dir}")
        shots = self.wait_for_files()
        all_inputs_path = self.run_root / "expected_inputs.txt"
        inputs = self.write_input_list(shots, all_inputs_path)
        selected_gpu, lock_handle = self.wait_for_gpu()
        try:
            smoke_inputs_path = self.run_root / "smoke_inputs.txt"
            smoke_inputs_path.write_text("\n".join(inputs[:3]) + "\n", encoding="utf-8")
            smoke_dir = self.run_root / "smoke"
            returncode, elapsed = self.run_container(
                gpu=selected_gpu,
                input_file=smoke_inputs_path,
                output_dir=smoke_dir,
                stage="SMOKE_3_SHOTS",
                total_inputs=3,
            )
            if returncode != 0:
                raise RuntimeError(f"3-shot Podman smoke failed with rc={returncode}")
            self.validate_run(
                output_dir=smoke_dir,
                expected_input_file=smoke_inputs_path,
                count=3,
                elapsed=elapsed,
            )
            self.state("SMOKE_PASS", selected_gpu=selected_gpu, smoke_wall_seconds=elapsed)

            full_dir = self.run_root / "full_518"
            returncode, elapsed = self.run_container(
                gpu=selected_gpu,
                input_file=all_inputs_path,
                output_dir=full_dir,
                stage="FULL_518_SHOT_RUN",
                total_inputs=self.expected_count,
            )
            if returncode != 0:
                raise RuntimeError(f"518-shot Podman run failed with rc={returncode}")
            self.state("VALIDATING_FULL_518", selected_gpu=selected_gpu, wall_seconds=elapsed)
            self.validate_run(
                output_dir=full_dir,
                expected_input_file=all_inputs_path,
                count=self.expected_count,
                elapsed=elapsed,
            )
            summary = json.loads((full_dir / "summary.json").read_text(encoding="utf-8"))
            self.state(
                "COMPLETE",
                selected_gpu=selected_gpu,
                wall_seconds=elapsed,
                summary=summary,
                result_dir=str(full_dir),
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def parse_candidates(value: str) -> List[int]:
    candidates = [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not candidates:
        raise ValueError("at least one GPU candidate is required")
    if len(candidates) != len(set(candidates)):
        raise ValueError("GPU candidate list contains duplicates")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--shot-dir", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--image", default="nba-yolo-shot-tagger:mvp")
    parser.add_argument("--expected-count", type=int, default=518)
    parser.add_argument("--gpu", default="auto", help="physical GPU index or 'auto'; only genuinely free GPUs are selected")
    parser.add_argument("--gpu-candidates", default="7,0,1,2,3,4,5,6")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-file-wait-seconds", type=float, default=1800.0)
    parser.add_argument("--max-gpu-wait-seconds", type=float, default=86400.0)
    parser.add_argument("--inference-fps", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    controller = Controller(
        repo=args.repo.resolve(),
        shot_dir=args.shot_dir.resolve(),
        run_root=args.run_root.resolve(),
        image=args.image,
        expected_count=args.expected_count,
        requested_gpu=args.gpu,
        candidates=parse_candidates(args.gpu_candidates),
        poll_seconds=args.poll_seconds,
        max_file_wait_seconds=args.max_file_wait_seconds,
        max_gpu_wait_seconds=args.max_gpu_wait_seconds,
        inference_fps=args.inference_fps,
        batch_size=args.batch_size,
    )
    try:
        controller.run()
    except Exception as exc:
        controller.state("FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
