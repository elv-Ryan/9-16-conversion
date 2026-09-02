#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

ALLOWED_TYPES = {"tag", "progress", "progress_ratio", "error"}
ALLOWED_FAMILIES = {
    "active_speaker",
    "gameplay_follow",
    "graphic_text_lock",
    "person_subject",
    "safe_center",
    "split_screen",
    "static_composition",
}


def _load_expected(path: Path) -> List[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"expected input list is empty: {path}")
    if len(values) != len(set(values)):
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        raise ValueError(f"expected input list contains duplicates: {duplicates[:10]}")
    return values


def _read_messages(path: Path) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if message.get("type") not in ALLOWED_TYPES:
                raise ValueError(
                    f"{path}:{line_number}: unsupported message type {message.get('type')!r}"
                )
            if not isinstance(message.get("data"), dict):
                raise ValueError(f"{path}:{line_number}: data must be a JSON object")
            messages.append(message)
    return messages


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def validate(
    *,
    jsonl_path: Path,
    expected_path: Path,
    output_dir: Path,
    expected_count: Optional[int],
    wall_seconds: Optional[float],
    require_zero_errors: bool,
) -> Dict[str, Any]:
    expected_sources = _load_expected(expected_path)
    if expected_count is not None and len(expected_sources) != expected_count:
        raise ValueError(
            f"input count mismatch: list={len(expected_sources)} expected={expected_count}"
        )
    expected_set = set(expected_sources)
    messages = _read_messages(jsonl_path)

    progress: Counter[str] = Counter()
    errors: MutableMapping[str, List[str]] = defaultdict(list)
    vertical: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    unexpected_sources: Counter[str] = Counter()

    for message in messages:
        kind = str(message["type"])
        data = message["data"]
        source = data.get("source_media")
        if source is not None and source not in expected_set:
            unexpected_sources[str(source)] += 1
        if kind == "progress":
            progress[str(source)] += 1
        elif kind == "error":
            errors[str(source)].append(str(data.get("message", "unknown error")))
        elif kind == "tag" and data.get("track") == "vertical_video":
            vertical[str(source)].append(message)

    validation_errors: List[str] = []
    per_shot: List[Dict[str, Any]] = []
    x_payload_shots: List[Dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    successful_source_seconds = 0.0
    all_x_values: List[float] = []

    for source in expected_sources:
        terminal_count = progress[source] + len(errors[source])
        if terminal_count != 1:
            validation_errors.append(
                f"{source}: expected exactly one terminal progress/error, found {terminal_count}"
            )

        if errors[source]:
            if vertical[source]:
                validation_errors.append(f"{source}: emitted vertical tag and error")
            per_shot.append(
                {
                    "source_media": source,
                    "status": "ERROR",
                    "family": "",
                    "category": "",
                    "family_confidence": "",
                    "source_fps": "",
                    "frame_count": "",
                    "duration_seconds": "",
                    "x_min": "",
                    "x_max": "",
                    "x_mean": "",
                    "x_std": "",
                    "x_range": "",
                    "focus_samples": "",
                    "error": " | ".join(errors[source]),
                }
            )
            continue

        if progress[source] != 1:
            validation_errors.append(f"{source}: expected one progress row, found {progress[source]}")
        if len(vertical[source]) != 1:
            validation_errors.append(
                f"{source}: expected exactly one vertical_video tag, found {len(vertical[source])}"
            )
            continue

        data = vertical[source][0]["data"]
        info = data.get("additional_info") or {}
        family = str(info.get("family", data.get("tag", "")))
        category = str(info.get("category", ""))
        if family not in ALLOWED_FAMILIES:
            validation_errors.append(f"{source}: invalid family {family!r}")

        values = info.get("x-coordinates")
        frame_count = info.get("frame_count")
        if not isinstance(frame_count, int) or frame_count <= 0:
            validation_errors.append(f"{source}: invalid frame_count {frame_count!r}")
            continue
        if not isinstance(values, list) or len(values) != frame_count:
            validation_errors.append(
                f"{source}: x/frame mismatch values={len(values) if isinstance(values, list) else None} "
                f"frames={frame_count}"
            )
            continue
        if not values:
            validation_errors.append(f"{source}: empty x-coordinate list")
            continue

        try:
            lower = float(info["legal_x_center_min"])
            upper = float(info["legal_x_center_max"])
        except (KeyError, TypeError, ValueError) as exc:
            validation_errors.append(f"{source}: invalid legal X range: {exc}")
            continue
        if not (0.0 <= lower <= upper <= 1.0):
            validation_errors.append(f"{source}: invalid legal X range [{lower}, {upper}]")

        numeric: List[float] = []
        for index, value in enumerate(values):
            if not _finite_number(value):
                validation_errors.append(f"{source}: x[{index}] is not finite numeric: {value!r}")
                break
            x = float(value)
            if x < lower - 1e-9 or x > upper + 1e-9:
                validation_errors.append(
                    f"{source}: x[{index}]={x} outside [{lower}, {upper}]"
                )
                break
            numeric.append(x)
        if len(numeric) != frame_count:
            continue

        source_fps = float(info.get("source_fps") or 0.0)
        if not math.isfinite(source_fps) or source_fps <= 0.0:
            validation_errors.append(f"{source}: invalid source_fps {source_fps!r}")
            continue
        duration_seconds = frame_count / source_fps
        successful_source_seconds += duration_seconds
        family_counts[family] += 1
        category_counts[category] += 1
        all_x_values.extend(numeric)
        mean = statistics.fmean(numeric)
        std = statistics.pstdev(numeric) if len(numeric) > 1 else 0.0
        focus_samples = info.get("focus_samples")
        focus_count = len(focus_samples) if isinstance(focus_samples, list) else 0
        confidence = float(info.get("family_confidence") or 0.0)

        per_shot.append(
            {
                "source_media": source,
                "status": "PASS",
                "family": family,
                "category": category,
                "family_confidence": confidence,
                "source_fps": source_fps,
                "frame_count": frame_count,
                "duration_seconds": duration_seconds,
                "x_min": min(numeric),
                "x_max": max(numeric),
                "x_mean": mean,
                "x_std": std,
                "x_range": max(numeric) - min(numeric),
                "focus_samples": focus_count,
                "error": "",
            }
        )
        x_payload_shots.append(
            {
                "source_media": source,
                "shot_id": info.get("shot_id"),
                "start_time_ms": data.get("start_time"),
                "end_time_ms": data.get("end_time"),
                "family": family,
                "category": category,
                "family_confidence": confidence,
                "source_fps": source_fps,
                "source_frame_start": info.get("source_frame_start"),
                "frame_count": frame_count,
                "x_center_norm": numeric,
            }
        )

    if unexpected_sources:
        validation_errors.append(
            f"unexpected source_media values: {dict(unexpected_sources.most_common(20))}"
        )

    successful = sum(row["status"] == "PASS" for row in per_shot)
    failed = sum(row["status"] == "ERROR" for row in per_shot)
    missing_vertical = len(expected_sources) - successful - failed
    protocol_complete = all(progress[source] + len(errors[source]) == 1 for source in expected_sources)
    zero_errors = failed == 0
    strict_pass = (
        not validation_errors
        and protocol_complete
        and successful == len(expected_sources)
        and (zero_errors or not require_zero_errors)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_shot_fields = [
        "source_media",
        "status",
        "family",
        "category",
        "family_confidence",
        "source_fps",
        "frame_count",
        "duration_seconds",
        "x_min",
        "x_max",
        "x_mean",
        "x_std",
        "x_range",
        "focus_samples",
        "error",
    ]
    with (output_dir / "per_shot_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_shot_fields)
        writer.writeheader()
        writer.writerows(per_shot)

    candidates = [row for row in per_shot if row["status"] == "PASS"]
    candidates.sort(
        key=lambda row: (
            float(row["family_confidence"]),
            -float(row["x_std"]),
            -float(row["duration_seconds"]),
        )
    )
    with (output_dir / "review_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "source_media",
            "family",
            "category",
            "family_confidence",
            "duration_seconds",
            "x_std",
            "x_range",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates[:100])

    (output_dir / "x-output.json").write_text(
        json.dumps(
            {
                "schema_version": "eluvio.nba-yolo-x-json.v1",
                "shots": x_payload_shots,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "errors.json").write_text(
        json.dumps({source: values for source, values in errors.items() if values}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    throughput = None
    if wall_seconds is not None and wall_seconds > 0.0:
        throughput = successful_source_seconds / wall_seconds

    summary: Dict[str, Any] = {
        "status": "PASS" if strict_pass else "FAIL",
        "jsonl": str(jsonl_path),
        "expected_inputs": len(expected_sources),
        "messages": len(messages),
        "successful_shots": successful,
        "error_shots": failed,
        "missing_or_invalid_shots": missing_vertical,
        "protocol_complete": protocol_complete,
        "require_zero_errors": require_zero_errors,
        "validation_error_count": len(validation_errors),
        "validation_errors": validation_errors[:200],
        "family_counts": dict(sorted(family_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "successful_source_seconds": successful_source_seconds,
        "successful_source_hours": successful_source_seconds / 3600.0,
        "wall_seconds": wall_seconds,
        "video_seconds_per_wall_second": throughput,
        "x_realtime": throughput,
        "all_x_min": min(all_x_values) if all_x_values else None,
        "all_x_max": max(all_x_values) if all_x_values else None,
        "all_x_mean": statistics.fmean(all_x_values) if all_x_values else None,
        "all_x_p05": _quantile(all_x_values, 0.05),
        "all_x_p95": _quantile(all_x_values, 0.95),
        "artifacts": {
            "per_shot_summary": str(output_dir / "per_shot_summary.csv"),
            "review_candidates": str(output_dir / "review_candidates.csv"),
            "x_output": str(output_dir / "x-output.json"),
            "errors": str(output_dir / "errors.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a long-lived Eluvio JSONL run across a directory of shots."
    )
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--expected-inputs", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--wall-seconds", type=float)
    parser.add_argument("--allow-model-errors", action="store_true")
    args = parser.parse_args()

    try:
        result = validate(
            jsonl_path=args.jsonl,
            expected_path=args.expected_inputs,
            output_dir=args.output_dir,
            expected_count=args.expected_count,
            wall_seconds=args.wall_seconds,
            require_zero_errors=not args.allow_model_errors,
        )
    except Exception as exc:
        raise SystemExit(f"validation failed: {type(exc).__name__}: {exc}") from exc

    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
