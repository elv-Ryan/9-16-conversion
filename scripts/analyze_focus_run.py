#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import cv2
except ImportError:  # Metrics still work without local media access.
    cv2 = None


@dataclass(frozen=True)
class ShotTrajectory:
    mode: str
    source: str
    start_frame: int
    start_ms: int
    end_ms: int
    label: str
    xs: Tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryMetrics:
    x_min: float
    x_max: float
    x_range: float
    path: float
    mean_abs_speed_per_s: float
    p95_abs_speed_per_s: float
    max_abs_speed_per_s: float
    max_abs_accel_per_s2: float
    max_abs_jerk_per_s3: float
    reversals: int
    moving_fraction: float
    crop_limit_fraction: Optional[float]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, q))
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def finite_stats(values: Sequence[float]) -> Dict[str, float | int]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {"count": 0, "min": 0.0, "q1": 0.0, "median": 0.0, "q3": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "q1": percentile(values, 0.25),
        "median": median(values),
        "q3": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": mean(values),
    }


def same_source(left: str, right: str) -> bool:
    return left == right or Path(left).name == Path(right).name


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def load_compact(mode: str, path: Path) -> tuple[List[ShotTrajectory], Dict[str, Any]]:
    messages = read_jsonl(path)
    focus_labels: Dict[tuple[str, int], str] = {}
    tracks = Counter()
    terminals = Counter()
    tag_after_terminal: List[str] = []
    terminal_seen = set()
    for message in messages:
        kind = str(message.get("type", ""))
        data = message.get("data") or {}
        source = str(data.get("source_media", ""))
        if kind in {"progress", "error"}:
            terminals[(source, kind)] += 1
            terminal_seen.add(source)
            continue
        if kind != "tag":
            continue
        if source in terminal_seen:
            tag_after_terminal.append(source)
        track = str(data.get("track", ""))
        tracks[track] += 1
        if track == "focus":
            start_frame = int((data.get("frame_info") or {}).get("frame_idx", 0))
            focus_labels[(source, start_frame)] = str(data.get("tag", "focus"))

    shots: List[ShotTrajectory] = []
    for message in messages:
        if message.get("type") != "tag":
            continue
        data = message.get("data") or {}
        if data.get("track") != "vertical_video":
            continue
        source = str(data.get("source_media", ""))
        start_frame = int((data.get("frame_info") or {}).get("frame_idx", 0))
        xs = tuple(float(value) for value in ((data.get("additional_info") or {}).get("x-coordinates") or []))
        shots.append(
            ShotTrajectory(
                mode=mode,
                source=source,
                start_frame=start_frame,
                start_ms=int(data.get("start_time", 0)),
                end_ms=int(data.get("end_time", 0)),
                label=focus_labels.get((source, start_frame), str(data.get("tag", "focus"))),
                xs=xs,
            )
        )
    summary = {
        "message_count": len(messages),
        "tracks": dict(tracks),
        "terminals": {f"{source}|{kind}": count for (source, kind), count in terminals.items()},
        "tag_after_terminal_sources": sorted(set(tag_after_terminal)),
    }
    return shots, summary


def load_debug(mode: str, path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    for row in rows:
        row.setdefault("mode", mode)
    return rows


def source_crop_limits(source: str) -> Optional[tuple[float, float]]:
    if cv2 is None or not Path(source).is_file():
        return None
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0.0 or height <= 0.0:
        return None
    crop_width = min(1.0, (9.0 / 16.0) * (height / width))
    return crop_width / 2.0, 1.0 - crop_width / 2.0


def times_for_shot(
    shot: ShotTrajectory,
    debug_by_mode_source_frame: Mapping[tuple[str, str, int], float],
) -> List[float]:
    times: List[float] = []
    for offset in range(len(shot.xs)):
        key = (shot.mode, shot.source, shot.start_frame + offset)
        if key in debug_by_mode_source_frame:
            times.append(float(debug_by_mode_source_frame[key]))
        else:
            times = []
            break
    if times:
        return times
    duration_s = max(1e-6, (shot.end_ms - shot.start_ms) / 1000.0)
    if len(shot.xs) <= 1:
        return [shot.start_ms / 1000.0]
    dt = duration_s / len(shot.xs)
    start_s = shot.start_ms / 1000.0
    return [start_s + index * dt for index in range(len(shot.xs))]


def count_reversals(xs: Sequence[float], times: Sequence[float]) -> int:
    if len(xs) < 3:
        return 0
    # Measure direction over approximately 150 ms, then require at least 0.002
    # normalized displacement. This suppresses frame-scale numerical noise.
    directions: List[tuple[float, int]] = []
    for index in range(1, len(xs)):
        previous = index - 1
        while previous > 0 and times[index] - times[previous] < 0.15:
            previous -= 1
        displacement = xs[index] - xs[previous]
        if abs(displacement) < 0.002:
            continue
        directions.append((times[index], 1 if displacement > 0.0 else -1))
    if not directions:
        return 0
    runs: List[tuple[int, float, float]] = []
    run_sign = directions[0][1]
    run_start = directions[0][0]
    run_end = run_start
    for timestamp, sign in directions[1:]:
        if sign == run_sign:
            run_end = timestamp
            continue
        runs.append((run_sign, run_start, run_end))
        run_sign = sign
        run_start = timestamp
        run_end = timestamp
    runs.append((run_sign, run_start, run_end))
    sustained = [run for run in runs if run[2] - run[1] >= 0.10]
    return max(0, len(sustained) - 1)


def trajectory_metrics(xs: Sequence[float], times: Sequence[float], limits: Optional[tuple[float, float]]) -> TrajectoryMetrics:
    if not xs:
        return TrajectoryMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, None)
    speeds_signed: List[float] = []
    for index in range(1, len(xs)):
        dt = max(1e-6, times[index] - times[index - 1])
        speeds_signed.append((xs[index] - xs[index - 1]) / dt)
    accelerations: List[float] = []
    for index in range(1, len(speeds_signed)):
        dt = max(1e-6, times[index + 1] - times[index])
        accelerations.append((speeds_signed[index] - speeds_signed[index - 1]) / dt)
    jerks: List[float] = []
    for index in range(1, len(accelerations)):
        dt = max(1e-6, times[index + 2] - times[index + 1])
        jerks.append((accelerations[index] - accelerations[index - 1]) / dt)
    abs_speeds = [abs(value) for value in speeds_signed]
    moving = sum(value >= 0.005 for value in abs_speeds)
    crop_fraction: Optional[float] = None
    if limits is not None:
        low, high = limits
        crop_fraction = sum(abs(x - low) <= 0.005 or abs(x - high) <= 0.005 for x in xs) / len(xs)
    return TrajectoryMetrics(
        x_min=min(xs),
        x_max=max(xs),
        x_range=max(xs) - min(xs),
        path=sum(abs(xs[index] - xs[index - 1]) for index in range(1, len(xs))),
        mean_abs_speed_per_s=mean(abs_speeds) if abs_speeds else 0.0,
        p95_abs_speed_per_s=percentile(abs_speeds, 0.95),
        max_abs_speed_per_s=max(abs_speeds, default=0.0),
        max_abs_accel_per_s2=max((abs(value) for value in accelerations), default=0.0),
        max_abs_jerk_per_s3=max((abs(value) for value in jerks), default=0.0),
        reversals=count_reversals(xs, times),
        moving_fraction=moving / len(abs_speeds) if abs_speeds else 0.0,
        crop_limit_fraction=crop_fraction,
    )


def metrics_row(shot: ShotTrajectory, metrics: TrajectoryMetrics, *, shot_index: int) -> Dict[str, Any]:
    return {
        "mode": shot.mode,
        "source": shot.source,
        "clip": Path(shot.source).stem,
        "shot_index": shot_index,
        "start_frame": shot.start_frame,
        "start_ms": shot.start_ms,
        "end_ms": shot.end_ms,
        "label": shot.label,
        "frame_count": len(shot.xs),
        **metrics.__dict__,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def summarize_detection(debug_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scores: Dict[str, List[float]] = defaultdict(list)
    areas: Dict[str, List[float]] = defaultdict(list)
    sources = Counter()
    frame_labels = Counter()
    states = Counter()
    evidence = Counter()
    regimes = Counter()
    detection_updates = 0
    for row in debug_rows:
        if row.get("detection_update"):
            detection_updates += 1
            for detection in row.get("detections") or []:
                label = str(detection.get("label", "unknown"))
                box = detection.get("box") or [0.0, 0.0, 0.0, 0.0]
                area = max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))
                scores[label].append(float(detection.get("score", 0.0)))
                areas[label].append(area)
                sources[str(detection.get("source", "unknown"))] += 1
        selected = row.get("selected") or {}
        frame_labels[str(selected.get("label", "unknown"))] += 1
        states[str(selected.get("scene_state", "unknown"))] += 1
        evidence[str(selected.get("evidence_kind", "unknown"))] += 1
        regimes[str(selected.get("smoothing_regime", "unknown"))] += 1
    labels = {}
    for label in sorted(set(scores) | set(areas)):
        labels[label] = {
            "score": finite_stats(scores[label]),
            "area": finite_stats(areas[label]),
        }
    return {
        "frames": len(debug_rows),
        "detection_updates": detection_updates,
        "raw_detection_sources": dict(sources),
        "raw_labels": labels,
        "selected_frame_labels": dict(frame_labels),
        "scene_states": dict(states),
        "evidence_kinds": dict(evidence),
        "smoothing_regimes": dict(regimes),
        "ground_truth_metrics": {
            "ball_recall": "not computed: requires visible-ball ground truth",
            "false_ball_focus_rate": "not computed: requires ball identity ground truth",
            "primary_subject_inclusion": "not computed: requires narrative-subject ground truth",
            "background_face_takeovers": "not computed automatically: review frame-level QA previews",
        },
    }


def summarize_tracks(debug_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    unique_ids: Dict[str, set[str]] = defaultdict(set)
    observed_updates = Counter()
    predicted_frames = Counter()
    selected_switches = Counter()
    semantic_ball_id_transfers = Counter()
    predicted_ball_run_durations: Dict[str, List[float]] = defaultdict(list)
    last_key: Dict[str, str] = {}
    last_ball_id: Dict[str, str] = {}
    active_predicted_start: Dict[str, float] = {}
    last_time: Dict[str, float] = {}

    for row in debug_rows:
        source = str(row.get("source", ""))
        timestamp = float(row.get("time_ms", 0)) / 1000.0
        last_time[source] = timestamp
        for track in row.get("tracks") or []:
            label = str(track.get("label", "unknown"))
            track_id = str(track.get("id", ""))
            unique_ids[label].add(track_id)
            if track.get("observed"):
                observed_updates[label] += 1
            else:
                predicted_frames[label] += 1
        selected = row.get("selected") or {}
        key = str(selected.get("key", ""))
        if source in last_key and key != last_key[source]:
            selected_switches[source] += 1
        last_key[source] = key
        focus_ids = [str(item) for item in selected.get("focus_ids") or []]
        ball_ids = [item for item in focus_ids if item.startswith("ball:")]
        if key == "sports_ball" and ball_ids:
            ball_id = ball_ids[0]
            if source in last_ball_id and ball_id != last_ball_id[source]:
                semantic_ball_id_transfers[source] += 1
            last_ball_id[source] = ball_id
        predicted = selected.get("evidence_kind") == "predicted_ball"
        if predicted and source not in active_predicted_start:
            active_predicted_start[source] = timestamp
        if not predicted and source in active_predicted_start:
            predicted_ball_run_durations[source].append(timestamp - active_predicted_start.pop(source))
    for source, start in active_predicted_start.items():
        predicted_ball_run_durations[source].append(last_time.get(source, start) - start)

    return {
        "unique_track_ids_by_label": {label: len(ids) for label, ids in unique_ids.items()},
        "observed_track_updates_by_label": dict(observed_updates),
        "predicted_track_frames_by_label": dict(predicted_frames),
        "selected_key_switches_by_source": dict(selected_switches),
        "semantic_ball_numeric_id_transfers_by_source": dict(semantic_ball_id_transfers),
        "predicted_ball_run_seconds": {
            source: {**finite_stats(values), "values": [round(value, 6) for value in values]}
            for source, values in predicted_ball_run_durations.items()
        },
        "note": "Numeric track counts are diagnostics, not recall: detection ground truth is required to classify fragmentation or false tracks.",
    }



def summarize_follow_error(debug_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Measure controller and selected-box lag without claiming target correctness.

    Camera-target error is the horizontal difference between the policy's selected
    target and the emitted crop center. Selected-box observation error compares a
    selected tracked box with the nearest raw detection of the same label on
    detection-update frames. These metrics diagnose smoothing/tracker lag, but do
    not establish that the selected subject or ball is semantically correct.
    """

    target_errors: Dict[str, List[float]] = defaultdict(list)
    raw_target_errors: Dict[str, List[float]] = defaultdict(list)
    box_dx: Dict[str, List[float]] = defaultdict(list)
    box_center_distance: Dict[str, List[float]] = defaultdict(list)
    stale_box_frames = Counter()
    same_key_label_transitions = Counter()
    hold_evidence = {
        "no_evidence_hold",
        "temporary_face_or_person_loss",
        "pending_subject",
        "live_without_local_evidence",
        "held_composition",
    }

    def center(box: Sequence[float]) -> tuple[float, float]:
        return (
            0.5 * (float(box[0]) + float(box[2])),
            0.5 * (float(box[1]) + float(box[3])),
        )

    for row in debug_rows:
        selected = row.get("selected") or {}
        evidence = str(selected.get("evidence_kind", "unknown"))
        label = str(selected.get("label", "unknown"))
        regime = str(selected.get("smoothing_regime", "unknown"))
        selected_x = float(selected.get("center_x", row.get("output_x", 0.5)))
        output_x = float(row.get("output_x", 0.5))
        crop_width = min(1.0, max(0.0, float(row.get("crop_width", 1.0))))
        half = min(0.5, 0.5 * crop_width)
        legal_selected_x = min(1.0 - half, max(half, selected_x))
        error = abs(legal_selected_x - output_x)
        raw_error = abs(selected_x - output_x)
        target_errors["all"].append(error)
        target_errors[f"evidence:{evidence}"].append(error)
        target_errors[f"label:{label}"].append(error)
        target_errors[f"regime:{regime}"].append(error)
        raw_target_errors["all"].append(raw_error)
        raw_target_errors[f"evidence:{evidence}"].append(raw_error)
        raw_target_errors[f"label:{label}"].append(raw_error)
        raw_target_errors[f"regime:{regime}"].append(raw_error)

        selected_boxes = selected.get("boxes") or []
        if evidence in hold_evidence and selected_boxes:
            stale_box_frames[evidence] += 1

        if not row.get("detection_update") or not selected_boxes:
            continue
        detections_by_label: Dict[str, List[Sequence[float]]] = defaultdict(list)
        for detection in row.get("detections") or []:
            box = detection.get("box") or []
            if len(box) == 4:
                detections_by_label[str(detection.get("label", "unknown"))].append(box)
        for selected_box in selected_boxes:
            selected_values = selected_box.get("box") or []
            selected_label = str(selected_box.get("label", "unknown"))
            if len(selected_values) != 4 or not detections_by_label.get(selected_label):
                continue
            sx, sy = center(selected_values)
            nearest = min(
                detections_by_label[selected_label],
                key=lambda box: math.hypot(center(box)[0] - sx, center(box)[1] - sy),
            )
            dx, dy = center(nearest)
            box_dx[selected_label].append(abs(dx - sx))
            box_center_distance[selected_label].append(math.hypot(dx - sx, dy - sy))

    movement_runs: Dict[str, List[float]] = defaultdict(list)
    material_switch_settle: Dict[str, List[float]] = defaultdict(list)
    unresolved_material_switches = Counter()
    rows_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in debug_rows:
        rows_by_source[str(row.get("source", ""))].append(row)

    for source, source_rows in rows_by_source.items():
        source_rows = sorted(
            source_rows,
            key=lambda row: (
                int(row.get("global_time_ms", row.get("time_ms", 0))),
                int(row.get("frame", 0)),
            ),
        )

        run_start: Optional[float] = None
        run_end: Optional[float] = None
        for index in range(1, len(source_rows)):
            previous = source_rows[index - 1]
            current = source_rows[index]
            if current.get("shot_id") != previous.get("shot_id"):
                if run_start is not None and run_end is not None:
                    movement_runs[source].append(run_end - run_start)
                run_start = None
                run_end = None
                continue
            previous_time = float(previous.get("time_ms", 0)) / 1000.0
            current_time = float(current.get("time_ms", 0)) / 1000.0
            dt = max(1e-6, current_time - previous_time)
            speed = abs(
                float(current.get("output_x", 0.5))
                - float(previous.get("output_x", 0.5))
            ) / dt
            moving = speed >= 0.005
            if moving and run_start is None:
                run_start = previous_time
            if moving:
                run_end = current_time
            elif run_start is not None and run_end is not None:
                movement_runs[source].append(run_end - run_start)
                run_start = None
                run_end = None

            previous_selected = previous.get("selected") or {}
            current_selected = current.get("selected") or {}
            if (
                str(previous_selected.get("key", ""))
                == str(current_selected.get("key", ""))
                and str(previous_selected.get("label", ""))
                != str(current_selected.get("label", ""))
            ):
                same_key_label_transitions[source] += 1

        if run_start is not None and run_end is not None:
            movement_runs[source].append(run_end - run_start)

        for index in range(1, len(source_rows)):
            previous = source_rows[index - 1]
            current = source_rows[index]
            if current.get("shot_id") != previous.get("shot_id"):
                continue
            previous_selected = previous.get("selected") or {}
            current_selected = current.get("selected") or {}
            if str(previous_selected.get("key", "")) == str(
                current_selected.get("key", "")
            ):
                continue

            def legal_target(row: Mapping[str, Any]) -> float:
                selected = row.get("selected") or {}
                raw = float(selected.get("center_x", row.get("output_x", 0.5)))
                crop_width = min(
                    1.0, max(0.0, float(row.get("crop_width", 1.0)))
                )
                half = min(0.5, 0.5 * crop_width)
                return min(1.0 - half, max(half, raw))

            if abs(legal_target(current) - legal_target(previous)) < 0.08:
                continue

            start_time = float(current.get("time_ms", 0)) / 1000.0
            settled = False
            for candidate_index in range(index, min(len(source_rows), index + 120)):
                candidate = source_rows[candidate_index]
                if candidate.get("shot_id") != current.get("shot_id"):
                    break
                error = abs(
                    float(candidate.get("output_x", 0.5))
                    - legal_target(candidate)
                )
                if error > 0.02:
                    continue
                confirmation = source_rows[
                    candidate_index : min(len(source_rows), candidate_index + 3)
                ]
                if len(confirmation) < 2:
                    continue
                if all(
                    row.get("shot_id") == current.get("shot_id")
                    and abs(
                        float(row.get("output_x", 0.5)) - legal_target(row)
                    )
                    <= 0.025
                    for row in confirmation
                ):
                    end_time = float(candidate.get("time_ms", 0)) / 1000.0
                    material_switch_settle[source].append(end_time - start_time)
                    settled = True
                    break
            if not settled:
                unresolved_material_switches[source] += 1

    return {
        "camera_to_selected_target_abs_x": {
            key: finite_stats(values) for key, values in sorted(target_errors.items())
        },
        "camera_to_raw_unclamped_selected_target_abs_x": {
            key: finite_stats(values)
            for key, values in sorted(raw_target_errors.items())
        },
        "selected_track_box_to_nearest_raw_detection_abs_x": {
            key: finite_stats(values) for key, values in sorted(box_dx.items())
        },
        "selected_track_box_to_nearest_raw_detection_center_distance": {
            key: finite_stats(values)
            for key, values in sorted(box_center_distance.items())
        },
        "held_frames_with_stale_selected_boxes": dict(stale_box_frames),
        "camera_movement_episode_seconds": {
            source: finite_stats(values)
            for source, values in sorted(movement_runs.items())
        },
        "material_target_switch_settle_seconds": {
            source: finite_stats(values)
            for source, values in sorted(material_switch_settle.items())
        },
        "unresolved_material_target_switches": dict(
            unresolved_material_switches
        ),
        "same_key_label_transitions_by_source": dict(
            same_key_label_transitions
        ),
        "note": (
            "Camera lag uses the legal crop-center target after edge clamping. "
            "Raw unclamped target error is reported separately. These are lag "
            "diagnostics, not semantic accuracy or ground-truth inclusion metrics."
        ),
    }

def focus_summary(shots: Sequence[ShotTrajectory], debug_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    shot_labels = Counter(shot.label for shot in shots)
    frame_labels = Counter()
    states = Counter()
    evidence = Counter()
    regimes = Counter()
    for row in debug_rows:
        selected = row.get("selected") or {}
        frame_labels[str(selected.get("label", "unknown"))] += 1
        states[str(selected.get("scene_state", "unknown"))] += 1
        evidence[str(selected.get("evidence_kind", "unknown"))] += 1
        regimes[str(selected.get("smoothing_regime", "unknown"))] += 1
    return {
        "shot_dominant_labels": dict(shot_labels),
        "frame_level_labels": dict(frame_labels),
        "frame_level_scene_states": dict(states),
        "frame_level_evidence": dict(evidence),
        "frame_level_smoothing_regimes": dict(regimes),
    }


def aggregate_clip_rows(shot_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in shot_rows:
        grouped[(str(row["mode"]), str(row["source"]))].append(row)
    result: List[Dict[str, Any]] = []
    for (mode, source), rows in sorted(grouped.items()):
        frame_count = sum(int(row["frame_count"]) for row in rows)
        moving_denominator = sum(max(0, int(row["frame_count"]) - 1) for row in rows)
        crop_rows = [row for row in rows if row["crop_limit_fraction"] is not None]
        crop_weight = sum(int(row["frame_count"]) for row in crop_rows)
        result.append({
            "mode": mode,
            "source": source,
            "clip": Path(source).stem,
            "shot_count": len(rows),
            "frame_count": frame_count,
            "x_min": min(float(row["x_min"]) for row in rows),
            "x_max": max(float(row["x_max"]) for row in rows),
            "x_range": max(float(row["x_max"]) for row in rows) - min(float(row["x_min"]) for row in rows),
            "path": sum(float(row["path"]) for row in rows),
            "mean_abs_speed_per_s": (
                sum(float(row["mean_abs_speed_per_s"]) * max(0, int(row["frame_count"]) - 1) for row in rows)
                / moving_denominator if moving_denominator else 0.0
            ),
            "p95_abs_speed_per_s": max(float(row["p95_abs_speed_per_s"]) for row in rows),
            "max_abs_speed_per_s": max(float(row["max_abs_speed_per_s"]) for row in rows),
            "max_abs_accel_per_s2": max(float(row["max_abs_accel_per_s2"]) for row in rows),
            "max_abs_jerk_per_s3": max(float(row["max_abs_jerk_per_s3"]) for row in rows),
            "reversals": sum(int(row["reversals"]) for row in rows),
            "moving_fraction": (
                sum(float(row["moving_fraction"]) * max(0, int(row["frame_count"]) - 1) for row in rows)
                / moving_denominator if moving_denominator else 0.0
            ),
            "crop_limit_fraction": (
                sum(float(row["crop_limit_fraction"]) * int(row["frame_count"]) for row in crop_rows) / crop_weight
                if crop_weight else None
            ),
        })
    return result


def source_frame_counts(
    shots: Sequence[ShotTrajectory],
) -> Dict[tuple[str, str], int]:
    counts = Counter()
    for shot in shots:
        counts[(shot.mode, shot.source)] += len(shot.xs)
    return dict(counts)


def build_acceptance_report(
    compact: Mapping[str, Dict[str, Any]],
    shots: Sequence[ShotTrajectory],
    debug_rows: Sequence[Dict[str, Any]],
    clip_rows: Sequence[Mapping[str, Any]],
    follow_error: Mapping[str, Dict[str, Any]],
) -> str:
    expected_tracks = {"vertical_video", "focus", "focus_bbox"}
    lines = [
        "# YOLO26 v5 automated acceptance report",
        "",
        "This report contains only checks computable without human or annotated ground truth. Ball identity, visible-ball inclusion, narrative-primary-subject inclusion, and background-face takeover gates remain manual/annotated-video checks.",
        "",
        "## Structural checks",
        "",
    ]
    overall_pass = True
    for mode in ("sports", "movie"):
        summary = compact[mode]
        tracks = set(summary.get("tracks", {}))
        tracks_ok = tracks == expected_tracks
        order_ok = not summary.get("tag_after_terminal_sources")
        mode_shots = [shot for shot in shots if shot.mode == mode]
        xs_ok = all(shot.xs and all(0.0 <= value <= 1.0 for value in shot.xs) for shot in mode_shots)
        overall_pass = overall_pass and tracks_ok and order_ok and xs_ok
        lines.extend([
            f"- **{mode} tracks exactly preserved:** {'PASS' if tracks_ok else 'FAIL'} — {sorted(tracks)}",
            f"- **{mode} tags precede terminal messages:** {'PASS' if order_ok else 'FAIL'}",
            f"- **{mode} X arrays non-empty and normalized:** {'PASS' if xs_ok else 'FAIL'}",
        ])

    compact_counts = source_frame_counts(shots)
    debug_counts = Counter(
        (str(row.get("mode", "")), str(row.get("source", "")))
        for row in debug_rows
    )
    count_checks = []
    for (mode, source), count in sorted(compact_counts.items()):
        matching_debug = next(
            (
                value
                for (debug_mode, debug_source), value in debug_counts.items()
                if debug_mode == mode and same_source(debug_source, source)
            ),
            0,
        )
        count_checks.append(
            (mode, source, count, matching_debug, count == matching_debug)
        )
    frame_count_ok = all(item[4] for item in count_checks)
    overall_pass = overall_pass and frame_count_ok
    lines.append(
        f"- **Compact X count matches debug frame count:** "
        f"{'PASS' if frame_count_ok else 'FAIL'}"
    )
    for mode, source, compact_count, debug_count, ok in count_checks:
        lines.append(
            f"  - `{mode}/{Path(source).name}`: compact={compact_count}, "
            f"debug={debug_count}, {'PASS' if ok else 'FAIL'}"
        )

    locked_deltas: List[float] = []
    previous: Dict[tuple[str, str], tuple[float, str]] = {}
    for row in debug_rows:
        source = str(row.get("source", ""))
        shot_id = str(row.get("shot_id", ""))
        selected = row.get("selected") or {}
        regime = str(selected.get("smoothing_regime", ""))
        x = float(row.get("output_x", 0.5))
        key = (source, shot_id)
        if key in previous and regime == "locked" and previous[key][1] == "locked":
            locked_deltas.append(abs(x - previous[key][0]))
        previous[key] = (x, regime)
    max_locked_delta = max(locked_deltas, default=0.0)
    locked_ok = max_locked_delta <= 1e-9
    overall_pass = overall_pass and locked_ok
    lines.append(f"- **Consecutive locked frames have zero output drift:** {'PASS' if locked_ok else 'FAIL'} — max Δ={max_locked_delta:.9f}")

    lines.extend([
        "",
        f"**Automated structural result: {'PASS' if overall_pass else 'FAIL'}**",
        "",
        "## Trajectory summary",
        "",
        "| Mode | Clip | Shots | X range | Path | Moving | Reversals | Crop-limit occupancy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in clip_rows:
        crop = row["crop_limit_fraction"]
        crop_text = "n/a" if crop is None else f"{100.0 * float(crop):.2f}%"
        lines.append(
            f"| {row['mode']} | {row['clip']} | {row['shot_count']} | {float(row['x_range']):.6f} | {float(row['path']):.6f} | {100.0 * float(row['moving_fraction']):.2f}% | {row['reversals']} | {crop_text} |"
        )

    lines.extend([
        "",
        "## Follow-lag diagnostics",
        "",
        "These values measure lag relative to the policy-selected target, not whether that target is semantically correct.",
        "",
        "| Mode | Selected evidence | Median crop-target error | P95 crop-target error | Held frames with stale boxes |",
        "|---|---|---:|---:|---:|",
    ])
    for mode in ("sports", "movie"):
        mode_summary = follow_error.get(mode, {})
        target_summary = mode_summary.get("camera_to_selected_target_abs_x", {})
        preferred = (
            ("evidence:observed_ball", "observed_ball")
            if mode == "sports"
            else ("evidence:frontal_face", "frontal_face")
        )
        stats = target_summary.get(preferred[0]) or target_summary.get("all") or {}
        stale = sum(
            int(value)
            for value in (
                mode_summary.get("held_frames_with_stale_selected_boxes", {}) or {}
            ).values()
        )
        lines.append(
            f"| {mode} | {preferred[1]} | {float(stats.get('median', 0.0)):.6f} | "
            f"{float(stats.get('p95', 0.0)):.6f} | {stale} |"
        )

    lines.extend([
        "",
        "## Ground-truth-dependent gates",
        "",
        "Pending visual or annotated evaluation on the side-by-side and pure-vertical previews:",
        "",
        "- Sports visible-ball-in-crop rate, crop-to-ball error, false-ball focus duration, and reacquisition latency.",
        "- Movie primary-subject inclusion, unwanted switches, unexplained dialogue movement, and background/poster/head takeover count.",
        "- Cut detection precision/recall on real footage.",
        "",
        "No performance claim should be made from this automated report alone.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sports-jsonl", type=Path, required=True)
    parser.add_argument("--movie-jsonl", type=Path, required=True)
    parser.add_argument("--sports-debug", type=Path, required=True)
    parser.add_argument("--movie-debug", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_shots: List[ShotTrajectory] = []
    all_debug: List[Dict[str, Any]] = []
    compact_summaries: Dict[str, Dict[str, Any]] = {}
    per_mode_debug: Dict[str, List[Dict[str, Any]]] = {}
    for mode, compact_path, debug_path in (
        ("sports", args.sports_jsonl, args.sports_debug),
        ("movie", args.movie_jsonl, args.movie_debug),
    ):
        shots, compact_summary = load_compact(mode, compact_path)
        debug = load_debug(mode, debug_path)
        all_shots.extend(shots)
        all_debug.extend(debug)
        compact_summaries[mode] = compact_summary
        per_mode_debug[mode] = debug

    debug_times: Dict[tuple[str, str, int], float] = {}
    for row in all_debug:
        debug_times[
            (
                str(row.get("mode", "")),
                str(row.get("source", "")),
                int(row.get("frame", 0)),
            )
        ] = float(row.get("time_ms", 0)) / 1000.0

    shot_rows: List[Dict[str, Any]] = []
    shot_indexes = Counter()
    limit_cache: Dict[str, Optional[tuple[float, float]]] = {}
    for shot in sorted(all_shots, key=lambda item: (item.mode, item.source, item.start_frame)):
        key = (shot.mode, shot.source)
        index = shot_indexes[key]
        shot_indexes[key] += 1
        limits = limit_cache.setdefault(shot.source, source_crop_limits(shot.source))
        times = times_for_shot(shot, debug_times)
        shot_rows.append(metrics_row(shot, trajectory_metrics(shot.xs, times, limits), shot_index=index))
    clip_rows = aggregate_clip_rows(shot_rows)
    write_csv(args.output_dir / "trajectory_by_shot.csv", shot_rows)
    write_csv(args.output_dir / "trajectory_by_clip.csv", clip_rows)

    detection_summary = {
        mode: summarize_detection(per_mode_debug[mode]) for mode in ("sports", "movie")
    }
    track_summary = {
        mode: summarize_tracks(per_mode_debug[mode]) for mode in ("sports", "movie")
    }
    focus_labels = {
        mode: focus_summary(
            [shot for shot in all_shots if shot.mode == mode], per_mode_debug[mode]
        )
        for mode in ("sports", "movie")
    }
    follow_error = {
        mode: summarize_follow_error(per_mode_debug[mode])
        for mode in ("sports", "movie")
    }
    (args.output_dir / "detection_summary.json").write_text(
        json.dumps(detection_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "track_continuity.json").write_text(
        json.dumps(track_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "focus_label_summary.json").write_text(
        json.dumps(focus_labels, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "follow_error_summary.json").write_text(
        json.dumps(follow_error, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = build_acceptance_report(
        compact_summaries, all_shots, all_debug, clip_rows, follow_error
    )
    (args.output_dir / "acceptance_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote metrics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
