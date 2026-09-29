from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np


STATIC_FAMILIES = {"graphic_text_lock", "split_screen", "static_composition", "safe_center"}

# Two people in frame produce two boxes of the same family, and the detector's
# preference between them is not stable, so the raw X series alternates. A
# continuously-smoothed trajectory chases that alternation and the crop slides
# back and forth across the frame -- which is what an interview looks like when
# it goes wrong. The answer is not more smoothing: it is to decide *who* the
# shot is on, hold that framing still, and cut when the subject really changes.
#
# How far X may wander and still be the same subject, as a fraction of the crop
# width: inside a quarter of a crop the subject has not left the frame.
HOLD_SAME_SUBJECT_FRACTION = 0.25
HOLD_MIN_TOLERANCE = 0.02
# A challenger must carry this multiple of the incumbent's weight inside the
# dwell window before the framing cuts to it. At 1.0 the framing would follow
# every momentary majority, which is the oscillation this exists to stop.
HOLD_SWITCH_RATIO = 1.5


def legal_crop_geometry(width: int, height: int, target_width_over_height: float) -> tuple[float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    crop_width_norm = min(1.0, height * target_width_over_height / width)
    half = 0.5 * crop_width_norm
    return crop_width_norm, half, 1.0 - half


def _fill_missing(values: Sequence[Optional[float]]) -> np.ndarray:
    count = len(values)
    if count == 0:
        return np.empty((0,), dtype=np.float64)
    valid_indices = [index for index, value in enumerate(values) if value is not None and np.isfinite(value)]
    if not valid_indices:
        return np.full((count,), 0.5, dtype=np.float64)
    valid_values = [float(values[index]) for index in valid_indices]
    return np.interp(np.arange(count), valid_indices, valid_values)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Confidence-weighted median: the value that splits the weight in half."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0.0:
        return float(np.median(values))
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def _subject_clusters(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Label each sample with the subject it belongs to.

    Single-linkage over the values: two samples are the same subject while
    successive sorted values sit within ``tolerance`` of each other. Two people
    standing apart give two labels no matter how often the detector alternates
    between them, which is the property the run-length view lacked.
    """
    count = values.size
    labels = np.zeros(count, dtype=np.int64)
    if count == 0:
        return labels
    order = np.argsort(values, kind="stable")
    label = 0
    labels[order[0]] = 0
    for previous, index in zip(order, order[1:]):
        if float(values[index] - values[previous]) > tolerance:
            label += 1
        labels[index] = label
    return labels


def _held_subject(
    labels: np.ndarray, weights: np.ndarray, min_dwell: int, switch_ratio: float
) -> np.ndarray:
    """Which subject the framing sits on at each sample.

    A weighted vote over a window of ``min_dwell`` samples centred on each
    position, with hysteresis: the incumbent keeps the framing unless a
    challenger carries ``switch_ratio`` times its weight in that window. When
    the detector alternates between two people every sample or two, neither
    ever clears the bar and the framing simply stays put -- which is the whole
    point. A speaker who actually takes over holds the window outright within
    half a dwell and the framing cuts to them.
    """
    count = labels.size
    held = np.zeros(count, dtype=np.int64)
    if count == 0:
        return held
    half = max(1, min_dwell // 2)

    def tally(start: int, end: int) -> dict:
        totals: dict = {}
        for label, weight in zip(labels[start:end], weights[start:end]):
            totals[int(label)] = totals.get(int(label), 0.0) + float(weight)
        return totals

    opening = tally(0, min(count, min_dwell))
    current = max(opening, key=lambda key: (opening[key], -key))
    for index in range(count):
        totals = tally(max(0, index - half), min(count, index + half + 1))
        challenger = max(totals, key=lambda key: (totals[key], -key))
        if challenger != current:
            incumbent = totals.get(current, 0.0)
            if totals[challenger] > incumbent * switch_ratio:
                current = challenger
        held[index] = current
    return held


def _constant_runs(held: np.ndarray) -> List[tuple[int, int]]:
    """Maximal spans over which the framing stays on the same subject."""
    if held.size == 0:
        return []
    bounds = [0]
    for index in range(1, held.size):
        if held[index] != held[index - 1]:
            bounds.append(index)
    bounds.append(held.size)
    return [(start, end) for start, end in zip(bounds, bounds[1:]) if end > start]


def _median_filter(values: np.ndarray, radius: int = 1) -> np.ndarray:
    if values.size <= 2 or radius <= 0:
        return values.copy()
    output = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        output[index] = float(np.median(values[start:end]))
    return output


def _bidirectional_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if values.size <= 1:
        return values.copy()
    alpha = min(1.0, max(0.001, float(alpha)))
    forward = values.copy()
    for index in range(1, values.size):
        forward[index] = alpha * values[index] + (1.0 - alpha) * forward[index - 1]
    backward = values.copy()
    for index in range(values.size - 2, -1, -1):
        backward[index] = alpha * values[index] + (1.0 - alpha) * backward[index + 1]
    return 0.5 * (forward + backward)


def _speed_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    if values.size <= 1:
        return values.copy()
    output = values.copy()
    for index in range(1, values.size):
        delta = output[index] - output[index - 1]
        output[index] = output[index - 1] + min(max(delta, -max_step), max_step)
    for index in range(values.size - 2, -1, -1):
        delta = output[index] - output[index + 1]
        output[index] = output[index + 1] + min(max(delta, -max_step), max_step)
    return output


def _speaker_segments(values: np.ndarray, jump_threshold: float = 0.18) -> List[tuple[int, int]]:
    if values.size < 5:
        return [(0, values.size)]
    boundaries = [0]
    index = 2
    while index < values.size - 2:
        left = float(np.median(values[max(0, index - 3):index]))
        right_window = values[index:min(values.size, index + 3)]
        right = float(np.median(right_window))
        stable = float(np.max(right_window) - np.min(right_window)) <= 0.10
        if stable and abs(right - left) >= jump_threshold:
            boundaries.append(index)
            index += 2
        index += 1
    boundaries.append(values.size)
    return [(start, end) for start, end in zip(boundaries, boundaries[1:]) if end > start]


def smooth_samples(
    *,
    family: str,
    raw_x: Sequence[Optional[float]],
    confidences: Sequence[float],
    inference_fps: float,
    legal_min: float,
    legal_max: float,
    hold_and_cut: bool = False,
    min_hold_seconds: float = 0.7,
) -> np.ndarray:
    values = _fill_missing(raw_x)
    if values.size == 0:
        return values
    values = np.clip(values, legal_min, legal_max)

    if family in STATIC_FAMILIES:
        valid = [
            (float(value), max(0.001, float(confidence)))
            for value, confidence in zip(values, confidences)
        ]
        expanded = np.asarray([value for value, _ in valid], dtype=np.float64)
        weights = np.asarray([weight for _, weight in valid], dtype=np.float64)
        locked = _weighted_median(expanded, weights)
        return np.full(values.shape, np.clip(locked, legal_min, legal_max), dtype=np.float64)

    if hold_and_cut:
        # Hold the framing on one subject, then cut. No continuous smoothing
        # across a change of subject: a talking head does not need the crop to
        # breathe with it, and any drift here reads as the camera wandering.
        # Deliberately on the unfiltered series. _median_filter exists to
        # despike a continuous signal, and at the array edges its two-sample
        # window returns a mean -- which invents a position halfway between
        # two speakers that nobody is standing at. The dwell does the
        # despiking here, and it does it by ignoring outliers rather than
        # averaging them in.
        crop_width = 2.0 * legal_min
        tolerance = max(HOLD_MIN_TOLERANCE, HOLD_SAME_SUBJECT_FRACTION * crop_width)
        min_dwell = max(2, int(round(min_hold_seconds * max(1.0, inference_fps))))
        weights = np.asarray(
            [max(0.001, float(confidence)) for confidence in confidences],
            dtype=np.float64,
        )
        if weights.size != values.size:
            weights = np.full(values.shape, 0.001, dtype=np.float64)

        labels = _subject_clusters(values, tolerance)
        held = _held_subject(labels, weights, min_dwell, HOLD_SWITCH_RATIO)
        smoothed = values.copy()
        for start, end in _constant_runs(held):
            subject = held[start]
            mine = np.flatnonzero(labels[start:end] == subject) + start
            if mine.size == 0:
                mine = np.arange(start, end)
            window = values[mine]
            if float(window.max() - window.min()) <= tolerance:
                # Stationary subject: one framing for the whole dwell. Samples
                # belonging to the *other* person are ignored rather than
                # averaged in -- they are what the framing is refusing to
                # follow.
                smoothed[start:end] = _weighted_median(window, weights[mine])
            else:
                # This subject is genuinely crossing the frame, so follow it --
                # but only inside the dwell, so the cut at its edge stays a cut,
                # and only along its own samples. A box belonging to the other
                # person is interpolated over rather than chased: it is not a
                # position this subject was ever at. (Without this the framing
                # lunges at a single stray box, which is the same failure the
                # hold is here to prevent, only briefer.)
                own = np.full(end - start, np.nan, dtype=np.float64)
                own[mine - start] = values[mine]
                series = _fill_missing(
                    [None if np.isnan(value) else float(value) for value in own]
                )
                smoothed[start:end] = _bidirectional_ema(
                    _median_filter(series, radius=1), alpha=0.55
                )
        return np.clip(smoothed, legal_min, legal_max)

    filtered = _median_filter(values, radius=1)
    if family == "gameplay_follow":
        smoothed = _bidirectional_ema(filtered, alpha=0.42)
        max_step = 1.25 / max(1.0, inference_fps)
        smoothed = _speed_limit(smoothed, max_step=max_step)
    elif family == "active_speaker":
        smoothed = filtered.copy()
        for start, end in _speaker_segments(filtered):
            smoothed[start:end] = _bidirectional_ema(filtered[start:end], alpha=0.70)
    else:
        smoothed = _bidirectional_ema(filtered, alpha=0.55)
        max_step = 0.90 / max(1.0, inference_fps)
        smoothed = _speed_limit(smoothed, max_step=max_step)
    return np.clip(smoothed, legal_min, legal_max)


def expand_to_source_frames(
    *,
    sample_frame_indices: Sequence[int],
    sample_x: Sequence[float],
    start_frame: int,
    end_frame: int,
    legal_min: float,
    legal_max: float,
    sample_events: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Botics frame geometry, with optional v4 speaker-jump boundaries.

    Legacy callers without events retain their exact interpolation behavior.
    Event 2 holds the left sample until the jump frame, then uses the new X.
    """
    if sample_events is not None:
        if len(sample_events) != len(sample_frame_indices) or len(sample_events) != len(sample_x):
            raise ValueError("sample_events, sample_frame_indices and sample_x must align")
        indices = np.asarray(sample_frame_indices, dtype=float)
        values = np.asarray(sample_x, dtype=float)
        if not np.isfinite(indices).all() or not np.isfinite(values).all():
            raise ValueError("Reviewed reconstruction requires finite samples")
        if len(indices) > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError("Reviewed sample frame indices must strictly increase")
    frame_count = max(0, end_frame - start_frame)
    if frame_count == 0:
        return np.empty((0,), dtype=np.float64)
    # sample_x is normally a NumPy array from smooth_samples().  Never use
    # NumPy arrays in boolean context: multi-element arrays raise ValueError.
    if len(sample_frame_indices) == 0 or np.asarray(sample_x).size == 0:
        return np.full((frame_count,), np.clip(0.5, legal_min, legal_max), dtype=np.float64)
    x_axis = np.asarray(sample_frame_indices, dtype=np.float64)
    y_axis = np.asarray(sample_x, dtype=np.float64)
    target = np.arange(start_frame, end_frame, dtype=np.float64)
    expanded = np.interp(target, x_axis, y_axis)
    if sample_events is not None and len(x_axis) > 1:
        hi = np.searchsorted(x_axis, target, side="right")
        interior = (hi > 0) & (hi < len(x_axis))
        positions = np.flatnonzero(interior)
        if len(positions):
            right = hi[positions]
            jumps = np.asarray(sample_events, dtype=int)[right] == 2
            use = positions[jumps]
            expanded[use] = y_axis[hi[use] - 1]
    return np.clip(expanded, legal_min, legal_max)
