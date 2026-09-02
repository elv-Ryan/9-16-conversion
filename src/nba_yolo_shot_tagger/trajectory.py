from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np


STATIC_FAMILIES = {"graphic_text_lock", "split_screen", "static_composition", "safe_center"}


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
        order = np.argsort(expanded)
        cumulative = np.cumsum(weights[order])
        locked = float(expanded[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])
        return np.full(values.shape, np.clip(locked, legal_min, legal_max), dtype=np.float64)

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
) -> np.ndarray:
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
    return np.clip(expanded, legal_min, legal_max)
