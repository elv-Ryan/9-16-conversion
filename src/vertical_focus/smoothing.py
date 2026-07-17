from __future__ import annotations

import math

from .geometry import clamp, clamp_crop_center


class SmoothCamera:
    """Causal, shot-local, acceleration-limited camera trajectory.

    The output is a normalized X center. There is no Y movement and no zoom.
    """

    def __init__(
        self,
        *,
        crop_width: float,
        response_time_s: float,
        deadband: float,
        max_speed_per_s: float,
        max_accel_per_s2: float,
    ) -> None:
        self.crop_width = float(crop_width)
        self.response_time_s = max(0.05, float(response_time_s))
        self.deadband = max(0.0, float(deadband))
        self.max_speed = max(1e-4, float(max_speed_per_s))
        self.max_accel = max(1e-4, float(max_accel_per_s2))
        self.x = 0.5
        self.v = 0.0
        self.initialized = False

    def reset(self, target_x: float) -> float:
        self.x = clamp_crop_center(target_x, self.crop_width)
        self.v = 0.0
        self.initialized = True
        return self.x

    def update(self, target_x: float, dt_s: float) -> float:
        target = clamp_crop_center(target_x, self.crop_width)
        if not self.initialized:
            return self.reset(target)

        dt = clamp(float(dt_s), 1.0 / 240.0, 0.25)
        error = target - self.x
        if abs(error) <= self.deadband:
            target = self.x
            error = 0.0

        # Critically damped second-order response, bounded in physical units.
        omega = 2.0 / self.response_time_s
        acceleration = (omega * omega * error) - (2.0 * omega * self.v)
        acceleration = clamp(acceleration, -self.max_accel, self.max_accel)
        next_v = clamp(self.v + acceleration * dt, -self.max_speed, self.max_speed)
        next_x = self.x + next_v * dt

        # Never overshoot the requested focus point.
        if error != 0.0 and (target - next_x) * error <= 0.0:
            next_x = target
            next_v = 0.0

        bounded = clamp_crop_center(next_x, self.crop_width)
        if bounded != next_x:
            next_v = 0.0

        self.x = bounded
        self.v = next_v
        return self.x
