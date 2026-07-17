from __future__ import annotations

from .geometry import clamp, clamp_crop_center


class SmoothCamera:
    """Causal horizontal camera controller with an explicit fast-catchup mode.

    The output is a normalized X center. There is no Y movement and no zoom.
    When the target enters the deadband, velocity is set to zero immediately so
    the crop cannot continue drifting after the evidence has stabilized.
    """

    def __init__(
        self,
        *,
        crop_width: float,
        response_time_s: float,
        deadband: float,
        max_speed_per_s: float,
        max_accel_per_s2: float,
        fast_response_time_s: float | None = None,
        fast_max_speed_per_s: float | None = None,
        fast_max_accel_per_s2: float | None = None,
        fast_error_threshold: float = 0.08,
        fast_boost_s: float = 0.30,
    ) -> None:
        self.crop_width = float(crop_width)
        self.response_time_s = max(0.05, float(response_time_s))
        self.deadband = max(0.0, float(deadband))
        self.max_speed = max(1e-4, float(max_speed_per_s))
        self.max_accel = max(1e-4, float(max_accel_per_s2))
        self.fast_response_time_s = max(
            0.04,
            float(fast_response_time_s if fast_response_time_s is not None else self.response_time_s * 0.5),
        )
        self.fast_max_speed = max(
            self.max_speed,
            float(fast_max_speed_per_s if fast_max_speed_per_s is not None else self.max_speed * 2.0),
        )
        self.fast_max_accel = max(
            self.max_accel,
            float(fast_max_accel_per_s2 if fast_max_accel_per_s2 is not None else self.max_accel * 2.5),
        )
        self.fast_error_threshold = max(self.deadband, float(fast_error_threshold))
        self.fast_boost_s = max(0.0, float(fast_boost_s))
        self._fast_remaining_s = 0.0
        self.x = 0.5
        self.v = 0.0
        self.initialized = False

    def reset(self, target_x: float) -> float:
        self.x = clamp_crop_center(target_x, self.crop_width)
        self.v = 0.0
        self._fast_remaining_s = 0.0
        self.initialized = True
        return self.x

    def hold(self) -> float:
        """Freeze the crop at its current position with no residual velocity."""
        self.v = 0.0
        self._fast_remaining_s = 0.0
        return self.x

    def update(self, target_x: float, dt_s: float, *, urgent: bool = False) -> float:
        target = clamp_crop_center(target_x, self.crop_width)
        if not self.initialized:
            return self.reset(target)

        dt = clamp(float(dt_s), 1.0 / 240.0, 0.25)
        error = target - self.x

        if abs(error) <= self.deadband:
            self.v = 0.0
            self._fast_remaining_s = 0.0
            return self.x

        if urgent or abs(error) >= self.fast_error_threshold:
            self._fast_remaining_s = max(self._fast_remaining_s, self.fast_boost_s)

        fast = self._fast_remaining_s > 0.0
        response_time = self.fast_response_time_s if fast else self.response_time_s
        max_speed = self.fast_max_speed if fast else self.max_speed
        max_accel = self.fast_max_accel if fast else self.max_accel

        # Command a bounded velocity toward the target, then acceleration-limit
        # the change in velocity. This is predictable, causal, and does not
        # continue drifting when the target becomes stationary.
        desired_v = clamp(error / response_time, -max_speed, max_speed)
        max_delta_v = max_accel * dt
        next_v = self.v + clamp(desired_v - self.v, -max_delta_v, max_delta_v)
        next_x = self.x + next_v * dt

        # Never overshoot the requested focus point.
        if (target - next_x) * error <= 0.0:
            next_x = target
            next_v = 0.0

        bounded = clamp_crop_center(next_x, self.crop_width)
        if bounded != next_x:
            next_v = 0.0

        self.x = bounded
        self.v = next_v
        self._fast_remaining_s = max(0.0, self._fast_remaining_s - dt)
        return self.x
