from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

from .geometry import clamp, clamp_crop_center


class SmoothCamera:
    """Causal horizontal camera with explicit hold, follow, and reframe modes.

    The legacy acceleration-limited controller remains available for callers
    without a named mode. V5.3 named profiles use either:

    * ``follow``: non-inertial exponential tracking with a bounded per-frame
      step. It has no residual velocity, so a stopped target cannot cause a
      trailing pan.
    * ``reframe``: a finite-duration quintic move to a frozen destination. The
      move starts and ends at zero velocity, then stops exactly.

    Both modes support a composition zone. Targets inside that zone cause an
    exact hold instead of continuous micro-corrections.
    """

    _PROFILE_MODES = frozenset({"legacy", "follow", "reframe"})

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
        target_filter_s: float = 0.14,
        fast_on_large_error: bool = False,
        profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.crop_width = float(crop_width)
        self.response_time_s = max(0.05, float(response_time_s))
        self.deadband = max(0.0, float(deadband))
        self.max_speed = max(1e-4, float(max_speed_per_s))
        self.max_accel = max(1e-4, float(max_accel_per_s2))
        self.fast_response_time_s = max(
            0.04,
            float(
                fast_response_time_s
                if fast_response_time_s is not None
                else self.response_time_s * 0.5
            ),
        )
        self.fast_max_speed = max(
            self.max_speed,
            float(
                fast_max_speed_per_s
                if fast_max_speed_per_s is not None
                else self.max_speed * 2.0
            ),
        )
        self.fast_max_accel = max(
            self.max_accel,
            float(
                fast_max_accel_per_s2
                if fast_max_accel_per_s2 is not None
                else self.max_accel * 2.5
            ),
        )
        self.fast_error_threshold = max(self.deadband, float(fast_error_threshold))
        self.fast_boost_s = max(0.0, float(fast_boost_s))
        self.target_filter_s = max(0.0, float(target_filter_s))
        self.fast_on_large_error = bool(fast_on_large_error)
        self.profiles: Dict[str, Dict[str, Any]] = {}
        for name, values in (profiles or {}).items():
            self.profiles[str(name)] = self._normalize_profile(values)

        self._fast_remaining_s = 0.0
        self._reframe: Optional[Dict[str, Any]] = None
        self.x = 0.5
        self.v = 0.0
        self.filtered_target = 0.5
        self.initialized = False

    def _normalize_profile(self, values: Mapping[str, Any]) -> Dict[str, Any]:
        mode = str(values.get("mode", "legacy")).strip().lower()
        if mode not in self._PROFILE_MODES:
            raise ValueError(f"unsupported camera profile mode: {mode!r}")
        minimum = max(0.04, float(values.get("reframe_min_seconds", 0.18)))
        maximum = max(
            minimum, float(values.get("reframe_max_seconds", max(0.30, minimum)))
        )
        return {
            "mode": mode,
            "response_time_seconds": max(
                0.02,
                float(values.get("response_time_seconds", self.response_time_s)),
            ),
            "deadband": max(0.0, float(values.get("deadband", self.deadband))),
            "max_speed_normalized_per_second": max(
                1e-4,
                float(
                    values.get(
                        "max_speed_normalized_per_second", self.max_speed
                    )
                ),
            ),
            "max_acceleration_normalized_per_second2": max(
                1e-4,
                float(
                    values.get(
                        "max_acceleration_normalized_per_second2", self.max_accel
                    )
                ),
            ),
            "target_filter_seconds": max(
                0.0,
                float(values.get("target_filter_seconds", self.target_filter_s)),
            ),
            "tracking_zone": max(0.0, float(values.get("tracking_zone", 0.0))),
            "snap_epsilon": max(
                0.0,
                float(values.get("snap_epsilon", values.get("deadband", self.deadband))),
            ),
            "reframe_min_seconds": minimum,
            "reframe_max_seconds": maximum,
            "reframe_speed_normalized_per_second": max(
                1e-4,
                float(values.get("reframe_speed_normalized_per_second", 1.25)),
            ),
            "retarget_threshold": max(
                0.0, float(values.get("retarget_threshold", 0.08))
            ),
        }

    @property
    def reframing(self) -> bool:
        return self._reframe is not None

    @property
    def reframe_target(self) -> Optional[float]:
        return None if self._reframe is None else float(self._reframe["target"])

    def profile_deadband(self, profile_name: Optional[str]) -> float:
        if profile_name and profile_name in self.profiles:
            profile = self.profiles[profile_name]
            return max(float(profile["deadband"]), float(profile["tracking_zone"]))
        return self.deadband

    def _clear_reframe(self) -> None:
        self._reframe = None

    def reset(self, target_x: float) -> float:
        self.x = clamp_crop_center(target_x, self.crop_width)
        self.filtered_target = self.x
        self.v = 0.0
        self._fast_remaining_s = 0.0
        self._clear_reframe()
        self.initialized = True
        return self.x

    def hold(self) -> float:
        self.v = 0.0
        self.filtered_target = self.x
        self._fast_remaining_s = 0.0
        self._clear_reframe()
        return self.x

    def _zone_destination(self, raw_target: float, zone: float) -> float:
        error = raw_target - self.x
        if abs(error) <= zone:
            return self.x
        return clamp_crop_center(
            raw_target - math.copysign(zone, error), self.crop_width
        )

    @staticmethod
    def _quintic_smoothstep(value: float) -> float:
        t = clamp(float(value), 0.0, 1.0)
        return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))

    def _update_follow(
        self, raw_target: float, dt: float, profile: Mapping[str, Any]
    ) -> float:
        self._clear_reframe()
        zone = float(profile["tracking_zone"])
        destination = self._zone_destination(raw_target, zone)
        error = destination - self.x
        deadband = float(profile["deadband"])
        if abs(error) <= deadband:
            return self.hold()

        filter_s = float(profile["target_filter_seconds"])
        if filter_s <= 0.0:
            self.filtered_target = destination
        else:
            alpha = 1.0 - math.exp(-dt / filter_s)
            self.filtered_target += alpha * (destination - self.filtered_target)
            self.filtered_target = clamp_crop_center(
                self.filtered_target, self.crop_width
            )
        destination = self.filtered_target
        error = destination - self.x

        alpha = 1.0 - math.exp(
            -dt / max(1e-4, float(profile["response_time_seconds"]))
        )
        max_step = float(profile["max_speed_normalized_per_second"]) * dt
        step = clamp(alpha * error, -max_step, max_step)
        next_x = clamp_crop_center(self.x + step, self.crop_width)
        snap = float(profile["snap_epsilon"])
        if abs(destination - next_x) <= snap or (destination - next_x) * error <= 0.0:
            next_x = destination
            step = next_x - self.x

        self.x = next_x
        self.v = step / max(dt, 1e-6)
        self._fast_remaining_s = 0.0
        return self.x

    def _start_reframe(
        self,
        *,
        target: float,
        profile_name: str,
        profile: Mapping[str, Any],
    ) -> None:
        distance = abs(target - self.x)
        duration = clamp(
            distance
            / max(
                1e-4,
                float(profile["reframe_speed_normalized_per_second"]),
            ),
            float(profile["reframe_min_seconds"]),
            float(profile["reframe_max_seconds"]),
        )
        self._reframe = {
            "profile_name": profile_name,
            "profile": dict(profile),
            "start": self.x,
            "target": target,
            "elapsed": 0.0,
            "duration": duration,
        }
        self.filtered_target = target
        self.v = 0.0

    def _advance_reframe(self, dt: float) -> float:
        assert self._reframe is not None
        move = self._reframe
        previous_x = self.x
        move["elapsed"] = float(move["elapsed"]) + dt
        progress = min(1.0, float(move["elapsed"]) / float(move["duration"]))
        eased = self._quintic_smoothstep(progress)
        self.x = clamp_crop_center(
            float(move["start"])
            + (float(move["target"]) - float(move["start"])) * eased,
            self.crop_width,
        )
        self.v = (self.x - previous_x) / max(dt, 1e-6)
        self.filtered_target = float(move["target"])
        if progress >= 1.0:
            self.x = float(move["target"])
            self.v = 0.0
            self._clear_reframe()
        return self.x

    def _update_reframe(
        self,
        raw_target: float,
        dt: float,
        profile_name: str,
        profile: Mapping[str, Any],
    ) -> float:
        zone = float(profile["tracking_zone"])
        destination = self._zone_destination(raw_target, zone)
        stop_epsilon = max(
            float(profile["deadband"]),
            float(profile["snap_epsilon"]),
            1e-9,
        )
        if (
            self._reframe is None
            and abs(destination - self.x) <= stop_epsilon
        ):
            return self.hold()

        if self._reframe is None:
            self._start_reframe(
                target=destination,
                profile_name=profile_name,
                profile=profile,
            )
        else:
            active_target = float(self._reframe["target"])
            threshold = max(
                float(profile["retarget_threshold"]),
                float(self._reframe["profile"].get("retarget_threshold", 0.0)),
            )
            if abs(destination - active_target) > threshold:
                self._start_reframe(
                    target=destination,
                    profile_name=profile_name,
                    profile=profile,
                )
        return self._advance_reframe(dt)

    def update(
        self,
        target_x: float,
        dt_s: float,
        *,
        urgent: bool = False,
        profile_name: Optional[str] = None,
    ) -> float:
        raw_target = clamp_crop_center(target_x, self.crop_width)
        if not self.initialized:
            return self.reset(raw_target)

        dt = clamp(float(dt_s), 1.0 / 240.0, 0.25)
        explicit_profile = (
            self.profiles.get(profile_name) if profile_name is not None else None
        )

        if explicit_profile is not None:
            mode = str(explicit_profile["mode"])
            if mode == "follow":
                # A one-frame confirmed reframe must not collapse back to a
                # slower follow profile on the next frame. Continue the active
                # finite move while the new target remains consistent.
                if self._reframe is not None:
                    destination = self._zone_destination(
                        raw_target, float(explicit_profile["tracking_zone"])
                    )
                    active_target = float(self._reframe["target"])
                    threshold = max(
                        float(explicit_profile["retarget_threshold"]),
                        float(
                            self._reframe["profile"].get(
                                "retarget_threshold", 0.0
                            )
                        ),
                    )
                    if abs(destination - active_target) <= threshold:
                        return self._advance_reframe(dt)
                return self._update_follow(raw_target, dt, explicit_profile)
            if mode == "reframe":
                return self._update_reframe(
                    raw_target,
                    dt,
                    str(profile_name),
                    explicit_profile,
                )

        # Legacy acceleration-limited controller retained for compatibility.
        self._clear_reframe()
        if explicit_profile is not None:
            response_time = float(explicit_profile["response_time_seconds"])
            deadband = float(explicit_profile["deadband"])
            max_speed = float(explicit_profile["max_speed_normalized_per_second"])
            max_accel = float(
                explicit_profile["max_acceleration_normalized_per_second2"]
            )
            target_filter_s = float(explicit_profile["target_filter_seconds"])
            self._fast_remaining_s = 0.0
        else:
            if urgent or (
                self.fast_on_large_error
                and abs(raw_target - self.x) >= self.fast_error_threshold
            ):
                self._fast_remaining_s = max(
                    self._fast_remaining_s, self.fast_boost_s
                )
            fast = self._fast_remaining_s > 0.0
            response_time = (
                self.fast_response_time_s if fast else self.response_time_s
            )
            deadband = self.deadband
            max_speed = self.fast_max_speed if fast else self.max_speed
            max_accel = self.fast_max_accel if fast else self.max_accel
            target_filter_s = 0.0 if urgent else self.target_filter_s

        if target_filter_s <= 0.0:
            self.filtered_target = raw_target
        else:
            alpha = 1.0 - math.exp(-dt / target_filter_s)
            self.filtered_target += alpha * (raw_target - self.filtered_target)
            self.filtered_target = clamp_crop_center(
                self.filtered_target, self.crop_width
            )

        target = self.filtered_target
        error = target - self.x
        if abs(error) <= deadband:
            self.v = 0.0
            self._fast_remaining_s = 0.0
            return self.x

        desired_v = clamp(error / response_time, -max_speed, max_speed)
        max_delta_v = max_accel * dt
        next_v = self.v + clamp(
            desired_v - self.v, -max_delta_v, max_delta_v
        )
        next_x = self.x + next_v * dt

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
