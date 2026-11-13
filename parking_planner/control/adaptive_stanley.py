
"""Adaptive Stanley controller for autonomous parking."""

from __future__ import annotations

from parking_planner.control.stanley_controller import StanleyController


class AdaptiveStanleyController(StanleyController):
    """Stanley controller with adaptive feedback gain."""

    def __init__(
        self,
        wheelbase=2.7,
        base_gain=0.5,
        min_gain=0.4,
        max_gain=1.0,
        error_gain=0.45,
        speed_gain=0.12,
        error_scale=0.25,
        speed_scale=0.5,
        softening_speed=0.5,
        max_steer=0.610865,
        search_back=2,
        search_ahead=24,
    ):
        super().__init__(
            wheelbase=wheelbase,
            gain=base_gain,
            softening_speed=softening_speed,
            max_steer=max_steer,
            search_back=search_back,
            search_ahead=search_ahead,
        )

        if not (0 < min_gain <= base_gain <= max_gain):
            raise ValueError("Invalid gain limits")

        if error_gain < 0 or speed_gain < 0:
            raise ValueError("Adaptive gains must be non-negative")

        if error_scale <= 0 or speed_scale <= 0:
            raise ValueError("Scales must be positive")

        self.base_gain = base_gain
        self.min_gain = min_gain
        self.max_gain = max_gain

        self.error_gain = error_gain
        self.speed_gain = speed_gain

        self.error_scale = error_scale
        self.speed_scale = speed_scale

        self.gain_history = []

    def _effective_gain(self, state, cross_track_error):
        """Adapt gain according to tracking error and speed."""

        # Larger lateral error produces stronger correction.
        error_ratio = (
            abs(cross_track_error)
            / (abs(cross_track_error) + self.error_scale)
        )

        # Higher speed increases the feedback gain.
        speed_ratio = min(
            abs(state.v) / self.speed_scale,
            1.0,
        )

        gain = (
            self.base_gain
            + self.error_gain * error_ratio
            + self.speed_gain * speed_ratio
        )

        gain = max(
            self.min_gain,
            min(self.max_gain, gain),
        )

        self.gain_history.append(gain)

        return gain
