"""Direction-aware Stanley path tracking for forward and reverse parking."""

from __future__ import annotations

import math
from dataclasses import dataclass

from parking_planner.vehicle.ackermann import VehicleState


@dataclass(frozen=True)
class StanleyOutput:
    steering: float
    target_index: int
    cross_track_error: float
    heading_error: float


class StanleyController:
    """Stanley controller using the rear-axle path convention.

    Forward tracking measures the front axle against the corresponding
    reference front-axle path. Reverse tracking measures the rear axle,
    which is the leading axle in the direction of reverse motion.

    ``PathPoint.yaw`` is the vehicle body heading for both gears. The travel
    heading is body heading for forward motion and body heading + pi in reverse.
    """

    def __init__(
        self,
        wheelbase: float = 2.7,
        gain: float = 0.5,
        softening_speed: float = 0.5,
        max_steer: float = math.radians(35.0),
        search_back: int = 2,
        search_ahead: int = 24,
    ):
        if wheelbase <= 0:
            raise ValueError("wheelbase must be positive")
        if gain <= 0 or softening_speed <= 0:
            raise ValueError("gain and softening_speed must be positive")
        if not 0 < max_steer < math.pi / 2:
            raise ValueError("max_steer must be between 0 and 90 degrees")
        if search_back < 0 or search_ahead < 1:
            raise ValueError("invalid nearest-point search window")

        self.wheelbase = wheelbase
        self.gain = gain
        self.softening_speed = softening_speed
        self.max_steer = max_steer
        self.search_back = search_back
        self.search_ahead = search_ahead

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _effective_gain(
        self,
        state: VehicleState,
        cross_track_error: float,
    ) -> float:
        """Return feedback gain; subclasses may override."""
        return self.gain

    def _tracking_position(self, state: VehicleState, direction: int) -> tuple[float, float]:
        if direction > 0:
            return (
                state.x + self.wheelbase * math.cos(state.yaw),
                state.y + self.wheelbase * math.sin(state.yaw),
            )
        return state.x, state.y

    def _reference_position(self, point, direction: int) -> tuple[float, float]:
        if direction > 0:
            return (
                point.x + self.wheelbase * math.cos(point.yaw),
                point.y + self.wheelbase * math.sin(point.yaw),
            )
        return point.x, point.y

    def compute(
        self,
        state: VehicleState,
        path,
        direction: int,
        last_index: int = 0,
    ) -> StanleyOutput:
        """Compute steering and tracking errors for one direction segment."""
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 (forward) or -1 (reverse)")
        if not path:
            raise ValueError("path segment must not be empty")

        px, py = self._tracking_position(state, direction)
        lo = max(0, last_index - self.search_back)
        hi = min(len(path), last_index + self.search_ahead + 1)
        if lo >= hi:
            lo, hi = 0, len(path)

        target_index = min(
            range(lo, hi),
            key=lambda i: (
                self._reference_position(path[i], direction)[0] - px
            ) ** 2
            + (
                self._reference_position(path[i], direction)[1] - py
            ) ** 2,
        )
        target = path[target_index]
        rx, ry = self._reference_position(target, direction)

        travel_heading = self.normalize_angle(
            target.yaw + (math.pi if direction < 0 else 0.0)
        )
        actual_travel_heading = self.normalize_angle(
            state.yaw + (math.pi if direction < 0 else 0.0)
        )

        # Signed error is positive when the reference lies to the left of
        # the vehicle's current travel direction.
        left_normal_x = -math.sin(travel_heading)
        left_normal_y = math.cos(travel_heading)
        cross_track_error = (
            (rx - px) * left_normal_x
            + (ry - py) * left_normal_y
        )
        heading_error = self.normalize_angle(
            travel_heading - actual_travel_heading
        )

        effective_gain = self._effective_gain(
            state,
            cross_track_error,
        )

        correction = math.atan2(
            effective_gain * cross_track_error,
            abs(state.v) + self.softening_speed,
        )
        # Use the steering primitive for the path interval ahead of the
        # current reference sample as curvature feedforward.
        feedforward_index = min(target_index + 1, len(path) - 1)
        feedforward_steering = path[feedforward_index].steering
        # Reverse motion reverses the steering-to-yaw response, so only the
        # feedback term is multiplied by the travel direction.
        steering = feedforward_steering + direction * (heading_error + correction)
        steering = max(-self.max_steer, min(self.max_steer, steering))

        return StanleyOutput(
            steering=steering,
            target_index=target_index,
            cross_track_error=cross_track_error,
            heading_error=heading_error,
        )
