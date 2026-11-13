
"""Curvature-constrained speed planning for autonomous parking."""

import math


class CurvatureSpeedPlanner:
    """Generate a speed profile from path curvature and acceleration limits."""

    def __init__(
        self,
        wheelbase=2.7,
        max_forward_speed=0.65,
        max_reverse_speed=0.50,
        max_lateral_accel=0.08,
        max_acceleration=0.45,
        max_deceleration=0.65,
        preview_points=3,
    ):
        if wheelbase <= 0:
            raise ValueError("wheelbase must be positive")

        if min(
            max_forward_speed,
            max_reverse_speed,
            max_lateral_accel,
            max_acceleration,
            max_deceleration,
        ) <= 0:
            raise ValueError("Speed and acceleration limits must be positive")

        if preview_points < 0:
            raise ValueError("preview_points must be non-negative")

        self.wheelbase = wheelbase

        self.max_forward_speed = max_forward_speed
        self.max_reverse_speed = max_reverse_speed

        self.max_lateral_accel = max_lateral_accel
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration

        self.preview_points = preview_points

    def generate(self, path, direction):
        """Generate a speed profile for one direction segment."""

        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")

        if not path:
            return []

        n = len(path)

        if n == 1:
            return [0.0]

        max_speed = (
            self.max_forward_speed
            if direction > 0
            else self.max_reverse_speed
        )

        # Calculate distance between adjacent path points.
        distances = []

        for i in range(n - 1):
            ds = math.hypot(
                path[i + 1].x - path[i].x,
                path[i + 1].y - path[i].y,
            )

            distances.append(max(ds, 1e-6))

        # Calculate curvature-based speed limits.
        speeds = []

        for point in path:
            curvature = abs(
                math.tan(point.steering) / self.wheelbase
            )

            curve_speed = math.sqrt(
                self.max_lateral_accel
                / max(curvature, 1e-6)
            )

            speeds.append(
                min(max_speed, curve_speed)
            )

        # The vehicle must stop at the end of each gear segment.
        speeds[-1] = 0.0

        # Backward pass: enforce braking constraints.
        for i in range(n - 2, -1, -1):

            reachable_speed = math.sqrt(
                speeds[i + 1] ** 2
                + 2.0 * self.max_deceleration * distances[i]
            )

            speeds[i] = min(
                speeds[i],
                reachable_speed,
            )

        # Start from standstill.
        speeds[0] = 0.0

        # Forward pass: enforce acceleration constraints.
        for i in range(1, n):

            reachable_speed = math.sqrt(
                speeds[i - 1] ** 2
                + 2.0 * self.max_acceleration * distances[i - 1]
            )

            speeds[i] = min(
                speeds[i],
                reachable_speed,
            )

        return speeds

    def target_speed(self, speed_profile, last_index):
        """Return a previewed speed magnitude."""

        if not speed_profile:
            return 0.0

        index = min(
            max(0, last_index + self.preview_points),
            len(speed_profile) - 1,
        )

        return speed_profile[index]
