
"""Direction-aware Pure Pursuit controller."""

import math

from parking_planner.control.stanley_controller import StanleyOutput


class PurePursuitController:

    def __init__(
        self,
        wheelbase=2.7,
        lookahead_distance=0.6,
        max_steer=math.radians(35.0),
        search_back=2,
        search_ahead=24,
    ):
        if wheelbase <= 0 or lookahead_distance <= 0:
            raise ValueError("Invalid controller parameters")

        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance
        self.max_steer = max_steer
        self.search_back = search_back
        self.search_ahead = search_ahead

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def compute(self, state, path, direction, last_index=0):

        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")

        if not path:
            raise ValueError("Path cannot be empty")

        lo = max(0, last_index - self.search_back)
        hi = min(len(path), last_index + self.search_ahead + 1)

        if lo >= hi:
            lo, hi = 0, len(path)

        # Find nearest path point.
        nearest = min(
            range(lo, hi),
            key=lambda i:
                (path[i].x - state.x) ** 2
                + (path[i].y - state.y) ** 2,
        )

        # Find lookahead target.
        target_index = nearest
        accumulated = 0.0

        while (
            target_index < len(path) - 1
            and accumulated < self.lookahead_distance
        ):
            a = path[target_index]
            b = path[target_index + 1]

            accumulated += math.hypot(
                b.x - a.x,
                b.y - a.y,
            )

            target_index += 1

        target = path[target_index]

        dx = target.x - state.x
        dy = target.y - state.y

        lookahead = max(
            math.hypot(dx, dy),
            1e-6,
        )

        # Travel heading differs between forward and reverse.
        travel_heading = self.normalize_angle(
            state.yaw
            + (math.pi if direction < 0 else 0.0)
        )

        alpha = self.normalize_angle(
            math.atan2(dy, dx) - travel_heading
        )

        curvature = 2.0 * math.sin(alpha) / lookahead

        # Reverse motion reverses steering-to-yaw response.
        steering = direction * math.atan(
            self.wheelbase * curvature
        )

        steering = max(
            -self.max_steer,
            min(self.max_steer, steering),
        )

        # Calculate tracking errors at nearest point.
        nearest_point = path[nearest]

        reference_heading = self.normalize_angle(
            nearest_point.yaw
            + (math.pi if direction < 0 else 0.0)
        )

        left_normal_x = -math.sin(reference_heading)
        left_normal_y = math.cos(reference_heading)

        cross_track_error = (
            (nearest_point.x - state.x) * left_normal_x
            + (nearest_point.y - state.y) * left_normal_y
        )

        heading_error = self.normalize_angle(
            reference_heading - travel_heading
        )

        return StanleyOutput(
            steering=steering,
            target_index=nearest,
            cross_track_error=cross_track_error,
            heading_error=heading_error,
        )
