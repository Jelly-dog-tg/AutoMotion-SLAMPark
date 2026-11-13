"""Path interpolation and geometric metrics for planned parking paths."""

import math

from parking_planner.planning.hybrid_astar import PathPoint


def resample_path(path: list[PathPoint], spacing: float = 0.1) -> list[PathPoint]:
    """Densify the planner path with short line segments.

    The planner checks each motion primitive at substeps. This function is for
    visualization and downstream path tracking; it does not claim to smooth
    or change the collision-checked geometric path.
    """
    if not path:
        return []
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if len(path) == 1:
        return list(path)

    dense = [path[0]]

    for previous, current in zip(path[:-1], path[1:]):
        distance = math.hypot(
            current.x - previous.x,
            current.y - previous.y,
        )
        steps = max(1, math.ceil(distance / spacing))

        yaw_delta = math.atan2(
            math.sin(current.yaw - previous.yaw),
            math.cos(current.yaw - previous.yaw),
        )

        for index in range(1, steps + 1):
            ratio = index / steps
            yaw = previous.yaw + ratio * yaw_delta
            dense.append(
                PathPoint(
                    x=previous.x + ratio * (current.x - previous.x),
                    y=previous.y + ratio * (current.y - previous.y),
                    yaw=math.atan2(math.sin(yaw), math.cos(yaw)),
                    direction=current.direction,
                    steering=current.steering,
                )
            )

    return dense


def path_metrics(path: list[PathPoint]) -> dict:
    """Calculate path length, reverse distance and gear-switch count."""
    total_length = 0.0
    reverse_distance = 0.0
    gear_switches = 0

    for previous, current in zip(path[:-1], path[1:]):
        distance = math.hypot(
            current.x - previous.x,
            current.y - previous.y,
        )
        total_length += distance

        if current.direction < 0:
            reverse_distance += distance

        if current.direction != previous.direction:
            gear_switches += 1

    return {
        "path_length": total_length,
        "reverse_distance": reverse_distance,
        "gear_switches": gear_switches,
    }
