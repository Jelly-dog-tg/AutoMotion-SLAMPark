
"""Simple 2D LiDAR simulator using ray casting."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LidarConfig:
    num_beams: int = 720
    fov_deg: float = 270.0
    min_range_m: float = 0.12
    max_range_m: float = 12.0
    range_std_m: float = 0.015

    def __post_init__(self):
        if self.num_beams < 2:
            raise ValueError("num_beams must be >= 2")

        if not 0 < self.fov_deg <= 360:
            raise ValueError("Invalid field of view")

        if not 0 < self.min_range_m < self.max_range_m:
            raise ValueError("Invalid range limits")

        if self.range_std_m < 0:
            raise ValueError("range_std_m must be non-negative")


class Lidar2D:

    def __init__(self, config=None):
        self.config = config or LidarConfig()

    @staticmethod
    def _axis_interval(origin, directions, lower, upper):

        parallel = np.abs(directions) < 1e-12

        t1 = np.full_like(directions, -np.inf)
        t2 = np.full_like(directions, np.inf)

        np.divide(
            lower - origin,
            directions,
            out=t1,
            where=~parallel,
        )

        np.divide(
            upper - origin,
            directions,
            out=t2,
            where=~parallel,
        )

        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)

        outside = parallel & (
            (origin < lower) | (origin > upper)
        )

        t_min[parallel] = -np.inf
        t_max[parallel] = np.inf

        t_min[outside] = np.inf
        t_max[outside] = -np.inf

        return t_min, t_max

    def scan(self, pose, obstacles, rng=None):

        if rng is None:
            rng = np.random.default_rng()

        cfg = self.config

        # Angles in the LiDAR local coordinate frame.
        half_fov = math.radians(cfg.fov_deg) / 2.0

        angles = np.linspace(
            -half_fov,
            half_fov,
            cfg.num_beams,
        )

        world_angles = angles + pose.yaw

        dx = np.cos(world_angles)
        dy = np.sin(world_angles)

        ranges = np.full(
            cfg.num_beams,
            cfg.max_range_m,
            dtype=float,
        )

        hit_mask = np.zeros(
            cfg.num_beams,
            dtype=bool,
        )

        # Ray-rectangle intersection.
        for obstacle in obstacles:

            x0, y0, x1, y1 = map(float, obstacle)

            if x0 >= x1 or y0 >= y1:
                raise ValueError("Invalid obstacle rectangle")

            tx_min, tx_max = self._axis_interval(
                pose.x, dx, x0, x1
            )

            ty_min, ty_max = self._axis_interval(
                pose.y, dy, y0, y1
            )

            t_enter = np.maximum(tx_min, ty_min)
            t_exit = np.minimum(tx_max, ty_max)

            distance = np.where(
                t_enter >= cfg.min_range_m,
                t_enter,
                t_exit,
            )

            valid = (
                (t_exit >= np.maximum(t_enter, 0.0))
                & (t_exit >= cfg.min_range_m)
                & (t_enter <= cfg.max_range_m)
                & (distance >= cfg.min_range_m)
                & (distance < ranges)
            )

            ranges[valid] = distance[valid]
            hit_mask[valid] = True

        # Simulate measurement noise.
        if cfg.range_std_m > 0 and np.any(hit_mask):

            ranges[hit_mask] += rng.normal(
                0.0,
                cfg.range_std_m,
                size=int(np.sum(hit_mask)),
            )

            ranges[hit_mask] = np.clip(
                ranges[hit_mask],
                cfg.min_range_m,
                cfg.max_range_m,
            )

        # Convert measurements to local Cartesian points.
        local_x = ranges[hit_mask] * np.cos(
            angles[hit_mask]
        )

        local_y = ranges[hit_mask] * np.sin(
            angles[hit_mask]
        )

        points_local = np.column_stack(
            (local_x, local_y)
        )

        # Convert measurements to world coordinates.
        world_x = pose.x + ranges[hit_mask] * np.cos(
            world_angles[hit_mask]
        )

        world_y = pose.y + ranges[hit_mask] * np.sin(
            world_angles[hit_mask]
        )

        points_world = np.column_stack(
            (world_x, world_y)
        )

        return {
            "angles_rad": angles,
            "ranges_m": ranges,
            "valid": hit_mask,
            "points_local": points_local,
            "points_world": points_world,
        }
