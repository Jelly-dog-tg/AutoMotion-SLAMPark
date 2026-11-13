
"""P11.6.1: 2D LiDAR ray casting in the live PyBullet 3D scene.

The sensor uses the vehicle rear-axle XY reference.
Output fields are compatible with the existing OnlineSLAM2D.
"""

import math

import numpy as np
import pybullet as p

from parking_planner.sensing.lidar_2d import LidarConfig


class PyBulletLidar2D:
    def __init__(
        self,
        client_id,
        vehicle_body,
        config=None,
        height_m=1.25,
    ):
        self.cid = client_id
        self.vehicle_body = vehicle_body
        self.config = config or LidarConfig(num_beams=180)
        self.height_m = float(height_m)

        half_fov = math.radians(self.config.fov_deg) / 2.0

        self.angles = np.linspace(
            -half_fov,
            half_fov,
            self.config.num_beams,
        )

        self.rng = np.random.default_rng(20261112)

    def scan(self, state):
        """Capture a fresh LiDAR scan from PhysicsVehicle.state()."""

        x, y, yaw = (
            float(state[key])
            for key in ("x", "y", "yaw")
        )

        headings = self.angles + yaw

        start = [x, y, self.height_m]

        far = [
            [
                x + self.config.max_range_m * math.cos(a),
                y + self.config.max_range_m * math.sin(a),
                self.height_m,
            ]
            for a in headings
        ]

        hits = p.rayTestBatch(
            [start] * len(far),
            far,
            physicsClientId=self.cid,
        )

        ranges = np.full(
            len(far),
            self.config.max_range_m,
            dtype=float,
        )

        valid = np.zeros(len(far), dtype=bool)
        self_hits = 0

        for i, result in enumerate(hits):
            body = int(result[0])
            fraction = float(result[2])

            if body == self.vehicle_body:
                self_hits += 1
                continue

            if body < 0:
                continue

            distance = self.config.max_range_m * fraction

            if self.config.min_range_m <= distance < self.config.max_range_m:
                ranges[i] = distance
                valid[i] = True

        # Simulated measurement noise.
        if valid.any() and self.config.range_std_m > 0:
            ranges[valid] = np.clip(
                ranges[valid]
                + self.rng.normal(
                    0.0,
                    self.config.range_std_m,
                    int(valid.sum()),
                ),
                self.config.min_range_m,
                self.config.max_range_m - 1e-5,
            )

        local = np.column_stack(
            (
                ranges[valid] * np.cos(self.angles[valid]),
                ranges[valid] * np.sin(self.angles[valid]),
            )
        )

        world = np.column_stack(
            (
                x + ranges[valid] * np.cos(headings[valid]),
                y + ranges[valid] * np.sin(headings[valid]),
            )
        )

        return {
            "angles_rad": self.angles.copy(),
            "ranges_m": ranges,
            "valid": valid,
            "points_local": local,
            "points_world": world,
            "self_hits": self_hits,
            "origin_xyz": np.asarray(start, dtype=float),
        }
