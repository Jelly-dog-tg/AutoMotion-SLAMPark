"""Online 2D LiDAR SLAM front-end: odometry prediction, ICP, map update.

The caller supplies measured wheel-odometry increments and LiDAR scans.
No ground-truth pose is accepted by step().
"""
import math
import numpy as np

from parking_planner.vehicle.ackermann import VehicleState
from parking_planner.localization.point_to_line_icp import (
    PointToLineConfig, point_to_line_icp,
)
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from parking_planner.sensing.lidar_2d import LidarConfig


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def compose_pose(pose, delta):
    """delta=(forward, left, dyaw) in the previous vehicle frame."""
    x, y, yaw = map(float, pose)
    dx, dy, dyaw = map(float, delta)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([x + c * dx - s * dy,
                     y + s * dx + c * dy,
                     wrap_angle(yaw + dyaw)], dtype=float)


class OnlineSLAM2D:
    def __init__(self, initial_pose, map_config=None, lidar_config=None,
                 icp_config=None):
        pose = np.asarray(initial_pose, dtype=float)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("initial_pose must be finite [x, y, yaw]")
        self.pose = pose.copy()
        self.mapper = OccupancyGridMapper(**(map_config or {
            "width": 22.0, "height": 22.0,
            "resolution": 0.05, "origin_x": 0.0, "origin_y": 0.0,
        }))
        self.lidar_config = lidar_config or LidarConfig()
        self.icp_config = icp_config or PointToLineConfig()
        self.history = []

    def step(self, scan, odometry_increment=None):
        """Consume one fresh scan and the wheel-odometry increment since last scan.

        First scan bootstraps the map. Subsequent steps predict pose from odom,
        attempt scan-to-map alignment, apply quality gating, then update the map.
        """
        angles = np.asarray(scan["angles_rad"], dtype=float)
        ranges = np.asarray(scan["ranges_m"], dtype=float)
        hits = np.asarray(scan["valid"], dtype=bool)
        if not (angles.ndim == ranges.ndim == hits.ndim == 1
                and len(angles) == len(ranges) == len(hits)
                == self.lidar_config.num_beams):
            raise ValueError("LiDAR scan dimensions do not match configuration")
        if not (np.all(np.isfinite(angles)) and np.all(np.isfinite(ranges))):
            raise ValueError("Nonfinite LiDAR angle or range")

        if not self.history:
            if odometry_increment is not None:
                raise ValueError("First scan must not have an odometry increment")
            prediction = self.pose.copy()
            status, reason, match = "BOOTSTRAP", None, None
            map_size = 0
        else:
            if odometry_increment is None:
                raise ValueError("Missing odometry increment after first scan")
            delta = np.asarray(odometry_increment, dtype=float)
            if delta.shape != (3,) or not np.all(np.isfinite(delta)):
                raise ValueError("Invalid odometry increment")
            prediction = compose_pose(self.pose, delta)
            map_rows, map_cols = np.where(self.mapper.states == 1)
            map_size = len(map_rows)
            status, reason, match = "ODOM_FALLBACK", "map_not_ready", None
            if int(hits.sum()) >= self.icp_config.min_correspondences and map_size >= 40:
                map_points = np.column_stack((
                    self.mapper.origin_x + (map_cols + 0.5) * self.mapper.resolution,
                    self.mapper.origin_y + (map_rows + 0.5) * self.mapper.resolution,
                ))
                local = np.column_stack((
                    ranges[hits] * np.cos(angles[hits]),
                    ranges[hits] * np.sin(angles[hits]),
                ))
                try:
                    match = point_to_line_icp(
                        scan=local, initial_pose=prediction,
                        map_points=map_points, config=self.icp_config,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    reason = f"icp_exception:{type(exc).__name__}"
                else:
                    if match["success"]:
                        status, reason = "ICP_ACCEPT", None
                    else:
                        reason = match["rejection_reason"]
            if status == "ICP_ACCEPT":
                self.pose = np.asarray(match["pose"], dtype=float).copy()
            else:
                self.pose = prediction.copy()

        # Crucial ordering: match against the old map, update only afterward.
        vehicle_pose = VehicleState(
            float(self.pose[0]), float(self.pose[1]), float(self.pose[2]), 0.0,
        )
        self.mapper.update(vehicle_pose, scan, self.lidar_config)
        entry = {
            "scan_index": len(self.history),
            "status": status, "reason": reason,
            "pose": self.pose.tolist(),
            "prediction": prediction.tolist(),
            "hit_count": int(hits.sum()),
            "occupied_map_points_before_update": int(map_size),
            "icp_correspondences": int(match["correspondences"]) if match else 0,
            "icp_fitness": float(match["fitness"]) if match else None,
            "icp_eigen_ratio": float(match["information_eigen_ratio"]) if match else None,
            "icp_line_rmse_m": match["line_rmse_m"] if match else None,
        }
        self.history.append(entry)
        return entry
