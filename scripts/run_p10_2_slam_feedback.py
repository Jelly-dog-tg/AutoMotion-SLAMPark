"""P10.2: Stanley follows the forward part of the P10.1 SLAM-map plan.

Uses the *saved exploration map* as a localization prior, but creates NEW
LiDAR measurements while the Ackermann simulator moves. Stanley receives a
SLAM/odometry-extrapolated pose, not the simulator's ground-truth x/y/yaw.

Run from project root:
    python scripts/run_p10_2_slam_feedback.py

Forward-segment-only integration check; reverse/gear shifting is P10.3.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.control.stanley_controller import StanleyController
from parking_planner.control.speed_controller import SpeedController
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from parking_planner.integration.slam_map_adapter import build_planner_grid
from parking_planner.planning.hybrid_astar import HybridAStarPlanner, PathPoint
from parking_planner.sensing.lidar_2d import Lidar2D, LidarConfig
from parking_planner.slam.online_slam import OnlineSLAM2D, compose_pose, wrap_angle
from parking_planner.vehicle.ackermann import AckermannModel, VehicleState

RESULTS = ROOT / "results"
PATH_FILE = RESULTS / "p10_1_slam_map_planning_path.npz"
MAP_FILE = RESULTS / "p9_5_3_keyboard_map.npz"
METRICS_FILE = RESULTS / "p10_2_slam_feedback_metrics.json"
PLOT_FILE = RESULTS / "p10_2_slam_feedback.png"
DT = 0.05
SCAN_EVERY = 8                         # 0.4 simulated seconds
MAX_STEPS = 750                        # Safety limit, 37.5 simulated seconds
SPEED_LIMIT = 0.42                     # Forward-segment test only (m/s)
STOP_TOLERANCE = 0.18                  # Distance to forward->reverse junction
STOP_SPEED = 0.06
SEED = 2026


def relative_pose(previous, current):
    """SE(2) increment: current odometry pose in previous odometry frame."""
    x0, y0, a0 = map(float, previous)
    x1, y1, a1 = map(float, current)
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(a0), math.sin(a0)
    return np.array([c * dx + s * dy, -s * dx + c * dy,
                     wrap_angle(a1 - a0)], dtype=float)


def pose_error(estimate, truth):
    pos = math.hypot(estimate[0] - truth.x, estimate[1] - truth.y)
    ang = abs(math.degrees(wrap_angle(estimate[2] - truth.yaw)))
    return pos, ang


def load_forward_segment():
    if not PATH_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {PATH_FILE}; run P10.1 before P10.2."
        )
    with np.load(PATH_FILE, allow_pickle=False) as data:
        raw = np.asarray(data["path"], dtype=float)
        start = np.asarray(data["start_pose"], dtype=float)
        goal = np.asarray(data["goal_pose"], dtype=float)
    if (raw.ndim != 2 or raw.shape[1] != 5 or len(raw) < 2
            or start.shape != (3,) or goal.shape != (3,)
            or not np.all(np.isfinite(raw)) or not np.all(np.isfinite(start))):
        raise ValueError("Invalid P10.1 saved path format")
    directions = raw[:, 3].astype(int)
    if not np.all(np.isin(directions, [-1, 1])):
        raise ValueError("Invalid path direction labels")
    changes = np.flatnonzero(np.diff(directions) != 0)
    if len(changes) != 1 or directions[0] != 1 or directions[-1] != -1:
        raise ValueError("P10.2 expects one forward->reverse switch in P10.1 path")
    # Only the first gear segment is used here. Do not start the reverse segment.
    segment_array = raw[:int(changes[0]) + 1]
    path = [PathPoint(float(x), float(y), float(yaw), int(direction), float(steer))
            for x, y, yaw, direction, steer in segment_array]
    return path, start, goal


def remaining_distances(path):
    suffix = np.zeros(len(path), dtype=float)
    for i in range(len(path)-2, -1, -1):
        suffix[i] = suffix[i+1] + math.hypot(
            path[i+1].x - path[i].x, path[i+1].y - path[i].y
        )
    return suffix


def main():
    RESULTS.mkdir(exist_ok=True, parents=True)
    if not MAP_FILE.is_file():
        raise FileNotFoundError(f"Missing saved exploration map: {MAP_FILE}")

    path, start, full_goal = load_forward_segment()
    scenario = build_perpendicular_parking_scenario()
    rng = np.random.default_rng(SEED)
    mapper = OccupancyGridMapper.load(MAP_FILE)
    planning_grid = build_planner_grid(mapper, planner_resolution=0.2,
                                       unknown_is_occupied=True)
    # Footprint shape copied from the existing planner for collision auditing only.
    planner = HybridAStarPlanner(grid=planning_grid,
                                geometry=scenario["geometry"],
                                config=scenario["planner_config"])

    model = AckermannModel(wheelbase=scenario["geometry"].wheelbase,
                           max_speed=1.0, max_acceleration=0.8)
    controller = StanleyController(wheelbase=model.wheelbase,
                                   max_steer=model.max_steer)
    pid = SpeedController(kp=2.5, ki=0.0, kd=0.0,
                          max_acceleration=0.8, max_speed=1.0)
    lidar_config = LidarConfig()
    lidar = Lidar2D(lidar_config)
    slam = OnlineSLAM2D(initial_pose=start,
                        lidar_config=lidar_config)
    # Start from the actual P9.5.3 map as a localization prior, in memory only.
    # The file is never overwritten, and no original GT obstacle grid is fed to SLAM.
    slam.mapper = mapper
    true = VehicleState(float(start[0]), float(start[1]), float(start[2]), 0.0)
    wheel = start.copy()  # Simulated wheel odometry; not the true x/y/yaw state.
    last_wheel_scan = wheel.copy()
    estimated = start.copy()
    prev_measured_speed = 0.0
    last_index = 0
    suffix = remaining_distances(path)
    scans = 0
    records = []
    status = "MAX_STEPS_REACHED"
    physical_collision = False
    planning_map_collision = False

    # Initial scan seeds the SLAM front end using *new* LiDAR measurements.
    slam.step(lidar.scan(true, scenario["parked_vehicles"], rng=rng), None)
    scans += 1

    for k in range(MAX_STEPS):
        # Extrapolate the latest corrected SLAM pose to this 20 Hz control tick.
        # This is crucial because LiDAR corrections occur only every 0.4 seconds.
        estimated = compose_pose(
            slam.pose, relative_pose(last_wheel_scan, wheel)
        )
        feedback = VehicleState(
            float(estimated[0]), float(estimated[1]), float(estimated[2]),
            float(prev_measured_speed)
        )
        output = controller.compute(
            state=feedback, path=path, direction=1, last_index=last_index
        )
        last_index = max(last_index, int(output.target_index))
        endpoint = path[-1]
        estimated_distance = math.hypot(
            endpoint.x - feedback.x, endpoint.y - feedback.y
        )
        remaining = float(suffix[last_index])
        # Brake using the estimated position only; ground truth is not consulted.
        target_v = min(SPEED_LIMIT, math.sqrt(
            2.0 * 0.7 * max(0.0, remaining - 0.05)
        ))
        if remaining <= 0.05 or estimated_distance <= STOP_TOLERANCE:
            target_v = 0.0
        accel = pid.compute(target_v, feedback.v, dt=DT)

        # Stop when the *estimated* position/speed satisfies the segment target.
        if estimated_distance <= STOP_TOLERANCE and abs(feedback.v) <= STOP_SPEED:
            status = "FORWARD_SEGMENT_SUCCESS"
            break

        candidate = model.step(true, accel, output.steering, DT)
        physical_collision = physical_collision or scenario["grid"].is_collision(
            candidate, planner.local_x, planner.local_y
        )
        planning_map_collision = planning_map_collision or planning_grid.is_collision(
            candidate, planner.local_x, planner.local_y
        )
        if physical_collision or planning_map_collision:
            status = "COLLISION_ABORT"
            break
        true = candidate

        # Sensor measurements may observe physical speed/steering, but not GT pose.
        measured_speed = (true.v * 1.025 + rng.normal(0.0, 0.015)
                          if abs(true.v) > 1e-7 else 0.0)
        measured_steer = output.steering + math.radians(0.5) + rng.normal(
            0.0, math.radians(0.30)
        )
        dyaw = measured_speed / model.wheelbase * math.tan(measured_steer) * DT
        middle = wheel[2] + dyaw / 2.0
        wheel[:2] += measured_speed * DT * np.array([
            math.cos(middle), math.sin(middle)
        ])
        wheel[2] = wrap_angle(wheel[2] + dyaw)
        prev_measured_speed = float(measured_speed)

        if (k+1) % SCAN_EVERY == 0:
            scan = lidar.scan(true, scenario["parked_vehicles"], rng=rng)
            slam.step(scan, relative_pose(last_wheel_scan, wheel))
            last_wheel_scan = wheel.copy()
            scans += 1

        # Capture truth for evaluation/plotting only AFTER controls are applied.
        estimated_after = compose_pose(
            slam.pose, relative_pose(last_wheel_scan, wheel)
        )
        pos_error, heading_error = pose_error(estimated_after, true)
        records.append({
            "time_s": (k+1) * DT,
            "true_x": true.x, "true_y": true.y, "true_yaw": true.yaw,
            "estimate_x": float(estimated_after[0]),
            "estimate_y": float(estimated_after[1]),
            "estimate_yaw": float(estimated_after[2]),
            "position_error_m": pos_error,
            "heading_error_deg": heading_error,
            "estimated_cte_m": float(output.cross_track_error),
            "steering_deg": math.degrees(output.steering),
            "speed_mps": float(true.v),
            "target_speed_mps": float(target_v),
        })

    if not records:
        raise RuntimeError("No simulation samples were generated")

    final_est = compose_pose(slam.pose, relative_pose(last_wheel_scan, wheel))
    final_truth_pos_err = math.hypot(true.x-path[-1].x, true.y-path[-1].y)
    final_est_pos_err = math.hypot(final_est[0]-path[-1].x,
                                   final_est[1]-path[-1].y)
    arr = {key: np.asarray([r[key] for r in records], dtype=float)
           for key in records[0]}
    status_counts = Counter(x["status"] for x in slam.history)
    metrics = {
        "stage": "P10.2 Stanley with live SLAM pose feedback, forward-segment only",
        "status": status,
        "path_source": str(PATH_FILE.relative_to(ROOT)),
        "map_source": str(MAP_FILE.relative_to(ROOT)),
        "controller_position_source": "OnlineSLAM2D + wheel odometry extrapolation",
        "truth_used_for_controller": False,
        "segment": "forward only; P10.3 will implement gear change and reverse",
        "reference_path_points": len(path),
        "control_steps": len(records),
        "elapsed_sim_time_s": float(arr["time_s"][-1]),
        "lidar_scans": scans,
        "icp_updates": dict(status_counts),
        "collision_in_physical_map": physical_collision,
        "collision_in_slam_planning_map": planning_map_collision,
        "localization_translation_rmse_m": float(np.sqrt(np.mean(
            arr["position_error_m"]**2))),
        "localization_heading_rmse_deg": float(np.sqrt(np.mean(
            arr["heading_error_deg"]**2))),
        "estimated_cross_track_rmse_m": float(np.sqrt(np.mean(
            arr["estimated_cte_m"]**2))),
        "final_estimated_segment_endpoint_error_m": final_est_pos_err,
        "final_true_segment_endpoint_error_m": final_truth_pos_err,
        "final_speed_mps": float(true.v),
        "full_parking_goal_pose": full_goal.tolist(),
        "full_parking_executed": False,
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    for xmin, ymin, xmax, ymax in scenario["parked_vehicles"]:
        ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                               color="gray", alpha=.6))
    sx0, sy0, sx1, sy1 = scenario["parking_slot"]
    ax.add_patch(Rectangle((sx0, sy0), sx1-sx0, sy1-sy0,
                           edgecolor="green", fill=False, ls="--"))
    ax.plot([p.x for p in path], [p.y for p in path], "b--", label="P10.1 path")
    ax.plot(arr["true_x"], arr["true_y"], label="Actual motion")
    ax.plot(arr["estimate_x"], arr["estimate_y"], label="SLAM estimate")
    ax.scatter(path[-1].x, path[-1].y, marker="x", s=65, label="Gear-change point")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Forward Tracking: SLAM Feedback")
    ax.set_aspect("equal"); ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(arr["time_s"], arr["position_error_m"], label="Position error (m)")
    ax.plot(arr["time_s"], arr["heading_error_deg"], label="Heading error (deg)")
    ax.set(xlabel="Time (s)", title="SLAM Localization Error")
    ax.legend(); ax.grid(alpha=.25)

    ax = axes[1, 0]
    ax.plot(arr["time_s"], arr["estimated_cte_m"])
    ax.axhline(0, color="black", lw=.6)
    ax.set(xlabel="Time (s)", ylabel="m", title="Estimated Cross-track Error")
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    ax.plot(arr["time_s"], arr["speed_mps"], label="Actual speed")
    ax.plot(arr["time_s"], arr["target_speed_mps"], "--", label="Target speed")
    ax.set(xlabel="Time (s)", ylabel="m/s", title="Forward Speed")
    ax.legend(); ax.grid(alpha=.25)
    fig.suptitle(f"P10.2 SLAM-to-Stanley Integration | {status}")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=170)
    plt.close(fig)

    print("\n========== P10.2 RESULTS ==========")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Visualization: {PLOT_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    if status != "FORWARD_SEGMENT_SUCCESS":
        print("P10.2 needs review; do not proceed to P10.3 yet.")


if __name__ == "__main__":
    main()
