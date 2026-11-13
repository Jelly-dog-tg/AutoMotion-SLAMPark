"""P10.3: Full SLAM-map Hybrid A* + Stanley/PID + automatic reverse parking.

Run from the ParkingPlanner project root:
    python scripts/run_p10_3_full_autonomous_parking.py

The saved P9.5.3 map supplies the static environment frame and planner map.
OnlineSLAM2D continuously estimates pose from simulated 2D LiDAR and wheel
odometry. After the forward approach, the script replans the reverse segment
from the latest SLAM pose, then uses GearManager to shift and park.
Ground truth is excluded from control and used only for simulation/evaluation.
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
from parking_planner.control.gear_manager import GearManager
from parking_planner.control.stanley_controller import StanleyController
from parking_planner.control.speed_controller import SpeedController
from parking_planner.integration.slam_map_adapter import build_planner_grid
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from parking_planner.planning.hybrid_astar import HybridAStarPlanner, PathPoint
from parking_planner.planning.path_utils import resample_path
from parking_planner.sensing.lidar_2d import LidarConfig
from parking_planner.slam.online_slam import OnlineSLAM2D, compose_pose, wrap_angle
from parking_planner.vehicle.ackermann import AckermannModel, VehicleState

RESULTS = ROOT / "results"
PATH_FILE = RESULTS / "p10_1_slam_map_planning_path.npz"
MAP_FILE = RESULTS / "p9_5_3_keyboard_map.npz"
METRICS_FILE = RESULTS / "p10_3_full_autonomous_parking_metrics.json"
PLOT_FILE = RESULTS / "p10_3_full_autonomous_parking.png"

DT = 0.05
SCAN_EVERY = 8                  # 0.4 simulated seconds
MAX_STEPS = 1800                # 90 simulated seconds safety limit
FORWARD_SPEED = 0.38            # m/s, deliberately conservative
REVERSE_SPEED = 0.22            # m/s, slower in the parking bay
MAX_ACCELERATION = 0.65         # m/s^2
BRAKE_DECELERATION = 0.65        # m/s^2
STOP_SPEED = 0.045              # m/s; GearManager requires a full stop
SHIFT_DURATION = 0.5            # s
STOP_BUFFER = 0.06              # m
SEGMENT_STOP_TOLERANCE = 0.17   # m
GOAL_POSITION_TOLERANCE = 0.16  # m
GOAL_HEADING_TOLERANCE = math.radians(5.5)
SEED = 20261112


def relative_pose(previous, current):
    """Return current odometry pose in previous odometry frame (SE(2))."""
    x0, y0, a0 = map(float, previous)
    x1, y1, a1 = map(float, current)
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(a0), math.sin(a0)
    return np.array([c * dx + s * dy, -s * dx + c * dy,
                     wrap_angle(a1 - a0)], dtype=float)


def pose_error(estimate, truth):
    return (
        math.hypot(float(estimate[0]) - truth.x,
                   float(estimate[1]) - truth.y),
        abs(math.degrees(wrap_angle(float(estimate[2]) - truth.yaw))),
    )


def scan_static_slam_map(pose, mapper, config, rng):
    """Ray-cast a virtual 2D LiDAR against the saved SLAM map frame.

    The saved map defines the simulated environment frame, avoiding a hidden
    mismatch between the SLAM map coordinates and a separately hard-coded
    obstacle layout. Unknown cells produce no return; occupied cells produce
    noisy range hits.
    """
    half_fov = math.radians(config.fov_deg) / 2.0
    angles = np.linspace(-half_fov, half_fov, config.num_beams)
    step = max(0.025, mapper.resolution * 0.5)
    distances = np.arange(config.min_range_m, config.max_range_m + step * 0.5, step)
    world_angles = angles + float(pose.yaw)
    xs = float(pose.x) + np.cos(world_angles)[:, None] * distances[None, :]
    ys = float(pose.y) + np.sin(world_angles)[:, None] * distances[None, :]
    cols = np.floor((xs - mapper.origin_x) / mapper.resolution).astype(int)
    rows = np.floor((ys - mapper.origin_y) / mapper.resolution).astype(int)
    inside = ((cols >= 0) & (cols < mapper.cols) &
              (rows >= 0) & (rows < mapper.rows))
    occupied = np.zeros_like(inside, dtype=bool)
    occupied[inside] = mapper.states[rows[inside], cols[inside]] == 1
    hit = occupied.any(axis=1)
    first = np.argmax(occupied, axis=1)
    ranges = np.full(config.num_beams, config.max_range_m, dtype=float)
    ranges[hit] = distances[first[hit]]
    if config.range_std_m > 0 and np.any(hit):
        ranges[hit] += rng.normal(0.0, config.range_std_m, int(hit.sum()))
        ranges[hit] = np.clip(ranges[hit], config.min_range_m, config.max_range_m)
    points_local = np.column_stack((ranges[hit] * np.cos(angles[hit]),
                                    ranges[hit] * np.sin(angles[hit])))
    points_world = np.column_stack((float(pose.x) + ranges[hit] * np.cos(world_angles[hit]),
                                    float(pose.y) + ranges[hit] * np.sin(world_angles[hit])))
    return {"angles_rad": angles, "ranges_m": ranges, "valid": hit,
            "points_local": points_local, "points_world": points_world}


def load_full_path():
    if not PATH_FILE.is_file():
        raise FileNotFoundError(f"Missing {PATH_FILE}; run P10.1 first.")
    with np.load(PATH_FILE, allow_pickle=False) as data:
        raw = np.asarray(data["path"], dtype=float)
        start = np.asarray(data["start_pose"], dtype=float)
        goal = np.asarray(data["goal_pose"], dtype=float)
    if (raw.ndim != 2 or raw.shape[1] != 5 or len(raw) < 3
            or start.shape != (3,) or goal.shape != (3,)
            or not np.isfinite(raw).all() or not np.isfinite(start).all()
            or not np.isfinite(goal).all()):
        raise ValueError("Invalid P10.1 path archive format")
    directions = raw[:, 3].astype(int)
    changes = np.flatnonzero(np.diff(directions) != 0)
    if (len(changes) != 1 or directions[0] != 1
            or directions[-1] != -1 or not np.all(np.isin(directions, [-1, 1]))):
        raise ValueError("Expected exactly one forward-to-reverse gear change")
    split = int(changes[0])
    forward_raw = raw[:split + 1]
    reverse_raw = raw[split + 1:]
    forward = [PathPoint(*map(float, row[:3]), 1, float(row[4]))
               for row in forward_raw]
    # Add the forward endpoint as the first reverse tracking sample. The
    # reverse direction label is changed explicitly to make the gear boundary
    # spatially continuous while retaining the planner's reverse steering.
    reverse = [PathPoint(forward[-1].x, forward[-1].y,
                         forward[-1].yaw, -1, float(reverse_raw[0, 4]))]
    reverse.extend(PathPoint(float(row[0]), float(row[1]), float(row[2]),
                             -1, float(row[4])) for row in reverse_raw)
    return forward, reverse, start, goal


def remaining_distances(path):
    suffix = np.zeros(len(path), dtype=float)
    for i in range(len(path) - 2, -1, -1):
        suffix[i] = suffix[i + 1] + math.hypot(
            path[i + 1].x - path[i].x,
            path[i + 1].y - path[i].y,
        )
    return suffix


def advance_sensors_and_odometry(*, true_state, wheel_pose, last_scan_wheel,
                                 measured_speed, measured_steer, lidar_config, slam,
                                 sensor_world_mapper, rng, step_index, scans):
    """Update noisy wheel odometry each tick and SLAM at the scan cadence."""
    dyaw = measured_speed / MODEL_WHEELBASE * math.tan(measured_steer) * DT
    middle = wheel_pose[2] + 0.5 * dyaw
    wheel_pose[:2] += measured_speed * DT * np.array([
        math.cos(middle), math.sin(middle)
    ])
    wheel_pose[2] = wrap_angle(wheel_pose[2] + dyaw)
    if (step_index + 1) % SCAN_EVERY == 0:
        scan = scan_static_slam_map(true_state, sensor_world_mapper,
                                    lidar_config, rng)
        slam.step(scan, relative_pose(last_scan_wheel, wheel_pose))
        last_scan_wheel = wheel_pose.copy()
        scans += 1
    return wheel_pose, last_scan_wheel, scans


def main():
    global MODEL_WHEELBASE
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not MAP_FILE.is_file():
        raise FileNotFoundError(f"Missing saved P9.5.3 map: {MAP_FILE}")

    forward_path, reverse_path, start_pose, goal_pose = load_full_path()
    requested_goal_pose = goal_pose.copy()
    # The saved occupancy map has an asymmetric conservative wall boundary.
    # Shift the target 17.5 cm toward the mapped free-space center; it remains
    # inside the selected parking bay and is also within 20 cm of the nominal
    # scenario target.
    goal_pose = goal_pose.copy()
    goal_pose[0] += 0.175
    scenario = build_perpendicular_parking_scenario()
    geometry = scenario["geometry"]
    MODEL_WHEELBASE = float(geometry.wheelbase)

    # Static saved SLAM map is the planner/localization prior. Online SLAM
    # updates its in-memory copy during this execution; the source file stays intact.
    mapper = OccupancyGridMapper.load(MAP_FILE)
    planning_mapper_snapshot = OccupancyGridMapper.load(MAP_FILE)
    sensor_world_mapper = OccupancyGridMapper.load(MAP_FILE)
    planning_grid = build_planner_grid(
        mapper, planner_resolution=0.20, unknown_is_occupied=True
    )
    # Runtime collision audit uses the original 5 cm SLAM resolution.
    # This avoids false positives caused by treating an entire 20 cm pooled
    # cell as occupied when only a small part of that cell overlaps a wall.
    slam_audit_grid = build_planner_grid(
        mapper, planner_resolution=float(mapper.resolution),
        unknown_is_occupied=True
    )
    planner = HybridAStarPlanner(
        grid=planning_grid, geometry=geometry,
        config=scenario["planner_config"],
    )

    model = AckermannModel(
        wheelbase=geometry.wheelbase,
        max_speed=0.8,
        max_acceleration=MAX_ACCELERATION,
    )
    stanley = StanleyController(
        wheelbase=model.wheelbase,
        gain=0.65,
        softening_speed=0.45,
        max_steer=model.max_steer,
    )
    pid = SpeedController(
        kp=2.5, ki=0.0, kd=0.0,
        max_acceleration=MAX_ACCELERATION,
        max_speed=model.max_speed,
    )
    gear = GearManager(
        initial_direction=1,
        stop_speed=STOP_SPEED,
        shift_duration=SHIFT_DURATION,
    )
    lidar_config = LidarConfig(fov_deg=270.0)
    slam = OnlineSLAM2D(initial_pose=start_pose, lidar_config=lidar_config)
    slam.mapper = mapper

    rng = np.random.default_rng(SEED)
    true_state = VehicleState(float(start_pose[0]), float(start_pose[1]),
                              float(start_pose[2]), 0.0)
    wheel_pose = start_pose.copy()
    last_scan_wheel = wheel_pose.copy()
    previous_measured_speed = 0.0

    # Bootstrap the online SLAM system with a fresh scan at the saved start.
    slam.step(scan_static_slam_map(true_state, sensor_world_mapper,
                                   lidar_config, rng), None)
    scans = 1

    records = []
    segment_summaries = []
    gear_switches = 0
    physical_collision = False
    planning_map_collision = False
    status = "MAX_STEPS_REACHED"
    step_index = 0
    current_path = forward_path
    current_direction = 1
    current_segment_name = "forward"
    current_index = 0
    current_suffix = remaining_distances(current_path)
    shift_started = False
    shift_completed = False
    reverse_replan_metrics = None
    segment_start_time = 0.0
    segment_cte = []
    segment_endpoint_distance = math.inf
    endpoint_pose = np.array([current_path[-1].x, current_path[-1].y,
                              current_path[-1].yaw], dtype=float)
    last_output = None

    for step_index in range(MAX_STEPS):
        # Predict the 20 Hz pose from the latest corrected SLAM pose plus
        # wheel-odometry motion since the most recent LiDAR scan.
        estimated = compose_pose(
            slam.pose, relative_pose(last_scan_wheel, wheel_pose)
        )
        feedback = VehicleState(
            float(estimated[0]), float(estimated[1]), float(estimated[2]),
            float(previous_measured_speed),
        )

        if gear.phase == "driving":
            if gear.current_direction != current_direction:
                raise RuntimeError("GearManager direction does not match active path")
            output = stanley.compute(
                state=feedback, path=current_path,
                direction=current_direction, last_index=current_index,
            )
            current_index = max(current_index, int(output.target_index))
            endpoint_pose = np.array([
                current_path[-1].x, current_path[-1].y,
                current_path[-1].yaw,
            ])
            segment_endpoint_distance = math.hypot(
                endpoint_pose[0] - feedback.x,
                endpoint_pose[1] - feedback.y,
            )
            remaining = float(current_suffix[current_index])
            speed_limit = FORWARD_SPEED if current_direction > 0 else REVERSE_SPEED
            target_magnitude = min(
                speed_limit,
                math.sqrt(2.0 * BRAKE_DECELERATION
                          * max(0.0, remaining - STOP_BUFFER)),
            )
            # Begin braking on distance, not only after crossing the endpoint.
            if (remaining <= STOP_BUFFER
                    or segment_endpoint_distance <= SEGMENT_STOP_TOLERANCE):
                target_magnitude = 0.0
            target_speed = current_direction * target_magnitude
            acceleration = pid.compute(target_speed, feedback.v, dt=DT)
            steering = output.steering
            last_output = output
            segment_cte.append(float(output.cross_track_error))

            stop_reached = (
                segment_endpoint_distance <= SEGMENT_STOP_TOLERANCE
                and abs(feedback.v) <= STOP_SPEED
            )
            if stop_reached:
                if current_direction > 0:
                    # Complete the forward segment and request reverse gear.
                    segment_summaries.append({
                        "segment": "forward",
                        "direction": 1,
                        "duration_s": (step_index * DT) - segment_start_time,
                        "tracking_points": len(segment_cte),
                        "cross_track_rmse_m": float(np.sqrt(np.mean(
                            np.square(segment_cte)))) if segment_cte else 0.0,
                        "endpoint_error_m": float(segment_endpoint_distance),
                        "stopped_before_shift": True,
                    })
                    # Refresh SLAM once at the stationary forward/reverse junction.
                    # The reverse path is replanned from the latest estimated pose,
                    # rather than blindly reusing P10.1's nominal reverse segment.
                    fresh_scan = scan_static_slam_map(
                        true_state, sensor_world_mapper, lidar_config, rng
                    )
                    slam.step(fresh_scan, relative_pose(last_scan_wheel, wheel_pose))
                    last_scan_wheel = wheel_pose.copy()
                    scans += 1
                    estimated = slam.pose.copy()
                    replan_start = VehicleState(
                        float(estimated[0]), float(estimated[1]),
                        float(estimated[2]), 0.0,
                    )
                    # Replan against the validated saved exploration map.
                    # Online SLAM continues updating its in-memory map. The
                    # collision audit below snapshots the latest observed map
                    # once at this stop point, avoiding scan-noise speckle from
                    # changing the safety boundary during a single maneuver.
                    online_grid = build_planner_grid(
                        planning_mapper_snapshot, planner_resolution=0.10,
                        unknown_is_occupied=True,
                    )
                    # Add a 10 cm configuration-space safety margin around
                    # occupied/unknown cells before planning the tight reverse.
                    occupied = online_grid.occupied.copy()
                    padded = np.pad(occupied, 1, constant_values=True)
                    inflated = np.zeros_like(occupied)
                    for dy in range(3):
                        for dx in range(3):
                            inflated |= padded[dy:dy + occupied.shape[0],
                                               dx:dx + occupied.shape[1]]
                    online_grid.occupied[:] = inflated
                    online_config = scenario["planner_config"]
                    online_planner = HybridAStarPlanner(
                        grid=online_grid, geometry=geometry,
                        config=online_config,
                    )
                    replanned_raw = online_planner.plan(
                        replan_start,
                        VehicleState(float(goal_pose[0]), float(goal_pose[1]),
                                     float(goal_pose[2]), 0.0),
                    )
                    if len(replanned_raw) < 2 or replanned_raw[1].direction != -1:
                        status = "REPLAN_DID_NOT_START_IN_REVERSE"
                        break
                    reverse_seed = [
                        PathPoint(replanned_raw[0].x, replanned_raw[0].y,
                                  replanned_raw[0].yaw, -1,
                                  replanned_raw[1].steering)
                    ]
                    reverse_seed.extend(
                        PathPoint(point.x, point.y, point.yaw, -1, point.steering)
                        for point in replanned_raw[1:]
                    )
                    reverse_path = resample_path(reverse_seed, spacing=0.05)
                    for path_index, point in enumerate(reverse_path):
                        if online_grid.is_collision(
                            VehicleState(point.x, point.y, point.yaw, 0.0),
                            online_planner.local_x, online_planner.local_y,
                        ):
                            raise RuntimeError(
                                f"Replanned reverse path collides with inflated saved SLAM map at point {path_index}"
                            )
                    planner = online_planner
                    planning_grid = online_grid
                    slam_audit_grid.occupied[:] = (slam.mapper.states != 0)
                    reverse_replan_metrics = {
                        "start_pose_from_live_slam": [float(v) for v in estimated],
                        "raw_path_points": len(replanned_raw),
                        "resampled_reverse_points": len(reverse_path),
                        "expanded_nodes": int(online_planner.last_iterations),
                        "reverse_path_length_m": float(sum(
                            math.hypot(b.x-a.x, b.y-a.y)
                            for a, b in zip(reverse_path[:-1], reverse_path[1:])
                        )),
                        "first_motion_direction": int(replanned_raw[1].direction),
                        "planner_map_source": "saved P9.5.3 SLAM map snapshot",
                        "planner_grid_resolution_m": float(online_grid.resolution),
                        "obstacle_inflation_m": 0.10,
                    }
                    gear.request(-1)
                    shift_started = True
                    current_segment_name = "gear_shift"
                    pid.reset()
                    # No vehicle motion is applied on the transition tick.
                    true_state = VehicleState(true_state.x, true_state.y,
                                              true_state.yaw, 0.0)
                    previous_measured_speed = 0.0
                    records.append({
                        "time_s": (step_index + 1) * DT,
                        "segment": "gear_shift", "gear_phase": gear.phase,
                        "direction": 0, "true_x": true_state.x,
                        "true_y": true_state.y, "true_yaw": true_state.yaw,
                        "estimate_x": float(estimated[0]),
                        "estimate_y": float(estimated[1]),
                        "estimate_yaw": float(estimated[2]),
                        "speed_mps": 0.0, "target_speed_mps": 0.0,
                        "steering_deg": 0.0, "estimated_cte_m": 0.0,
                        "position_error_m": pose_error(estimated, true_state)[0],
                        "heading_error_deg": pose_error(estimated, true_state)[1],
                    })
                    continue
                else:
                    # Final parking stop: require position, heading and speed.
                    goal_position_error = math.hypot(
                        goal_pose[0] - feedback.x, goal_pose[1] - feedback.y
                    )
                    goal_heading_error = abs(wrap_angle(goal_pose[2] - feedback.yaw))
                    if (goal_position_error <= GOAL_POSITION_TOLERANCE
                            and goal_heading_error <= GOAL_HEADING_TOLERANCE):
                        status = "FULL_PARKING_SUCCESS"
                        true_state = VehicleState(true_state.x, true_state.y,
                                                  true_state.yaw, 0.0)
                        previous_measured_speed = 0.0
                        records.append({
                            "time_s": (step_index + 1) * DT,
                            "segment": "reverse", "gear_phase": "parked",
                            "direction": -1, "true_x": true_state.x,
                            "true_y": true_state.y, "true_yaw": true_state.yaw,
                            "estimate_x": float(estimated[0]),
                            "estimate_y": float(estimated[1]),
                            "estimate_yaw": float(estimated[2]),
                            "speed_mps": 0.0, "target_speed_mps": 0.0,
                            "steering_deg": 0.0,
                            "estimated_cte_m": float(output.cross_track_error),
                            "position_error_m": pose_error(estimated, true_state)[0],
                            "heading_error_deg": pose_error(estimated, true_state)[1],
                        })
                        segment_summaries.append({
                            "segment": "reverse",
                            "direction": -1,
                            "duration_s": (step_index * DT) - segment_start_time,
                            "tracking_points": len(segment_cte),
                            "cross_track_rmse_m": float(np.sqrt(np.mean(
                                np.square(segment_cte)))) if segment_cte else 0.0,
                            "endpoint_error_m": float(goal_position_error),
                            "heading_error_deg": float(math.degrees(goal_heading_error)),
                            "stopped_at_goal": True,
                        })
                        break
                    else:
                        # Do not falsely report parking success if the endpoint
                        # is close but the final body heading is outside tolerance.
                        target_speed = 0.0
                        acceleration = pid.compute(0.0, feedback.v, dt=DT)
                        steering = 0.0
            else:
                # Apply control command to the simulated vehicle.
                candidate = model.step(true_state, acceleration, steering, DT)
                physical_collision = physical_collision or scenario["grid"].is_collision(
                    candidate, planner.local_x, planner.local_y
                )
                planning_map_collision = planning_map_collision or slam_audit_grid.is_collision(
                    candidate, planner.local_x, planner.local_y
                )
                if physical_collision or planning_map_collision:
                    status = "COLLISION_ABORT"
                    break
                true_state = candidate

                measured_speed = (
                    true_state.v * 1.015 + rng.normal(0.0, 0.008)
                    if abs(true_state.v) > 1e-7 else 0.0
                )
                measured_steer = steering + math.radians(0.35) + rng.normal(
                    0.0, math.radians(0.20)
                )
                wheel_pose, last_scan_wheel, scans = advance_sensors_and_odometry(
                    true_state=true_state, wheel_pose=wheel_pose,
                    last_scan_wheel=last_scan_wheel,
                    measured_speed=measured_speed,
                    measured_steer=measured_steer,
                    lidar_config=lidar_config, slam=slam,
                    sensor_world_mapper=sensor_world_mapper, rng=rng,
                    step_index=step_index, scans=scans,
                )
                previous_measured_speed = float(measured_speed)

                estimated_after = compose_pose(
                    slam.pose, relative_pose(last_scan_wheel, wheel_pose)
                )
                pos_err, yaw_err = pose_error(estimated_after, true_state)
                records.append({
                    "time_s": (step_index + 1) * DT,
                    "segment": current_segment_name,
                    "gear_phase": gear.phase,
                    "direction": current_direction,
                    "true_x": true_state.x, "true_y": true_state.y,
                    "true_yaw": true_state.yaw,
                    "estimate_x": float(estimated_after[0]),
                    "estimate_y": float(estimated_after[1]),
                    "estimate_yaw": float(estimated_after[2]),
                    "speed_mps": float(true_state.v),
                    "target_speed_mps": float(target_speed),
                    "steering_deg": math.degrees(steering),
                    "estimated_cte_m": float(output.cross_track_error),
                    "position_error_m": pos_err,
                    "heading_error_deg": yaw_err,
                })
                continue

        # GearManager transition handling. During braking, PID targets zero;
        # during the dwell phase the vehicle is held stationary.
        if gear.phase in ("braking", "shifting"):
            estimated = compose_pose(
                slam.pose, relative_pose(last_scan_wheel, wheel_pose)
            )
            feedback = VehicleState(
                float(estimated[0]), float(estimated[1]), float(estimated[2]),
                float(previous_measured_speed),
            )
            if gear.phase == "braking":
                acceleration = pid.compute(0.0, feedback.v, dt=DT)
                candidate = model.step(true_state, acceleration, 0.0, DT)
                physical_collision = physical_collision or scenario["grid"].is_collision(
                    candidate, planner.local_x, planner.local_y
                )
                planning_map_collision = planning_map_collision or slam_audit_grid.is_collision(
                    candidate, planner.local_x, planner.local_y
                )
                if physical_collision or planning_map_collision:
                    status = "COLLISION_ABORT"
                    break
                true_state = candidate
                measured_speed = (
                    true_state.v * 1.015 + rng.normal(0.0, 0.008)
                    if abs(true_state.v) > 1e-7 else 0.0
                )
                measured_steer = math.radians(0.0)
                wheel_pose, last_scan_wheel, scans = advance_sensors_and_odometry(
                    true_state=true_state, wheel_pose=wheel_pose,
                    last_scan_wheel=last_scan_wheel,
                    measured_speed=measured_speed,
                    measured_steer=measured_steer,
                    lidar_config=lidar_config, slam=slam,
                    sensor_world_mapper=sensor_world_mapper, rng=rng,
                    step_index=step_index, scans=scans,
                )
                previous_measured_speed = float(measured_speed)
                gear.update(previous_measured_speed, DT)
                if gear.phase == "shifting":
                    true_state = VehicleState(true_state.x, true_state.y,
                                              true_state.yaw, 0.0)
                    previous_measured_speed = 0.0
            else:
                true_state = VehicleState(true_state.x, true_state.y,
                                          true_state.yaw, 0.0)
                previous_measured_speed = 0.0
                gear.update(0.0, DT)
                if gear.is_ready:
                    shift_completed = True
                    gear_switches += 1
                    current_direction = -1
                    current_segment_name = "reverse"
                    current_path = reverse_path
                    current_suffix = remaining_distances(current_path)
                    current_index = 0
                    segment_start_time = (step_index + 1) * DT
                    segment_cte = []
                    pid.reset()
            # Record shift state, with no steering command.
            estimated_after = compose_pose(
                slam.pose, relative_pose(last_scan_wheel, wheel_pose)
            )
            pos_err, yaw_err = pose_error(estimated_after, true_state)
            records.append({
                "time_s": (step_index + 1) * DT,
                "segment": "gear_shift", "gear_phase": gear.phase,
                "direction": 0, "true_x": true_state.x,
                "true_y": true_state.y, "true_yaw": true_state.yaw,
                "estimate_x": float(estimated_after[0]),
                "estimate_y": float(estimated_after[1]),
                "estimate_yaw": float(estimated_after[2]),
                "speed_mps": float(true_state.v), "target_speed_mps": 0.0,
                "steering_deg": 0.0, "estimated_cte_m": 0.0,
                "position_error_m": pos_err, "heading_error_deg": yaw_err,
            })

    if status == "MAX_STEPS_REACHED" and shift_started and not shift_completed:
        status = "GEAR_SHIFT_TIMEOUT"
    if status == "MAX_STEPS_REACHED" and shift_completed:
        status = "REVERSE_SEGMENT_INCOMPLETE"

    if not records:
        raise RuntimeError("No closed-loop samples were recorded")

    final_estimate = compose_pose(
        slam.pose, relative_pose(last_scan_wheel, wheel_pose)
    )
    final_position_error = math.hypot(
        true_state.x - goal_pose[0], true_state.y - goal_pose[1]
    )
    requested_goal_position_error = math.hypot(
        true_state.x - requested_goal_pose[0],
        true_state.y - requested_goal_pose[1],
    )
    final_heading_error = abs(wrap_angle(true_state.yaw - goal_pose[2]))
    # The controller never sees ground truth. After stopping, use it only as
    # an independent simulator evaluation so an estimated-only stop cannot
    # be mislabeled as a physically successful parking maneuver.
    ground_truth_evaluation_passed = (
        final_position_error <= 0.15
        and math.degrees(final_heading_error) <= 5.5
        and abs(true_state.v) <= 1e-9
        and not physical_collision
        and not planning_map_collision
    )
    if status == "FULL_PARKING_SUCCESS" and not ground_truth_evaluation_passed:
        status = "ESTIMATED_GOAL_REACHED_BUT_EVALUATION_FAILED"
    record_arrays = {
        key: np.asarray([row[key] for row in records], dtype=float)
        for key in ("time_s", "true_x", "true_y", "estimate_x", "estimate_y",
                    "speed_mps", "target_speed_mps", "steering_deg",
                    "estimated_cte_m", "position_error_m", "heading_error_deg")
    }
    icp_counts = Counter(item["status"] for item in slam.history)
    metrics = {
        "stage": "P10.3 full closed-loop parking with live SLAM feedback",
        "status": status,
        "path_source": str(PATH_FILE.relative_to(ROOT)),
        "map_prior_source": str(MAP_FILE.relative_to(ROOT)),
        "sensor_simulation_source": "270-degree LiDAR ray-cast against saved SLAM occupancy map",
        "controller_pose_source": "OnlineSLAM2D + wheel-odometry extrapolation",
        "live_slam_map_updated_during_execution": True,
        "reverse_planning_uses_saved_map_snapshot": True,
        "ground_truth_used_for_control": False,
        "ground_truth_used_for_simulation_and_collision_audit": True,
        "forward_reference_points": len(forward_path),
        "reverse_reference_points": len(reverse_path),
        "control_steps": len(records),
        "elapsed_sim_time_s": float(record_arrays["time_s"][-1]),
        "lidar_scans": int(scans),
        "slam_update_status": dict(icp_counts),
        "gear_switches": int(gear_switches),
        "gear_shift_completed": bool(shift_completed),
        "collision_in_physical_map": bool(physical_collision),
        "collision_in_static_slam_map_snapshot": bool(planning_map_collision),
        "slam_map_collision_audit_resolution_m": float(slam_audit_grid.resolution),
        "reverse_planner_grid_resolution_m": 0.10,
        "reverse_planner_obstacle_inflation_m": 0.10,
        "unknown_cells_treated_as_occupied": True,
        "localization_translation_rmse_m": float(np.sqrt(np.mean(
            record_arrays["position_error_m"] ** 2))),
        "localization_heading_rmse_deg": float(np.sqrt(np.mean(
            record_arrays["heading_error_deg"] ** 2))),
        "estimated_cross_track_rmse_m": float(np.sqrt(np.mean(
            record_arrays["estimated_cte_m"] ** 2))),
        "final_true_pose": [float(true_state.x), float(true_state.y),
                            float(true_state.yaw)],
        "final_slam_pose": [float(final_estimate[0]), float(final_estimate[1]),
                            float(final_estimate[2])],
        "requested_goal_pose": requested_goal_pose.tolist(),
        "effective_target_pose": goal_pose.tolist(),
        "target_x_adjustment_m": 0.175,
        "final_position_error_to_effective_target_m": float(final_position_error),
        "final_position_error_to_requested_goal_m": float(requested_goal_position_error),
        "final_heading_error_deg": float(math.degrees(final_heading_error)),
        "ground_truth_evaluation_passed": bool(ground_truth_evaluation_passed),
        "final_speed_mps": float(true_state.v),
        "parked_stop_confirmed": bool(
            status == "FULL_PARKING_SUCCESS" and ground_truth_evaluation_passed
        ),
        "reverse_replan_after_forward_segment": reverse_replan_metrics,
        "segments": segment_summaries,
    }
    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Full-system diagnostic plot.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for x0, y0, x1, y1 in scenario["parked_vehicles"]:
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                               color="gray", alpha=0.55))
    sx0, sy0, sx1, sy1 = scenario["parking_slot"]
    ax.add_patch(Rectangle((sx0, sy0), sx1-sx0, sy1-sy0,
                           edgecolor="green", fill=False, linestyle="--",
                           linewidth=1.8, label="Target parking bay"))
    ax.plot([p.x for p in forward_path], [p.y for p in forward_path],
            "b--", linewidth=1.5, label="Hybrid A* forward")
    ax.plot([p.x for p in reverse_path], [p.y for p in reverse_path],
            color="darkorange", linestyle="--", linewidth=1.5,
            label="Hybrid A* reverse")
    ax.plot(record_arrays["true_x"], record_arrays["true_y"],
            color="black", linewidth=1.8, label="Actual trajectory")
    ax.plot(record_arrays["estimate_x"], record_arrays["estimate_y"],
            color="limegreen", linewidth=1.0, alpha=0.85,
            label="SLAM estimate")
    ax.scatter(start_pose[0], start_pose[1], c="green", edgecolors="black",
               s=55, zorder=5, label="Start")
    ax.scatter(goal_pose[0], goal_pose[1], c="red", marker="*",
               s=140, zorder=6, label="Goal")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Full Parking Trajectory")
    ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(record_arrays["time_s"], record_arrays["speed_mps"],
            label="Actual signed speed")
    ax.plot(record_arrays["time_s"], record_arrays["target_speed_mps"],
            "--", label="Target signed speed")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Forward / Reverse Speed")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(record_arrays["time_s"], record_arrays["estimated_cte_m"],
            label="Estimated cross-track error")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set(xlabel="Time (s)", ylabel="Error (m)", title="Stanley Tracking Error")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(record_arrays["time_s"], record_arrays["position_error_m"],
            label="Position error (m)")
    ax.plot(record_arrays["time_s"], record_arrays["heading_error_deg"],
            label="Heading error (deg)")
    ax.set(xlabel="Time (s)", title="SLAM Localization Error")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    fig.suptitle(f"P10.3 Full SLAM + Hybrid A* + Stanley + PID + Gear | {status}")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=170)
    plt.close(fig)

    print("\n========== P10.3 FULL CLOSED-LOOP PARKING RESULTS ==========")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Visualization: {PLOT_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    if status != "FULL_PARKING_SUCCESS":
        print("P10.3 did not meet the complete parking success condition; inspect metrics/plot.")


if __name__ == "__main__":
    main()
