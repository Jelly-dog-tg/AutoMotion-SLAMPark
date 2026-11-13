"""P9.5.2: Online LiDAR + odometry + Scan-to-Map ICP + occupancy mapping.

This is a sequential simulated online pipeline (not hardware or wall-clock real-time).
Only the LiDAR simulator and evaluation inspect the true vehicle pose.
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.vehicle.ackermann import AckermannModel, VehicleState
from parking_planner.sensing.lidar_2d import Lidar2D, LidarConfig
from parking_planner.slam.online_slam import OnlineSLAM2D, wrap_angle

RESULTS = ROOT / "results"
DT = 0.05
SCAN_STEPS = 8   # 0.4 seconds at 20 Hz vehicle update
TOTAL_STEPS = 260  # 13 seconds
SEED = 2026


def relative_pose(previous, current):
    x0, y0, yaw0 = previous
    x1, y1, yaw1 = current
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(yaw0), math.sin(yaw0)
    return np.array([c*dx + s*dy, -s*dx + c*dy,
                     wrap_angle(yaw1-yaw0)])


def calculate_error(estimated, reference):
    t = np.linalg.norm(estimated[:, :2]-reference[:, :2], axis=1)
    yaw = np.degrees(np.abs(np.arctan2(
        np.sin(estimated[:,2]-reference[:,2]),
        np.cos(estimated[:,2]-reference[:,2]),
    )))
    return {"translation_rmse_m": float(np.sqrt(np.mean(t*t))),
            "heading_rmse_deg": float(np.sqrt(np.mean(yaw*yaw))),
            "final_position_error_m": float(t[-1]),
            "final_heading_error_deg": float(yaw[-1])}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    scenario = build_perpendicular_parking_scenario()
    true = scenario["start"]
    wheel_pose = np.array([true.x, true.y, true.yaw], dtype=float)
    last_wheel_at_scan = wheel_pose.copy()

    vehicle = AckermannModel(wheelbase=scenario["geometry"].wheelbase)
    lidar_config = LidarConfig()
    lidar = Lidar2D(lidar_config)
    slam = OnlineSLAM2D(
        initial_pose=wheel_pose,
        lidar_config=lidar_config,
    )
    true_poses, odom_poses, est_poses = [], [], []

    for step in range(TOTAL_STEPS+1):
        # Produce a new observation from the CURRENT simulated vehicle state.
        # No saved P9.3/P9.5.1 scans or map are loaded.
        if step % SCAN_STEPS == 0:
            scan = lidar.scan(true, scenario["parked_vehicles"], rng=rng)
            delta = None if step == 0 else relative_pose(
                last_wheel_at_scan, wheel_pose,
            )
            slam.step(scan, delta)
            last_wheel_at_scan = wheel_pose.copy()
            true_poses.append([true.x, true.y, true.yaw])
            odom_poses.append(wheel_pose.copy())
            est_poses.append(slam.pose.copy())

        if step == TOTAL_STEPS:
            break

        # Simple scripted exploration in the unobstructed parking aisle.
        # This is NOT autonomous frontier exploration; keyboard control is P9.5.3.
        target_v = 0.45 if step < 225 else 0.0
        steering_command = 0.0
        accel = np.clip(3.0*(target_v-true.v), -0.8, 0.8)
        true = vehicle.step(true, float(accel), steering_command, dt=DT)

        # Simulated wheel-speed and steering measurements, not true x/y/yaw.
        measured_speed = true.v * 1.025 + rng.normal(0.0, 0.025)
        measured_steering = steering_command + math.radians(1.5) + rng.normal(0.0, math.radians(0.30))
        delta_yaw = (measured_speed / vehicle.wheelbase
                     * math.tan(measured_steering) * DT)
        mid_yaw = wheel_pose[2] + delta_yaw/2.0
        wheel_pose[:2] += measured_speed*DT*np.array(
            [math.cos(mid_yaw), math.sin(mid_yaw)]
        )
        wheel_pose[2] = wrap_angle(wheel_pose[2]+delta_yaw)

    reference = np.asarray(true_poses, dtype=float)
    odometry = np.asarray(odom_poses, dtype=float)
    estimates = np.asarray(est_poses, dtype=float)
    counts = Counter(x["status"] for x in slam.history)
    reasons = Counter(x["reason"] for x in slam.history if x["reason"])
    states = slam.mapper.states

    summary = {
        "stage": "P9.5.2 Online SLAM sequential simulation",
        "scenario": "straight aisle exploration; scripted controls",
        "scan_count": len(slam.history),
        "lidar_beams": lidar_config.num_beams,
        "vehicle_dt_s": DT, "scan_interval_s": DT*SCAN_STEPS,
        "update_status": dict(counts), "fallback_reasons": dict(reasons),
        "odometry_errors": calculate_error(odometry, reference),
        "slam_errors": calculate_error(estimates, reference),
        "map_cells": {"occupied": int(np.sum(states==1)),
                      "free": int(np.sum(states==0)),
                      "unknown": int(np.sum(states==-1))},
        "ground_truth_usage": "LiDAR simulation and post-run evaluation only",
        "history": slam.history,
    }

    (RESULTS / "p9_5_2_online_slam_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(RESULTS / "p9_5_2_online_slam_map.npz",
                        **slam.mapper.arrays(),
                        estimated_poses=estimates,
                        odometry_poses=odometry,
                        reference_poses=reference)

    fig, axes = plt.subplots(1,3,figsize=(16,5))
    ax = axes[0]
    ax.imshow(states + 1, origin="lower",
              extent=[slam.mapper.origin_x, slam.mapper.origin_x+slam.mapper.width,
                      slam.mapper.origin_y, slam.mapper.origin_y+slam.mapper.height],
              cmap=ListedColormap(["#aaaaaa", "white", "black"]),
              vmin=0,vmax=2,interpolation="nearest")
    for traj, label in [(reference,"Ground truth"),
                        (odometry,"Wheel odometry"),
                        (estimates,"Online SLAM")]:
        ax.plot(traj[:,0],traj[:,1],label=label)
    ax.set(xlabel="X (m)",ylabel="Y (m)",title="Online occupancy mapping")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    t_axis = np.arange(len(reference))*DT*SCAN_STEPS
    for ax, idx, title in [(axes[1],0,"Translation error (m)"),
                           (axes[2],1,"Heading error (deg)")]:
        for data, label in [(odometry,"Wheel odometry"),
                            (estimates,"Online SLAM")]:
            if idx == 0:
                e = np.linalg.norm(data[:,:2]-reference[:,:2],axis=1)
            else:
                e = np.degrees(np.abs(np.arctan2(
                    np.sin(data[:,2]-reference[:,2]),
                    np.cos(data[:,2]-reference[:,2]))))
            ax.plot(t_axis,e,label=label)
        ax.set(xlabel="Simulation time (s)", title=title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("P9.5.2 Online LiDAR SLAM (scripted aisle exploration)")
    fig.tight_layout()
    fig.savefig(RESULTS / "p9_5_2_online_slam.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='history'},
                     ensure_ascii=False, indent=2))
    print("Output: results/p9_5_2_online_slam_*.{json,npz,png}")


if __name__ == "__main__":
    main()
