
"""P11.6.4: Live PyBullet driving and online SLAM.

P: pause/run
W/S: speed
A/D: steering
C: center steering
Space: brake
M: save
Q/Esc: brake, save and quit
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)
from parking_planner.simulation.pybullet.scene import ParkingScene
from parking_planner.simulation.pybullet.physics_vehicle import PhysicsVehicle
from parking_planner.simulation.pybullet.display import WINDOW_OPTIONS
from parking_planner.simulation.pybullet.hud import EdgeStatusBars
from parking_planner.sensing.lidar_2d import LidarConfig
from parking_planner.sensing.pybullet_lidar import PyBulletLidar2D
from parking_planner.slam.online_slam import (
    OnlineSLAM2D,
    compose_pose,
    wrap_angle,
)

DT = 1.0 / 240.0
SCAN_EVERY = 96
MAP_REDRAW_EVERY = 2
STEER_STEP_DEG = 5.0
SPEED_STEP_MPS = 0.10
MAX_MANUAL_SPEED = 0.45

OUTPUT = ROOT / "results"
PREFIX = "p11_6_4"


def pressed(events, key):
    return any(
        events.get(ord(k), 0) & p.KEY_WAS_TRIGGERED
        for k in (key.lower(), key.upper())
    )


def relative_pose(a, b):
    delta = np.asarray(b[:2]) - np.asarray(a[:2])
    c = math.cos(float(a[2]))
    s = math.sin(float(a[2]))

    return np.array([
        c * delta[0] + s * delta[1],
        -s * delta[0] + c * delta[1],
        wrap_angle(float(b[2] - a[2])),
    ])


def rms(values):
    a = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else None


class LiveMapWindow:
    """Display the incrementally updated SLAM map."""

    def __init__(self, mapper):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm

        self.plt = plt
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(7, 6.8))

        manager = self.fig.canvas.manager
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("P11.6.4 Online SLAM Map")

        x0 = mapper.origin_x
        y0 = mapper.origin_y

        self.image = self.ax.imshow(
            mapper.states,
            origin="lower",
            interpolation="nearest",
            extent=(
                x0,
                x0 + mapper.width,
                y0,
                y0 + mapper.height,
            ),
            cmap=ListedColormap([
                "#979ca5",
                "#ffffff",
                "#16181c",
            ]),
            norm=BoundaryNorm(
                [-1.5, -0.5, 0.5, 1.5],
                3,
            ),
        )

        (self.track,) = self.ax.plot(
            [], [],
            color="forestgreen",
            linewidth=1.5,
            label="SLAM trajectory",
        )

        (self.car,) = self.ax.plot(
            [], [],
            "ro",
            markersize=6,
            label="Estimated vehicle",
        )

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.13)
        self.ax.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def update(self, mapper, poses, info):
        if not self.plt.fignum_exists(self.fig.number):
            return

        self.image.set_data(mapper.states)

        poses = np.asarray(poses, dtype=float).reshape(-1, 3)

        self.track.set_data(
            poses[:, 0],
            poses[:, 1],
        )
        self.car.set_data(
            [poses[-1, 0]],
            [poses[-1, 1]],
        )

        self.ax.set_title(
            f"Live SLAM | Scans: {len(poses)} | "
            f"ICP: {info['icp']} | "
            f"Odom fallback: {info['odom']}\n"
            "Gray: Unknown | White: Free | Black: Occupied"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save(self, path):
        if self.plt.fignum_exists(self.fig.number):
            self.fig.savefig(path, dpi=160)

    def close(self):
        if self.plt.fignum_exists(self.fig.number):
            self.plt.close(self.fig)


class PhysicalSlamExplorer:
    """Physical vehicle, live LiDAR and online SLAM."""

    def __init__(self):
        self.cid = p.connect(
            p.GUI,
            options=WINDOW_OPTIONS,
        )

        if self.cid < 0:
            raise RuntimeError("PyBullet GUI initialization failed")

        p.setTimeStep(DT, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(
            numSolverIterations=100,
            deterministicOverlappingPairs=1,
            physicsClientId=self.cid,
        )

        self.scenario = build_perpendicular_parking_scenario()
        self.scene = ParkingScene(self.cid, self.scenario)

        start = self.scenario["start"]

        self.vehicle = PhysicsVehicle(
            self.cid,
            self.scenario["geometry"],
            (start.x, start.y, start.yaw),
        )

        self.lidar_config = LidarConfig(
            num_beams=180,
            fov_deg=270.0,
            min_range_m=0.12,
            max_range_m=12.0,
            range_std_m=0.015,
        )

        self.lidar = PyBulletLidar2D(
            self.cid,
            self.vehicle.body,
            config=self.lidar_config,
            height_m=1.25,
        )

        self.rng = np.random.default_rng(20261112)

        # Settle the physical vehicle before exploration.
        for _ in range(240):
            self.vehicle.command(0.0, 0.0, DT)
            p.stepSimulation(physicsClientId=self.cid)

        initial_state = self.vehicle.state()

        self.initial_pose = np.array([
            initial_state["x"],
            initial_state["y"],
            initial_state["yaw"],
        ], dtype=float)

        self.odom_pose = self.initial_pose.copy()
        self.scan_odom_pose = self.odom_pose.copy()

        self.slam = OnlineSLAM2D(
            initial_pose=self.initial_pose,
            lidar_config=self.lidar_config,
        )

        self.hud = EdgeStatusBars(self.cid)
        self.map_window = None

        self.scene.camera_target = [11.8, 11.4, 0.0]
        self.scene.configure_gui()

        self.paused = True
        self.closing = False
        self.closing_at_step = None

        self.target_speed = 0.0
        self.target_steer = 0.0

        self.steps = 0
        self.time_s = 0.0

        self.collision_samples = 0
        self.self_hit_total = 0

        self.status_counter = Counter()
        self.reason_counter = Counter()

        self.estimated_poses = []
        self.odometry_poses = []
        self.reference_poses = []
        self.scan_times = []
        self.scan_ranges = []
        self.scan_valid = []
        self.physical_trace = []

        self.slam_position_errors = []
        self.odom_position_errors = []
        self.slam_heading_errors = []
        self.odom_heading_errors = []

        self.last_scan = None
        self.last_info = None
        self.ray_lines = []

        # Initial live LiDAR scan.
        self.capture_scan()

        # Open the separate live mapping window.
        self.map_window = LiveMapWindow(self.slam.mapper)
        self.refresh_map()
        self.update_hud()

    def integrate_odometry(self, measured_state):
        """Integrate noisy wheel speed and steering measurements."""

        speed = (
            float(measured_state["speed"]) * 1.02
            + self.rng.normal(0.0, 0.008)
        )

        steering = (
            float(measured_state["steering"])
            + math.radians(0.3)
            + self.rng.normal(0.0, math.radians(0.2))
        )

        wheelbase = self.scenario["geometry"].wheelbase

        dtheta = (
            speed / wheelbase
            * math.tan(steering)
            * DT
        )

        heading_mid = self.odom_pose[2] + 0.5 * dtheta

        self.odom_pose[0] += (
            speed * DT * math.cos(heading_mid)
        )

        self.odom_pose[1] += (
            speed * DT * math.sin(heading_mid)
        )

        self.odom_pose[2] = wrap_angle(
            self.odom_pose[2] + dtheta
        )

    def estimated_current_pose(self):
        """SLAM estimate extrapolated by wheel odometry."""

        delta = relative_pose(
            self.scan_odom_pose,
            self.odom_pose,
        )

        return compose_pose(
            self.slam.pose,
            delta,
        )

    def draw_lidar(self, scan):
        for uid in self.ray_lines:
            p.removeUserDebugItem(
                uid,
                physicsClientId=self.cid,
            )

        self.ray_lines.clear()

        state = self.vehicle.state()

        x = state["x"]
        y = state["y"]
        yaw = state["yaw"]

        indices = np.linspace(
            0,
            self.lidar_config.num_beams - 1,
            24,
            dtype=int,
        )

        for i in indices:
            if not scan["valid"][i]:
                continue

            angle = yaw + scan["angles_rad"][i]
            distance = float(scan["ranges_m"][i])

            uid = p.addUserDebugLine(
                [x, y, self.lidar.height_m],
                [
                    x + distance * math.cos(angle),
                    y + distance * math.sin(angle),
                    self.lidar.height_m,
                ],
                [0.2, 0.88, 0.96],
                lineWidth=1.0,
                physicsClientId=self.cid,
            )

            self.ray_lines.append(uid)

    def capture_scan(self):
        """Capture a new physical LiDAR scan and update SLAM."""

        state = self.vehicle.state()

        truth_pose = np.array([
            state["x"],
            state["y"],
            state["yaw"],
        ], dtype=float)

        scan = self.lidar.scan(state)

        if not self.estimated_poses:
            delta = None
        else:
            delta = relative_pose(
                self.scan_odom_pose,
                self.odom_pose,
            )

        info = self.slam.step(scan, delta)

        self.scan_odom_pose = self.odom_pose.copy()

        self.status_counter[info["status"]] += 1
        self.reason_counter[
            info.get("reason") or "accepted_or_bootstrap"
        ] += 1

        self.self_hit_total += int(scan["self_hits"])

        self.estimated_poses.append(self.slam.pose.copy())
        self.odometry_poses.append(self.odom_pose.copy())
        self.reference_poses.append(truth_pose)

        self.scan_times.append(self.time_s)
        self.scan_ranges.append(
            np.asarray(scan["ranges_m"], dtype=np.float32).copy()
        )
        self.scan_valid.append(
            np.asarray(scan["valid"], dtype=bool).copy()
        )

        self.last_scan = scan
        self.last_info = info

        self.draw_lidar(scan)

        # Ground truth is evaluation-only, not SLAM feedback.
        self.slam_position_errors.append(
            float(np.linalg.norm(
                self.slam.pose[:2] - truth_pose[:2]
            ))
        )

        self.odom_position_errors.append(
            float(np.linalg.norm(
                self.odom_pose[:2] - truth_pose[:2]
            ))
        )

        self.slam_heading_errors.append(
            math.degrees(abs(wrap_angle(
                self.slam.pose[2] - truth_pose[2]
            )))
        )

        self.odom_heading_errors.append(
            math.degrees(abs(wrap_angle(
                self.odom_pose[2] - truth_pose[2]
            )))
        )

        if (
            self.map_window is not None
            and len(self.estimated_poses) % MAP_REDRAW_EVERY == 0
        ):
            self.refresh_map()

    def refresh_map(self):
        if self.map_window is None:
            return

        self.map_window.update(
            self.slam.mapper,
            self.estimated_poses,
            {
                "icp": self.status_counter["ICP_ACCEPT"],
                "odom": self.status_counter["ODOM_FALLBACK"],
            },
        )

    def step(self):
        """Advance actual vehicle physics, odometry and SLAM."""

        self.vehicle.command(
            self.target_speed,
            self.target_steer,
            DT,
        )

        p.stepSimulation(physicsClientId=self.cid)

        state = self.vehicle.state()

        self.steps += 1
        self.time_s += DT

        self.integrate_odometry(state)

        self.physical_trace.append([
            self.time_s,
            state["x"],
            state["y"],
            state["yaw"],
            state["speed"],
            state["steering"],
        ])

        contacts = p.getContactPoints(
            bodyA=self.vehicle.body,
            physicsClientId=self.cid,
        )

        if any(
            contact[2] in self.scene.obstacle_ids
            for contact in contacts
        ):
            self.collision_samples += 1
            self.target_speed = 0.0

        if self.steps % SCAN_EVERY == 0:
            self.capture_scan()

    def update_hud(self):
        state = self.vehicle.state()
        estimate = self.estimated_current_pose()

        status = (
            "STOPPING" if self.closing
            else "PAUSED" if self.paused
            else "EXPLORING"
        )

        self.hud.update(
            (
                f"P11.6.4 LIVE SLAM | {status} "
                f"| {self.time_s:.1f}s "
                f"| scans={len(self.scan_times)} "
                f"| ICP={self.status_counter['ICP_ACCEPT']} "
                f"| fallback={self.status_counter['ODOM_FALLBACK']}"
            ),
            (
                f"v={state['speed']:+.2f} m/s "
                f"| target={self.target_speed:+.2f} m/s "
                f"| steer={math.degrees(state['steering']):+.1f} deg "
                f"| SLAM=({estimate[0]:.2f},{estimate[1]:.2f},"
                f"{math.degrees(estimate[2]):+.0f} deg)"
            ),
            (
                "Click 3D window | P pause | W/S speed "
                "| A/D steer | C center | SPACE brake "
                "| M save | Q quit"
            ),
            (
                "CYAN: LiDAR hits | MAP: gray unknown / "
                "white free / black occupied"
            ),
        )

    def save(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        self.refresh_map()

        mapper = self.slam.mapper

        np.savez_compressed(
            OUTPUT / f"{PREFIX}_live_slam_map.npz",
            **mapper.arrays(),
            estimated_poses=np.asarray(
                self.estimated_poses, dtype=float
            ),
            odometry_poses=np.asarray(
                self.odometry_poses, dtype=float
            ),
            reference_poses=np.asarray(
                self.reference_poses, dtype=float
            ),
            scan_timestamps_s=np.asarray(
                self.scan_times, dtype=float
            ),
            angles_rad=self.lidar.angles,
            ranges_m=np.asarray(
                self.scan_ranges, dtype=np.float32
            ),
            valid=np.asarray(
                self.scan_valid, dtype=bool
            ),
            physical_trace=np.asarray(
                self.physical_trace, dtype=float
            ).reshape(-1, 6),
            trace_columns=np.asarray([
                "time_s", "x", "y", "yaw", "speed", "steering"
            ], dtype="U20"),
        )

        states = mapper.states

        summary = {
            "stage": "P11.6.4 live physical driving + online SLAM",
            "status": "LIVE_MAP_SAVED_LOCALIZATION_UNOPTIMIZED",
            "scan_count": len(self.scan_times),
            "scan_period_s": SCAN_EVERY * DT,
            "lidar_beams": self.lidar_config.num_beams,
            "scan_valid_hits_total": int(np.sum(self.scan_valid)),
            "scan_self_hits": self.self_hit_total,
            "icp_accepts": self.status_counter["ICP_ACCEPT"],
            "odom_fallbacks": self.status_counter["ODOM_FALLBACK"],
            "status_counts": dict(self.status_counter),
            "rejection_reasons": dict(self.reason_counter),
            "occupied_cells": int(np.count_nonzero(states == 1)),
            "free_cells": int(np.count_nonzero(states == 0)),
            "unknown_cells": int(np.count_nonzero(states == -1)),
            "odom_position_rmse_m": rms(self.odom_position_errors),
            "slam_position_rmse_m": rms(self.slam_position_errors),
            "odom_heading_rmse_deg": rms(self.odom_heading_errors),
            "slam_heading_rmse_deg": rms(self.slam_heading_errors),
            "collision_contact_samples": self.collision_samples,
            "simulation_time_s": self.time_s,
            "final_true_pose_evaluation_only": (
                self.reference_poses[-1].tolist()
            ),
            "final_slam_pose": self.slam.pose.tolist(),
            "final_wheel_odometry_pose": self.odom_pose.tolist(),
            "ground_truth_used_for_planning_or_slam_updates": False,
            "ground_truth_uses": (
                "Virtual sensor ray casting, initial pose anchor, "
                "and evaluation only"
            ),
            "saved_p10_reference_loaded": False,
            "known_limit": (
                "ICP unchanged; review map and localization "
                "before autonomous planning"
            ),
        }

        metrics_file = OUTPUT / f"{PREFIX}_live_slam_metrics.json"

        metrics_file.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.map_window:
            self.map_window.save(
                OUTPUT / f"{PREFIX}_live_slam_map.png"
            )

        print(json.dumps(summary, indent=2, ensure_ascii=False))

    def run(self):
        print(
            "P11.6.4 ready. Click the PyBullet 3D window "
            "to enable keyboard controls."
        )

        frame = 0

        while p.isConnected(self.cid):
            frame_start = time.monotonic()
            frame += 1

            events = p.getKeyboardEvents(physicsClientId=self.cid)

            if (
                pressed(events, "q")
                or events.get(27, 0) & p.KEY_WAS_TRIGGERED
            ):
                if not self.closing:
                    self.closing_at_step = self.steps

                self.target_speed = 0.0
                self.closing = True
                self.paused = False

            if pressed(events, "p") and not self.closing:
                self.paused = not self.paused

            if pressed(events, "w") and not self.closing:
                self.target_speed = min(
                    MAX_MANUAL_SPEED,
                    self.target_speed + SPEED_STEP_MPS,
                )

            if pressed(events, "s") and not self.closing:
                self.target_speed = max(
                    -MAX_MANUAL_SPEED,
                    self.target_speed - SPEED_STEP_MPS,
                )

            if pressed(events, "a") and not self.closing:
                self.target_steer = min(
                    self.vehicle.config.max_steering_rad,
                    self.target_steer + math.radians(STEER_STEP_DEG),
                )

            if pressed(events, "d") and not self.closing:
                self.target_steer = max(
                    -self.vehicle.config.max_steering_rad,
                    self.target_steer - math.radians(STEER_STEP_DEG),
                )

            if pressed(events, "c"):
                self.target_steer = 0.0

            if events.get(32, 0) & p.KEY_WAS_TRIGGERED:
                self.target_speed = 0.0

            if pressed(events, "m"):
                self.save()

            if not self.paused:
                self.step()

            if frame % 12 == 0:
                self.update_hud()

            if self.closing and (
                abs(self.vehicle.state()["speed"]) < 0.01
                or self.steps - self.closing_at_step >= 5 * 240
            ):
                break

            time.sleep(
                max(
                    0.0,
                    DT - (time.monotonic() - frame_start),
                )
            )

        self.save()

    def close(self):
        if self.map_window:
            self.map_window.close()

        if p.isConnected(self.cid):
            self.hud.close()
            p.disconnect(self.cid)


def main():
    app = None

    try:
        app = PhysicalSlamExplorer()
        app.run()
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
