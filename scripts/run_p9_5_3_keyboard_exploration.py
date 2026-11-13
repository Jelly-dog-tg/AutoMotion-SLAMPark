"""P9.5.3: Keyboard-controlled 2D parking-lot exploration with online LiDAR SLAM.

Run in VSCode PowerShell from the repository root:
    python scripts/run_p9_5_3_keyboard_exploration.py

Commands (click the plot once to focus):
    W/S  increase/decrease target speed by 0.1 m/s (reverse is negative)
    A/D  increase/decrease steering by 5 degrees
    C    center steering
    SPACE brake to zero speed
    P    pause/resume
    M    save current map, trajectory and summary
    Q    save and exit

No saved map or replay scans are used. Simulation truth is used only for LiDAR
ray casting, physics collision checking and the optional reference overlay.
SLAM receives fresh scans plus simulated wheel odometry increments only.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon, Rectangle
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
SCAN_EVERY = 8  # 0.4 seconds of simulation time
SPEED_INCREMENT = 0.10
STEER_INCREMENT_DEG = 5.0
MAX_MANUAL_SPEED = 0.50
MAX_MANUAL_STEER_DEG = 35.0
RNG_SEED = 2026


def relative_pose(previous, current):
    """Return body-frame transform from previous pose to current pose."""
    x0, y0, a0 = map(float, previous)
    x1, y1, a1 = map(float, current)
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(a0), math.sin(a0)
    return np.array([c * dx + s * dy, -s * dx + c * dy,
                     wrap_angle(a1 - a0)], dtype=float)


def vehicle_polygon(state, geometry):
    """Four physical vehicle corners in world coordinates (rear-axle reference)."""
    front, rear, half_width = (
        geometry.front_extent, geometry.rear_overhang, geometry.width / 2.0
    )
    local = np.array([
        [-rear, -half_width], [front, -half_width],
        [front, half_width], [-rear, half_width],
    ], dtype=float)
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([state.x, state.y])


def rectangles_overlap(vehicle_corners, box):
    """Separating axis theorem for a rotated vehicle and axis-aligned rectangle."""
    x0, y0, x1, y1 = box
    obstacle = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    for corners in (vehicle_corners, obstacle):
        for i in range(4):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            if (max(vehicle_corners @ axis) < min(obstacle @ axis) - 1e-9
                    or max(obstacle @ axis) < min(vehicle_corners @ axis) - 1e-9):
                return False
    return True


def collision(state, scenario):
    corners = vehicle_polygon(state, scenario["geometry"])
    if np.any(corners[:, 0] < 0.0) or np.any(corners[:, 0] > 22.0):
        return True
    if np.any(corners[:, 1] < 0.0) or np.any(corners[:, 1] > 22.0):
        return True
    return any(rectangles_overlap(corners, box) for box in scenario["parked_vehicles"])


def errors(estimated, truth):
    dxy = np.linalg.norm(estimated[:, :2] - truth[:, :2], axis=1)
    yaw = np.degrees(np.abs(np.arctan2(
        np.sin(estimated[:, 2] - truth[:, 2]),
        np.cos(estimated[:, 2] - truth[:, 2])
    )))
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(dxy ** 2))),
        "heading_rmse_deg": float(np.sqrt(np.mean(yaw ** 2))),
        "final_translation_error_m": float(dxy[-1]),
        "final_heading_error_deg": float(yaw[-1]),
    }


class KeyboardExplorer:
    def __init__(self, show_gui=True):
        self.scenario = build_perpendicular_parking_scenario()
        self.model = AckermannModel(
            wheelbase=self.scenario["geometry"].wheelbase,
            max_steer=math.radians(MAX_MANUAL_STEER_DEG),
            max_speed=1.0,
        )
        self.lidar_config = LidarConfig()
        self.lidar = Lidar2D(self.lidar_config)
        self.rng = np.random.default_rng(RNG_SEED)
        start = self.scenario["start"]
        self.true = VehicleState(start.x, start.y, start.yaw, 0.0)
        self.odom = np.array([start.x, start.y, start.yaw], dtype=float)
        self.last_odom_at_scan = self.odom.copy()
        self.slam = OnlineSLAM2D(
            initial_pose=self.odom.copy(), lidar_config=self.lidar_config
        )
        self.target_speed = 0.0
        self.steering = 0.0
        self.paused = True  # allow time to read instructions before moving
        self.step_index = 0
        self.last_scan = None
        self.reference = []
        self.wheel_history = []
        self.slam_history = []
        self.status = "PAUSED (press P to start)"
        self.saved = False
        self.closed = False
        self.show_gui = show_gui

        self._scan_now()  # bootstrap map from a fresh scan
        if show_gui:
            self._create_window()

    def _scan_now(self):
        scan = self.lidar.scan(
            self.true, self.scenario["parked_vehicles"], rng=self.rng
        )
        delta = (None if len(self.slam.history) == 0
                 else relative_pose(self.last_odom_at_scan, self.odom))
        result = self.slam.step(scan, delta)
        self.last_odom_at_scan = self.odom.copy()
        self.last_scan = scan
        self.reference.append([self.true.x, self.true.y, self.true.yaw])
        self.wheel_history.append(self.odom.copy())
        self.slam_history.append(self.slam.pose.copy())
        self.status = result["status"] if not self.paused else self.status

    def apply_key(self, key):
        if key is None:
            return
        key = key.lower()
        if key == "w":
            self.target_speed = min(MAX_MANUAL_SPEED,
                                    round(self.target_speed + SPEED_INCREMENT, 3))
        elif key == "s":
            self.target_speed = max(-MAX_MANUAL_SPEED,
                                    round(self.target_speed - SPEED_INCREMENT, 3))
        elif key == "a":
            self.steering = min(math.radians(MAX_MANUAL_STEER_DEG),
                                self.steering + math.radians(STEER_INCREMENT_DEG))
        elif key == "d":
            self.steering = max(-math.radians(MAX_MANUAL_STEER_DEG),
                                self.steering - math.radians(STEER_INCREMENT_DEG))
        elif key == "c":
            self.steering = 0.0
        elif key in (" ", "space"):
            self.target_speed = 0.0
        elif key == "p":
            self.paused = not self.paused
            self.status = "PAUSED" if self.paused else "RUNNING"
        elif key == "m":
            self.save()
            self.status = "SAVED"
        elif key == "q":
            self.save()
            self.closed = True
            if self.show_gui:
                plt.close(self.fig)
        if self.show_gui and not self.closed:
            self._draw()
            self.fig.canvas.draw_idle()

    def _advance_one_step(self):
        if self.paused or self.closed:
            return
        accel = float(np.clip(3.0 * (self.target_speed - self.true.v), -0.8, 0.8))
        # Force a complete stop when speed is already very small.
        if self.target_speed == 0.0 and abs(self.true.v) < 0.015:
            self.true = VehicleState(self.true.x, self.true.y, self.true.yaw, 0.0)
            accel = 0.0

        candidate = self.model.step(self.true, accel, self.steering, dt=DT)
        if collision(candidate, self.scenario):
            self.target_speed = 0.0
            self.true = VehicleState(self.true.x, self.true.y, self.true.yaw, 0.0)
            self.paused = True
            self.status = "COLLISION BLOCKED: steer away then press P"
            if self.show_gui:
                self._draw()
                self.fig.canvas.draw_idle()
            return

        self.true = candidate
        self.saved = False  # a later close must preserve changes after the last manual save
        # Simulate wheel-speed + steering measurements, not true x/y/yaw.
        measured_speed = (0.0 if abs(self.true.v) < 1e-6 else
                          self.true.v * 1.025 + self.rng.normal(0, 0.015))
        measured_steering = (
            self.steering + math.radians(0.5)
            + self.rng.normal(0, math.radians(0.30))
        )
        dyaw = measured_speed / self.model.wheelbase * math.tan(measured_steering) * DT
        mid_yaw = self.odom[2] + dyaw / 2.0
        self.odom[:2] += measured_speed * DT * np.array([
            math.cos(mid_yaw), math.sin(mid_yaw)
        ])
        self.odom[2] = wrap_angle(self.odom[2] + dyaw)
        self.step_index += 1
        if self.step_index % SCAN_EVERY == 0:
            self._scan_now()

    def _create_window(self):
        # Matplotlib reserves the S key for its built-in "save figure" dialog.
        # Disable that shortcut so our own S handler can decrease speed / reverse.
        plt.rcParams["keymap.save"] = []
        self.fig, (self.ax_world, self.ax_map) = plt.subplots(
            1, 2, figsize=(13, 6.3)
        )
        self.fig.suptitle("P9.5.3 Keyboard Exploration | P: run/pause | W/S: speed | A/D: steer | C: center | Space: brake | M: save | Q: exit", fontsize=10)
        self.fig.canvas.mpl_connect("key_press_event", lambda event: self.apply_key(event.key))
        self.fig.canvas.mpl_connect("close_event", lambda event: self._on_close())
        self.fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        self.fig.subplots_adjust(bottom=0.11)
        self.info = self.fig.text(0.5, 0.025, "", ha="center", va="bottom", fontsize=10)
        self._draw()
        self.timer = self.fig.canvas.new_timer(interval=int(DT * 1000))
        self.timer.add_callback(self._on_timer)
        self.timer.start()

    def _draw(self):
        self.ax_world.clear()
        self.ax_map.clear()
        for ax in (self.ax_world, self.ax_map):
            ax.set(xlim=(0, 22), ylim=(0, 22), xlabel="X (m)", ylabel="Y (m)")
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)

        world = self.ax_world
        for box in self.scenario["parked_vehicles"]:
            x0, y0, x1, y1 = box
            world.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                      facecolor="gray", edgecolor="black", alpha=0.7))
        x0, y0, x1, y1 = self.scenario["parking_slot"]
        world.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  fill=False, linestyle="--", edgecolor="green", lw=2))
        world.add_patch(Polygon(vehicle_polygon(self.true, self.scenario["geometry"]),
                                closed=True, facecolor="#3182ce", edgecolor="navy", alpha=0.6))
        world.arrow(self.true.x, self.true.y, math.cos(self.true.yaw) * 1.5,
                    math.sin(self.true.yaw) * 1.5,
                    head_width=0.20, color="navy", length_includes_head=True)
        if self.last_scan is not None and len(self.last_scan["points_world"]):
            pts = self.last_scan["points_world"]
            world.scatter(pts[:, 0], pts[:, 1], c="darkorange", s=4, alpha=0.8,
                          label="Fresh LiDAR hits")
        ref = np.asarray(self.reference)
        est = np.asarray(self.slam_history)
        if len(ref) > 1:
            world.plot(ref[:, 0], ref[:, 1], "k--", lw=1, label="GT (display only)")
            world.plot(est[:, 0], est[:, 1], "g-", lw=1.5, label="SLAM estimate")
        world.set_title("Simulation + LiDAR")
        world.legend(loc="upper right", fontsize=7)

        mapped = self.ax_map
        states = self.slam.mapper.states
        mapped.imshow(
            states + 1, origin="lower", extent=(0, 22, 0, 22),
            cmap=ListedColormap(["#999999", "#ffffff", "#1a1a1a"]),
            vmin=0, vmax=2, interpolation="nearest", zorder=0
        )
        mapped.plot(est[:, 0], est[:, 1], "#008a3c", lw=1.5, label="SLAM trajectory")
        mapped.plot(est[-1, 0], est[-1, 1], "ro", ms=4, label="Estimated vehicle")
        mapped.set_title("Online Occupancy Grid (0.05 m)")
        mapped.legend(loc="upper right", fontsize=7)
        self.info.set_text(
            f"{self.status}  |  speed target {self.target_speed:+.2f} m/s  "
            f"|  steering {math.degrees(self.steering):+.0f}°  "
            f"|  scans {len(self.slam.history)}  "
            f"|  simulated time {self.step_index * DT:.1f} s"
        )

    def _on_timer(self):
        if self.closed:
            return False
        self._advance_one_step()
        # Render at the LiDAR rate, not every physics step.
        if not self.paused and self.step_index % SCAN_EVERY == 0:
            self._draw()
            self.fig.canvas.draw_idle()
        return True

    def _on_close(self):
        if not self.saved:
            self.save()
        self.closed = True

    def save(self):
        RESULTS.mkdir(parents=True, exist_ok=True)
        reference = np.asarray(self.reference, dtype=float)
        wheel = np.asarray(self.wheel_history, dtype=float)
        estimate = np.asarray(self.slam_history, dtype=float)
        counts = Counter(item["status"] for item in self.slam.history)
        reasons = Counter(item["reason"] for item in self.slam.history if item["reason"])
        summary = {
            "stage": "P9.5.3 Keyboard exploration with incremental LiDAR SLAM",
            "scan_count": len(self.slam.history),
            "simulation_duration_s": round(self.step_index * DT, 3),
            "lidar_beams": self.lidar_config.num_beams,
            "map_resolution_m": self.slam.mapper.resolution,
            "update_status": dict(counts),
            "fallback_reasons": dict(reasons),
            "odometry_errors": errors(wheel, reference),
            "slam_errors": errors(estimate, reference),
            "map_cells": {"occupied": int(np.sum(self.slam.mapper.states == 1)),
                          "free": int(np.sum(self.slam.mapper.states == 0)),
                          "unknown": int(np.sum(self.slam.mapper.states == -1))},
            "ground_truth_usage": "LiDAR simulation, collision physics and evaluation only",
            "note": "Operator keyboard exploration, not autonomous frontier exploration",
        }
        (RESULTS / "p9_5_3_keyboard_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez_compressed(
            RESULTS / "p9_5_3_keyboard_map.npz",
            **self.slam.mapper.arrays(),
            reference_poses=reference,
            odometry_poses=wheel,
            estimated_poses=estimate,
        )
        if self.show_gui and hasattr(self, "fig"):
            self._draw()
            self.fig.savefig(RESULTS / "p9_5_3_keyboard_exploration.png", dpi=160)
        self.saved = True
        print("\n========== P9.5.3 SAVED ==========\n"
              + json.dumps({k: v for k, v in summary.items() if k not in ("note",)},
                           ensure_ascii=False, indent=2))
        return summary


def main():
    # Internal non-GUI smoke mode for quick reproducible test; user runs GUI mode.
    smoke = "--smoke" in sys.argv
    explorer = KeyboardExplorer(show_gui=not smoke)
    if smoke:
        explorer.apply_key("p")
        explorer.apply_key("w")
        explorer.apply_key("w")
        explorer.apply_key("w")
        for _ in range(64):
            explorer._advance_one_step()
        explorer.save()
    else:
        print("Click the window once to focus. P starts/pauses.")
        print("W/S speed, A/D steering, C center, Space brake, M save, Q exit.")
        plt.show()


if __name__ == "__main__":
    main()
