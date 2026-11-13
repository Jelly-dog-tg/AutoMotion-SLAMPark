"""P10.4.1: Explore -> click parking bay -> plan, without resetting the vehicle.

Run from ParkingPlanner root on Windows:
    python scripts/run_p10_4_1_explore_select_plan.py

Controls: P pause/resume, W/S speed, A/D steer (5 deg per key, +/-45 deg),
          C center, SPACE brake, G select target bay, click the green bay,
          X clear a plan, M save, Q save and quit.

The live HUD displays signed vehicle speed, speed target, front-wheel steering
command and body heading. This stage plans, but DOES NOT execute the path.
Online execution is P10.4.2.
Truth is used for simulated physics/LiDAR and display, not planning.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_p9_5_3_keyboard_exploration import KeyboardExplorer
from parking_planner.integration.slam_map_adapter import build_planner_grid
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import resample_path, path_metrics
from parking_planner.vehicle.ackermann import VehicleState

OUT = ROOT / "results"
PREFIX = "p10_4_1"
MAX_MANUAL_STEER_DEG = 45.0


class ExplorePlanApp(KeyboardExplorer):
    """Keep the very same online SLAM, vehicle, odometry and LiDAR instances."""

    def __init__(self, show_gui=True):
        # Set BEFORE super().__init__ because it dynamically calls self._draw.
        self.selection_active = False
        self.plan_points = None
        self.plan_start_pose = None
        self.plan_goal_pose = None
        self.plan_meta = None
        super().__init__(show_gui=show_gui)
        # Widen only this app instance's manual steering limit.
        # P9.5.3 and the shared scenario/planner configuration remain unchanged.
        self.model.max_steer = math.radians(MAX_MANUAL_STEER_DEG)

    def _create_window(self):
        super()._create_window()
        self.fig.suptitle(
            "P10.4.1  Explore -> Stop -> G -> Click Green Bay -> Plan   |   "
            "P pause  W/S speed  A/D steer (+/-45 deg, 5 deg/press)  "
            "C center  Space brake  X clear  M save  Q quit",
            fontsize=9,
        )
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def apply_key(self, key):
        if not key:
            return
        key = key.lower()
        if key == "g":
            if abs(self.true.v) > 0.045 or abs(self.target_speed) > 0.001:
                self.status = "STOP FIRST: Space brake, wait for |v|<0.045, then P"
            elif not self.paused:
                self.status = "PRESS P to pause before selecting a bay"
            else:
                self.selection_active = True
                self.status = "SELECT BAY: click inside the green rectangle in LEFT panel"
        elif key == "x":
            self.selection_active = False
            self.plan_points = None
            self.plan_start_pose = None
            self.plan_goal_pose = None
            self.plan_meta = None
            self.status = "PLAN CLEARED - manual exploration available"
        elif self.plan_points is not None and key not in ("m", "q"):
            self.status = "PLAN READY: X clears; automatic execution comes in P10.4.2"
        elif key in ("a", "d"):
            limit = math.radians(MAX_MANUAL_STEER_DEG)
            increment = math.radians(5.0)
            if key == "a":
                self.steering = min(limit, self.steering + increment)
            else:
                self.steering = max(-limit, self.steering - increment)
            self.status = (
                f"MANUAL STEER {math.degrees(self.steering):+.0f} deg "
                f"(limit +/-{MAX_MANUAL_STEER_DEG:.0f} deg)"
            )
        else:
            super().apply_key(key)
            return
        if self.show_gui and not self.closed:
            self._draw()
            self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if not self.selection_active or self.closed or event.inaxes != self.ax_world:
            return
        if event.xdata is None or event.ydata is None:
            return
        xmin, ymin, xmax, ymax = self.scenario["parking_slot"]
        if not (xmin <= event.xdata <= xmax and ymin <= event.ydata <= ymax):
            self.status = "CLICK INSIDE GREEN PARKING BAY (left panel)"
        else:
            self.selection_active = False
            # Click selects a known bay; its calibrated parking pose is a
            # manually supplied target, NOT a LiDAR-based bay detector.
            goal = self.scenario["goal"]
            self.plan_to_goal([goal.x, goal.y, goal.yaw])
        if self.show_gui:
            self._draw()
            self.fig.canvas.draw_idle()

    def plan_to_goal(self, goal_xyz):
        """Plan from CURRENT corrected SLAM pose and CURRENT in-memory map."""
        if not self.paused or abs(self.true.v) > 0.045:
            self.status = "PLAN REFUSED: stop the vehicle and pause first"
            return False
        if self.plan_points is not None:
            self.status = "PLAN ALREADY EXISTS: X to clear"
            return False
        # Incorporate the last odometry interval with a fresh scan. This
        # prevents using a stale SLAM pose after the vehicle stops.
        self._scan_now()
        self.saved = False
        estimate = self.slam.pose.copy()
        start = VehicleState(float(estimate[0]), float(estimate[1]),
                             float(estimate[2]), 0.0)
        goal = VehicleState(*[float(v) for v in goal_xyz], 0.0)
        # NO use of scenario['grid'] or scenario['parked_vehicles'] for planning.
        # Unknown remains occupied. The live SLAM map is used directly.
        grid = build_planner_grid(
            self.slam.mapper, planner_resolution=0.20,
            unknown_is_occupied=True,
        )
        planner = HybridAStarPlanner(
            grid=grid, geometry=self.scenario["geometry"],
            config=self.scenario["planner_config"],
        )
        if grid.is_collision(start, planner.local_x, planner.local_y):
            self.status = "PLAN REFUSED: estimated start footprint blocked/unknown"
            return False
        if grid.is_collision(goal, planner.local_x, planner.local_y):
            self.status = "PLAN REFUSED: goal footprint blocked/unknown; explore more"
            return False
        try:
            path = planner.plan(start, goal)
        except (ValueError, RuntimeError) as exc:
            self.status = f"PLAN FAILED: {type(exc).__name__} - explore more"
            print(self.status, exc)
            return False
        if not path:
            self.status = "NO SAFE PATH: move/explore more, then try again"
            return False
        dense = resample_path(path, spacing=0.05)
        for i, point in enumerate(dense):
            if grid.is_collision(VehicleState(point.x, point.y, point.yaw, 0.0),
                                 planner.local_x, planner.local_y):
                self.status = f"PLAN REFUSED: footprint collision at point {i}"
                return False
        self.plan_points = dense
        self.plan_start_pose = estimate.copy()
        self.plan_goal_pose = np.asarray(goal_xyz, dtype=float)
        pm = path_metrics(path)
        self.plan_meta = {
            "stage": "P10.4.1 explore-to-plan live handoff",
            "status": "PATH_PLANNED_NOT_EXECUTED",
            "map_source": "current OnlineSLAM2D in-memory map",
            "start_pose_source": "latest SLAM pose after a fresh stop scan",
            "start_pose": estimate.tolist(),
            "target": self.plan_goal_pose.tolist(),
            "goal_source": "manually selected known parking slot",
            "exploration_steps": int(self.step_index),
            "slam_scans_at_planning": len(self.slam.history),
            "planner_resolution_m": float(grid.resolution),
            "unknown_is_occupied": True,
            "planning_expanded_nodes": int(planner.last_iterations),
            "raw_path_points": len(path),
            "dense_path_points": len(dense),
            "path_length_m": float(pm["path_length"]),
            "reverse_distance_m": float(pm["reverse_distance"]),
            "gear_switches": int(pm["gear_switches"]),
            "path_collision_free_on_live_slam_map": True,
            "ground_truth_used_for_planning": False,
            "vehicle_reset_between_modes": False,
            "automated_path_execution": False,
            "manual_steering_limit_deg": MAX_MANUAL_STEER_DEG,
            "manual_steering_increment_deg": 5.0,
        }
        self.status = f"LIVE PLAN READY: {len(dense)} points; press M to save"
        print(json.dumps(self.plan_meta, ensure_ascii=False, indent=2))
        return True

    def _draw(self):
        super()._draw()
        if self.plan_points:
            x = np.array([p.x for p in self.plan_points])
            y = np.array([p.y for p in self.plan_points])
            d = np.array([p.direction for p in self.plan_points])
            for ax in (self.ax_world, self.ax_map):
                for direction, color, name in ((1, "royalblue", "Forward plan"),
                                               (-1, "darkorange", "Reverse plan")):
                    mask = d == direction
                    if np.any(mask):
                        ax.plot(np.where(mask, x, np.nan),
                                np.where(mask, y, np.nan), color=color,
                                lw=2, label=name)
                ax.plot(self.plan_goal_pose[0], self.plan_goal_pose[1],
                        "r*", ms=12, label="Selected goal")
                ax.legend(loc="upper right", fontsize=7)
        self._update_hud()

    def _update_hud(self):
        """Show live measured simulation state and steering command."""
        if not hasattr(self, "info"):
            return
        self.info.set_text(
            f"{self.status}\n"
            f"Current speed: {self.true.v:+.2f} m/s  |  "
            f"Target speed: {self.target_speed:+.2f} m/s  |  "
            f"Front-wheel steering: {math.degrees(self.steering):+.1f} deg "
            f"(limit +/-{MAX_MANUAL_STEER_DEG:.0f} deg)  |  "
            f"Vehicle heading: {math.degrees(self.true.yaw):+.1f} deg  |  "
            f"LiDAR scans: {len(self.slam.history)}  |  "
            f"Sim time: {self.step_index * 0.05:.1f} s"
        )

    def _on_timer(self):
        """Refresh the telemetry HUD every physics tick (20 Hz)."""
        keep_running = super()._on_timer()
        if self.show_gui and not self.closed:
            self._update_hud()
            self.fig.canvas.draw_idle()
        return keep_running

    def save(self):
        """Do NOT overwrite the successful P9.5.3 saved map or P10.1 path."""
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT / f"{PREFIX}_exploration_map.npz",
            **self.slam.mapper.arrays(),
            estimated_poses=np.asarray(self.slam_history, dtype=float),
            odometry_poses=np.asarray(self.wheel_history, dtype=float),
            reference_poses=np.asarray(self.reference, dtype=float),
        )
        summary = {
            "stage": "P10.4.1 interactive exploration and planning",
            "status": "PATH_PLANNED_NOT_EXECUTED" if self.plan_points else "EXPLORED_ONLY",
            "live_scan_count": len(self.slam.history),
            "sim_duration_s": round(self.step_index * 0.05, 3),
            "vehicle_not_reset_between_modes": True,
            "manual_steering_limit_deg": MAX_MANUAL_STEER_DEG,
            "manual_steering_increment_deg": 5.0,
            "plan": self.plan_meta,
        }
        if self.plan_points:
            np.savez_compressed(
                OUT / f"{PREFIX}_live_plan.npz",
                path=np.asarray([[p.x, p.y, p.yaw, p.direction, p.steering]
                                 for p in self.plan_points], dtype=float),
                start_pose=self.plan_start_pose,
                goal_pose=self.plan_goal_pose,
            )
        (OUT / f"{PREFIX}_metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self.show_gui and hasattr(self, "fig"):
            self._draw()
            self.fig.savefig(OUT / f"{PREFIX}_explore_select_plan.png", dpi=160)
        self.saved = True
        print("P10.4.1 saved:", OUT / f"{PREFIX}_metrics.json")
        return summary


def main():
    app = ExplorePlanApp(show_gui=True)
    print("P10.4.1 keys: P pause, W/S speed, A/D steer +/-45 deg (5 deg/press), Space brake,")
    print("G click green bay when stopped/paused, X clear, M save, Q quit.")
    plt.show()


if __name__ == "__main__":
    main()
