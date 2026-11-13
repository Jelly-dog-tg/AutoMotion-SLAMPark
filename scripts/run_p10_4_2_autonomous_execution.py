
"""P10.4.2: Autonomous execution of the saved P10.4.1 SLAM Hybrid A* path.

Controls:
    E       Start / resume autonomous execution
    P       Pause / resume
    SPACE   Controlled emergency stop
    M       Save P10.4.2 results
    Q       Save and quit
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.control.gear_manager import GearManager
from parking_planner.control.speed_controller import SpeedController
from parking_planner.control.stanley_controller import StanleyController

from parking_planner.mapping.occupancy_grid_mapper import (
    OccupancyGridMapper,
)

from parking_planner.planning.hybrid_astar import (
    HybridAStarPlanner,
    OccupancyGrid,
    PathPoint,
)

from parking_planner.slam.online_slam import (
    OnlineSLAM2D,
    compose_pose,
    wrap_angle,
)

from parking_planner.vehicle.ackermann import (
    AckermannModel,
    VehicleState,
)

from scripts.run_p9_5_3_keyboard_exploration import (
    DT,
    SCAN_EVERY,
    KeyboardExplorer,
    collision,
    relative_pose,
)


# ============================================================
# 1. Paths
# ============================================================

RESULTS = ROOT / "results"

PLAN_FILE = RESULTS / "p10_4_1_live_plan.npz"
MAP_FILE = RESULTS / "p10_4_1_exploration_map.npz"
METRICS_FILE = RESULTS / "p10_4_1_metrics.json"

OUT_PREFIX = "p10_4_2"

OUT_MAP = RESULTS / f"{OUT_PREFIX}_execution_map.npz"
OUT_TRAJECTORY = RESULTS / f"{OUT_PREFIX}_execution_trajectory.npz"
OUT_METRICS = RESULTS / f"{OUT_PREFIX}_execution_metrics.json"
OUT_PLOT = RESULTS / f"{OUT_PREFIX}_autonomous_execution.png"


# ============================================================
# 2. Controller configuration
# ============================================================

FORWARD_SPEED = 0.24
REVERSE_SPEED = 0.18

MAX_ACCELERATION = 0.65
BRAKE_DECELERATION = 0.55

STOP_SPEED = 0.045
SHIFT_DURATION = 0.50

STOP_BUFFER = 0.05

SEGMENT_REMAINING_TOLERANCE = 0.12
SEGMENT_ENDPOINT_TOLERANCE = 0.25

GOAL_POSITION_TOLERANCE = 0.22
GOAL_HEADING_TOLERANCE = math.radians(7.0)

MAX_STEPS = 2400

RNG_SEED = 20261113


# ============================================================
# 3. Path processing
# ============================================================

def path_suffix(path: list[PathPoint]) -> np.ndarray:

    suffix = np.zeros(len(path), dtype=float)

    for i in range(len(path) - 2, -1, -1):

        suffix[i] = (
            suffix[i + 1]
            + math.hypot(
                path[i + 1].x - path[i].x,
                path[i + 1].y - path[i].y,
            )
        )

    return suffix


def split_path_by_edge_direction(
    raw: np.ndarray,
) -> list[tuple[int, list[PathPoint]]]:

    if raw.ndim != 2 or raw.shape[1] != 5 or len(raw) < 3:
        raise ValueError(
            "P10.4.1 path must have shape (N, 5), N >= 3"
        )

    if not np.isfinite(raw).all():
        raise ValueError(
            "P10.4.1 path contains non-finite values"
        )

    edge_directions = np.rint(raw[1:, 3]).astype(int)

    if not np.all(np.isin(edge_directions, (-1, 1))):
        raise ValueError(
            "Path direction values must be +1 or -1"
        )

    segments = []

    start_index = 0
    current_direction = int(edge_directions[0])

    for edge_index in range(1, len(edge_directions)):

        next_direction = int(edge_directions[edge_index])

        if next_direction != current_direction:

            rows = raw[start_index:edge_index + 1]

            points = [
                PathPoint(
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    current_direction,
                    float(row[4]),
                )
                for row in rows
            ]

            if len(points) < 2:
                raise ValueError(
                    "Path contains a zero-length gear segment"
                )

            segments.append(
                (current_direction, points)
            )

            start_index = edge_index
            current_direction = next_direction

    rows = raw[start_index:]

    points = [
        PathPoint(
            float(row[0]),
            float(row[1]),
            float(row[2]),
            current_direction,
            float(row[4]),
        )
        for row in rows
    ]

    if len(points) < 2:
        raise ValueError(
            "Final path segment is too short"
        )

    segments.append(
        (current_direction, points)
    )

    return segments


def signed_remaining_distance(
    controller,
    state,
    path,
    direction,
    target_index,
    suffix,
):

    index = min(
        max(0, int(target_index)),
        len(path) - 1,
    )

    px, py = controller._tracking_position(
        state,
        direction,
    )

    rx, ry = controller._reference_position(
        path[index],
        direction,
    )

    travel_yaw = (
        path[index].yaw
        + (math.pi if direction < 0 else 0.0)
    )

    signed_gap = (
        (rx - px) * math.cos(travel_yaw)
        + (ry - py) * math.sin(travel_yaw)
    )

    return max(
        0.0,
        float(suffix[index]) + signed_gap,
    )


def pose_errors(
    estimate: np.ndarray,
    truth: VehicleState,
):

    position = math.hypot(
        float(estimate[0]) - truth.x,
        float(estimate[1]) - truth.y,
    )

    heading = abs(
        math.degrees(
            wrap_angle(
                float(estimate[2]) - truth.yaw
            )
        )
    )

    return position, heading


# ============================================================
# 4. Autonomous execution application
# ============================================================

class AutonomousExecutionApp(KeyboardExplorer):

    def __init__(self, show_gui=True):

        super().__init__(show_gui=show_gui)

        if show_gui and hasattr(self, "timer"):
            self.timer.stop()

        self._load_saved_handoff()

        if show_gui and hasattr(self, "timer"):
            self._draw()
            self.timer.start()


    # --------------------------------------------------------
    # Load P10.4.1 results
    # --------------------------------------------------------

    def _load_saved_handoff(self):

        for required in (
            PLAN_FILE,
            MAP_FILE,
            METRICS_FILE,
        ):

            if not required.is_file():

                raise FileNotFoundError(
                    f"Missing {required}. "
                    "Complete and save P10.4.1 first."
                )

        with np.load(
            PLAN_FILE,
            allow_pickle=False,
        ) as data:

            raw_path = np.asarray(
                data["path"],
                dtype=float,
            )

            start_pose = np.asarray(
                data["start_pose"],
                dtype=float,
            )

            goal_pose = np.asarray(
                data["goal_pose"],
                dtype=float,
            )

        with np.load(
            MAP_FILE,
            allow_pickle=False,
        ) as data:

            saved_odom = np.asarray(
                data["odometry_poses"],
                dtype=float,
            )

            saved_estimates = np.asarray(
                data["estimated_poses"],
                dtype=float,
            )

            saved_reference = np.asarray(
                data["reference_poses"],
                dtype=float,
            )

        handoff_metrics = json.loads(
            METRICS_FILE.read_text(
                encoding="utf-8"
            )
        )

        if start_pose.shape != (3,):
            raise ValueError(
                "Invalid P10.4.1 start_pose"
            )

        if goal_pose.shape != (3,):
            raise ValueError(
                "Invalid P10.4.1 goal_pose"
            )

        if (
            not len(saved_odom)
            or not len(saved_estimates)
            or not len(saved_reference)
        ):

            raise ValueError(
                "P10.4.1 map archive has incomplete pose history"
            )

        if np.linalg.norm(
            raw_path[0, :2] - start_pose[:2]
        ) > 0.05:

            raise ValueError(
                "Saved path start does not match saved SLAM start pose"
            )

        self.path_raw = raw_path

        self.path_segments = (
            split_path_by_edge_direction(raw_path)
        )

        self.plan_start_pose = start_pose.copy()
        self.plan_goal_pose = goal_pose.copy()

        self.plan_points = [
            PathPoint(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                int(round(row[3])),
                float(row[4]),
            )
            for row in raw_path
        ]

        # Restore saved SLAM map.

        mapper = OccupancyGridMapper.load(
            MAP_FILE
        )

        self.map_collision_grid = OccupancyGrid(
            width=mapper.width,
            height=mapper.height,
            resolution=mapper.resolution,
            origin_x=mapper.origin_x,
            origin_y=mapper.origin_y,
        )

        # Occupied and unknown cells are treated as blocked.

        self.map_collision_grid.occupied[:] = (
            mapper.states != 0
        )

        self.map_snapshot_states = (
            self.map_collision_grid.occupied.copy()
        )

        scenario = self.scenario

        # Vehicle model.

        self.model = AckermannModel(
            wheelbase=scenario["geometry"].wheelbase,
            max_steer=math.radians(35.0),
            max_acceleration=MAX_ACCELERATION,
            max_speed=0.8,
        )

        # Stanley controller.

        self.stanley = StanleyController(
            wheelbase=self.model.wheelbase,
            gain=0.65,
            softening_speed=0.45,
            max_steer=self.model.max_steer,
        )

        # PID speed controller.

        self.pid = SpeedController(
            kp=2.5,
            ki=0.0,
            kd=0.0,
            max_acceleration=MAX_ACCELERATION,
            max_speed=self.model.max_speed,
        )

        # Gear management.

        self.gear_manager = GearManager(
            initial_direction=self.path_segments[0][0],
            stop_speed=STOP_SPEED,
            shift_duration=SHIFT_DURATION,
        )

        # Resume physical simulation pose.

        physics_resume_pose = (
            saved_reference[-1].astype(float).copy()
        )

        self.true = VehicleState(
            float(physics_resume_pose[0]),
            float(physics_resume_pose[1]),
            float(physics_resume_pose[2]),
            0.0,
        )

        self.physics_resume_pose = (
            physics_resume_pose
        )

        self.initial_localization_error_m = float(
            np.linalg.norm(
                start_pose[:2] - physics_resume_pose[:2]
            )
        )

        self.odom = saved_odom[-1].astype(float).copy()

        self.last_odom_at_scan = self.odom.copy()

        self.slam = OnlineSLAM2D(
            initial_pose=start_pose,
            lidar_config=self.lidar_config,
        )

        self.slam.mapper = mapper

        self.rng = np.random.default_rng(
            RNG_SEED
        )

        self.slam_history = [
            row.copy()
            for row in saved_estimates
        ]

        self.wheel_history = [
            row.copy()
            for row in saved_odom
        ]

        self.reference = [
            row.copy()
            for row in saved_reference[:-1]
        ]

        self.last_scan = None

        self.scan_count_offset = len(
            saved_estimates
        )

        self.sim_time_offset_s = float(
            handoff_metrics.get(
                "sim_duration_s",
                0.0,
            )
        )

        self.step_index = 0

        # Bootstrap one fresh scan.

        self._scan_now()

        # Initialize path tracking.

        self.current_segment_index = 0

        self.current_direction, self.current_path = (
            self.path_segments[0]
        )

        self.current_suffix = path_suffix(
            self.current_path
        )

        self.current_index = 0

        self.segment_start_step = 0

        self.current_segment_cte = []

        self.segment_summaries = []

        self.all_cte = []

        self.localization_position_errors = []

        self.localization_heading_errors = []

        self.execution_records = []

        self.execution_trajectory = [[
            self.true.x,
            self.true.y,
            self.true.yaw,
        ]]

        self.previous_measured_speed = 0.0

        self.target_speed = 0.0

        self.steering = 0.0

        self.auto_active = False

        self.emergency_stop_requested = False

        self.gear_switches = 0

        self.collision_physical = False

        self.collision_slam_map = False

        self.terminal_status = None

        self.saved = False

        self.paused = True

        self.closed = False

        self.status = (
            f"P10.4.2 READY | "
            f"{len(self.path_segments)} direction segments | "
            "press E to execute saved P10.4.1 path"
        )

        # Path safety audit.

        footprint_planner = HybridAStarPlanner(
            grid=self.map_collision_grid,
            geometry=scenario["geometry"],
            config=scenario["planner_config"],
        )

        self.footprint_local_x = (
            footprint_planner.local_x
        )

        self.footprint_local_y = (
            footprint_planner.local_y
        )

        map_bad = []
        physical_bad = []

        for seg_i, (
            direction,
            segment,
        ) in enumerate(self.path_segments):

            for point_i, point in enumerate(segment):

                state = VehicleState(
                    point.x,
                    point.y,
                    point.yaw,
                    0.0,
                )

                if self.map_collision_grid.is_collision(
                    state,
                    footprint_planner.local_x,
                    footprint_planner.local_y,
                ):

                    map_bad.append(
                        (seg_i, point_i)
                    )

                if collision(
                    state,
                    scenario,
                ):

                    physical_bad.append(
                        (seg_i, point_i)
                    )

        self.path_precheck = {
            "path_collision_free_on_saved_slam_map": (
                not map_bad
            ),

            "path_collision_free_in_simulated_physical_map": (
                not physical_bad
            ),

            "map_collision_samples": map_bad[:10],

            "physical_collision_samples": (
                physical_bad[:10]
            ),
        }

        if map_bad or physical_bad:

            self.terminal_status = (
                "PATH_SAFETY_PRECHECK_FAILED"
            )

            self.status = (
                "SAFETY REFUSED: saved path fails collision audit; "
                "do not press E"
            )


    # --------------------------------------------------------
    # GUI
    # --------------------------------------------------------

    def _create_window(self):

        super()._create_window()

        self.fig.suptitle(
            "P10.4.2 Saved SLAM Path -> Autonomous Execution | "
            "E start/resume | P pause | "
            "Space emergency stop | M save | Q quit",
            fontsize=9,
        )


    def _draw(self):

        super()._draw()

        if not hasattr(self, "path_segments"):
            return

        labels_seen = set()

        for direction, segment in self.path_segments:

            xs = [
                point.x
                for point in segment
            ]

            ys = [
                point.y
                for point in segment
            ]

            color = (
                "royalblue"
                if direction > 0
                else "darkorange"
            )

            label = (
                "Planned forward"
                if direction > 0
                else "Planned reverse"
            )

            for ax in (
                self.ax_world,
                self.ax_map,
            ):

                ax.plot(
                    xs,
                    ys,
                    color=color,
                    lw=2.0,
                    alpha=0.85,
                    label=(
                        label
                        if label not in labels_seen
                        else None
                    ),
                )

            labels_seen.add(label)

        if len(self.execution_trajectory) >= 2:

            trajectory = np.asarray(
                self.execution_trajectory,
                dtype=float,
            )

            for ax in (
                self.ax_world,
                self.ax_map,
            ):

                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color="black",
                    lw=1.8,
                    label="Executed trajectory",
                )

        for ax in (
            self.ax_world,
            self.ax_map,
        ):

            ax.plot(
                self.plan_goal_pose[0],
                self.plan_goal_pose[1],
                "r*",
                ms=12,
                label="Parking goal",
            )

            ax.legend(
                loc="upper right",
                fontsize=7,
            )

        self._update_hud()


    def _update_hud(self):

        if not hasattr(self, "info"):
            return

        segment_count = len(
            getattr(self, "path_segments", [])
        )

        segment_index = getattr(
            self,
            "current_segment_index",
            0,
        )

        if (
            segment_count
            and segment_index < segment_count
        ):

            direction = (
                self.path_segments[segment_index][0]
            )

            gear_text = (
                "FORWARD"
                if direction > 0
                else "REVERSE"
            )

            segment_text = (
                f"segment {segment_index + 1}/"
                f"{segment_count} {gear_text}"
            )

        elif segment_count:

            segment_text = (
                f"segments complete "
                f"{segment_count}/{segment_count}"
            )

        else:

            segment_text = "loading path"

        scans = (
            getattr(self, "scan_count_offset", 0)
            + len(self.slam.history)
        )

        elapsed = (
            getattr(self, "sim_time_offset_s", 0.0)
            + getattr(self, "step_index", 0) * DT
        )

        self.info.set_text(
            f"{self.status} | {segment_text} | "
            f"Speed: {self.true.v:+.2f} m/s | "
            f"Target: {self.target_speed:+.2f} m/s | "
            f"Steering: {math.degrees(self.steering):+.1f} deg | "
            f"Heading: {math.degrees(self.true.yaw):+.1f} deg | "
            f"LiDAR scans: {scans} | "
            f"Sim time: {elapsed:.1f} s"
        )


    # --------------------------------------------------------
    # Keyboard controls
    # --------------------------------------------------------

    def apply_key(self, key):

        if not key:
            return

        key = key.lower()

        if key == "e":

            if self.terminal_status is not None:

                self.status = (
                    f"EXECUTION LOCKED: {self.terminal_status}"
                )

            elif not self.path_precheck[
                "path_collision_free_on_saved_slam_map"
            ]:

                self.status = (
                    "START REFUSED: saved SLAM map collision audit failed"
                )

            elif not self.path_precheck[
                "path_collision_free_in_simulated_physical_map"
            ]:

                self.status = (
                    "START REFUSED: physical-map collision audit failed"
                )

            else:

                self.auto_active = True

                self.paused = False

                self.emergency_stop_requested = False

                self.saved = False

                self.status = (
                    "AUTONOMOUS EXECUTION RUNNING"
                )

        elif key == "p":

            if self.auto_active:

                self.paused = not self.paused

                self.status = (
                    "PAUSED (press P or E to resume)"
                    if self.paused
                    else "AUTONOMOUS EXECUTION RUNNING"
                )

            else:

                self.status = (
                    "NOT RUNNING: press E to start"
                )

        elif key in (" ", "space"):

            if self.auto_active:

                self.emergency_stop_requested = True

                self.status = (
                    "EMERGENCY BRAKING TO A CONTROLLED STOP"
                )

            else:

                self.status = (
                    "VEHICLE ALREADY STOPPED"
                )

        elif key == "m":

            self.save()

            self.status = (
                "P10.4.2 RESULTS SAVED"
            )

        elif key == "q":

            self.save()

            self.closed = True

            if self.show_gui:
                plt.close(self.fig)

        else:

            self.status = (
                "AUTO MODE: E start/resume | P pause | "
                "Space emergency stop | M save | Q quit"
            )

        if self.show_gui and not self.closed:

            self._draw()

            self.fig.canvas.draw_idle()


    # --------------------------------------------------------
    # SLAM feedback
    # --------------------------------------------------------

    def _estimated_pose(self):

        return compose_pose(
            self.slam.pose,
            relative_pose(
                self.last_odom_at_scan,
                self.odom,
            ),
        )


    def _record_step(
        self,
        estimate,
        cte,
        direction,
    ):

        position_error, heading_error = pose_errors(
            estimate,
            self.true,
        )

        self.localization_position_errors.append(
            position_error
        )

        self.localization_heading_errors.append(
            heading_error
        )

        record = {

            "time_s": (
                self.sim_time_offset_s
                + self.step_index * DT
            ),

            "segment_index": int(
                self.current_segment_index
            ),

            "direction": int(direction),

            "gear_phase": self.gear_manager.phase,

            "true_x": float(self.true.x),

            "true_y": float(self.true.y),

            "true_yaw": float(self.true.yaw),

            "estimate_x": float(estimate[0]),

            "estimate_y": float(estimate[1]),

            "estimate_yaw": float(estimate[2]),

            "speed_mps": float(self.true.v),

            "target_speed_mps": float(self.target_speed),

            "steering_deg": float(
                math.degrees(self.steering)
            ),

            "cross_track_error_m": float(cte),

            "position_error_m": float(position_error),

            "heading_error_deg": float(heading_error),
        }

        self.execution_records.append(record)

        self.execution_trajectory.append([
            self.true.x,
            self.true.y,
            self.true.yaw,
        ])

        self.saved = False


    # --------------------------------------------------------
    # Vehicle physics and odometry
    # --------------------------------------------------------

    def _advance_vehicle(
        self,
        acceleration,
        steering,
    ):

        candidate = self.model.step(
            self.true,
            acceleration,
            steering,
            DT,
        )

        if collision(
            candidate,
            self.scenario,
        ):

            self.collision_physical = True

            self._abort(
                "PHYSICAL_COLLISION_ABORT"
            )

            return False

        if self.map_collision_grid.is_collision(
            candidate,
            self.footprint_local_x,
            self.footprint_local_y,
        ):

            self.collision_slam_map = True

            self._abort(
                "SAVED_SLAM_MAP_COLLISION_ABORT"
            )

            return False

        self.true = candidate

        measured_speed = (

            self.true.v * 1.015
            + self.rng.normal(0.0, 0.008)

            if abs(self.true.v) > 1e-7
            else 0.0
        )

        measured_steering = (
            steering
            + math.radians(0.35)
            + self.rng.normal(
                0.0,
                math.radians(0.20),
            )
        )

        dyaw = (
            measured_speed
            / self.model.wheelbase
            * math.tan(measured_steering)
            * DT
        )

        middle_yaw = (
            self.odom[2] + 0.5 * dyaw
        )

        self.odom[:2] += (
            measured_speed
            * DT
            * np.array([
                math.cos(middle_yaw),
                math.sin(middle_yaw),
            ])
        )

        self.odom[2] = wrap_angle(
            self.odom[2] + dyaw
        )

        self.previous_measured_speed = float(
            measured_speed
        )

        self.step_index += 1

        if self.step_index % SCAN_EVERY == 0:

            self._scan_now()

            if self.auto_active and not self.paused:

                self.status = (
                    "EMERGENCY BRAKING TO A CONTROLLED STOP"
                    if self.emergency_stop_requested
                    else "AUTONOMOUS EXECUTION RUNNING"
                )

        return True


    # --------------------------------------------------------
    # Segment summary
    # --------------------------------------------------------

    def _summarize_segment(
        self,
        endpoint_error,
    ):

        errors = np.asarray(
            self.current_segment_cte,
            dtype=float,
        )

        self.segment_summaries.append({

            "segment_index": int(
                self.current_segment_index
            ),

            "direction": int(
                self.current_direction
            ),

            "direction_name": (
                "forward"
                if self.current_direction > 0
                else "reverse"
            ),

            "duration_s": float(
                (self.step_index - self.segment_start_step)
                * DT
            ),

            "tracking_points": int(
                len(errors)
            ),

            "cross_track_rmse_m": (
                float(np.sqrt(np.mean(errors ** 2)))
                if len(errors)
                else 0.0
            ),

            "endpoint_error_m": float(
                endpoint_error
            ),

            "reached_endpoint": True,
        })


    # --------------------------------------------------------
    # Abort
    # --------------------------------------------------------

    def _abort(self, reason):

        self.true = VehicleState(
            self.true.x,
            self.true.y,
            self.true.yaw,
            0.0,
        )

        self.previous_measured_speed = 0.0

        self.target_speed = 0.0

        self.steering = 0.0

        self.auto_active = False

        self.paused = True

        self.terminal_status = reason

        self.status = reason

        self.saved = False


    # --------------------------------------------------------
    # Main autonomous controller
    # --------------------------------------------------------

    def _advance_auto_one_step(self):

        if (
            not self.auto_active
            or self.paused
            or self.closed
        ):

            return

        estimate = self._estimated_pose()

        feedback = VehicleState(
            float(estimate[0]),
            float(estimate[1]),
            float(estimate[2]),
            float(self.previous_measured_speed),
        )

        # Emergency stop.

        if (
            self.emergency_stop_requested
            and self.gear_manager.phase == "driving"
        ):

            self.target_speed = 0.0

            acceleration = self.pid.compute(
                0.0,
                feedback.v,
                dt=DT,
            )

            steering = self.steering

            if not self._advance_vehicle(
                acceleration,
                steering,
            ):

                return

            estimate_after = self._estimated_pose()

            self._record_step(
                estimate_after,
                0.0,
                self.current_direction,
            )

            if abs(self.previous_measured_speed) <= STOP_SPEED:

                self.true = VehicleState(
                    self.true.x,
                    self.true.y,
                    self.true.yaw,
                    0.0,
                )

                self.previous_measured_speed = 0.0

                self.auto_active = False

                self.paused = True

                self.emergency_stop_requested = False

                self.status = (
                    "EMERGENCY STOPPED | press E to resume"
                )

            return

        # Normal path tracking.

        if self.gear_manager.phase == "driving":

            direction = self.current_direction

            path = self.current_path

            output = self.stanley.compute(
                state=feedback,
                path=path,
                direction=direction,
                last_index=self.current_index,
            )

            self.current_index = max(
                self.current_index,
                int(output.target_index),
            )

            endpoint = path[-1]

            endpoint_distance = math.hypot(
                endpoint.x - feedback.x,
                endpoint.y - feedback.y,
            )

            remaining = signed_remaining_distance(
                self.stanley,
                feedback,
                path,
                direction,
                self.current_index,
                self.current_suffix,
            )

            speed_limit = (
                FORWARD_SPEED
                if direction > 0
                else REVERSE_SPEED
            )

            braking_distance = max(
                0.0,
                remaining - STOP_BUFFER,
            )

            target_magnitude = min(
                speed_limit,
                math.sqrt(
                    2.0
                    * BRAKE_DECELERATION
                    * braking_distance
                ),
            )

            if remaining <= STOP_BUFFER:
                target_magnitude = 0.0

            self.target_speed = (
                direction * target_magnitude
            )

            self.steering = float(
                output.steering
            )

            self.current_segment_cte.append(
                float(output.cross_track_error)
            )

            self.all_cte.append(
                float(output.cross_track_error)
            )

            acceleration = self.pid.compute(
                self.target_speed,
                feedback.v,
                dt=DT,
            )

            # Endpoint stop condition.

            stopped_at_endpoint = (

                remaining <= SEGMENT_REMAINING_TOLERANCE

                and endpoint_distance <= SEGMENT_ENDPOINT_TOLERANCE

                and abs(feedback.v) <= STOP_SPEED
            )

            if stopped_at_endpoint:

                self.true = VehicleState(
                    self.true.x,
                    self.true.y,
                    self.true.yaw,
                    0.0,
                )

                self.previous_measured_speed = 0.0

                self.target_speed = 0.0

                self.steering = 0.0

                self.pid.reset()

                self._summarize_segment(
                    endpoint_distance
                )

                # Change gear if another segment exists.

                if (
                    self.current_segment_index
                    < len(self.path_segments) - 1
                ):

                    next_direction = (
                        self.path_segments[
                            self.current_segment_index + 1
                        ][0]
                    )

                    self.gear_manager.request(
                        next_direction
                    )

                    self.status = (
                        f"STOPPED AT SEGMENT "
                        f"{self.current_segment_index + 1}; "
                        f"shifting to "
                        f"{'FORWARD' if next_direction > 0 else 'REVERSE'}"
                    )

                    estimate_after = self._estimated_pose()

                    self._record_step(
                        estimate_after,
                        output.cross_track_error,
                        direction,
                    )

                    return

                # Final goal check.

                goal_position_error = math.hypot(
                    self.plan_goal_pose[0] - feedback.x,
                    self.plan_goal_pose[1] - feedback.y,
                )

                goal_heading_error = abs(
                    wrap_angle(
                        self.plan_goal_pose[2]
                        - feedback.yaw
                    )
                )

                if (
                    goal_position_error
                    <= GOAL_POSITION_TOLERANCE

                    and goal_heading_error
                    <= GOAL_HEADING_TOLERANCE
                ):

                    self.true = VehicleState(
                        self.true.x,
                        self.true.y,
                        self.true.yaw,
                        0.0,
                    )

                    self.previous_measured_speed = 0.0

                    self.target_speed = 0.0

                    self.steering = 0.0

                    self.auto_active = False

                    self.paused = True

                    self.terminal_status = (
                        "FULL_AUTONOMOUS_PARKING_SUCCESS"
                    )

                    self.status = (
                        "FULL AUTONOMOUS PARKING SUCCESS"
                    )

                else:

                    self._abort(
                        "FINAL_GOAL_TOLERANCE_NOT_MET"
                    )

                estimate_after = self._estimated_pose()

                self._record_step(
                    estimate_after,
                    output.cross_track_error,
                    direction,
                )

                return

            # Detect missed endpoint.

            if (
                remaining <= STOP_BUFFER

                and endpoint_distance > SEGMENT_ENDPOINT_TOLERANCE

                and abs(feedback.v) <= STOP_SPEED
            ):

                self._abort(
                    "SEGMENT_ENDPOINT_MISSED"
                )

                return

            if not self._advance_vehicle(
                acceleration,
                self.steering,
            ):

                return

            estimate_after = self._estimated_pose()

            self._record_step(
                estimate_after,
                output.cross_track_error,
                direction,
            )

            return

        # ----------------------------------------------------
        # Stop-before-shift state machine
        # ----------------------------------------------------

        self.true = VehicleState(
            self.true.x,
            self.true.y,
            self.true.yaw,
            0.0,
        )

        self.previous_measured_speed = 0.0

        self.target_speed = 0.0

        self.steering = 0.0

        changed = self.gear_manager.update(
            0.0,
            DT,
        )

        self.step_index += 1

        if changed:

            self.gear_switches += 1

            self.current_segment_index += 1

            (
                self.current_direction,
                self.current_path,
            ) = self.path_segments[
                self.current_segment_index
            ]

            self.current_suffix = path_suffix(
                self.current_path
            )

            self.current_index = 0

            self.current_segment_cte = []

            self.segment_start_step = self.step_index

            self.pid.reset()

            self.status = (
                f"GEAR SHIFT COMPLETE | segment "
                f"{self.current_segment_index + 1}/"
                f"{len(self.path_segments)} | "
                f"{'FORWARD' if self.current_direction > 0 else 'REVERSE'}"
            )

        estimate_after = self._estimated_pose()

        self._record_step(
            estimate_after,
            0.0,
            0,
        )


    # --------------------------------------------------------
    # Timer
    # --------------------------------------------------------

    def _on_timer(self):

        if self.closed:
            return False

        if self.auto_active and not self.paused:

            self._advance_auto_one_step()

            if (
                self.show_gui
                and self.step_index % 4 == 0
                and not self.closed
            ):

                self._draw()

                self.fig.canvas.draw_idle()

        return True


    # --------------------------------------------------------
    # Start execution
    # --------------------------------------------------------

    def start_execution(self):

        if self.terminal_status is not None:
            return False

        if not self.path_precheck[
            "path_collision_free_on_saved_slam_map"
        ]:

            return False

        self.auto_active = True

        self.paused = False

        self.emergency_stop_requested = False

        self.status = (
            "AUTONOMOUS EXECUTION RUNNING"
        )

        self.saved = False

        return True


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def metrics(self):

        estimate = self._estimated_pose()

        goal_position_error = math.hypot(
            self.plan_goal_pose[0] - estimate[0],
            self.plan_goal_pose[1] - estimate[1],
        )

        goal_heading_error = abs(
            math.degrees(
                wrap_angle(
                    self.plan_goal_pose[2] - estimate[2]
                )
            )
        )

        loc_pos = np.asarray(
            self.localization_position_errors,
            dtype=float,
        )

        loc_yaw = np.asarray(
            self.localization_heading_errors,
            dtype=float,
        )

        cte = np.asarray(
            self.all_cte,
            dtype=float,
        )

        return {

            "stage": (
                "P10.4.2 autonomous execution of "
                "P10.4.1 live-SLAM Hybrid A* path"
            ),

            "status": (
                self.terminal_status
                or (
                    "RUNNING"
                    if self.auto_active and not self.paused
                    else "PAUSED"
                    if self.auto_active and self.paused
                    else "READY_NOT_STARTED"
                )
            ),

            "source_map": str(
                MAP_FILE.relative_to(ROOT)
            ),

            "source_path": str(
                PLAN_FILE.relative_to(ROOT)
            ),

            "resume_pose_source": (
                "P10.4.1 saved handoff: SLAM pose for control, "
                "reference pose for simulation physics only"
            ),

            "controller_start_pose": (
                self.plan_start_pose.tolist()
            ),

            "simulation_physics_resume_pose": (
                self.physics_resume_pose.tolist()
            ),

            "initial_localization_error_m": (
                float(self.initial_localization_error_m)
            ),

            "requested_goal_pose": (
                self.plan_goal_pose.tolist()
            ),

            "path_direction_segments": [
                int(direction)
                for direction, _ in self.path_segments
            ],

            "path_segment_count": (
                len(self.path_segments)
            ),

            "path_point_count": (
                int(len(self.path_raw))
            ),

            "path_length_m": float(
                sum(
                    math.hypot(
                        b[0] - a[0],
                        b[1] - a[1],
                    )
                    for a, b in zip(
                        self.path_raw[:-1],
                        self.path_raw[1:],
                    )
                )
            ),

            "live_slam_map_updated_during_execution": True,

            "ground_truth_used_for_planning": False,

            "ground_truth_used_for_control": False,

            "ground_truth_used_for_simulation_and_collision_audit": True,

            "controller_pose_source": (
                "OnlineSLAM2D + wheel-odometry extrapolation"
            ),

            "controller": (
                "direction-aware Stanley + PID speed + GearManager"
            ),

            "initial_gear_direction": int(
                self.path_segments[0][0]
            ),

            "gear_switches_completed": int(
                self.gear_switches
            ),

            "lidar_scans_during_execution": int(
                len(self.slam.history) - 1
            ),

            "total_lidar_scans_including_handoff": int(
                self.scan_count_offset
                + len(self.slam.history)
            ),

            "control_steps": int(
                self.step_index
            ),

            "execution_duration_s": float(
                self.step_index * DT
            ),

            "collision_in_physical_map": bool(
                self.collision_physical
            ),

            "collision_in_saved_slam_map": bool(
                self.collision_slam_map
            ),

            **self.path_precheck,

            "localization_translation_rmse_m": (
                float(np.sqrt(np.mean(loc_pos ** 2)))
                if len(loc_pos)
                else None
            ),

            "localization_heading_rmse_deg": (
                float(np.sqrt(np.mean(loc_yaw ** 2)))
                if len(loc_yaw)
                else None
            ),

            "estimated_cross_track_rmse_m": (
                float(np.sqrt(np.mean(cte ** 2)))
                if len(cte)
                else None
            ),

            "final_true_pose": [
                float(self.true.x),
                float(self.true.y),
                float(self.true.yaw),
            ],

            "final_slam_pose": [
                float(v)
                for v in estimate
            ],

            "final_estimated_position_error_to_requested_goal_m": (
                float(goal_position_error)
            ),

            "final_estimated_heading_error_deg": (
                float(goal_heading_error)
            ),

            "final_simulated_true_position_error_to_requested_goal_m": (
                float(
                    math.hypot(
                        self.plan_goal_pose[0] - self.true.x,
                        self.plan_goal_pose[1] - self.true.y,
                    )
                )
            ),

            "final_simulated_true_heading_error_deg": (
                float(
                    abs(
                        math.degrees(
                            wrap_angle(
                                self.plan_goal_pose[2]
                                - self.true.yaw
                            )
                        )
                    )
                )
            ),

            "final_speed_mps": float(
                self.true.v
            ),

            "parked_stop_confirmed": bool(
                self.terminal_status
                == "FULL_AUTONOMOUS_PARKING_SUCCESS"
                and abs(self.true.v) < 1e-9
            ),

            "segments": self.segment_summaries,
        }


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    def save(self):

        RESULTS.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Save updated SLAM map separately.

        np.savez_compressed(
            OUT_MAP,

            **self.slam.mapper.arrays(),

            estimated_poses=np.asarray(
                self.slam_history,
                dtype=float,
            ),

            odometry_poses=np.asarray(
                self.wheel_history,
                dtype=float,
            ),

            reference_poses=np.asarray(
                self.reference,
                dtype=float,
            ).reshape(-1, 3),
        )

        records = self.execution_records

        if records:

            record_keys = list(
                records[0].keys()
            )

            numeric_keys = [
                key
                for key in record_keys
                if isinstance(
                    records[0][key],
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                )
            ]

            record_matrix = np.asarray(
                [
                    [
                        row[key]
                        for key in numeric_keys
                    ]
                    for row in records
                ],
                dtype=float,
            )

        else:

            record_matrix = np.empty(
                (0, 0),
                dtype=float,
            )

            numeric_keys = []

        np.savez_compressed(
            OUT_TRAJECTORY,

            trajectory=np.asarray(
                self.execution_trajectory,
                dtype=float,
            ),

            records=record_matrix,

            record_columns=np.asarray(
                numeric_keys,
                dtype="U64",
            ),

            path=self.path_raw,

            start_pose=self.plan_start_pose,

            goal_pose=self.plan_goal_pose,
        )

        summary = self.metrics()

        OUT_METRICS.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if (
            self.show_gui
            and hasattr(self, "fig")
        ):

            self._draw()

            self.fig.savefig(
                OUT_PLOT,
                dpi=160,
            )

        self.saved = True

        print("P10.4.2 saved:")

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

        return summary


# ============================================================
# 5. Main
# ============================================================

def main():

    smoke = "--smoke" in sys.argv

    app = AutonomousExecutionApp(
        show_gui=not smoke
    )

    print(
        "P10.4.2 controls: "
        "E start/resume, "
        "P pause, "
        "Space emergency stop, "
        "M save, "
        "Q quit"
    )

    print(
        "P10.4.1 map and path are inputs; "
        "this script writes only p10_4_2_* outputs."
    )

    if smoke:

        if not app.start_execution():

            raise RuntimeError(
                "P10.4.2 safety precheck refused execution"
            )

        for _ in range(MAX_STEPS):

            if (
                not app.auto_active
                or app.paused
            ):

                break

            app._advance_auto_one_step()

        summary = app.save()

        if (
            summary["status"]
            != "FULL_AUTONOMOUS_PARKING_SUCCESS"
        ):

            raise RuntimeError(
                "Headless execution did not pass: "
                f"{summary['status']}"
            )

    else:

        plt.show()


if __name__ == "__main__":

    main()
