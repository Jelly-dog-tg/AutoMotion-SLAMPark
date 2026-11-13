"""P10.1: Hybrid A* path planning on a saved SLAM occupancy map.

Run from ParkingPlanner root:
    python scripts/run_p10_1_slam_map_planning.py

This is an OFFLINE mapping-to-planning integration check, not live parking.
The map is created in P9.5.3; its true obstacle geometry is NOT used.
The configured target parking pose is a manually supplied goal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
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
from parking_planner.integration.slam_map_adapter import build_planner_grid
from parking_planner.mapping.occupancy_grid_mapper import OccupancyGridMapper
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import path_metrics, resample_path
from parking_planner.vehicle.ackermann import VehicleState

MAP_FILENAME = "p9_5_3_keyboard_map.npz"
OUTPUT_PREFIX = "p10_1_slam_map_planning"


def heading_error_deg(a: float, b: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(a - b), math.cos(a - b))))


def path_segments(path):
    """Return connected forward/reverse segments for plotting."""
    if not path:
        return []
    groups = []
    direction = path[0].direction
    current = [path[0]]
    for point in path[1:]:
        if point.direction == direction:
            current.append(point)
        else:
            groups.append((direction, current))
            current = [current[-1], point]
            direction = point.direction
    groups.append((direction, current))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-index", type=int, default=0,
        help="Index in recorded SLAM estimated_poses; default=0 reproduces a full parking maneuver. "
             "Use -1 to try the last explored pose.",
    )
    args = parser.parse_args()

    results = ROOT / "results"
    map_path = results / MAP_FILENAME
    if not map_path.is_file():
        raise FileNotFoundError(
            f"Cannot find {map_path}. Save the P9.5.3 exploration map first."
        )

    # Only P9.5.3 SLAM data supplies the collision map and start pose.
    mapper = OccupancyGridMapper.load(map_path)
    with np.load(map_path, allow_pickle=False) as archive:
        if "estimated_poses" not in archive.files:
            raise KeyError("The saved exploration map has no 'estimated_poses' array")
        poses = np.asarray(archive["estimated_poses"], dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) == 0:
        raise ValueError("estimated_poses must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(poses)):
        raise ValueError("Non-finite pose recorded in SLAM map")
    if not -len(poses) <= args.start_index < len(poses):
        raise IndexError("start-index is outside the saved SLAM pose history")

    start_pose = poses[args.start_index]
    start = VehicleState(float(start_pose[0]), float(start_pose[1]),
                         float(start_pose[2]), 0.0)
    scenario = build_perpendicular_parking_scenario()
    # This target represents a user-selected parking bay in the final system.
    # It is not used as an obstacle map or as a source of vehicle localization.
    goal = scenario["goal"]

    grid = build_planner_grid(mapper, planner_resolution=0.20,
                              unknown_is_occupied=True)
    config = replace(
        scenario["planner_config"],
        goal_xy_tolerance=0.10,
        goal_yaw_tolerance=math.radians(3.0),
    )
    planner = HybridAStarPlanner(
        grid=grid, geometry=scenario["geometry"], config=config
    )

    t0 = time.perf_counter()
    try:
        path = planner.plan(start, goal)
    except ValueError as exc:
        raise RuntimeError(
            "Planning start/goal is blocked on the SAVED SLAM map. "
            "Check exploration coverage or selected pose; do not automatically "
            "change unknown-space to free."
        ) from exc
    elapsed = time.perf_counter() - t0
    if not path:
        raise RuntimeError(
            "No path on the SLAM map. Inspect free-space coverage, start/goal "
            "placement and parking-bay geometry; do not disable safety gating."
        )

    dense = resample_path(path, spacing=0.05)
    # Footprint audit on the dense display/control reference, not just nodes.
    for i, p in enumerate(dense):
        pose = VehicleState(p.x, p.y, p.yaw, 0.0)
        if grid.is_collision(pose, planner.local_x, planner.local_y):
            raise RuntimeError(
                f"Dense planned path collides with the SLAM planning grid "
                f"at point {i}; refusing to export an unsafe reference."
            )

    pm = path_metrics(path)
    metric_file = results / (OUTPUT_PREFIX + "_metrics.json")
    plot_file = results / (OUTPUT_PREFIX + ".png")
    path_file = results / (OUTPUT_PREFIX + "_path.npz")
    results.mkdir(parents=True, exist_ok=True)

    # Persist an explicitly typed path for the next P10 controller-integration stage.
    np.savez_compressed(
        path_file,
        path=np.asarray(
            [[p.x, p.y, p.yaw, p.direction, p.steering] for p in dense],
            dtype=float,
        ),
        start_pose=np.asarray([start.x, start.y, start.yaw]),
        goal_pose=np.asarray([goal.x, goal.y, goal.yaw]),
        source_map=np.asarray(str(map_path.relative_to(ROOT))),
    )

    summary = {
        "stage": "P10.1 SLAM-map Hybrid A* offline planning",
        "status": "SUCCESS",
        "source_map": str(map_path.relative_to(ROOT)),
        "slam_resolution_m": float(mapper.resolution),
        "planner_resolution_m": float(grid.resolution),
        "planner_grid_shape": list(grid.occupied.shape),
        "blocked_cells_including_unknown": int(grid.occupied.sum()),
        "unknown_is_occupied": True,
        "start_pose_source": f"saved SLAM estimated_poses[{args.start_index}]",
        "start_pose": [start.x, start.y, start.yaw],
        "goal_pose_source": "configured parking bay (manual target placeholder)",
        "goal_pose": [goal.x, goal.y, goal.yaw],
        "expanded_nodes": int(planner.last_iterations),
        "planning_wall_time_s": round(elapsed, 4),
        "raw_path_points": len(path),
        "resampled_path_points": len(dense),
        "path_length_m": float(pm["path_length"]),
        "reverse_distance_m": float(pm["reverse_distance"]),
        "gear_switches": int(pm["gear_switches"]),
        "final_position_error_m": float(
            math.hypot(path[-1].x - goal.x, path[-1].y - goal.y)
        ),
        "final_heading_error_deg": heading_error_deg(path[-1].yaw, goal.yaw),
        "dense_path_collision_free_on_slam_grid": True,
        "ground_truth_trajectory_used_for_planning": False,
        "vehicle_closed_loop_executed": False,
        "output_path": str(path_file.relative_to(ROOT)),
    }
    metric_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    ax.imshow(
        grid.occupied.astype(np.uint8),
        origin="lower",
        extent=(
            grid.origin_x, grid.origin_x + grid.width,
            grid.origin_y, grid.origin_y + grid.height,
        ),
        cmap=ListedColormap(["white", "#555555"]), vmin=0, vmax=1,
        interpolation="nearest",
    )
    # Parking-bay outline is a target annotation, not a collision-map input.
    x0, y0, x1, y1 = scenario["parking_slot"]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            "g--", linewidth=1.8, label="Target parking bay")
    for direction, points in path_segments(dense):
        xy = np.asarray([[p.x, p.y] for p in points])
        ax.plot(xy[:, 0], xy[:, 1], linewidth=2.4,
                color="#2563eb" if direction == 1 else "#f97316",
                label="Forward" if direction == 1 else "Reverse")
    ax.scatter(start.x, start.y, c="limegreen", edgecolors="black",
               s=70, zorder=5, label="SLAM start")
    ax.scatter(goal.x, goal.y, c="red", marker="*",
               s=160, zorder=6, label="Selected goal")
    ax.scatter(path[-1].x, path[-1].y, c="cyan", edgecolors="black",
               s=50, zorder=6, label="Planned endpoint")
    ax.set(xlim=(grid.origin_x, grid.origin_x + grid.width),
           ylim=(grid.origin_y, grid.origin_y + grid.height),
           xlabel="X (m)", ylabel="Y (m)",
           title="P10.1 Hybrid A* on saved SLAM Occupancy Grid")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=160)
    plt.close(fig)

    print("\n========== P10.1 SLAM-MAP PLANNING ==========")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {metric_file}\n       {path_file}\n       {plot_file}")


if __name__ == "__main__":
    main()
