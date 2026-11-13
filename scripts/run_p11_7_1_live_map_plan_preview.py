"""P11.7.1: Hybrid A* preview on the latest P11.6.4 live SLAM map.

This is a planning/safety preview only. It does NOT move the physical car.
It never loads the P10.4.2 fixed path or uses reference_poses for planning.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import OccupancyGrid, HybridAStarPlanner
from parking_planner.planning.path_utils import resample_path, path_metrics
from parking_planner.vehicle.ackermann import VehicleState

RESULTS = ROOT / "results"
MAP_FILE = RESULTS / "p11_6_4_live_slam_map.npz"
PREFIX = "p11_7_1_live_map_plan_preview"
PLANNER_RESOLUTION = 0.20


def make_planner_grid(states: np.ndarray, cfg: dict) -> OccupancyGrid:
    """Downsample 5 cm SLAM cells to 20 cm; unknown and occupied both block."""
    source_res = float(cfg["resolution"])
    ratio_float = PLANNER_RESOLUTION / source_res
    ratio = int(round(ratio_float))
    if ratio < 1 or not math.isclose(ratio_float, ratio, abs_tol=1e-8):
        raise ValueError("Planner resolution must be an integer multiple of SLAM resolution")
    rows, cols = states.shape
    if rows % ratio or cols % ratio:
        raise ValueError("SLAM grid shape is not divisible by the downsampling ratio")
    blocks = states.reshape(rows // ratio, ratio, cols // ratio, ratio)
    blocked = np.any(blocks != 0, axis=(1, 3))  # unknown=-1 and occupied=1
    grid = OccupancyGrid(
        width=float(cfg["width"]), height=float(cfg["height"]),
        resolution=PLANNER_RESOLUTION,
        origin_x=float(cfg["origin_x"]), origin_y=float(cfg["origin_y"]),
    )
    if blocked.shape != grid.occupied.shape:
        raise ValueError(f"Planner grid shape mismatch: {blocked.shape} vs {grid.occupied.shape}")
    grid.occupied[:] = blocked
    return grid


def footprint_polygon(pose: VehicleState, geometry):
    local = np.array([
        [-geometry.rear_overhang, -geometry.width / 2],
        [geometry.front_extent, -geometry.width / 2],
        [geometry.front_extent, geometry.width / 2],
        [-geometry.rear_overhang, geometry.width / 2],
        [-geometry.rear_overhang, -geometry.width / 2],
    ])
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return np.column_stack((
        pose.x + c * local[:, 0] - s * local[:, 1],
        pose.y + s * local[:, 0] + c * local[:, 1],
    ))


def split_path(path):
    if not path:
        return []
    result, direction, segment = [], path[0].direction, [path[0]]
    for point in path[1:]:
        if point.direction == direction:
            segment.append(point)
        else:
            result.append((direction, segment))
            segment = [segment[-1], point]
            direction = point.direction
    result.append((direction, segment))
    return result


def main():
    if not MAP_FILE.is_file():
        raise FileNotFoundError(f"Missing live SLAM map: {MAP_FILE}")

    with np.load(MAP_FILE, allow_pickle=False) as data:
        cfg = json.loads(str(data["config"].item()))
        states = np.asarray(data["states"], dtype=np.int8)
        poses = np.asarray(data["estimated_poses"], dtype=float)

    if states.ndim != 2 or poses.ndim != 2 or poses.shape[1] != 3 or len(poses) == 0:
        raise ValueError("Invalid SLAM map or estimated_poses in P11.6.4 archive")
    if not np.all(np.isfinite(poses)):
        raise ValueError("Non-finite SLAM pose in P11.6.4 archive")

    scenario = build_perpendicular_parking_scenario()
    estimate = poses[-1].copy()
    start = VehicleState(float(estimate[0]), float(estimate[1]), float(estimate[2]), 0.0)
    g = scenario["goal"]
    goal = VehicleState(float(g.x), float(g.y), float(g.yaw), 0.0)

    grid = make_planner_grid(states, cfg)
    planner = HybridAStarPlanner(
        grid=grid, geometry=scenario["geometry"], config=scenario["planner_config"]
    )

    start_blocked = grid.is_collision(start, planner.local_x, planner.local_y)
    goal_blocked = grid.is_collision(goal, planner.local_x, planner.local_y)
    path = None
    planning_error = None

    if not start_blocked and not goal_blocked:
        try:
            path = planner.plan(start, goal)
            if not path:
                planning_error = "planner_returned_empty_path"
        except (ValueError, RuntimeError) as exc:
            planning_error = f"{type(exc).__name__}: {exc}"

    dense = resample_path(path, spacing=0.05) if path else []
    dense_collision = False
    if dense:
        for point in dense:
            pose = VehicleState(float(point.x), float(point.y), float(point.yaw), 0.0)
            if grid.is_collision(pose, planner.local_x, planner.local_y):
                dense_collision = True
                break
        if dense_collision:
            path = None
            dense = []
            planning_error = "dense_path_footprint_collision"

    if start_blocked:
        status = "SAFETY_REFUSED_START_FOOTPRINT_BLOCKED"
    elif goal_blocked:
        status = "SAFETY_REFUSED_GOAL_FOOTPRINT_BLOCKED"
    elif path:
        status = "PLAN_SUCCESS_NOT_EXECUTED"
    else:
        status = "NO_SAFE_PATH"

    RESULTS.mkdir(parents=True, exist_ok=True)
    map_cells = {str(int(v)): int((states == v).sum()) for v in (-1, 0, 1)}
    summary = {
        "stage": "P11.7.1 Hybrid A* preview on P11.6.4 live SLAM map",
        "status": status,
        "source_map": str(MAP_FILE.relative_to(ROOT)),
        "start_pose_source": "latest estimated_poses[-1] from P11.6.4",
        "start_pose": estimate.tolist(),
        "goal_pose_source": "configured target parking pose; not a fixed path",
        "goal_pose": [goal.x, goal.y, goal.yaw],
        "slam_resolution_m": float(cfg["resolution"]),
        "planner_resolution_m": PLANNER_RESOLUTION,
        "unknown_and_occupied_are_blocked": True,
        "slam_cell_counts": map_cells,
        "planner_blocked_cells": int(grid.occupied.sum()),
        "start_footprint_blocked": bool(start_blocked),
        "goal_footprint_blocked": bool(goal_blocked),
        "planning_expanded_nodes": int(planner.last_iterations),
        "planning_error": planning_error,
        "path_points": len(path) if path else 0,
        "dense_path_points": len(dense),
        "path_metrics": path_metrics(path) if path else None,
        "dense_path_collision_free": bool(path and not dense_collision),
        "ground_truth_used_for_planning": False,
        "P10_4_2_fixed_path_loaded": False,
        "reference_poses_used_for_planning": False,
        "physical_vehicle_moved": False,
    }

    if path:
        np.savez_compressed(
            RESULTS / f"{PREFIX}.npz",
            path=np.asarray([[p.x, p.y, p.yaw, p.direction, p.steering] for p in dense], dtype=float),
            start_pose=estimate,
            goal_pose=np.asarray([goal.x, goal.y, goal.yaw], dtype=float),
            source_map=np.asarray(str(MAP_FILE.relative_to(ROOT))),
        )

    fig, ax = plt.subplots(figsize=(9, 8))
    extent = (
        grid.origin_x, grid.origin_x + grid.width,
        grid.origin_y, grid.origin_y + grid.height,
    )
    ax.imshow(grid.occupied.astype(np.uint8), origin="lower", extent=extent,
              cmap=ListedColormap(["white", "#555555"]), vmin=0, vmax=1,
              interpolation="nearest")
    x0, y0, x1, y1 = scenario["parking_slot"]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            "g--", lw=1.8, label="Target parking bay")
    ax.plot(*footprint_polygon(start, scenario["geometry"]).T,
            color="limegreen", lw=2, label="Current SLAM footprint")
    ax.plot(*footprint_polygon(goal, scenario["geometry"]).T,
            color="red" if goal_blocked else "cyan", lw=2,
            label="Goal footprint (blocked)" if goal_blocked else "Goal footprint")
    ax.scatter(start.x, start.y, c="limegreen", edgecolors="black", s=65, zorder=5)
    ax.scatter(goal.x, goal.y, c="red", marker="*", s=150, zorder=6)
    if path:
        for direction, segment in split_path(dense):
            xy = np.asarray([[p.x, p.y] for p in segment])
            ax.plot(xy[:, 0], xy[:, 1], lw=2.4,
                    color="royalblue" if direction > 0 else "darkorange",
                    label="Forward" if direction > 0 else "Reverse")
    ax.set(xlim=(grid.origin_x, grid.origin_x + grid.width),
           ylim=(grid.origin_y, grid.origin_y + grid.height),
           xlabel="X (m)", ylabel="Y (m)",
           title=f"P11.7.1 Live SLAM map -> Hybrid A* | {status}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{PREFIX}.png", dpi=160)
    plt.close(fig)

    metrics_file = RESULTS / f"{PREFIX}_metrics.json"
    metrics_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {metrics_file}")
    print(f"Saved: {RESULTS / (PREFIX + '.png')}")
    if path:
        print(f"Saved: {RESULTS / (PREFIX + '.npz')}")
    else:
        print("No path was exported; safety refusal is intentional.")


if __name__ == "__main__":
    main()
