
"""P11.7.2: Diagnose goal footprint blocking from live SLAM map."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import (
    OccupancyGrid,
    HybridAStarPlanner,
)
from parking_planner.vehicle.ackermann import VehicleState


RESULTS = ROOT / "results"
MAP_FILE = RESULTS / "p11_6_4_live_slam_map.npz"

PLANNER_RESOLUTION = 0.20


def footprint_polygon(pose, geometry):
    local = np.array([
        [-geometry.rear_overhang, -geometry.width / 2],
        [geometry.front_extent, -geometry.width / 2],
        [geometry.front_extent, geometry.width / 2],
        [-geometry.rear_overhang, geometry.width / 2],
    ])

    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)

    return np.column_stack((
        pose.x + c * local[:, 0] - s * local[:, 1],
        pose.y + s * local[:, 0] + c * local[:, 1],
    ))


def polygon_intersects_cell(poly, x0, y0, x1, y1):
    """SAT collision test between polygon and an axis-aligned cell."""

    rect = np.array([
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
    ])

    axes = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
    ]

    for i in range(len(poly)):
        edge = poly[(i + 1) % len(poly)] - poly[i]
        axes.append(np.array([-edge[1], edge[0]]))

    for axis in axes:
        p_projection = poly @ axis
        r_projection = rect @ axis

        if (
            p_projection.max() < r_projection.min() - 1e-9
            or r_projection.max() < p_projection.min() - 1e-9
        ):
            return False

    return True


def make_planner_grid(states, cfg):

    ratio = int(round(
        PLANNER_RESOLUTION / float(cfg["resolution"])
    ))

    blocks = states.reshape(
        states.shape[0] // ratio,
        ratio,
        states.shape[1] // ratio,
        ratio,
    )

    blocked = np.any(blocks != 0, axis=(1, 3))

    grid = OccupancyGrid(
        width=float(cfg["width"]),
        height=float(cfg["height"]),
        resolution=PLANNER_RESOLUTION,
        origin_x=float(cfg["origin_x"]),
        origin_y=float(cfg["origin_y"]),
    )

    grid.occupied[:] = blocked

    return grid, blocks, ratio


def main():

    if not MAP_FILE.exists():
        raise FileNotFoundError(MAP_FILE)

    with np.load(MAP_FILE, allow_pickle=False) as data:
        cfg = json.loads(str(data["config"].item()))
        states = np.asarray(data["states"], dtype=np.int8)
        poses = np.asarray(data["estimated_poses"], dtype=float)

    scenario = build_perpendicular_parking_scenario()

    g = scenario["goal"]

    goal = VehicleState(
        float(g.x),
        float(g.y),
        float(g.yaw),
        0.0,
    )

    start = VehicleState(
        float(poses[-1, 0]),
        float(poses[-1, 1]),
        float(poses[-1, 2]),
        0.0,
    )

    geometry = scenario["geometry"]

    grid, blocks, ratio = make_planner_grid(states, cfg)

    planner = HybridAStarPlanner(
        grid=grid,
        geometry=geometry,
        config=scenario["planner_config"],
    )

    poly = footprint_polygon(goal, geometry)

    actual_collision = grid.is_collision(
        goal,
        planner.local_x,
        planner.local_y,
    )

    origin_x = grid.origin_x
    origin_y = grid.origin_y
    resolution = grid.resolution

    min_x, min_y = poly.min(axis=0)
    max_x, max_y = poly.max(axis=0)

    row_min = max(
        0, int(math.floor((min_y - origin_y) / resolution))
    )
    row_max = min(
        grid.occupied.shape[0] - 1,
        int(math.floor((max_y - origin_y) / resolution))
    )

    col_min = max(
        0, int(math.floor((min_x - origin_x) / resolution))
    )
    col_max = min(
        grid.occupied.shape[1] - 1,
        int(math.floor((max_x - origin_x) / resolution))
    )

    counts = {
        "free": 0,
        "unknown": 0,
        "occupied": 0,
        "mixed": 0,
    }

    blockers = []

    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):

            x0 = origin_x + c * resolution
            y0 = origin_y + r * resolution

            x1 = x0 + resolution
            y1 = y0 + resolution

            if not polygon_intersects_cell(
                poly, x0, y0, x1, y1
            ):
                continue

            source_block = blocks[r, :, c, :]

            unknown_count = int(np.sum(source_block == -1))
            occupied_count = int(np.sum(source_block == 1))
            free_count = int(np.sum(source_block == 0))

            if unknown_count and occupied_count:
                reason = "mixed"
            elif occupied_count:
                reason = "occupied"
            elif unknown_count:
                reason = "unknown"
            else:
                reason = "free"

            counts[reason] += 1

            if reason != "free":
                blockers.append({
                    "row": r,
                    "col": c,
                    "center_x": round((x0 + x1) / 2, 3),
                    "center_y": round((y0 + y1) / 2, 3),
                    "reason": reason,
                    "unknown_source_cells": unknown_count,
                    "occupied_source_cells": occupied_count,
                    "free_source_cells": free_count,
                })

    # Check whether the vehicle footprint fits inside the bay.

    x0, y0, x1, y1 = scenario["parking_slot"]

    margins = {
        "left": float(poly[:, 0].min() - x0),
        "right": float(x1 - poly[:, 0].max()),
        "bottom": float(poly[:, 1].min() - y0),
        "top": float(y1 - poly[:, 1].max()),
    }

    inside_slot = min(margins.values()) >= -1e-6

    summary = {
        "stage": "P11.7.2 Goal footprint diagnostic",
        "source_map": str(MAP_FILE.relative_to(ROOT)),
        "goal_pose": [goal.x, goal.y, goal.yaw],
        "start_pose": [start.x, start.y, start.yaw],
        "vehicle_geometry": {
            "width": geometry.width,
            "front_extent": geometry.front_extent,
            "rear_overhang": geometry.rear_overhang,
        },
        "goal_footprint_polygon": poly.tolist(),
        "planner_collision_checker": bool(actual_collision),
        "overlapping_planner_cells": sum(counts.values()),
        "overlapping_cell_classification": counts,
        "blocking_cells": len(blockers),
        "blockers": blockers,
        "parking_slot": [x0, y0, x1, y1],
        "footprint_inside_slot": bool(inside_slot),
        "slot_margins_m": margins,
        "physical_vehicle_moved": False,
        "planning_executed": False,
        "map_modified": False,
        "safety_policy_modified": False,
    }

    # Three-state map: free / unknown / occupied.

    display = np.zeros(states.shape, dtype=np.uint8)
    display[states == -1] = 1
    display[states == 1] = 2

    fig, ax = plt.subplots(figsize=(11, 9))

    ax.imshow(
        display,
        origin="lower",
        extent=(
            cfg["origin_x"],
            cfg["origin_x"] + cfg["width"],
            cfg["origin_y"],
            cfg["origin_y"] + cfg["height"],
        ),
        cmap=ListedColormap([
            "white",
            "#f2c66d",
            "#555555",
        ]),
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )

    # Target parking bay.
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        "g--",
        linewidth=2,
    )

    # Goal footprint.
    closed_poly = np.vstack([poly, poly[0]])

    ax.plot(
        closed_poly[:, 0],
        closed_poly[:, 1],
        color="red",
        linewidth=2.5,
    )

    ax.scatter(
        goal.x, goal.y,
        marker="*",
        color="red",
        s=140,
        zorder=5,
    )

    # Current vehicle footprint.
    start_poly = footprint_polygon(start, geometry)
    start_closed = np.vstack([start_poly, start_poly[0]])

    ax.plot(
        start_closed[:, 0],
        start_closed[:, 1],
        color="limegreen",
        linewidth=2,
    )

    # Mark every blocked planner cell overlapping the goal footprint.
    for item in blockers:

        color = (
            "#ff9800"
            if item["reason"] == "unknown"
            else "#e31a1c"
        )

        ax.add_patch(Rectangle(
            (
                item["center_x"] - resolution / 2,
                item["center_y"] - resolution / 2,
            ),
            resolution,
            resolution,
            fill=False,
            edgecolor=color,
            linewidth=1.5,
        ))

    handles = [
        Patch(facecolor="white", edgecolor="gray", label="Free"),
        Patch(facecolor="#f2c66d", label="Unknown"),
        Patch(facecolor="#555555", label="Occupied"),
        Line2D([0], [0], color="red", lw=2, label="Goal footprint"),
        Line2D([0], [0], color="limegreen", lw=2, label="Current footprint"),
        Line2D([0], [0], color="green", linestyle="--", label="Parking bay"),
        Patch(facecolor="none", edgecolor="#ff9800", label="Unknown blocker"),
        Patch(facecolor="none", edgecolor="#e31a1c", label="Occupied/mixed blocker"),
    ]

    ax.legend(handles=handles, loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        "P11.7.2 Goal Footprint Diagnostic | "
        f"collision={actual_collision}"
    )

    ax.grid(alpha=0.15)
    fig.tight_layout()

    image_file = RESULTS / "p11_7_2_goal_footprint_diagnostic.png"
    fig.savefig(image_file, dpi=160)
    plt.close(fig)

    metrics_file = RESULTS / "p11_7_2_goal_footprint_diagnostic.json"

    metrics_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {image_file}")
    print(f"Saved: {metrics_file}")


if __name__ == "__main__":
    main()
