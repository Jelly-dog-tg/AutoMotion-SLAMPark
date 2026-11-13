"""Plan and visualize a perpendicular reverse-parking maneuver."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import resample_path, path_metrics
from parking_planner.vehicle.ackermann import VehicleState


def vehicle_corners(state, geometry):
    local = [
        (-geometry.rear_overhang, -geometry.width / 2.0),
        (geometry.front_extent, -geometry.width / 2.0),
        (geometry.front_extent, geometry.width / 2.0),
        (-geometry.rear_overhang, geometry.width / 2.0),
    ]
    c = math.cos(state.yaw)
    s = math.sin(state.yaw)
    return [
        (state.x + c * x - s * y, state.y + s * x + c * y)
        for x, y in local
    ]


def draw_vehicle(ax, state, geometry, color, label):
    ax.add_patch(
        Polygon(
            vehicle_corners(state, geometry),
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            label=label,
            zorder=5,
        )
    )
    ax.annotate(
        "",
        xy=(
            state.x + 1.4 * math.cos(state.yaw),
            state.y + 1.4 * math.sin(state.yaw),
        ),
        xytext=(state.x, state.y),
        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.8},
        zorder=6,
    )


def main():
    scenario = build_perpendicular_parking_scenario()

    planner = HybridAStarPlanner(
        grid=scenario["grid"],
        wheelbase=scenario["geometry"].wheelbase,
        geometry=scenario["geometry"],
        config=scenario["planner_config"],
    )

    path = planner.plan(scenario["start"], scenario["goal"])
    if path is None:
        print("PARKING PLAN: FAILED")
        print(f"Expanded nodes: {planner.last_iterations}")
        raise SystemExit(1)

    dense_path = resample_path(path, spacing=0.1)
    metrics = path_metrics(dense_path)

    start = scenario["start"]
    goal = scenario["goal"]
    geometry = scenario["geometry"]
    slot_xmin, slot_ymin, slot_xmax, slot_ymax = scenario["parking_slot"]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Target parking bay outline.
    ax.add_patch(
        Rectangle(
            (slot_xmin, slot_ymin),
            slot_xmax - slot_xmin,
            slot_ymax - slot_ymin,
            facecolor="none",
            edgecolor="seagreen",
            linewidth=2.0,
            linestyle="--",
            label="Target parking bay",
        )
    )

    # Neighboring parked vehicles.
    for index, (ox0, oy0, ox1, oy1) in enumerate(scenario["parked_vehicles"]):
        ax.add_patch(
            Rectangle(
                (ox0, oy0),
                ox1 - ox0,
                oy1 - oy0,
                facecolor="dimgray",
                edgecolor="black",
                alpha=0.8,
                label="Parked vehicle" if index == 0 else None,
            )
        )

    # Forward and reverse path segments.
    forward_label_used = False
    reverse_label_used = False

    for previous, current in zip(dense_path[:-1], dense_path[1:]):
        if current.direction < 0:
            color = "darkorange"
            label = "Reverse" if not reverse_label_used else None
            reverse_label_used = True
        else:
            color = "royalblue"
            label = "Forward" if not forward_label_used else None
            forward_label_used = True

        ax.plot(
            [previous.x, current.x],
            [previous.y, current.y],
            color=color,
            linewidth=2.0,
            label=label,
            zorder=3,
        )

    ax.scatter(start.x, start.y, color="green", s=75, label="Start", zorder=7)
    ax.scatter(goal.x, goal.y, color="red", marker="*", s=150, label="Goal", zorder=7)

    draw_vehicle(ax, start, geometry, "green", "Start vehicle")
    draw_vehicle(ax, path[-1], geometry, "red", "Final planned pose")

    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.0, 22.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Hybrid A* - Perpendicular Reverse Parking")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "perpendicular_reverse_parking.png"
    fig.savefig(output_path, dpi=180)
    plt.show()

    final = path[-1]
    position_error = math.hypot(final.x - goal.x, final.y - goal.y)
    yaw_error = abs(
        math.atan2(
            math.sin(final.yaw - goal.yaw),
            math.cos(final.yaw - goal.yaw),
        )
    )

    # Independent collision check at every displayed path sample.
    collision_free = all(
        not scenario["grid"].is_collision(
            VehicleState(point.x, point.y, point.yaw, 0.0),
            planner.local_x,
            planner.local_y,
        )
        for point in dense_path
    )

    print("\n========== PERPENDICULAR PARKING RESULTS ==========")
    print("Planning status: SUCCESS")
    print(f"Expanded nodes: {planner.last_iterations}")
    print(f"Raw path points: {len(path)}")
    print(f"Resampled path points: {len(dense_path)}")
    print(f"Path length: {metrics['path_length']:.3f} m")
    print(f"Reverse distance: {metrics['reverse_distance']:.3f} m")
    print(f"Gear switches: {metrics['gear_switches']}")
    print(f"Final position error: {position_error:.3f} m")
    print(f"Final heading error: {math.degrees(yaw_error):.2f} deg")
    print(f"Resampled path collision-free: {collision_free}")
    print(f"Visualization: {output_path}")

    if not collision_free:
        raise SystemExit("Path validation failed: resampled path intersects an obstacle.")


if __name__ == "__main__":
    main()
