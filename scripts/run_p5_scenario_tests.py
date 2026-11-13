
"""P5: Multi-scenario autonomous parking evaluation."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import (
    HybridAStarPlanner,
    OccupancyGrid,
)
from parking_planner.planning.path_utils import path_metrics, resample_path
from parking_planner.simulation.closed_loop import (
    ClosedLoopParkingSimulator,
    SimulationConfig,
)
from parking_planner.vehicle.ackermann import AckermannModel, VehicleState


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def vehicle_corners(state, geometry):
    local = [
        (-geometry.rear_overhang, -geometry.width / 2),
        (geometry.front_extent, -geometry.width / 2),
        (geometry.front_extent, geometry.width / 2),
        (-geometry.rear_overhang, geometry.width / 2),
    ]

    c = math.cos(state.yaw)
    s = math.sin(state.yaw)

    return [
        (state.x + c * x - s * y, state.y + s * x + c * y)
        for x, y in local
    ]


def make_test_scenarios():
    """Create three tests without modifying the P4 baseline."""

    base = build_perpendicular_parking_scenario()

    # T1: Shift the initial vehicle position.
    initial_pose = dict(base)
    initial_pose["name"] = "T1_shifted_start"
    initial_pose["start"] = VehicleState(
        x=6.74, y=10.0, yaw=0.0, v=0.0
    )

    # T2: Move the parking bay and parked vehicles 0.5 m to the right.
    shift_x = 0.50
    moved_bay = dict(base)
    moved_bay["name"] = "T2_moved_bay"

    moved_obstacles = [
        (x0 + shift_x, y0, x1 + shift_x, y1)
        for x0, y0, x1, y1 in base["parked_vehicles"]
    ]

    moved_grid = OccupancyGrid(
        width=base["grid"].width,
        height=base["grid"].height,
        resolution=base["grid"].resolution,
    )

    for obstacle in moved_obstacles:
        moved_grid.add_obstacle_rect(*obstacle)

    moved_bay["grid"] = moved_grid
    moved_bay["parked_vehicles"] = moved_obstacles

    moved_bay["parking_slot"] = tuple(
        value + shift_x if index in (0, 2) else value
        for index, value in enumerate(base["parking_slot"])
    )

    moved_bay["goal"] = VehicleState(
        x=base["goal"].x + shift_x,
        y=base["goal"].y,
        yaw=base["goal"].yaw,
        v=0.0,
    )

    # T3: Narrow the parking bay from 2.6 m to 2.3 m.
    narrow_bay = dict(base)
    narrow_bay["name"] = "T3_narrow_bay"

    narrow_obstacles = [
        (7.4, 11.5, 9.35, 16.0),
        (11.65, 11.5, 13.6, 16.0),
    ]

    narrow_grid = OccupancyGrid(
        width=base["grid"].width,
        height=base["grid"].height,
        resolution=base["grid"].resolution,
    )

    for obstacle in narrow_obstacles:
        narrow_grid.add_obstacle_rect(*obstacle)

    narrow_bay["grid"] = narrow_grid
    narrow_bay["parked_vehicles"] = narrow_obstacles
    narrow_bay["parking_slot"] = (9.35, 10.8, 11.65, 17.0)

    return [initial_pose, moved_bay, narrow_bay]


def run_one_scenario(scenario):

    geometry = scenario["geometry"]

    planner = HybridAStarPlanner(
        grid=scenario["grid"],
        geometry=geometry,
        config=scenario["planner_config"],
    )

    raw_path = planner.plan(
        scenario["start"],
        scenario["goal"],
    )

    if raw_path is None:
        return {
            "name": scenario["name"],
            "planning_status": "FAILED",
            "closed_loop_status": "NOT_RUN",
            "expanded_nodes": planner.last_iterations,
            "collision_free": False,
            "passed": False,
        }, None, None

    reference_path = resample_path(raw_path, spacing=0.1)

    vehicle_model = AckermannModel(
        wheelbase=geometry.wheelbase,
        max_steer=math.radians(35.0),
        max_acceleration=2.0,
        max_speed=5.0,
    )

    simulator = ClosedLoopParkingSimulator(
        vehicle_model=vehicle_model,
        planner=planner,
        config=SimulationConfig(
            dt=0.05,
            forward_speed=0.5,
            reverse_speed=0.4,
            max_acceleration=2.0,
            max_speed=5.0,
            speed_kp=2.5,
            brake_deceleration=1.5,
            stop_buffer=0.04,
            endpoint_tolerance=0.16,
            stop_speed=0.06,
            max_steps_per_segment=1600,
            shift_duration=0.4,
        ),
    )

    result = simulator.run(
        scenario["start"],
        reference_path,
    )

    final_state = result["final_state"]
    goal = scenario["goal"]

    position_error = math.hypot(
        final_state.x - goal.x,
        final_state.y - goal.y,
    )

    heading_error = abs(
        normalize_angle(final_state.yaw - goal.yaw)
    )

    records = result["states"]

    cte = [
        abs(item["cross_track_error"])
        for item in records
        if item["steering"] != 0.0
        or item["cross_track_error"] != 0.0
    ]

    position_tolerance = 0.20
    heading_tolerance_deg = 5.0

    passed = (
        result["completed"]
        and result["collision_free"]
        and position_error <= position_tolerance
        and math.degrees(heading_error) <= heading_tolerance_deg
    )

    metrics = {
        "name": scenario["name"],
        "planning_status": "SUCCESS",
        "closed_loop_status": (
            "SUCCESS" if result["completed"] else "INCOMPLETE"
        ),
        "expanded_nodes": planner.last_iterations,
        "path_points": len(reference_path),
        "path_length_m": round(
            path_metrics(reference_path)["path_length"], 3
        ),
        "simulation_steps": len(records),
        "simulation_duration_s": round(result["duration"], 3),
        "gear_switches": result["gear_switches"],
        "final_position_error_m": round(position_error, 4),
        "final_heading_error_deg": round(
            math.degrees(heading_error), 3
        ),
        "cross_track_rmse_m": round(
            math.sqrt(
                sum(value * value for value in cte)
                / max(1, len(cte))
            ),
            4,
        ),
        "cross_track_max_m": round(max(cte, default=0.0), 4),
        "collision_free": bool(result["collision_free"]),
        "position_tolerance_m": position_tolerance,
        "heading_tolerance_deg": heading_tolerance_deg,
        "passed": bool(passed),
    }

    return metrics, reference_path, result


def draw_scenario(ax, scenario, metrics, reference_path, result):

    slot_xmin, slot_ymin, slot_xmax, slot_ymax = (
        scenario["parking_slot"]
    )

    ax.add_patch(Rectangle(
        (slot_xmin, slot_ymin),
        slot_xmax - slot_xmin,
        slot_ymax - slot_ymin,
        facecolor="none",
        edgecolor="seagreen",
        linestyle="--",
        linewidth=2,
        label="Parking bay",
    ))

    for i, (x0, y0, x1, y1) in enumerate(
        scenario["parked_vehicles"]
    ):
        ax.add_patch(Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="gray",
            edgecolor="black",
            alpha=0.75,
            label="Parked vehicle" if i == 0 else None,
        ))

    goal = scenario["goal"]

    ax.scatter(
        goal.x, goal.y,
        marker="*", color="red", s=100,
        label="Goal",
    )

    ax.scatter(
        scenario["start"].x,
        scenario["start"].y,
        color="green",
        s=35,
        label="Start",
    )

    if reference_path is not None and result is not None:

        ax.plot(
            [p.x for p in reference_path],
            [p.y for p in reference_path],
            "--",
            color="royalblue",
            linewidth=1.5,
            label="Planned path",
        )

        records = result["states"]

        actual_x = [scenario["start"].x] + [
            r["x"] for r in records
        ]

        actual_y = [scenario["start"].y] + [
            r["y"] for r in records
        ]

        ax.plot(
            actual_x,
            actual_y,
            color="darkorange",
            linewidth=2,
            label="Actual trajectory",
        )

        final_state = result["final_state"]

        ax.add_patch(Polygon(
            vehicle_corners(final_state, scenario["geometry"]),
            closed=True,
            fill=False,
            edgecolor="red",
            linewidth=1.8,
            label="Final vehicle",
        ))

    status = metrics.get(
        "closed_loop_status",
        metrics.get("planning_status", "UNKNOWN"),
    )

    ax.set_title(f"{scenario['name']}\n{status}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(4.5, 16.5)
    ax.set_ylim(7.0, 18.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)


def main():

    scenarios = make_test_scenarios()

    all_metrics = []
    plot_items = []

    for scenario in scenarios:

        print(f"\nRunning {scenario['name']} ...")

        metrics, reference_path, result = run_one_scenario(
            scenario
        )

        all_metrics.append(metrics)

        plot_items.append(
            (scenario, metrics, reference_path, result)
        )

        for key, value in metrics.items():
            print(f"  {key}: {value}")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "p5_scenario_metrics.json"

    metrics_path.write_text(
        json.dumps(all_metrics, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, item in zip(axes, plot_items):
        draw_scenario(ax, *item)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.92),
        fontsize=9,
    )

    fig.suptitle(
        "P5 Multi-scenario Autonomous Parking Evaluation",
        fontsize=15,
        y=0.99,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.84))

    figure_path = output_dir / "p5_multi_scenario_comparison.png"

    fig.savefig(figure_path, dpi=180)

    plt.show()

    passed = sum(
        1 for item in all_metrics
        if item.get("passed", False)
    )

    print("\n========== P5 MULTI-SCENARIO RESULTS ==========")
    print(f"Passed: {passed}/{len(all_metrics)}")
    print(f"Metrics: {metrics_path}")
    print(f"Visualization: {figure_path}")


if __name__ == "__main__":
    main()
