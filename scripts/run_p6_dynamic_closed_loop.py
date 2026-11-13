
"""P6: Compare kinematic and dynamic Ackermann closed-loop parking."""


import sys
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

# Add project root to Python module search path
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "results"

# Project imports
from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import (
    path_metrics,
    resample_path,
)

from parking_planner.simulation.closed_loop import (
    ClosedLoopParkingSimulator,
    SimulationConfig,
)

from parking_planner.vehicle.ackermann import AckermannModel
from parking_planner.vehicle.dynamic_ackermann import (
    DynamicAckermannModel,
)



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
        (
            state.x + c * x - s * y,
            state.y + s * x + c * y,
        )
        for x, y in local
    ]


def create_simulation_config():
    """Keep the same control parameters as P4."""

    return SimulationConfig(
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
    )


def run_simulation(vehicle_model, planner, start, path, config):
    """Run one closed-loop parking experiment."""

    simulator = ClosedLoopParkingSimulator(
        vehicle_model=vehicle_model,
        planner=planner,
        config=config,
    )

    return simulator.run(start, path)


def calculate_metrics(result, vehicle_model, goal):

    records = result["states"]
    final_state = result["final_state"]

    position_error = math.hypot(
        final_state.x - goal.x,
        final_state.y - goal.y,
    )

    heading_error = abs(
        normalize_angle(final_state.yaw - goal.yaw)
    )

    valid_cte = [
        abs(item["cross_track_error"])
        for item in records
        if item["steering"] != 0.0
        or item["cross_track_error"] != 0.0
    ]

    cte_rmse = math.sqrt(
        sum(value ** 2 for value in valid_cte)
        / max(1, len(valid_cte))
    )

    max_cte = max(valid_cte, default=0.0)

    max_commanded_steering = max(
        (abs(item["steering"]) for item in records),
        default=0.0,
    )

    max_speed = max(
        (abs(item["speed"]) for item in records),
        default=0.0,
    )

    # Dynamic model has its own actuator history.
    actuator_history = getattr(vehicle_model, "history", [])

    if actuator_history:
        max_actual_steering = max(
            abs(item["steering_actual"])
            for item in actuator_history
        )

        max_steering_rate = max(
            abs(item["steering_rate"])
            for item in actuator_history
        )

        max_actual_acceleration = max(
            abs(item["acceleration_actual"])
            for item in actuator_history
        )

        max_jerk = max(
            abs(item["jerk"])
            for item in actuator_history
        )

    else:
        # Original kinematic model responds immediately.
        max_actual_steering = max_commanded_steering
        max_steering_rate = None
        max_actual_acceleration = None
        max_jerk = None

    return {
        "closed_loop_status": (
            "SUCCESS" if result["completed"] else "INCOMPLETE"
        ),

        "collision_free": result["collision_free"],

        "simulation_steps": len(records),

        "simulation_duration_s": round(
            result["duration"], 3
        ),

        "gear_switches": result["gear_switches"],

        "final_x_m": round(final_state.x, 4),

        "final_y_m": round(final_state.y, 4),

        "final_yaw_deg": round(
            math.degrees(final_state.yaw), 3
        ),

        "final_speed_mps": round(final_state.v, 4),

        "final_position_error_m": round(
            position_error, 4
        ),

        "final_heading_error_deg": round(
            math.degrees(heading_error), 3
        ),

        "cross_track_rmse_m": round(cte_rmse, 4),

        "cross_track_max_m": round(max_cte, 4),

        "max_commanded_steering_deg": round(
            math.degrees(max_commanded_steering), 3
        ),

        "max_actual_steering_deg": round(
            math.degrees(max_actual_steering), 3
        ),

        "max_actual_steering_rate_deg_s": (
            round(math.degrees(max_steering_rate), 3)
            if max_steering_rate is not None
            else None
        ),

        "max_actual_acceleration_mps2": (
            round(max_actual_acceleration, 4)
            if max_actual_acceleration is not None
            else None
        ),

        "max_jerk_mps3": (
            round(max_jerk, 4)
            if max_jerk is not None
            else None
        ),

        "max_speed_mps": round(max_speed, 3),

        "segments": result["segments"],
    }


def plot_comparison(
    scenario,
    reference_path,
    baseline_result,
    dynamic_result,
    dynamic_model,
    baseline_metrics,
    dynamic_metrics,
):

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # --------------------------------------------------
    # 1. Trajectory comparison
    # --------------------------------------------------

    ax = axes[0, 0]

    xmin, ymin, xmax, ymax = scenario["parking_slot"]

    ax.add_patch(Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=False,
        edgecolor="green",
        linestyle="--",
        linewidth=2,
        label="Parking slot",
    ))

    for obstacle in scenario["parked_vehicles"]:
        x0, y0, x1, y1 = obstacle

        ax.add_patch(Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="gray",
            edgecolor="black",
            alpha=0.7,
        ))

    for direction, color, label in [
        (1, "royalblue", "Reference forward"),
        (-1, "darkorange", "Reference reverse"),
    ]:

        selected = [
            point for point in reference_path
            if point.direction == direction
        ]

        if selected:
            ax.plot(
                [p.x for p in selected],
                [p.y for p in selected],
                "--",
                color=color,
                label=label,
            )

    for result, color, label in [
        (baseline_result, "purple", "Kinematic model"),
        (dynamic_result, "black", "Dynamic model"),
    ]:

        records = result["states"]

        x = [scenario["start"].x] + [
            item["x"] for item in records
        ]

        y = [scenario["start"].y] + [
            item["y"] for item in records
        ]

        ax.plot(x, y, color=color, linewidth=2, label=label)

    ax.scatter(
        scenario["goal"].x,
        scenario["goal"].y,
        marker="*",
        color="red",
        s=120,
        label="Goal",
    )

    ax.set_title("P6 Closed-loop Trajectory Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --------------------------------------------------
    # 2. Speed comparison
    # --------------------------------------------------

    ax = axes[0, 1]

    for result, color, label in [
        (baseline_result, "purple", "Kinematic"),
        (dynamic_result, "black", "Dynamic"),
    ]:

        records = result["states"]

        t = [item["time"] for item in records]
        speed = [item["speed"] for item in records]

        ax.plot(t, speed, color=color, label=label)

    ax.set_title("Vehicle Speed Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --------------------------------------------------
    # 3. Steering actuator response
    # --------------------------------------------------

    ax = axes[1, 0]

    history = dynamic_model.history

    t = [item["time"] for item in history]

    command = [
        math.degrees(item["steering_command"])
        for item in history
    ]

    actual = [
        math.degrees(item["steering_actual"])
        for item in history
    ]

    ax.plot(t, command, "--", label="Steering command")
    ax.plot(t, actual, label="Actual steering")

    ax.set_title("Dynamic Steering Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --------------------------------------------------
    # 4. Acceleration response
    # --------------------------------------------------

    ax = axes[1, 1]

    command = [
        item["acceleration_command"]
        for item in history
    ]

    actual = [
        item["acceleration_actual"]
        for item in history
    ]

    ax.plot(t, command, "--", label="Acceleration command")
    ax.plot(t, actual, label="Actual acceleration")

    ax.set_title("Dynamic Acceleration Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --------------------------------------------------
    # 5. Jerk
    # --------------------------------------------------

    ax = axes[2, 0]

    jerk = [
        item["jerk"] for item in history
    ]

    ax.plot(t, jerk, label="Actual jerk")

    ax.axhline(
        5.0,
        linestyle="--",
        label="Jerk limit",
    )

    ax.axhline(-5.0, linestyle="--")

    ax.set_title("Dynamic Vehicle Jerk")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Jerk (m/s³)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --------------------------------------------------
    # 6. Performance summary
    # --------------------------------------------------

    ax = axes[2, 1]
    ax.axis("off")

    summary = [
        "P6 CLOSED-LOOP COMPARISON",
        "",
        "KINEMATIC MODEL",
        f"Status: {baseline_metrics['closed_loop_status']}",
        f"Collision free: {baseline_metrics['collision_free']}",
        f"Position error: {baseline_metrics['final_position_error_m']} m",
        f"Heading error: {baseline_metrics['final_heading_error_deg']} deg",
        f"Cross-track RMSE: {baseline_metrics['cross_track_rmse_m']} m",
        "",
        "DYNAMIC MODEL",
        f"Status: {dynamic_metrics['closed_loop_status']}",
        f"Collision free: {dynamic_metrics['collision_free']}",
        f"Position error: {dynamic_metrics['final_position_error_m']} m",
        f"Heading error: {dynamic_metrics['final_heading_error_deg']} deg",
        f"Cross-track RMSE: {dynamic_metrics['cross_track_rmse_m']} m",
        f"Max jerk: {dynamic_metrics['max_jerk_mps3']} m/s³",
    ]

    ax.text(
        0.03,
        0.97,
        "\n".join(summary),
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        family="monospace",
    )

    fig.tight_layout()

    figure_path = OUTPUT_DIR / "p6_dynamic_closed_loop_comparison.png"

    fig.savefig(figure_path, dpi=180)

    plt.show()

    print(f"Visualization: {figure_path}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build the same parking scenario used in P4/P5.
    scenario = build_perpendicular_parking_scenario()

    geometry = scenario["geometry"]

    planner = HybridAStarPlanner(
        grid=scenario["grid"],
        geometry=geometry,
        config=scenario["planner_config"],
    )

    # 2. Generate one shared reference path.
    raw_path = planner.plan(
        scenario["start"],
        scenario["goal"],
    )

    if raw_path is None:
        raise SystemExit("Hybrid A* planning failed.")

    reference_path = resample_path(raw_path, spacing=0.1)

    # 3. Shared simulation configuration.
    config = create_simulation_config()

    # 4. Baseline: original kinematic Ackermann model.
    baseline_model = AckermannModel(
        wheelbase=geometry.wheelbase,
        max_steer=math.radians(35.0),
        max_acceleration=2.0,
        max_speed=5.0,
    )

    baseline_result = run_simulation(
        baseline_model,
        planner,
        scenario["start"],
        reference_path,
        config,
    )

    # 5. Dynamic Ackermann model.
    dynamic_model = DynamicAckermannModel(
        wheelbase=geometry.wheelbase,
        max_steer=math.radians(35.0),
        max_acceleration=2.0,
        max_speed=5.0,
        steering_time_constant=0.25,
        max_steering_rate=math.radians(60.0),
        acceleration_time_constant=0.25,
        max_jerk=5.0,
    )

    dynamic_result = run_simulation(
        dynamic_model,
        planner,
        scenario["start"],
        reference_path,
        config,
    )

    # 6. Calculate metrics.
    baseline_metrics = calculate_metrics(
        baseline_result,
        baseline_model,
        scenario["goal"],
    )

    dynamic_metrics = calculate_metrics(
        dynamic_result,
        dynamic_model,
        scenario["goal"],
    )

    metrics = {
        "planning_status": "SUCCESS",
        "expanded_nodes": planner.last_iterations,
        "reference_path_points": len(reference_path),
        "path_length_m": round(
            path_metrics(reference_path)["path_length"], 3
        ),
        "baseline_kinematic": baseline_metrics,
        "dynamic_ackermann": dynamic_metrics,
        "overall_status": (
            "PASS"
            if (
                baseline_result["completed"]
                and baseline_result["collision_free"]
                and dynamic_result["completed"]
                and dynamic_result["collision_free"]
            )
            else "NEEDS_TUNING"
        ),
    }

    # 7. Save metrics.
    metrics_path = OUTPUT_DIR / "p6_dynamic_closed_loop_metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 8. Generate comparison plots.
    plot_comparison(
        scenario,
        reference_path,
        baseline_result,
        dynamic_result,
        dynamic_model,
        baseline_metrics,
        dynamic_metrics,
    )

    # 9. Print results.
    print("\n========== P6 DYNAMIC CLOSED-LOOP RESULTS ==========")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"\nMetrics: {metrics_path}")

    print(f"\nOverall status: {metrics['overall_status']}")


if __name__ == "__main__":
    main()
