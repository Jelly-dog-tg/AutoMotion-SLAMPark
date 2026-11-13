
"""P7: Hybrid A* and autonomous parking control optimization."""

import sys
import json
import math
from pathlib import Path
from dataclasses import replace

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "results"

from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.planning.hybrid_astar import (
    HybridAStarPlanner,
)

from parking_planner.planning.path_utils import (
    path_metrics,
    resample_path,
)

from parking_planner.control.stanley_controller import (
    StanleyController,
)

from parking_planner.control.speed_controller import (
    SpeedController,
)

from parking_planner.control.adaptive_stanley import (
    AdaptiveStanleyController,
)

from parking_planner.control.speed_profile import (
    CurvatureSpeedPlanner,
)

from parking_planner.simulation.closed_loop import (
    ClosedLoopParkingSimulator,
    SimulationConfig,
)

from parking_planner.vehicle.dynamic_ackermann import (
    DynamicAckermannModel,
)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def create_simulation_config():
    return SimulationConfig(
        dt=0.05,
        forward_speed=0.5,
        reverse_speed=0.4,
        max_acceleration=2.0,
        max_speed=5.0,
        speed_kp=2.5,
        speed_ki=0.0,
        speed_kd=0.0,
        brake_deceleration=1.5,
        stop_buffer=0.04,
        endpoint_tolerance=0.16,
        stop_speed=0.06,
        max_steps_per_segment=1600,
        shift_duration=0.4,
    )


def create_vehicle_model(geometry):
    """Create a fresh dynamic vehicle model for each experiment."""

    return DynamicAckermannModel(
        wheelbase=geometry.wheelbase,
        max_steer=math.radians(35.0),
        max_acceleration=2.0,
        max_speed=5.0,
        steering_time_constant=0.25,
        max_steering_rate=math.radians(60.0),
        acceleration_time_constant=0.25,
        max_jerk=5.0,
    )


def create_controller(geometry, optimized):
    if optimized:
        return AdaptiveStanleyController(
            wheelbase=geometry.wheelbase,
            base_gain=0.5,
            min_gain=0.4,
            max_gain=1.0,
            error_gain=0.45,
            speed_gain=0.12,
            max_steer=math.radians(35.0),
        )

    return StanleyController(
        wheelbase=geometry.wheelbase,
        gain=0.5,
        max_steer=math.radians(35.0),
    )


def create_speed_controller(optimized):
    if optimized:
        return SpeedController(
            kp=2.5,
            ki=0.0,
            kd=0.05,
            max_acceleration=0.8,
            max_speed=0.55,
        )

    return SpeedController(
        kp=2.5,
        ki=0.0,
        kd=0.0,
        max_acceleration=2.0,
        max_speed=5.0,
    )


def calculate_metrics(result, model, controller):
    records = result["states"]
    final_state = result["final_state"]

    goal = scenario_goal

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

    history = getattr(model, "history", [])

    max_jerk = max(
        (abs(item["jerk"]) for item in history),
        default=0.0,
    )

    max_actual_acceleration = max(
        (
            abs(item["acceleration_actual"])
            for item in history
        ),
        default=0.0,
    )

    max_actual_steering = max(
        (
            abs(item["steering_actual"])
            for item in history
        ),
        default=0.0,
    )

    gain_history = getattr(
        controller,
        "gain_history",
        [],
    )

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

        "final_position_error_m": round(
            position_error, 4
        ),

        "final_heading_error_deg": round(
            math.degrees(heading_error), 3
        ),

        "cross_track_rmse_m": round(
            cte_rmse, 4
        ),

        "cross_track_max_m": round(
            max(valid_cte, default=0.0), 4
        ),

        "max_speed_mps": round(
            max(
                (abs(item["speed"]) for item in records),
                default=0.0,
            ), 4
        ),

        "max_command_acceleration_mps2": round(
            max(
                (abs(item["acceleration"]) for item in records),
                default=0.0,
            ), 4
        ),

        "max_actual_acceleration_mps2": round(
            max_actual_acceleration, 4
        ),

        "max_steering_deg": round(
            math.degrees(max_actual_steering), 3
        ),

        "max_jerk_mps3": round(
            max_jerk, 4
        ),

        "adaptive_gain_mean": (
            round(
                sum(gain_history) / len(gain_history),
                4,
            )
            if gain_history else None
        ),

        "segments": result["segments"],
    }


def run_case(
    name,
    planner,
    path,
    scenario,
    config,
    optimized_control=False,
):
    """Run one complete closed-loop parking experiment."""

    geometry = scenario["geometry"]

    model = create_vehicle_model(geometry)

    stanley = create_controller(
        geometry,
        optimized_control,
    )

    speed_controller = create_speed_controller(
        optimized_control,
    )

    speed_planner = None

    if optimized_control:
        speed_planner = CurvatureSpeedPlanner(
            wheelbase=geometry.wheelbase,
            max_forward_speed=0.50,
            max_reverse_speed=0.40,
            max_lateral_accel=0.06,
            max_acceleration=0.30,
            max_deceleration=0.45,
            preview_points=1,
        )

    simulator = ClosedLoopParkingSimulator(
        vehicle_model=model,
        planner=planner,
        stanley=stanley,
        speed_controller=speed_controller,
        config=config,
        speed_planner=speed_planner,
    )

    result = simulator.run(
        scenario["start"],
        path,
    )

    metrics = calculate_metrics(
        result,
        model,
        stanley,
    )

    print(f"\n========== {name} ==========")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    return {
        "result": result,
        "model": model,
        "controller": stanley,
        "metrics": metrics,
    }


def calculate_planning_metrics(planner, raw_path):
    values = path_metrics(raw_path)

    return {
        "planning_status": "SUCCESS",
        "expanded_nodes": planner.last_iterations,
        "raw_path_points": len(raw_path),
        "path_length_m": round(
            values["path_length"], 4
        ),
        "reverse_distance_m": round(
            values["reverse_distance"], 4
        ),
        "gear_switches": values["gear_switches"],
    }


def plot_comparison(scenario, cases, output_path):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 11),
    )

    # 1. Trajectory comparison
    ax = axes[0, 0]

    xmin, ymin, xmax, ymax = scenario["parking_slot"]

    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="green",
            linestyle="--",
            linewidth=2,
            label="Parking bay",
        )
    )

    for obstacle in scenario["parked_vehicles"]:
        x0, y0, x1, y1 = obstacle

        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                color="gray",
                alpha=0.5,
            )
        )

    for name, data in cases.items():

        records = data["result"]["states"]

        x = [scenario["start"].x] + [
            item["x"] for item in records
        ]

        y = [scenario["start"].y] + [
            item["y"] for item in records
        ]

        ax.plot(x, y, linewidth=1.6, label=name)

    ax.scatter(
        scenario["goal"].x,
        scenario["goal"].y,
        marker="*",
        color="red",
        s=130,
        label="Goal",
    )

    ax.set_title("P7 Parking Trajectory Comparison")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Vehicle speed
    ax = axes[0, 1]

    for name, data in cases.items():

        records = data["result"]["states"]

        ax.plot(
            [item["time"] for item in records],
            [item["speed"] for item in records],
            label=name,
        )

    ax.set_title("Vehicle Speed Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Cross-track error
    ax = axes[1, 0]

    for name, data in cases.items():

        records = data["result"]["states"]

        ax.plot(
            [item["time"] for item in records],
            [item["cross_track_error"] for item in records],
            label=name,
        )

    ax.set_title("Cross-track Error Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 4. Steering command
    ax = axes[1, 1]

    for name, data in cases.items():

        records = data["result"]["states"]

        ax.plot(
            [item["time"] for item in records],
            [
                math.degrees(item["steering"])
                for item in records
            ],
            label=name,
        )

    ax.set_title("Steering Command Comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    plt.show()

    plt.close(fig)

    print(f"\nVisualization: {output_path}")


def main():

    global scenario_goal

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenario = build_perpendicular_parking_scenario()

    scenario_goal = scenario["goal"]

    geometry = scenario["geometry"]

    # --------------------------------------------------
    # 1. Baseline Hybrid A*
    # --------------------------------------------------

    baseline_planner = HybridAStarPlanner(
        grid=scenario["grid"],
        geometry=geometry,
        config=scenario["planner_config"],
    )

    baseline_raw_path = baseline_planner.plan(
        scenario["start"],
        scenario["goal"],
    )

    if baseline_raw_path is None:
        raise SystemExit("Baseline Hybrid A* planning failed.")

    baseline_path = resample_path(
        baseline_raw_path,
        spacing=0.1,
    )

    # --------------------------------------------------
    # 2. Heading-aware Hybrid A*
    # --------------------------------------------------

    optimized_config = replace(
        scenario["planner_config"],
        heading_heuristic_weight=0.065,
    )

    optimized_planner = HybridAStarPlanner(
        grid=scenario["grid"],
        geometry=geometry,
        config=optimized_config,
    )

    optimized_raw_path = optimized_planner.plan(
        scenario["start"],
        scenario["goal"],
    )

    if optimized_raw_path is None:
        raise SystemExit("Optimized Hybrid A* planning failed.")

    optimized_path = resample_path(
        optimized_raw_path,
        spacing=0.1,
    )

    planning_metrics = {
        "baseline": calculate_planning_metrics(
            baseline_planner,
            baseline_raw_path,
        ),

        "heading_aware": calculate_planning_metrics(
            optimized_planner,
            optimized_raw_path,
        ),
    }

    config = create_simulation_config()

    # --------------------------------------------------
    # 3. Four ablation experiments
    # --------------------------------------------------

    cases = {}

    # Original planning + original control.
    cases["baseline"] = run_case(
        "BASELINE",
        baseline_planner,
        baseline_path,
        scenario,
        config,
        optimized_control=False,
    )

    # Optimized planning + original control.
    cases["planning_only"] = run_case(
        "PLANNING ONLY",
        optimized_planner,
        optimized_path,
        scenario,
        config,
        optimized_control=False,
    )

    # Original planning + optimized control.
    cases["control_only"] = run_case(
        "CONTROL ONLY",
        baseline_planner,
        baseline_path,
        scenario,
        config,
        optimized_control=True,
    )

    # Optimized planning + optimized control.
    cases["full_p7"] = run_case(
        "FULL P7",
        optimized_planner,
        optimized_path,
        scenario,
        config,
        optimized_control=True,
    )

    # --------------------------------------------------
    # 4. Overall evaluation
    # --------------------------------------------------

    overall_status = (
        "PASS"
        if all(
            item["metrics"]["closed_loop_status"] == "SUCCESS"
            and item["metrics"]["collision_free"]
            for item in cases.values()
        )
        else "NEEDS_TUNING"
    )

    final_metrics = {
        "planning_comparison": planning_metrics,
        "closed_loop_comparison": {
            name: data["metrics"]
            for name, data in cases.items()
        },
        "overall_status": overall_status,
    }

    metrics_path = (
        OUTPUT_DIR
        / "p7_algorithm_optimization_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            final_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 5. Visualization
    # --------------------------------------------------

    plot_path = (
        OUTPUT_DIR
        / "p7_algorithm_optimization.png"
    )

    plot_comparison(
        scenario,
        cases,
        plot_path,
    )

    print("\n========== P7 FINAL RESULTS ==========")

    print(json.dumps(
        final_metrics,
        indent=2,
        ensure_ascii=False,
    ))

    print(f"\nMetrics: {metrics_path}")

    print(f"Overall status: {overall_status}")


if __name__ == "__main__":
    main()
