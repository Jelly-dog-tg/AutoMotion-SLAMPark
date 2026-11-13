
"""P6: Compare Stanley and Pure Pursuit controllers."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.control.pure_pursuit_controller import (
    PurePursuitController,
)

from parking_planner.control.stanley_controller import (
    StanleyController,
)

from parking_planner.planning.hybrid_astar import HybridAStarPlanner
from parking_planner.planning.path_utils import path_metrics, resample_path

from parking_planner.simulation.closed_loop import (
    ClosedLoopParkingSimulator,
    SimulationConfig,
)

from parking_planner.vehicle.ackermann import AckermannModel


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def run_controller(name, controller, scenario, planner, reference_path):

    geometry = scenario["geometry"]

    model = AckermannModel(
        wheelbase=geometry.wheelbase,
        max_steer=math.radians(35.0),
        max_acceleration=2.0,
        max_speed=5.0,
    )

    simulator = ClosedLoopParkingSimulator(
        vehicle_model=model,
        planner=planner,
        stanley=controller,
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

    final = result["final_state"]
    goal = scenario["goal"]

    position_error = math.hypot(
        final.x - goal.x,
        final.y - goal.y,
    )

    heading_error = abs(
        angle_diff(final.yaw, goal.yaw)
    )

    records = result["states"]

    cte = [
        abs(r["cross_track_error"])
        for r in records
        if r["steering"] != 0.0
        or r["cross_track_error"] != 0.0
    ]

    steer = [r["steering"] for r in records]

    steering_tv = sum(
        abs(b - a)
        for a, b in zip(steer[:-1], steer[1:])
    )

    metrics = {
        "controller": name,
        "completed": bool(result["completed"]),
        "collision_free": bool(result["collision_free"]),
        "path_length_m": round(
            path_metrics(reference_path)["path_length"], 3
        ),
        "simulation_duration_s": round(
            result["duration"], 3
        ),
        "simulation_steps": len(records),
        "gear_switches": result["gear_switches"],
        "final_position_error_m": round(position_error, 4),
        "final_heading_error_deg": round(
            math.degrees(heading_error), 3
        ),
        "cross_track_rmse_m": round(
            math.sqrt(
                sum(v * v for v in cte) / max(1, len(cte))
            ),
            4,
        ),
        "cross_track_max_m": round(
            max(cte, default=0.0), 4
        ),
        "max_steering_deg": round(
            math.degrees(
                max((abs(v) for v in steer), default=0.0)
            ),
            3,
        ),
        "steering_total_variation_deg": round(
            math.degrees(steering_tv), 3
        ),
    }

    return metrics, result


def main():

    scenario = build_perpendicular_parking_scenario()
    geometry = scenario["geometry"]

    # Generate one common Hybrid A* reference path.
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
        raise SystemExit("Hybrid A* failed to find a path")

    reference_path = resample_path(
        raw_path,
        spacing=0.1,
    )

    # Two controllers use identical simulation conditions.
    controllers = [
        (
            "Stanley",
            StanleyController(
                wheelbase=geometry.wheelbase,
                gain=0.5,
                softening_speed=0.5,
                max_steer=math.radians(35.0),
            ),
        ),
        (
            "Pure Pursuit",
            PurePursuitController(
                wheelbase=geometry.wheelbase,
                lookahead_distance=0.6,
                max_steer=math.radians(35.0),
            ),
        ),
    ]

    metrics_list = []
    results = {}

    for name, controller in controllers:

        metrics, result = run_controller(
            name,
            controller,
            scenario,
            planner,
            reference_path,
        )

        metrics_list.append(metrics)
        results[name] = result

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    metrics_path = (
        output_dir / "p6_controller_comparison.json"
    )

    metrics_path.write_text(
        json.dumps(metrics_list, indent=2),
        encoding="utf-8",
    )

    # Visualization.
    fig, axes = plt.subplots(
        1, 3, figsize=(19, 6)
    )

    ax = axes[0]

    for i, (x0, y0, x1, y1) in enumerate(
        scenario["parked_vehicles"]
    ):
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                color="gray",
                alpha=0.65,
                label="Parked vehicle" if i == 0 else None,
            )
        )

    x0, y0, x1, y1 = scenario["parking_slot"]

    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="green",
            linestyle="--",
            linewidth=2,
            label="Parking bay",
        )
    )

    ax.plot(
        [p.x for p in reference_path],
        [p.y for p in reference_path],
        "--",
        color="black",
        linewidth=1.2,
        label="Hybrid A* reference",
    )

    for name, color in (
        ("Stanley", "royalblue"),
        ("Pure Pursuit", "darkorange"),
    ):
        records = results[name]["states"]

        xs = [scenario["start"].x] + [
            r["x"] for r in records
        ]

        ys = [scenario["start"].y] + [
            r["y"] for r in records
        ]

        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.8,
            label=f"{name} trajectory",
        )

    ax.scatter(
        scenario["start"].x,
        scenario["start"].y,
        color="green",
        s=45,
        label="Start",
    )

    ax.scatter(
        scenario["goal"].x,
        scenario["goal"].y,
        color="red",
        marker="*",
        s=130,
        label="Goal",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(5.0, 15.5)
    ax.set_ylim(7.0, 18.0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Parking trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    # Steering response and cross-track error.
    for name, color in (
        ("Stanley", "royalblue"),
        ("Pure Pursuit", "darkorange"),
    ):
        records = results[name]["states"]

        times = [r["time"] for r in records]

        steering_deg = [
            math.degrees(r["steering"])
            for r in records
        ]

        cte = [
            r["cross_track_error"]
            for r in records
        ]

        axes[1].plot(
            times, steering_deg,
            color=color, linewidth=1.2, label=name,
        )

        axes[2].plot(
            times, cte,
            color=color, linewidth=1.2, label=name,
        )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Steering (deg)")
    axes[1].set_title("Steering response")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Cross-track error (m)")
    axes[2].set_title("Tracking error")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle(
        "P6 Stanley vs Pure Pursuit Controller Comparison",
        fontsize=14,
    )

    fig.tight_layout()

    figure_path = (
        output_dir / "p6_controller_comparison.png"
    )

    fig.savefig(figure_path, dpi=180)

    plt.show()

    print("\n========== P6 CONTROLLER COMPARISON RESULTS ==========")

    for metrics in metrics_list:
        print(json.dumps(metrics, indent=2))

    print(f"Metrics: {metrics_path}")
    print(f"Visualization: {figure_path}")


if __name__ == "__main__":
    main()
