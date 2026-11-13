
"""P5: PID speed control and stop-before-shift parking evaluation."""

import json
import math
import sys
from pathlib import Path

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from configs.perpendicular_parking import (
    build_perpendicular_parking_scenario,
)

from parking_planner.planning.hybrid_astar import (
    HybridAStarPlanner,
)

from parking_planner.planning.path_utils import (
    resample_path,
    path_metrics,
)

from parking_planner.simulation.closed_loop import (
    ClosedLoopParkingSimulator,
    SimulationConfig,
)

from parking_planner.vehicle.ackermann import AckermannModel


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def main():

    # -------------------------------------------------
    # 1. Build parking scenario
    # -------------------------------------------------

    scenario = build_perpendicular_parking_scenario()

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
        raise SystemExit("P5 failed: no feasible path.")

    reference_path = resample_path(
        raw_path,
        spacing=0.1,
    )

    # -------------------------------------------------
    # 2. Vehicle model
    # -------------------------------------------------

    vehicle_model = AckermannModel(
        wheelbase=geometry.wheelbase,
        max_steer=math.radians(35.0),
        max_acceleration=2.0,
        max_speed=5.0,
    )

    # -------------------------------------------------
    # 3. PID and gear control configuration
    # -------------------------------------------------

    config = SimulationConfig(
        dt=0.05,

        forward_speed=0.5,
        reverse_speed=0.4,

        max_acceleration=2.0,
        max_speed=5.0,

        speed_kp=2.5,
        speed_ki=0.35,
        speed_kd=0.05,

        brake_deceleration=1.5,
        stop_buffer=0.04,

        endpoint_tolerance=0.16,
        stop_speed=0.06,

        max_steps_per_segment=1600,
        shift_duration=0.4,
    )

    simulator = ClosedLoopParkingSimulator(
        vehicle_model=vehicle_model,
        planner=planner,
        config=config,
    )

    # -------------------------------------------------
    # 4. Execute closed-loop parking
    # -------------------------------------------------

    result = simulator.run(
        scenario["start"],
        reference_path,
    )

    records = result["states"]
    final_state = result["final_state"]
    goal = scenario["goal"]

    # -------------------------------------------------
    # 5. Calculate performance metrics
    # -------------------------------------------------

    position_error = math.hypot(
        final_state.x - goal.x,
        final_state.y - goal.y,
    )

    heading_error = abs(
        normalize_angle(
            final_state.yaw - goal.yaw
        )
    )

    speed_errors = [
        item["target_speed"] - item["speed"]
        for item in records
    ]

    speed_rmse = math.sqrt(
        sum(e * e for e in speed_errors)
        / max(1, len(speed_errors))
    )

    max_speed_error = max(
        (abs(e) for e in speed_errors),
        default=0.0,
    )

    driving_records = [
        item for item in records
        if item["gear_phase"] == "driving"
    ]

    cte_values = [
        item["cross_track_error"]
        for item in driving_records
    ]

    cte_rmse = math.sqrt(
        sum(e * e for e in cte_values)
        / max(1, len(cte_values))
    )

    max_cte = max(
        (abs(e) for e in cte_values),
        default=0.0,
    )

    shift_records = [
        item for item in records
        if item["gear_phase"] == "shifting"
    ]

    max_shift_speed = max(
        (abs(item["speed"]) for item in shift_records),
        default=0.0,
    )

    max_acceleration = max(
        (abs(item["acceleration"]) for item in records),
        default=0.0,
    )

    max_steering = max(
        (abs(item["steering"]) for item in records),
        default=0.0,
    )

    metrics = {
        "planning_status": "SUCCESS",

        "closed_loop_status": (
            "SUCCESS" if result["completed"] else "INCOMPLETE"
        ),

        "controller": "PID + Stanley",

        "pid_kp": config.speed_kp,
        "pid_ki": config.speed_ki,
        "pid_kd": config.speed_kd,

        "path_length_m": round(
            path_metrics(reference_path)["path_length"], 3
        ),

        "simulation_steps": len(records),
        "simulation_duration_s": round(result["duration"], 3),

        "gear_switches": result["gear_switches"],

        "gear_shift_duration_s": round(
            len(shift_records) * config.dt, 3
        ),

        "max_speed_during_shift_mps": round(
            max_shift_speed, 6
        ),

        "speed_rmse_mps": round(speed_rmse, 5),
        "max_speed_error_mps": round(max_speed_error, 5),

        "max_acceleration_mps2": round(
            max_acceleration, 4
        ),

        "max_steering_deg": round(
            math.degrees(max_steering), 3
        ),

        "cross_track_rmse_m": round(cte_rmse, 5),
        "cross_track_max_m": round(max_cte, 5),

        "final_position_error_m": round(
            position_error, 5
        ),

        "final_heading_error_deg": round(
            math.degrees(heading_error), 4
        ),

        "final_speed_mps": round(final_state.v, 5),

        "collision_free": result["collision_free"],
        "completed": result["completed"],
    }

    # -------------------------------------------------
    # 6. Save metrics
    # -------------------------------------------------

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    metrics_path = output_dir / "p5_pid_gear_metrics.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # 7. Visualization
    # -------------------------------------------------

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10),
    )

    # Trajectory
    ax = axes[0, 0]

    slot = scenario["parking_slot"]

    ax.add_patch(
        Rectangle(
            (slot[0], slot[1]),
            slot[2] - slot[0],
            slot[3] - slot[1],
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
                alpha=0.7,
            )
        )

    for direction, color, label in [
        (1, "blue", "Forward reference"),
        (-1, "orange", "Reverse reference"),
    ]:

        points = [
            p for p in reference_path
            if p.direction == direction
        ]

        if points:
            ax.plot(
                [p.x for p in points],
                [p.y for p in points],
                "--",
                color=color,
                label=label,
            )

    actual_x = [scenario["start"].x] + [
        item["x"] for item in records
    ]

    actual_y = [scenario["start"].y] + [
        item["y"] for item in records
    ]

    ax.plot(
        actual_x,
        actual_y,
        color="black",
        linewidth=2,
        label="Actual trajectory",
    )

    ax.scatter(
        goal.x,
        goal.y,
        color="red",
        marker="*",
        s=120,
        label="Goal",
    )

    ax.set_title("PID Closed-loop Parking")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Speed tracking
    time = [item["time"] for item in records]

    actual_speed = [
        item["speed"] for item in records
    ]

    target_speed = [
        item["target_speed"] for item in records
    ]

    axes[0, 1].plot(
        time,
        actual_speed,
        label="Actual speed",
    )

    axes[0, 1].plot(
        time,
        target_speed,
        "--",
        label="Target speed",
    )

    axes[0, 1].axhline(0, color="black", linewidth=0.7)

    axes[0, 1].set_title("PID Speed Tracking")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Acceleration response
    acceleration = [
        item["acceleration"] for item in records
    ]

    axes[1, 0].plot(
        time,
        acceleration,
        label="PID acceleration",
    )

    axes[1, 0].axhline(0, color="black", linewidth=0.7)

    axes[1, 0].set_title("Longitudinal Acceleration")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Acceleration (m/s²)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Cross-track error
    cte = [
        item["cross_track_error"]
        for item in records
    ]

    axes[1, 1].plot(
        time,
        cte,
        label="Cross-track error",
    )

    axes[1, 1].axhline(0, color="black", linewidth=0.7)

    axes[1, 1].set_title("Stanley Tracking Error")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Error (m)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(
        "P5 PID Speed Control and Gear Management",
        fontsize=15,
    )

    fig.tight_layout()

    figure_path = output_dir / "p5_pid_gear_control.png"

    fig.savefig(
        figure_path,
        dpi=180,
    )

    plt.show()

    # -------------------------------------------------
    # 8. Print results
    # -------------------------------------------------

    print("\n========== P5 PID GEAR CONTROL RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nMetrics: {metrics_path}")
    print(f"Visualization: {figure_path}")

    if not result["completed"] or not result["collision_free"]:
        raise SystemExit("P5 FAILED: parking execution incomplete.")

    if max_shift_speed > 1e-9:
        raise SystemExit("P5 FAILED: vehicle moved during gear shifting.")

    print("\nP5 PID GEAR CONTROL: PASSED")


if __name__ == "__main__":
    main()
