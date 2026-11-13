"""Execute the P3 parking path with the P4 closed-loop controllers."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import HybridAStarPlanner
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


def main():
    scenario = build_perpendicular_parking_scenario()
    geometry = scenario["geometry"]
    planner = HybridAStarPlanner(
        grid=scenario["grid"],
        geometry=geometry,
        config=scenario["planner_config"],
    )

    raw_path = planner.plan(scenario["start"], scenario["goal"])
    if raw_path is None:
        raise SystemExit("P4 aborted: P3 Hybrid A* failed to produce a path.")
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
    result = simulator.run(scenario["start"], reference_path)

    final_state = result["final_state"]
    goal = scenario["goal"]
    final_position_error = math.hypot(final_state.x - goal.x, final_state.y - goal.y)
    final_heading_error = abs(normalize_angle(final_state.yaw - goal.yaw))
    records = result["states"]
    valid_cte = [abs(item["cross_track_error"]) for item in records if item["steering"] != 0.0 or item["cross_track_error"] != 0.0]
    rms_cte = math.sqrt(sum(value * value for value in valid_cte) / max(1, len(valid_cte)))
    max_cte = max(valid_cte, default=0.0)
    max_steering = max((abs(item["steering"]) for item in records), default=0.0)
    max_speed = max((abs(item["speed"]) for item in records), default=0.0)

    metrics = {
        "planning_status": "SUCCESS",
        "closed_loop_status": "SUCCESS" if result["completed"] else "INCOMPLETE",
        "expanded_nodes": planner.last_iterations,
        "reference_path_points": len(reference_path),
        "path_length_m": round(path_metrics(reference_path)["path_length"], 3),
        "simulation_steps": len(records),
        "simulation_duration_s": round(result["duration"], 3),
        "gear_switches": result["gear_switches"],
        "final_x_m": round(final_state.x, 4),
        "final_y_m": round(final_state.y, 4),
        "final_yaw_deg": round(math.degrees(final_state.yaw), 3),
        "final_speed_mps": round(final_state.v, 4),
        "final_position_error_m": round(final_position_error, 4),
        "final_heading_error_deg": round(math.degrees(final_heading_error), 3),
        "cross_track_rmse_m": round(rms_cte, 4),
        "cross_track_max_m": round(max_cte, 4),
        "max_steering_deg": round(math.degrees(max_steering), 3),
        "max_speed_mps": round(max_speed, 3),
        "collision_free": result["collision_free"],
        "segments": result["segments"],
    }

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "p4_closed_loop_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    slot_xmin, slot_ymin, slot_xmax, slot_ymax = scenario["parking_slot"]
    ax.add_patch(Rectangle(
        (slot_xmin, slot_ymin), slot_xmax - slot_xmin, slot_ymax - slot_ymin,
        facecolor="none", edgecolor="seagreen", linestyle="--", linewidth=2,
        label="Target parking bay",
    ))
    for index, (x0, y0, x1, y1) in enumerate(scenario["parked_vehicles"]):
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor="gray", edgecolor="black", alpha=0.8,
            label="Parked vehicle" if index == 0 else None,
        ))

    for direction, color, label in ((1, "royalblue", "Reference forward"), (-1, "darkorange", "Reference reverse")):
        selected = [point for point in reference_path if point.direction == direction]
        if selected:
            ax.plot([p.x for p in selected], [p.y for p in selected],
                    color=color, linestyle="--", linewidth=1.8, label=label)

    actual_x = [scenario["start"].x] + [item["x"] for item in records]
    actual_y = [scenario["start"].y] + [item["y"] for item in records]
    ax.plot(actual_x, actual_y, color="black", linewidth=2.0, label="Actual closed-loop")
    ax.add_patch(Polygon(vehicle_corners(scenario["start"], geometry), closed=True,
                         fill=False, edgecolor="green", linewidth=2.0, label="Start vehicle"))
    ax.add_patch(Polygon(vehicle_corners(final_state, geometry), closed=True,
                         fill=False, edgecolor="red", linewidth=2.0, label="Final vehicle"))
    ax.scatter(goal.x, goal.y, marker="*", color="red", s=110, label="Goal")
    ax.set(xlim=(5.0, 15.5), ylim=(7.0, 18.0), xlabel="X (m)", ylabel="Y (m)",
           title="Reference vs actual parking trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    time = [item["time"] for item in records]
    speed = [item["speed"] for item in records]
    target_speed = [item["target_speed"] for item in records]
    axes[1].plot(time, speed, label="Actual speed")
    axes[1].plot(time, target_speed, "--", label="Target speed")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Longitudinal speed control")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    steering_deg = [math.degrees(item["steering"]) for item in records]
    cte = [item["cross_track_error"] for item in records]
    axes[2].plot(time, steering_deg, label="Steering (deg)")
    axes[2].plot(time, [math.degrees(item["heading_error"]) for item in records],
                 label="Heading error (deg)", alpha=0.8)
    axes[2].set(xlabel="Time (s)", ylabel="Angle (deg)", title="Steering and heading response")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    fig.suptitle("P4 Closed-loop Perpendicular Reverse Parking", fontsize=15)
    fig.tight_layout()
    figure_path = output_dir / "p4_closed_loop_parking.png"
    fig.savefig(figure_path, dpi=180)
    plt.show()

    print("\n========== P4 CLOSED-LOOP PARKING RESULTS ==========")
    for key, value in metrics.items():
        if key != "segments":
            print(f"{key}: {value}")
    print("Segment details:")
    for index, segment in enumerate(result["segments"], start=1):
        print(f"  Segment {index}: {segment}")
    print(f"Metrics: {output_dir / 'p4_closed_loop_metrics.json'}")
    print(f"Visualization: {figure_path}")

    if not result["completed"] or not result["collision_free"]:
        raise SystemExit("P4 closed-loop execution did not complete collision-free.")


if __name__ == "__main__":
    main()
