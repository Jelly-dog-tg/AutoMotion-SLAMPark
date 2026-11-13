"""P8: Multi-scenario robustness and performance evaluation for full P7 parking stack."""

import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.perpendicular_parking import build_perpendicular_parking_scenario
from parking_planner.planning.hybrid_astar import HybridAStarPlanner, OccupancyGrid
from parking_planner.planning.path_utils import path_metrics, resample_path
from parking_planner.control.adaptive_stanley import AdaptiveStanleyController
from parking_planner.control.speed_controller import SpeedController
from parking_planner.control.speed_profile import CurvatureSpeedPlanner
from parking_planner.simulation.closed_loop import ClosedLoopParkingSimulator, SimulationConfig
from parking_planner.vehicle.ackermann import VehicleState
from parking_planner.vehicle.dynamic_ackermann import DynamicAckermannModel

OUTPUT_DIR = ROOT / "results"


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def make_grid(base, obstacles):
    grid = OccupancyGrid(
        width=base["grid"].width,
        height=base["grid"].height,
        resolution=base["grid"].resolution,
        origin_x=base["grid"].origin_x,
        origin_y=base["grid"].origin_y,
    )
    for obstacle in obstacles:
        grid.add_obstacle_rect(*obstacle)
    return grid


def make_scenarios():
    base = build_perpendicular_parking_scenario()
    scenarios = []

    # T1: small initial-position perturbation; same map and target.
    t1 = dict(base)
    t1["name"] = "T1_initial_position_offset"
    t1["execution_start"] = VehicleState(
        x=base["start"].x + 0.10,
        y=base["start"].y,
        yaw=base["start"].yaw,
        v=0.0,
    )
    scenarios.append(t1)

    # T2: translate the whole parking bay and both parked vehicles.
    shift_x = 0.50
    t2 = dict(base)
    t2["name"] = "T2_moved_bay"
    t2_obstacles = [
        (x0 + shift_x, y0, x1 + shift_x, y1)
        for x0, y0, x1, y1 in base["parked_vehicles"]
    ]
    t2["grid"] = make_grid(base, t2_obstacles)
    t2["parked_vehicles"] = t2_obstacles
    t2["parking_slot"] = (
        base["parking_slot"][0] + shift_x,
        base["parking_slot"][1],
        base["parking_slot"][2] + shift_x,
        base["parking_slot"][3],
    )
    t2["goal"] = VehicleState(
        x=base["goal"].x + shift_x,
        y=base["goal"].y,
        yaw=base["goal"].yaw,
        v=0.0,
    )
    scenarios.append(t2)

    # T3: reduce bay width from 2.6 m to 2.3 m.
    t3 = dict(base)
    t3["name"] = "T3_narrow_bay"
    t3_obstacles = [
        (7.4, 11.5, 9.35, 16.0),
        (11.65, 11.5, 13.6, 16.0),
    ]
    t3["grid"] = make_grid(base, t3_obstacles)
    t3["parked_vehicles"] = t3_obstacles
    t3["parking_slot"] = (9.35, 10.8, 11.65, 17.0)
    scenarios.append(t3)

    # T4: move only the right-side parked vehicle outward by 0.25 m.
    # This changes the obstacle layout and widens the available bay asymmetrically.
    t4 = dict(base)
    t4["name"] = "T4_shifted_obstacle"
    t4_obstacles = [
        base["parked_vehicles"][0],
        (
            base["parked_vehicles"][1][0] + 0.25,
            base["parked_vehicles"][1][1],
            base["parked_vehicles"][1][2] + 0.25,
            base["parked_vehicles"][1][3],
        ),
    ]
    t4["grid"] = make_grid(base, t4_obstacles)
    t4["parked_vehicles"] = t4_obstacles
    t4["parking_slot"] = (9.2, 10.8, 12.05, 17.0)
    t4["goal"] = VehicleState(
        x=10.625,
        y=base["goal"].y,
        yaw=base["goal"].yaw,
        v=0.0,
    )
    scenarios.append(t4)

    # T5: perturb the initial vehicle heading by +5 degrees.
    t5 = dict(base)
    t5["name"] = "T5_initial_heading_offset"
    t5["execution_start"] = VehicleState(
        x=base["start"].x,
        y=base["start"].y,
        yaw=math.radians(5.0),
        v=0.0,
    )
    scenarios.append(t5)

    return scenarios


def make_simulation_config():
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


def run_scenario(scenario):
    geometry = scenario["geometry"]
    planner_config = replace(
        scenario["planner_config"],
        heading_heuristic_weight=0.065,
    )
    planner = HybridAStarPlanner(
        grid=scenario["grid"],
        geometry=geometry,
        config=planner_config,
    )

    try:
        raw_path = planner.plan(scenario["start"], scenario["goal"])
    except Exception as exc:
        return {
            "name": scenario["name"],
            "planning_status": "ERROR",
            "closed_loop_status": "NOT_RUN",
            "error": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }, None, None

    if raw_path is None:
        return {
            "name": scenario["name"],
            "planning_status": "FAILED",
            "closed_loop_status": "NOT_RUN",
            "expanded_nodes": planner.last_iterations,
            "passed": False,
        }, None, None

    reference_path = resample_path(raw_path, spacing=0.1)

    model = DynamicAckermannModel(
        wheelbase=geometry.wheelbase,
        max_steer=math.radians(35.0),
        max_acceleration=2.0,
        max_speed=5.0,
        steering_time_constant=0.25,
        max_steering_rate=math.radians(60.0),
        acceleration_time_constant=0.25,
        max_jerk=5.0,
    )
    controller = AdaptiveStanleyController(
        wheelbase=geometry.wheelbase,
        base_gain=0.5,
        min_gain=0.4,
        max_gain=1.0,
        error_gain=0.45,
        speed_gain=0.12,
        max_steer=math.radians(35.0),
    )
    speed_controller = SpeedController(
        kp=2.5,
        ki=0.0,
        kd=0.05,
        max_acceleration=0.8,
        max_speed=0.55,
    )
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
        stanley=controller,
        speed_controller=speed_controller,
        config=make_simulation_config(),
        speed_planner=speed_planner,
    )

    execution_start = scenario.get("execution_start", scenario["start"])
    try:
        result = simulator.run(execution_start, reference_path)
    except Exception as exc:
        return {
            "name": scenario["name"],
            "planning_status": "SUCCESS",
            "closed_loop_status": "ERROR",
            "expanded_nodes": planner.last_iterations,
            "path_points": len(reference_path),
            "error": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }, reference_path, None

    final_state = result["final_state"]
    goal = scenario["goal"]
    position_error = math.hypot(final_state.x - goal.x, final_state.y - goal.y)
    heading_error_deg = math.degrees(abs(normalize_angle(final_state.yaw - goal.yaw)))

    records = result["states"]
    cte_values = [
        abs(item["cross_track_error"])
        for item in records
        if item["steering"] != 0.0 or item["cross_track_error"] != 0.0
    ]
    history = getattr(model, "history", [])
    gain_history = getattr(controller, "gain_history", [])

    max_speed = max((abs(item["speed"]) for item in records), default=0.0)
    max_acceleration = max(
        (abs(item.get("acceleration_actual", 0.0)) for item in history),
        default=0.0,
    )
    max_jerk = max((abs(item.get("jerk", 0.0)) for item in history), default=0.0)

    execution_start = scenario.get("execution_start", scenario["start"])
    initial_position_offset = math.hypot(
        execution_start.x - scenario["start"].x,
        execution_start.y - scenario["start"].y,
    )
    initial_heading_offset_deg = math.degrees(abs(normalize_angle(
        execution_start.yaw - scenario["start"].yaw
    )))

    position_tolerance = 0.20
    heading_tolerance_deg = 5.0
    cross_track_max_tolerance = 0.25
    cross_track_max = max(cte_values, default=0.0)
    passed = (
        result["completed"]
        and result["collision_free"]
        and position_error <= position_tolerance
        and heading_error_deg <= heading_tolerance_deg
        and cross_track_max <= cross_track_max_tolerance
    )

    metrics = {
        "name": scenario["name"],
        "planning_status": "SUCCESS",
        "expanded_nodes": planner.last_iterations,
        "raw_path_points": len(raw_path),
        "path_points": len(reference_path),
        "path_length_m": round(path_metrics(raw_path)["path_length"], 4),
        "reverse_distance_m": round(path_metrics(raw_path)["reverse_distance"], 4),
        "closed_loop_status": "SUCCESS" if result["completed"] else "INCOMPLETE",
        "collision_free": bool(result["collision_free"]),
        "simulation_steps": len(records),
        "simulation_duration_s": round(result["duration"], 3),
        "gear_switches": result["gear_switches"],
        "initial_position_offset_m": round(initial_position_offset, 4),
        "initial_heading_offset_deg": round(initial_heading_offset_deg, 3),
        "final_position_error_m": round(position_error, 4),
        "final_heading_error_deg": round(heading_error_deg, 3),
        "cross_track_rmse_m": round(
            math.sqrt(sum(value * value for value in cte_values) / max(1, len(cte_values))), 4
        ),
        "cross_track_max_m": round(cross_track_max, 4),
        "max_speed_mps": round(max_speed, 4),
        "max_actual_acceleration_mps2": round(max_acceleration, 4),
        "max_jerk_mps3": round(max_jerk, 4),
        "adaptive_gain_mean": round(sum(gain_history) / len(gain_history), 4) if gain_history else None,
        "position_tolerance_m": position_tolerance,
        "heading_tolerance_deg": heading_tolerance_deg,
        "cross_track_max_tolerance_m": cross_track_max_tolerance,
        "passed": bool(passed),
        "segments": result["segments"],
    }
    return metrics, reference_path, result


def plot_results(items, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    axes = axes.ravel()

    for ax, (scenario, metrics, path, result) in zip(axes, items):
        for x0, y0, x1, y1 in scenario["parked_vehicles"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="gray", alpha=0.65))

        sx0, sy0, sx1, sy1 = scenario["parking_slot"]
        ax.add_patch(Rectangle(
            (sx0, sy0), sx1 - sx0, sy1 - sy0,
            fill=False, edgecolor="seagreen", linestyle="--", linewidth=2,
            label="Parking bay",
        ))
        execution_start = scenario.get("execution_start", scenario["start"])
        ax.scatter(execution_start.x, execution_start.y, c="green", s=35, label="Actual start")
        ax.scatter(scenario["goal"].x, scenario["goal"].y, c="red", marker="*", s=110, label="Goal")

        if path is not None:
            ax.plot([p.x for p in path], [p.y for p in path], "--", color="royalblue", linewidth=1.4, label="Planned")
        if result is not None:
            records = result["states"]
            xs = [execution_start.x] + [r["x"] for r in records]
            ys = [execution_start.y] + [r["y"] for r in records]
            ax.plot(xs, ys, color="darkorange", linewidth=1.8, label="Actual")

        status = "PASS" if metrics.get("passed") else "FAIL"
        ax.set_title(f"{scenario['name']} | {status}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(4.5, 16.5)
        ax.set_ylim(7.0, 18.5)

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.955), ncol=4, fontsize=9,
    )
    fig.suptitle(
        "P8 Multi-scenario Autonomous Parking Robustness Evaluation",
        fontsize=15, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180)
    plt.show()
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = make_scenarios()
    all_metrics = []
    plot_items = []

    for scenario in scenarios:
        print(f"\nRunning {scenario['name']} ...")
        metrics, path, result = run_scenario(scenario)
        all_metrics.append(metrics)
        plot_items.append((scenario, metrics, path, result))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    passed = sum(1 for item in all_metrics if item.get("passed", False))
    overall_status = "PASS" if passed == len(all_metrics) else "PARTIAL_PASS"

    metrics_path = OUTPUT_DIR / "p8_robustness_metrics.json"
    successful = [item for item in all_metrics if item.get("passed")]
    summary = {
        "mean_planning_expanded_nodes": round(
            sum(item.get("expanded_nodes", 0) for item in all_metrics) / max(1, len(all_metrics)), 1
        ),
        "mean_final_position_error_m": round(
            sum(item.get("final_position_error_m", 0.0) for item in successful) / max(1, len(successful)), 4
        ),
        "max_final_heading_error_deg": round(
            max((item.get("final_heading_error_deg", 0.0) for item in successful), default=0.0), 3
        ),
        "mean_cross_track_rmse_m": round(
            sum(item.get("cross_track_rmse_m", 0.0) for item in successful) / max(1, len(successful)), 4
        ),
        "max_cross_track_error_m": round(
            max((item.get("cross_track_max_m", 0.0) for item in successful), default=0.0), 4
        ),
        "mean_simulation_duration_s": round(
            sum(item.get("simulation_duration_s", 0.0) for item in successful) / max(1, len(successful)), 3
        ),
    }
    metrics_path.write_text(
        json.dumps({
            "overall_status": overall_status,
            "scenario_count": len(all_metrics),
            "passed_count": passed,
            "pass_rate_percent": round(100.0 * passed / max(1, len(all_metrics)), 2),
            "summary": summary,
            "scenarios": all_metrics,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = OUTPUT_DIR / "p8_robustness_metrics.csv"
    columns = [
        "name", "planning_status", "closed_loop_status", "passed",
        "expanded_nodes", "path_length_m", "reverse_distance_m",
        "simulation_duration_s", "initial_position_offset_m",
        "initial_heading_offset_deg", "final_position_error_m",
        "final_heading_error_deg", "cross_track_rmse_m", "cross_track_max_m",
        "max_speed_mps", "max_actual_acceleration_mps2", "max_jerk_mps3",
        "adaptive_gain_mean", "collision_free", "gear_switches",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)

    figure_path = OUTPUT_DIR / "p8_robustness_trajectories.png"
    plot_results(plot_items, figure_path)

    print("\n========== P8 ROBUSTNESS RESULTS ==========")
    print(f"Passed: {passed}/{len(all_metrics)}")
    print(f"Pass rate: {100.0 * passed / max(1, len(all_metrics)):.1f}%")
    print(f"Overall status: {overall_status}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Metrics CSV: {csv_path}")
    print(f"Visualization: {figure_path}")


if __name__ == "__main__":
    main()
