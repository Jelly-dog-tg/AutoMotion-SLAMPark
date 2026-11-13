"""Deterministic perpendicular reverse-parking scenario."""

import math

from parking_planner.vehicle.ackermann import VehicleState
from parking_planner.planning.hybrid_astar import (
    OccupancyGrid,
    PlannerConfig,
    VehicleGeometry,
)


def build_perpendicular_parking_scenario():
    """Create a parking aisle with two neighboring parked vehicles."""

    grid = OccupancyGrid(
        width=22.0,
        height=22.0,
        resolution=0.25,
    )

    # Neighboring parked cars; the gap between them is the target bay.
    parked_vehicles = [
        (7.4, 11.5, 9.2, 16.0),
        (11.8, 11.5, 13.6, 16.0),
    ]
    for obstacle in parked_vehicles:
        grid.add_obstacle_rect(*obstacle)

    geometry = VehicleGeometry(
        length=4.5,
        width=1.8,
        wheelbase=2.7,
        rear_overhang=1.0,
    )

    start = VehicleState(
        x=6.64,
        y=10.0,
        yaw=0.0,
        v=0.0,
    )

    # Rear-axle target pose; the vehicle faces out toward the driving aisle.
    goal = VehicleState(
        x=10.5,
        y=15.3,
        yaw=-math.pi / 2,
        v=0.0,
    )

    planner_config = PlannerConfig(
        xy_resolution=0.4,
        yaw_resolution=math.radians(15.0),
        step_size=0.8,
        steer_samples=5,
        max_steer=math.radians(35.0),
        reverse_penalty=1.5,
        gear_switch_penalty=1.5,
        steer_penalty=0.05,
        steer_change_penalty=0.2,
        goal_xy_tolerance=0.2,
        goal_yaw_tolerance=math.radians(5.0),
        max_iterations=60000,
        heuristic_weight=1.5,
    )

    parking_slot = (9.2, 10.8, 11.8, 17.0)

    return {
        "name": "perpendicular_reverse_parking",
        "grid": grid,
        "geometry": geometry,
        "start": start,
        "goal": goal,
        "planner_config": planner_config,
        "parked_vehicles": parked_vehicles,
        "parking_slot": parking_slot,
    }
