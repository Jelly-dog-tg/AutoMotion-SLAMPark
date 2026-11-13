
"""Hybrid A* path planner for autonomous parking."""

import heapq
import itertools
import math
from dataclasses import dataclass

import numpy as np

from parking_planner.vehicle.ackermann import VehicleState


@dataclass(frozen=True)
class VehicleGeometry:
    """Vehicle dimensions relative to the rear-axle reference point."""

    length: float = 4.5
    width: float = 1.8
    wheelbase: float = 2.7
    rear_overhang: float = 1.0

    def __post_init__(self):
        if self.length <= 0 or self.width <= 0 or self.wheelbase <= 0:
            raise ValueError("Vehicle dimensions and wheelbase must be positive.")
        if self.rear_overhang < 0 or self.rear_overhang >= self.length:
            raise ValueError("rear_overhang must be in [0, length).")
        if self.front_overhang < 0:
            raise ValueError("Vehicle length must be at least wheelbase + rear_overhang.")

    @property
    def front_extent(self):
        """Distance from rear axle to front bumper."""
        return self.length - self.rear_overhang

    @property
    def front_overhang(self):
        """Distance from front axle to front bumper."""
        return self.length - self.rear_overhang - self.wheelbase


@dataclass(frozen=True)
class PlannerConfig:
    """Hybrid A* configuration."""

    xy_resolution: float = 0.5
    yaw_resolution: float = math.radians(15.0)

    step_size: float = 0.8
    steer_samples: int = 5
    max_steer: float = math.radians(35.0)

    reverse_penalty: float = 2.0
    gear_switch_penalty: float = 1.5
    steer_penalty: float = 0.05
    steer_change_penalty: float = 0.2

    goal_xy_tolerance: float = 0.8
    goal_yaw_tolerance: float = math.radians(15.0)

    max_iterations: int = 60000
    heuristic_weight: float = 1.5

    heading_heuristic_weight: float = 0.0


@dataclass(frozen=True)
class PathPoint:
    """A planned vehicle pose and its driving direction."""

    x: float
    y: float
    yaw: float
    direction: int
    steering: float


@dataclass
class SearchNode:
    """Internal search node."""

    key: tuple
    state: VehicleState
    direction: int
    steering: float
    cost: float
    parent_id: int | None


class OccupancyGrid:
    """2D occupancy grid for collision checking."""

    def __init__(
        self,
        width: float,
        height: float,
        resolution: float = 0.25,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        if width <= 0 or height <= 0 or resolution <= 0:
            raise ValueError("Invalid map dimensions")

        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.cols = math.ceil(width / resolution)
        self.rows = math.ceil(height / resolution)

        self.occupied = np.zeros(
            (self.rows, self.cols),
            dtype=bool,
        )

    def add_obstacle_rect(
        self,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
    ):
        """Add an axis-aligned rectangular obstacle."""

        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid obstacle rectangle")

        ix0 = max(
            0,
            math.floor((xmin - self.origin_x) / self.resolution),
        )
        ix1 = min(
            self.cols,
            math.ceil((xmax - self.origin_x) / self.resolution),
        )

        iy0 = max(
            0,
            math.floor((ymin - self.origin_y) / self.resolution),
        )
        iy1 = min(
            self.rows,
            math.ceil((ymax - self.origin_y) / self.resolution),
        )

        if ix0 < ix1 and iy0 < iy1:
            self.occupied[iy0:iy1, ix0:ix1] = True

    def is_collision(self, state, local_x, local_y):
        """Check whether the transformed vehicle footprint collides."""

        cos_yaw = math.cos(state.yaw)
        sin_yaw = math.sin(state.yaw)

        world_x = (
            state.x
            + cos_yaw * local_x
            - sin_yaw * local_y
        )

        world_y = (
            state.y
            + sin_yaw * local_x
            + cos_yaw * local_y
        )

        ix = np.floor(
            (world_x - self.origin_x) / self.resolution
        ).astype(int)

        iy = np.floor(
            (world_y - self.origin_y) / self.resolution
        ).astype(int)

        # Treat map boundaries as obstacles.
        if (
            np.any(ix < 0)
            or np.any(ix >= self.cols)
            or np.any(iy < 0)
            or np.any(iy >= self.rows)
        ):
            return True

        return bool(np.any(self.occupied[iy, ix]))


class HybridAStarPlanner:
    """Hybrid A* planner using Ackermann vehicle kinematics."""

    def __init__(
        self,
        grid: OccupancyGrid,
        wheelbase: float | None = None,
        geometry: VehicleGeometry | None = None,
        config: PlannerConfig | None = None,
    ):
        self.grid = grid
        self.geometry = geometry or VehicleGeometry()
        self.config = config or PlannerConfig()

        if wheelbase is not None and not math.isclose(
            wheelbase, self.geometry.wheelbase, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "wheelbase must match VehicleGeometry.wheelbase; define it once."
            )
        self.wheelbase = self.geometry.wheelbase

        if (
            self.config.step_size <= 0
            or self.config.xy_resolution <= 0
            or self.config.yaw_resolution <= 0
            or self.config.steer_samples < 2
            or self.config.max_iterations <= 0
            or self.config.max_steer <= 0
            or self.config.reverse_penalty < 1.0
            or self.config.heuristic_weight < 1.0
        ):
            raise ValueError("Invalid planner configuration")

        # Precompute vehicle footprint sample points.
        spacing = grid.resolution * 0.5

        nx = math.ceil(self.geometry.length / spacing) + 1
        ny = math.ceil(self.geometry.width / spacing) + 1

        xs = np.linspace(
            -self.geometry.rear_overhang,
            self.geometry.front_extent,
            nx,
        )

        ys = np.linspace(
            -self.geometry.width / 2,
            self.geometry.width / 2,
            ny,
        )

        local_x, local_y = np.meshgrid(xs, ys)

        self.local_x = local_x.ravel()
        self.local_y = local_y.ravel()

        self.n_yaw_bins = max(
            1,
            round(2 * math.pi / self.config.yaw_resolution),
        )

        self.steer_values = np.linspace(
            -self.config.max_steer,
            self.config.max_steer,
            self.config.steer_samples,
        )

        self.last_iterations = 0

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    def _get_key(self, state, direction, steering_index):
        """Discretize pose, gear and previous steering primitive.

        Previous steering belongs in the state because the transition cost
        penalizes steering changes.
        """

        ix = round(
            (state.x - self.grid.origin_x)
            / self.config.xy_resolution
        )

        iy = round(
            (state.y - self.grid.origin_y)
            / self.config.xy_resolution
        )

        yaw = self.normalize_angle(state.yaw)

        yaw_bin = int(
            round(
                (yaw + math.pi)
                / (2 * math.pi)
                * self.n_yaw_bins
            )
        ) % self.n_yaw_bins

        return ix, iy, yaw_bin, direction, steering_index

    def _propagate(self, state, direction, steering):
        """Simulate one motion primitive."""

        substeps = max(
            1,
            math.ceil(
                self.config.step_size
                / (self.grid.resolution * 0.5)
            ),
        )

        ds = direction * self.config.step_size / substeps

        x = state.x
        y = state.y
        yaw = state.yaw

        for _ in range(substeps):

            x += ds * math.cos(yaw)
            y += ds * math.sin(yaw)

            yaw = self.normalize_angle(
                yaw
                + ds / self.wheelbase * math.tan(steering)
            )

            candidate = VehicleState(
                x=x,
                y=y,
                yaw=yaw,
                v=0.0,
            )

            if self.grid.is_collision(
                candidate,
                self.local_x,
                self.local_y,
            ):
                return None

        return VehicleState(x, y, yaw, 0.0)

    def _goal_reached(self, state, goal):
        distance = math.hypot(
            state.x - goal.x,
            state.y - goal.y,
        )

        yaw_error = abs(
            self.normalize_angle(state.yaw - goal.yaw)
        )

        return (
            distance <= self.config.goal_xy_tolerance
            and yaw_error <= self.config.goal_yaw_tolerance
        )

    def _heuristic(self, state, goal):
        """Position-and-heading heuristic for Hybrid A*."""

        # 1. Euclidean distance
        distance = math.hypot(
            state.x - goal.x,
            state.y - goal.y,
        )

        # 2. Heading difference
        heading_error = abs(
            self.normalize_angle(
                state.yaw - goal.yaw
            )
        )

        # 3. Convert heading difference into an equivalent
        #    maneuvering distance.
        heading_cost = (
            self.config.heading_heuristic_weight
            * self.wheelbase
            * heading_error
        )

        return distance + heading_cost

    def _reconstruct_path(self, nodes, node_id):
        path = []

        while node_id is not None:
            node = nodes[node_id]

            path.append(
                PathPoint(
                    x=node.state.x,
                    y=node.state.y,
                    yaw=node.state.yaw,
                    direction=node.direction,
                    steering=node.steering,
                )
            )

            node_id = node.parent_id

        path.reverse()
        return path

    def plan(self, start: VehicleState, goal: VehicleState):
        """Search a collision-free path from start to goal."""

        if self.grid.is_collision(
            start, self.local_x, self.local_y
        ):
            raise ValueError("Start pose is in collision or outside map")

        if self.grid.is_collision(
            goal, self.local_x, self.local_y
        ):
            raise ValueError("Goal pose is in collision or outside map")

        start_state = VehicleState(
            start.x,
            start.y,
            self.normalize_angle(start.yaw),
            0.0,
        )

        zero_steer_index = int(np.argmin(np.abs(self.steer_values)))
        start_key = self._get_key(start_state, 1, zero_steer_index)

        nodes = [
            SearchNode(
                key=start_key,
                state=start_state,
                direction=1,
                steering=0.0,
                cost=0.0,
                parent_id=None,
            )
        ]

        best_cost = {start_key: 0.0}

        counter = itertools.count()

        open_queue = [
            (
                self.config.heuristic_weight * self._heuristic(start_state, goal),
                next(counter),
                0,
            )
        ]

        self.last_iterations = 0

        while (
            open_queue
            and self.last_iterations < self.config.max_iterations
        ):
            _, _, node_id = heapq.heappop(open_queue)

            node = nodes[node_id]

            # Skip outdated search entries.
            if node.cost > best_cost.get(
                node.key, math.inf
            ) + 1e-9:
                continue

            self.last_iterations += 1

            if self._goal_reached(node.state, goal):
                return self._reconstruct_path(
                    nodes,
                    node_id,
                )

            # Forward and reverse motion.
            for direction in (1, -1):

                # Different steering primitives.
                for steering_index, steering in enumerate(self.steer_values):

                    next_state = self._propagate(
                        node.state,
                        direction,
                        float(steering),
                    )

                    if next_state is None:
                        continue

                    next_key = self._get_key(
                        next_state,
                        direction,
                        steering_index,
                    )

                    steer_norm = (
                        abs(float(steering))
                        / self.config.max_steer
                    )

                    steer_change = (
                        abs(float(steering) - node.steering)
                        / self.config.max_steer
                    )

                    # Base distance cost.
                    new_cost = (
                        node.cost + self.config.step_size
                    )

                    # Penalize reverse driving.
                    if direction < 0:
                        new_cost += (
                            self.config.step_size
                            * (self.config.reverse_penalty - 1.0)
                        )

                    # Penalize gear switching.
                    if direction != node.direction:
                        new_cost += self.config.gear_switch_penalty

                    # Penalize large steering angles and changes.
                    new_cost += (
                        self.config.steer_penalty * steer_norm
                    )

                    new_cost += (
                        self.config.steer_change_penalty
                        * steer_change
                    )

                    if new_cost + 1e-9 >= best_cost.get(
                        next_key, math.inf
                    ):
                        continue

                    best_cost[next_key] = new_cost

                    next_id = len(nodes)

                    nodes.append(
                        SearchNode(
                            key=next_key,
                            state=next_state,
                            direction=direction,
                            steering=float(steering),
                            cost=new_cost,
                            parent_id=node_id,
                        )
                    )

                    priority = (
                        new_cost
                        + self.config.heuristic_weight
                        * self._heuristic(next_state, goal)
                    )

                    heapq.heappush(
                        open_queue,
                        (
                            priority,
                            next(counter),
                            next_id,
                        ),
                    )

        return None
