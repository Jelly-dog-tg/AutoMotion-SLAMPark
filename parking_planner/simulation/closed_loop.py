"""Closed-loop execution of a direction-labelled parking reference path."""

from __future__ import annotations

import math
from dataclasses import dataclass

from parking_planner.control import GearManager, SpeedController, StanleyController
from parking_planner.vehicle.ackermann import AckermannModel, VehicleState


@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.05
    forward_speed: float = 0.5
    reverse_speed: float = 0.4
    max_acceleration: float = 2.0
    max_speed: float = 5.0
    speed_kp: float = 2.5
    speed_ki: float = 0.0
    speed_kd: float = 0.0
    brake_deceleration: float = 1.5
    stop_buffer: float = 0.04
    endpoint_tolerance: float = 0.16
    stop_speed: float = 0.06
    max_steps_per_segment: int = 1600
    shift_duration: float = 0.4


class ClosedLoopParkingSimulator:
    """Run a Stanley + speed + gear controller over a planned path."""

    def __init__(
        self,
        vehicle_model: AckermannModel,
        planner,
        stanley: StanleyController | None = None,
        speed_controller: SpeedController | None = None,
        gear_manager: GearManager | None = None,
        config: SimulationConfig | None = None,
        speed_planner=None,
    ):
        self.vehicle_model = vehicle_model
        self.planner = planner
        self.config = config or SimulationConfig()

        self.stanley = stanley or StanleyController(
            wheelbase=vehicle_model.wheelbase,
            max_steer=vehicle_model.max_steer,
        )

        self.speed_controller = speed_controller or SpeedController(
            kp=self.config.speed_kp,
            ki=self.config.speed_ki,
            kd=self.config.speed_kd,
            max_acceleration=self.config.max_acceleration,
            max_speed=self.config.max_speed,
        )

        self.gear_manager = gear_manager

        # P7: Curvature-constrained speed planner
        self.speed_planner = speed_planner
        self.vehicle_model = vehicle_model
        self.planner = planner
        self.config = config or SimulationConfig()
        self.stanley = stanley or StanleyController(
            wheelbase=vehicle_model.wheelbase,
            max_steer=vehicle_model.max_steer,
        )
        self.speed_controller = speed_controller or SpeedController(
            kp=self.config.speed_kp,
            ki=self.config.speed_ki,
            kd=self.config.speed_kd,
            max_acceleration=self.config.max_acceleration,
            max_speed=self.config.max_speed,
        )
        self.gear_manager = gear_manager

    @staticmethod
    def split_path_by_direction(path):
        if not path:
            raise ValueError("planned path must not be empty")
        segments = []
        start = 0
        direction = path[0].direction
        for index in range(1, len(path)):
            if path[index].direction != direction:
                segments.append((direction, path[start:index]))
                start = index
                direction = path[index].direction
        segments.append((direction, path[start:]))
        return [(direction, segment) for direction, segment in segments if segment]

    @staticmethod
    def _remaining_distance(path, index):
        if index >= len(path) - 1:
            return math.hypot(path[-1].x - path[index].x, path[-1].y - path[index].y)
        return sum(
            math.hypot(b.x - a.x, b.y - a.y)
            for a, b in zip(path[index:-1], path[index + 1 :])
        )

    def _is_collision(self, state):
        return self.planner.grid.is_collision(
            state,
            self.planner.local_x,
            self.planner.local_y,
        )

    def _make_record(self, time_s, state, direction, output, acceleration, target_speed):
        return {
            "time": time_s,
            "x": state.x,
            "y": state.y,
            "yaw": state.yaw,
            "speed": state.v,
            "direction": direction,
            "steering": output.steering if output is not None else 0.0,
            "cross_track_error": output.cross_track_error if output is not None else 0.0,
            "heading_error": output.heading_error if output is not None else 0.0,
            "acceleration": acceleration,
            "target_speed": target_speed,
            "gear_phase": self.gear_manager.phase,
        }

    
    def run(self, start: VehicleState, path):
        segments = self.split_path_by_direction(path)

        if not segments:
            raise ValueError("Cannot simulate an empty path")

        if self.gear_manager is None:
            self.gear_manager = GearManager(
                initial_direction=segments[0][0],
                stop_speed=self.config.stop_speed,
                shift_duration=self.config.shift_duration,
            )

        state = VehicleState(start.x, start.y, start.yaw, 0.0)

        time_s = 0.0
        records = []
        gear_switches = 0
        collision_free = not self._is_collision(state)
        segment_summaries = []

        for segment_number, (direction, segment) in enumerate(segments):

            # ==================================================
            # 1. Gear switching
            # ==================================================

            if segment_number > 0:
                self.gear_manager.request(direction)

                self.speed_controller.reset()

                shift_steps = 0

                while not self.gear_manager.is_ready:

                    if self.gear_manager.phase == "braking":

                        acceleration = self.speed_controller.compute(
                            0.0,
                            state.v,
                            dt=self.config.dt,
                        )

                        state = self.vehicle_model.step(
                            state,
                            acceleration=acceleration,
                            steering=0.0,
                            dt=self.config.dt,
                        )

                    else:
                        acceleration = 0.0

                        state = VehicleState(
                            state.x,
                            state.y,
                            state.yaw,
                            0.0,
                        )

                    time_s += self.config.dt

                    self.gear_manager.update(
                        state.v,
                        self.config.dt,
                    )

                    if self.gear_manager.phase == "shifting":

                        state = VehicleState(
                            state.x,
                            state.y,
                            state.yaw,
                            0.0,
                        )

                        self.speed_controller.reset()

                    records.append(
                        self._make_record(
                            time_s,
                            state,
                            self.gear_manager.current_direction,
                            None,
                            acceleration,
                            0.0,
                        )
                    )

                    collision_free = (
                        collision_free
                        and not self._is_collision(state)
                    )

                    shift_steps += 1

                    if shift_steps > 200:
                        raise RuntimeError(
                            "gear change did not complete"
                        )

                self.speed_controller.reset()
                gear_switches += 1

            if self.gear_manager.current_direction != direction:
                raise RuntimeError(
                    "gear manager direction does not match path segment"
                )

            # ==================================================
            # 2. P7: Generate curvature-constrained speed profile
            # ==================================================

            speed_profile = None

            if self.speed_planner is not None:

                speed_profile = self.speed_planner.generate(
                    segment,
                    direction,
                )

                if len(speed_profile) != len(segment):
                    raise RuntimeError(
                        "Speed profile length does not match path segment"
                    )

            # ==================================================
            # 3. Initialize segment tracking
            # ==================================================

            last_index = 0
            segment_start_time = time_s

            reached_endpoint = False

            max_abs_cte = 0.0
            cte_samples = []

            max_abs_steer = 0.0
            steps_in_segment = 0

            # ==================================================
            # 4. Closed-loop control
            # ==================================================

            for _ in range(self.config.max_steps_per_segment):

                steps_in_segment += 1

                # Stanley lateral controller
                output = self.stanley.compute(
                    state=state,
                    path=segment,
                    direction=direction,
                    last_index=last_index,
                )

                last_index = max(
                    last_index,
                    output.target_index,
                )

                # --------------------------------------------------
                # Calculate remaining distance
                # --------------------------------------------------

                endpoint = segment[-1]

                endpoint_distance = math.hypot(
                    endpoint.x - state.x,
                    endpoint.y - state.y,
                )

                remaining = self._remaining_distance(
                    segment,
                    last_index,
                )

                braking_distance = max(
                    0.0,
                    remaining - self.config.stop_buffer,
                )

                braking_speed = math.sqrt(
                    2.0
                    * self.config.brake_deceleration
                    * braking_distance
                )

                # --------------------------------------------------
                # P7: Curvature-aware target speed
                # --------------------------------------------------

                if speed_profile is None:

                    # Original P6 fixed-speed strategy
                    target_magnitude = min(
                        (
                            self.config.forward_speed
                            if direction > 0
                            else self.config.reverse_speed
                        ),
                        braking_speed,
                    )

                else:

                    # P7 curvature-constrained speed strategy
                    profile_speed = self.speed_planner.target_speed(
                        speed_profile,
                        last_index,
                    )

                    target_magnitude = min(
                        profile_speed,
                        braking_speed,
                    )

                # Endpoint stopping condition
                if (
                    remaining <= self.config.stop_buffer
                    or endpoint_distance <= self.config.endpoint_tolerance
                ):
                    target_magnitude = 0.0

                # Signed target speed
                target_speed = direction * target_magnitude

                # --------------------------------------------------
                # PID longitudinal controller
                # --------------------------------------------------

                acceleration = self.speed_controller.compute(
                    target_speed,
                    state.v,
                    dt=self.config.dt,
                )

                # --------------------------------------------------
                # Vehicle dynamics
                # --------------------------------------------------

                state = self.vehicle_model.step(
                    state,
                    acceleration=acceleration,
                    steering=output.steering,
                    dt=self.config.dt,
                )

                time_s += self.config.dt

                # --------------------------------------------------
                # Record simulation state
                # --------------------------------------------------

                records.append(
                    self._make_record(
                        time_s,
                        state,
                        direction,
                        output,
                        acceleration,
                        target_speed,
                    )
                )

                # --------------------------------------------------
                # Collision detection
                # --------------------------------------------------

                collision_free = (
                    collision_free
                    and not self._is_collision(state)
                )

                # --------------------------------------------------
                # Tracking metrics
                # --------------------------------------------------

                max_abs_cte = max(
                    max_abs_cte,
                    abs(output.cross_track_error),
                )

                cte_samples.append(
                    output.cross_track_error
                )

                max_abs_steer = max(
                    max_abs_steer,
                    abs(output.steering),
                )

                # --------------------------------------------------
                # Endpoint detection
                # --------------------------------------------------

                endpoint_distance_after = math.hypot(
                    endpoint.x - state.x,
                    endpoint.y - state.y,
                )

                if (
                    endpoint_distance_after
                    <= self.config.endpoint_tolerance
                    and abs(state.v) <= self.config.stop_speed
                ):

                    state = VehicleState(
                        state.x,
                        state.y,
                        state.yaw,
                        0.0,
                    )

                    reached_endpoint = True
                    break

                if not collision_free:
                    break

            # ==================================================
            # 5. Segment summary
            # ==================================================

            segment_summaries.append({

                "direction": direction,

                "steps": steps_in_segment,

                "duration": time_s - segment_start_time,

                "reached_endpoint": reached_endpoint,

                "max_abs_cross_track_error": max_abs_cte,

                "rms_cross_track_error": math.sqrt(
                    sum(
                        value * value
                        for value in cte_samples
                    ) / max(1, len(cte_samples))
                ),

                "max_abs_steering_deg": math.degrees(
                    max_abs_steer
                ),

                "final_endpoint_distance": math.hypot(
                    segment[-1].x - state.x,
                    segment[-1].y - state.y,
                ),

                "speed_profile_enabled": (
                    speed_profile is not None
                ),

                "speed_profile_max_mps": (
                    max(speed_profile)
                    if speed_profile else None
                ),

            })

            if not reached_endpoint:
                break

            if not collision_free:
                break

        # ==================================================
        # 6. Final simulation results
        # ==================================================

        return {

            "states": records,

            "final_state": state,

            "duration": time_s,

            "gear_switches": gear_switches,

            "collision_free": collision_free,

            "segments": segment_summaries,

            "completed": (
                len(segment_summaries) == len(segments)
                and all(
                    item["reached_endpoint"]
                    for item in segment_summaries
                )
                and collision_free
            ),
        }
