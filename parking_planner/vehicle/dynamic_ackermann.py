
"""Dynamic Ackermann vehicle model with actuator constraints."""

from __future__ import annotations

import math

from parking_planner.vehicle.ackermann import (
    AckermannModel,
    VehicleState,
)


class DynamicAckermannModel(AckermannModel):
    """
    Ackermann vehicle model with steering and longitudinal actuator dynamics.

    Features:
    1. First-order steering actuator response.
    2. Steering rate limitation.
    3. First-order acceleration actuator response.
    4. Jerk limitation.
    5. Midpoint integration for vehicle motion.
    """

    def __init__(
        self,
        wheelbase: float = 2.7,
        max_steer: float = math.radians(35.0),
        max_acceleration: float = 2.0,
        max_speed: float = 5.0,
        steering_time_constant: float = 0.25,
        max_steering_rate: float = math.radians(60.0),
        acceleration_time_constant: float = 0.25,
        max_jerk: float = 5.0,
    ):
        super().__init__(
            wheelbase=wheelbase,
            max_steer=max_steer,
            max_acceleration=max_acceleration,
            max_speed=max_speed,
        )

        if steering_time_constant <= 0:
            raise ValueError("steering_time_constant must be positive")

        if acceleration_time_constant <= 0:
            raise ValueError("acceleration_time_constant must be positive")

        if max_steering_rate <= 0 or max_jerk <= 0:
            raise ValueError("actuator rate limits must be positive")

        self.steering_time_constant = steering_time_constant
        self.max_steering_rate = max_steering_rate
        self.acceleration_time_constant = acceleration_time_constant
        self.max_jerk = max_jerk

        self.reset()

    def reset(self):
        """Reset internal actuator states and history."""

        self.actual_steering = 0.0
        self.actual_acceleration = 0.0

        self.time = 0.0
        self.history = []

    @staticmethod
    def _clip(value, lower, upper):
        return max(lower, min(upper, value))

    @staticmethod
    def _first_order_rate_limited(
        current,
        target,
        time_constant,
        max_rate,
        dt,
    ):
        """
        First-order actuator response with a maximum rate constraint.
        """

        alpha = 1.0 - math.exp(-dt / time_constant)

        filtered_target = current + alpha * (target - current)

        max_change = max_rate * dt

        change = DynamicAckermannModel._clip(
            filtered_target - current,
            -max_change,
            max_change,
        )

        return current + change

    def step(
        self,
        state: VehicleState,
        acceleration: float,
        steering: float,
        dt: float = 0.05,
    ) -> VehicleState:

        if dt <= 0:
            raise ValueError("dt must be positive")

        # 1. Limit input commands.
        acceleration_command = self._clip(
            acceleration,
            -self.max_acceleration,
            self.max_acceleration,
        )

        steering_command = self._clip(
            steering,
            -self.max_steer,
            self.max_steer,
        )

        previous_steering = self.actual_steering
        previous_acceleration = self.actual_acceleration

        # 2. Steering actuator dynamics.
        self.actual_steering = self._first_order_rate_limited(
            current=self.actual_steering,
            target=steering_command,
            time_constant=self.steering_time_constant,
            max_rate=self.max_steering_rate,
            dt=dt,
        )

        # 3. Longitudinal actuator dynamics.
        self.actual_acceleration = self._first_order_rate_limited(
            current=self.actual_acceleration,
            target=acceleration_command,
            time_constant=self.acceleration_time_constant,
            max_rate=self.max_jerk,
            dt=dt,
        )

        # 4. Update velocity.
        v_next = state.v + self.actual_acceleration * dt

        v_next = self._clip(
            v_next,
            -self.max_speed,
            self.max_speed,
        )

        # 5. Midpoint integration of the bicycle model.
        v_mid = 0.5 * (state.v + v_next)

        yaw_rate = (
            v_mid / self.wheelbase
            * math.tan(self.actual_steering)
        )

        yaw_mid = state.yaw + 0.5 * yaw_rate * dt

        x_next = (
            state.x + v_mid * math.cos(yaw_mid) * dt
        )

        y_next = (
            state.y + v_mid * math.sin(yaw_mid) * dt
        )

        yaw_next = self.normalize_angle(
            state.yaw + yaw_rate * dt
        )

        # 6. Record actuator response.
        self.time += dt

        steering_rate = (
            self.actual_steering - previous_steering
        ) / dt

        jerk = (
            self.actual_acceleration - previous_acceleration
        ) / dt

        self.history.append({
            "time": self.time,
            "steering_command": steering_command,
            "steering_actual": self.actual_steering,
            "steering_rate": steering_rate,
            "acceleration_command": acceleration_command,
            "acceleration_actual": self.actual_acceleration,
            "jerk": jerk,
        })

        return VehicleState(
            x=x_next,
            y=y_next,
            yaw=yaw_next,
            v=v_next,
        )
