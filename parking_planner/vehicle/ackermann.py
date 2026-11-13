
"""Ackermann steering vehicle kinematic model."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VehicleState:
    """Vehicle state at the rear axle center."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0


class AckermannModel:
    """Kinematic bicycle model for an Ackermann-steered vehicle."""

    def __init__(
        self,
        wheelbase: float = 2.7,
        max_steer: float = math.radians(35.0),
        max_acceleration: float = 2.0,
        max_speed: float = 5.0,
    ):
        if wheelbase <= 0:
            raise ValueError("wheelbase must be positive")
        if not 0 < max_steer < math.pi / 2:
            raise ValueError("max_steer must be between 0 and 90 degrees")
        if max_acceleration <= 0 or max_speed <= 0:
            raise ValueError("acceleration and speed limits must be positive")

        self.wheelbase = wheelbase
        self.max_steer = max_steer
        self.max_acceleration = max_acceleration
        self.max_speed = max_speed

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def step(
        self,
        state: VehicleState,
        acceleration: float,
        steering: float,
        dt: float = 0.1,
    ) -> VehicleState:
        """Advance vehicle state by one time step."""

        if dt <= 0:
            raise ValueError("dt must be positive")

        # Limit control inputs
        acceleration = max(
            -self.max_acceleration,
            min(self.max_acceleration, acceleration),
        )

        steering = max(
            -self.max_steer,
            min(self.max_steer, steering),
        )

        # Kinematic bicycle model
        x_next = (
            state.x
            + state.v * math.cos(state.yaw) * dt
        )

        y_next = (
            state.y
            + state.v * math.sin(state.yaw) * dt
        )

        yaw_rate = (
            state.v / self.wheelbase
            * math.tan(steering)
        )

        yaw_next = self.normalize_angle(
            state.yaw + yaw_rate * dt
        )

        v_next = state.v + acceleration * dt

        v_next = max(
            -self.max_speed,
            min(self.max_speed, v_next),
        )

        return VehicleState(
            x=x_next,
            y=y_next,
            yaw=yaw_next,
            v=v_next,
        )
