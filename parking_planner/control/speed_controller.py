
"""PID longitudinal speed controller for autonomous parking."""

from __future__ import annotations


class SpeedController:
    """PID speed controller with acceleration saturation and anti-windup."""

    def __init__(
        self,
        kp: float = 2.5,
        ki: float = 0.35,
        kd: float = 0.05,
        max_acceleration: float = 2.0,
        max_speed: float = 5.0,
    ):
        if kp <= 0:
            raise ValueError("kp must be positive")
        if ki < 0 or kd < 0:
            raise ValueError("ki and kd must be non-negative")
        if max_acceleration <= 0 or max_speed <= 0:
            raise ValueError("limits must be positive")

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.max_acceleration = max_acceleration
        self.max_speed = max_speed

        self.integral = 0.0
        self.previous_measurement = None

        self.last_error = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def reset(self):
        """Reset controller memory."""
        self.integral = 0.0
        self.previous_measurement = None

        self.last_error = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def compute(
        self,
        target_speed: float,
        current_speed: float,
        dt: float = 0.05,
    ) -> float:
        """Return signed acceleration command in m/s^2."""

        if dt <= 0:
            raise ValueError("dt must be positive")

        target_speed = max(
            -self.max_speed,
            min(self.max_speed, target_speed),
        )

        error = target_speed - current_speed

        # Proportional term
        p_term = self.kp * error

        # Derivative on measurement avoids derivative kick.
        if self.previous_measurement is None:
            measurement_rate = 0.0
        else:
            measurement_rate = (
                current_speed - self.previous_measurement
            ) / dt

        d_term = -self.kd * measurement_rate

        # Candidate integral
        candidate_integral = self.integral + error * dt

        candidate_output = (
            p_term
            + self.ki * candidate_integral
            + d_term
        )

        # Conditional integration anti-windup
        pushing_high = (
            candidate_output > self.max_acceleration
            and error > 0
        )

        pushing_low = (
            candidate_output < -self.max_acceleration
            and error < 0
        )

        if not (pushing_high or pushing_low):
            self.integral = candidate_integral

        i_term = self.ki * self.integral

        output = p_term + i_term + d_term

        # Acceleration saturation
        output = max(
            -self.max_acceleration,
            min(self.max_acceleration, output),
        )

        self.previous_measurement = current_speed

        self.last_error = error
        self.last_p = p_term
        self.last_i = i_term
        self.last_d = d_term

        return output
