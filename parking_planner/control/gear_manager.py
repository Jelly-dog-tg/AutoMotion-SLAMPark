"""Explicit stop-before-shift state machine for forward/reverse changes."""

from __future__ import annotations


class GearManager:
    """Manage forward/reverse gear changes with a full stop and dwell time."""

    def __init__(
        self,
        initial_direction: int = 1,
        stop_speed: float = 0.05,
        shift_duration: float = 0.4,
    ):
        if initial_direction not in (-1, 1):
            raise ValueError("initial_direction must be +1 or -1")
        if stop_speed <= 0 or shift_duration < 0:
            raise ValueError("invalid gear-manager timing parameters")

        self.current_direction = initial_direction
        self.pending_direction: int | None = None
        self.stop_speed = stop_speed
        self.shift_duration = shift_duration
        self.phase = "driving"
        self.shift_elapsed = 0.0

    def request(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        if direction == self.current_direction and self.phase == "driving":
            return
        if self.phase == "driving":
            self.pending_direction = direction
            self.phase = "braking"
            self.shift_elapsed = 0.0
        elif direction != self.pending_direction:
            raise RuntimeError("cannot request another gear while a shift is pending")

    def update(self, speed: float, dt: float) -> bool:
        """Advance the state machine; return True when a shift completes."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        if self.phase == "braking":
            if abs(speed) <= self.stop_speed:
                self.phase = "shifting"
                self.shift_elapsed = 0.0
        elif self.phase == "shifting":
            self.shift_elapsed += dt
            if self.shift_elapsed + 1e-12 >= self.shift_duration:
                self.current_direction = int(self.pending_direction)
                self.pending_direction = None
                self.phase = "driving"
                self.shift_elapsed = 0.0
                return True
        return False

    @property
    def is_ready(self) -> bool:
        return self.phase == "driving"
