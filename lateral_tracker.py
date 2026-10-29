"""Controller-owned reconstruction of ego lateral position.

This module intentionally depends only on Python. It can be used by the training
environment and by a deployed API controller without access to simulator-private
state.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_EGO_INITIAL_Y = 510.0
DEFAULT_ROAD_TOP = 40.0
DEFAULT_ROAD_BOTTOM = 1160.0
DEFAULT_CAR_HEIGHT = 179.0


def normalize_lateral_y(
    y: float,
    *,
    road_top: float = DEFAULT_ROAD_TOP,
    road_bottom: float = DEFAULT_ROAD_BOTTOM,
    car_height: float = DEFAULT_CAR_HEIGHT,
) -> float:
    """Map safe top-left ego y to roughly [-1, +1].

    -1 is the upper wall collision boundary, +1 the lower boundary, and 0 is
    the midpoint between them. Values beyond the safe region are allowed so the
    encoding remains continuous on a crash transition.
    """
    lower_boundary = float(road_bottom) - float(car_height)
    midpoint = (float(road_top) + lower_boundary) * 0.5
    half_span = (lower_boundary - float(road_top)) * 0.5
    if half_span <= 0.0:
        raise ValueError("invalid road/car geometry")
    return (float(y) - midpoint) / half_span


@dataclass
class LateralPositionTracker:
    """Exact ego-y reconstruction from legal velocity plus issued actions."""

    y: float = DEFAULT_EGO_INITIAL_Y
    vy: float = 0.0
    initial_y: float = DEFAULT_EGO_INITIAL_Y
    road_top: float = DEFAULT_ROAD_TOP
    road_bottom: float = DEFAULT_ROAD_BOTTOM
    car_height: float = DEFAULT_CAR_HEIGHT

    def reset(self) -> None:
        self.y = float(self.initial_y)
        self.vy = 0.0

    def sync_velocity(self, vy: float) -> None:
        """Use the API's legal velocity.y to eliminate any possible vy drift."""
        self.vy = float(vy)

    def advance(self, action: str) -> None:
        """Mirror Car.turn() followed by the ego Car.update() for one tick."""
        if action == "STEER_LEFT":
            self.vy -= 0.1
        elif action == "STEER_RIGHT":
            self.vy += 0.1
        self.y += self.vy

    @property
    def normalized_y(self) -> float:
        return normalize_lateral_y(
            self.y,
            road_top=self.road_top,
            road_bottom=self.road_bottom,
            car_height=self.car_height,
        )
