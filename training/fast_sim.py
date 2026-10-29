"""Pygame-free numeric replica of the race-car challenge dynamics.

This module intentionally mirrors the public simulator's physics and RNG call
order while removing rendering, sprites, sensors and per-step pygame.Rect
allocation. It is intended for privileged teacher training only.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
MAX_TICKS = 60 * 60

ROAD_TOP = 40.0
ROAD_BOTTOM = 1160.0
LANE_HEIGHT = (ROAD_BOTTOM - ROAD_TOP) / LANE_COUNT
LANE_CENTERS = tuple(
    ROAD_TOP + (i + 0.5) * LANE_HEIGHT for i in range(LANE_COUNT)
)

# All three source PNGs are 800x397. The public Car loader scales them to a
# target height int(224 * 0.8) == 179 and truncates the width to int(...).
CAR_SOURCE_WIDTH = 800
CAR_SOURCE_HEIGHT = 397
CAR_HEIGHT = int(LANE_HEIGHT * 0.8)  # 179
CAR_WIDTH = int((CAR_SOURCE_WIDTH / CAR_SOURCE_HEIGHT) * CAR_HEIGHT)  # 360

EGO_X = float((SCREEN_WIDTH // 2) - (CAR_WIDTH // 2))
EGO_INITIAL_Y = float(int(LANE_CENTERS[LANE_COUNT // 2] - CAR_HEIGHT / 2))
EGO_INITIAL_VX = 10.0

RETIRE_MIN_X = -1000.0
RETIRE_MAX_X = SCREEN_WIDTH + 1000.0

ACTIONS = (
    "NOTHING",
    "ACCELERATE",
    "DECELERATE",
    "STEER_LEFT",
    "STEER_RIGHT",
)

OBS_SIZE = 5 + LANE_COUNT * 4


def _rect_overlap(
    ax: int,
    ay: int,
    aw: int,
    ah: int,
    bx: int,
    by: int,
    bw: int,
    bh: int,
) -> bool:
    """Equivalent to pygame.Rect(...).colliderect(...)."""
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


class FastRaceCarSim:
    """Exact-state scalar simulator with the public RNG semantics."""

    __slots__ = (
        "rng",
        "seed_value",
        "ticks",
        "distance",
        "crashed",
        "ego_vx",
        "ego_vy",
        "ego_y",
        "car_x",
        "car_y",
        "car_vx",
        "car_lane",
        "car_active",
        "active_order",
        "car_bucket",
    )

    def __init__(self, seed_value: int = 0) -> None:
        self.rng = random.Random()
        self.seed_value = seed_value
        self.car_x = [0.0] * (LANE_COUNT - 1)
        self.car_y = [0.0] * (LANE_COUNT - 1)
        self.car_vx = [0.0] * (LANE_COUNT - 1)
        self.car_lane = [-1] * (LANE_COUNT - 1)
        self.car_active = [False] * (LANE_COUNT - 1)
        self.active_order: List[int] = []
        self.car_bucket: List[int] = []
        self.reset(seed_value)

    def reset(self, seed_value: int) -> None:
        self.seed_value = seed_value
        self.rng = random.Random(seed_value)

        self.ticks = 0
        self.distance = 0.0
        self.crashed = False
        self.ego_vx = EGO_INITIAL_VX
        self.ego_vy = 0.0
        self.ego_y = EGO_INITIAL_Y

        n = LANE_COUNT - 1
        for i in range(n):
            self.car_x[i] = 0.0
            self.car_y[i] = 0.0
            self.car_vx[i] = 8.0
            self.car_lane[i] = -1
            self.car_active[i] = False

        self.active_order = []
        self.car_bucket = list(range(n))

        # initialize_game_state() randomly chooses a blue/red sprite four times.
        # Color does not affect physics, but these calls must be consumed to keep
        # the RNG stream identical to the public simulator.
        for _ in range(n):
            self.rng.choice((0, 1))

    def _handle_action(self, action: str) -> None:
        if action == "ACCELERATE":
            self.ego_vx += 0.1
        elif action == "DECELERATE":
            self.ego_vx = max(0.0, self.ego_vx - 0.1)
        elif action == "STEER_LEFT":
            self.ego_vy -= 0.1
        elif action == "STEER_RIGHT":
            self.ego_vy += 0.1

    def _update_cars(self) -> None:
        self.ego_y += self.ego_vy

        # Match STATE.cars order: ego first, then active opponents in insertion
        # order. Each opponent consumes one rng.random() per tick.
        for idx in self.active_order:
            self.car_x[idx] += self.car_vx[idx] - self.ego_vx
            self.car_vx[idx] += (self.rng.random() - 0.5) / 5.0

    def _remove_passed_cars(self) -> None:
        keep: List[int] = []
        retire: List[int] = []

        for idx in self.active_order:
            x = self.car_x[idx]
            if x < RETIRE_MIN_X or x > RETIRE_MAX_X:
                retire.append(idx)
            else:
                keep.append(idx)

        for idx in retire:
            self.car_bucket.append(idx)
            self.car_lane[idx] = -1
            self.car_active[idx] = False

        self.active_order = keep

    def _place_car(self) -> None:
        # Public code checks len(STATE.cars) > LANE_COUNT. With four opponent
        # slots this is normally false, but preserve the same condition.
        if 1 + len(self.active_order) > LANE_COUNT:
            return

        occupied = {self.car_lane[idx] for idx in self.active_order}
        open_lanes = [i for i in range(LANE_COUNT) if i not in occupied]

        # Important: the public simulator draws lane/x/speed BEFORE checking
        # whether car_bucket is empty, so these RNG calls happen every tick.
        lane = self.rng.choice(open_lanes)
        x_offset = self.rng.choice((-0.5, 1.5))
        speed_coeff = self.rng.random() * 5.0

        if not self.car_bucket:
            return

        idx = self.car_bucket.pop()
        vx = (
            self.ego_vx + speed_coeff
            if x_offset == -0.5
            else self.ego_vx - speed_coeff
        )

        self.car_vx[idx] = vx
        self.car_x[idx] = (SCREEN_WIDTH * x_offset) - (CAR_WIDTH // 2)
        self.car_y[idx] = float(int(LANE_CENTERS[lane] - CAR_HEIGHT / 2))
        self.car_lane[idx] = lane
        self.car_active[idx] = True
        self.active_order.append(idx)

    def _check_collision(self) -> bool:
        ego_x = int(EGO_X)
        ego_y = int(self.ego_y)

        for idx in self.active_order:
            if _rect_overlap(
                ego_x,
                ego_y,
                CAR_WIDTH,
                CAR_HEIGHT,
                int(self.car_x[idx]),
                int(self.car_y[idx]),
                CAR_WIDTH,
                CAR_HEIGHT,
            ):
                return True

        # Pygame wall rectangles are [0,40) and [1160,1200).
        if ego_y < 40:
            return True
        if ego_y + CAR_HEIGHT > 1160:
            return True

        return False

    def step(self, action: str) -> None:
        if self.crashed:
            return

        self.ticks += 1
        self._handle_action(action)
        self.distance += self.ego_vx
        self._update_cars()
        self._remove_passed_cars()
        self._place_car()
        self.crashed = self._check_collision()

    def observation(self) -> np.ndarray:
        """Same privileged observation layout as training.teacher_env."""
        ego_cx = int(EGO_X) + CAR_WIDTH // 2
        ego_cy = int(self.ego_y) + CAR_HEIGHT // 2

        road_mid = (ROAD_TOP + ROAD_BOTTOM) * 0.5
        road_half = (ROAD_BOTTOM - ROAD_TOP) * 0.5

        obs = [
            np.clip(self.ego_vx / 400.0, 0.0, 2.0),
            np.clip((ego_cy - road_mid) / road_half, -2.0, 2.0),
            np.clip(self.ego_vy / 20.0, -2.0, 2.0),
            np.clip(1.0 - self.ticks / float(MAX_TICKS), 0.0, 1.0),
            np.clip(len(self.car_bucket) / float(LANE_COUNT - 1), 0.0, 1.0),
        ]

        slots = [[0.0, 0.0, 0.0, 0.0] for _ in range(LANE_COUNT)]
        for idx in self.active_order:
            lane = self.car_lane[idx]
            car_cx = int(self.car_x[idx]) + CAR_WIDTH // 2
            rel_x = float(car_cx - ego_cx)
            other_vx = self.car_vx[idx]
            rel_vx = other_vx - self.ego_vx
            slots[lane] = [
                1.0,
                float(np.clip(rel_x / 2000.0, -2.0, 2.0)),
                float(np.clip(rel_vx / 100.0, -4.0, 4.0)),
                float(np.clip(other_vx / 400.0, -1.0, 2.0)),
            ]

        for slot in slots:
            obs.extend(slot)

        return np.asarray(obs, dtype=np.float32)

    def info(self) -> Dict[str, object]:
        return {
            "distance": float(self.distance),
            "ticks": int(self.ticks),
            "crashed": bool(self.crashed),
            "speed": float(self.ego_vx),
            "lateral_speed": float(self.ego_vy),
            "sim_seed": self.seed_value,
        }
