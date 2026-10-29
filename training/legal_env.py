"""Exact original-simulator environment exposing only challenge-legal observations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import src.game.core as core
from lateral_tracker import LateralPositionTracker
from training.gpu_sim import (
    ACTIONS,
    LEGAL_OBS_SIZE,
    LEGAL_OBS_SIZE_TRACKED_Y,
    SENSOR_NAMES,
    SENSOR_RANGE,
)


def _ensure_repo_cwd() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (Path.cwd() / "public" / "assets").exists():
        os.chdir(root)


def update_sensors(state: core.GameState) -> None:
    for sensor in state.sensors:
        sensor.update()


def build_legal_observation(
    state: core.GameState,
    *,
    tracked_y: Optional[float] = None,
) -> np.ndarray:
    """Encode exactly the information available in RaceCarPredictRequestDto.

    Optional sensor readings are represented by a distance channel (None -> 1)
    plus a hit-mask channel, preserving the distinction from a real 1000 px hit.
    """
    sensor_by_name = {sensor.name: sensor.reading for sensor in state.sensors}

    distances = []
    hits = []
    for name in SENSOR_NAMES:
        reading = sensor_by_name.get(name)
        if reading is None:
            distances.append(1.0)
            hits.append(0.0)
        else:
            distances.append(float(np.clip(float(reading) / SENSOR_RANGE, 0.0, 1.0)))
            hits.append(1.0)

    extras = [
        float(np.clip(float(state.ego.velocity.x) / 400.0, 0.0, 2.0)),
        float(np.clip(float(state.ego.velocity.y) / 20.0, -2.0, 2.0)),
        float(np.clip(1.0 - state.ticks / float(core.MAX_TICKS), 0.0, 1.0)),
        float(np.clip(float(state.distance) / 400_000.0, 0.0, 2.0)),
    ]
    previous_action = [0.0] * len(ACTIONS)
    try:
        previous_action[ACTIONS.index(state.latest_action)] = 1.0
    except ValueError:
        previous_action[0] = 1.0

    values = distances + hits + extras + previous_action
    if tracked_y is not None:
        tracker = LateralPositionTracker(y=float(tracked_y))
        values.append(float(tracker.normalized_y))

    return np.asarray(values, dtype=np.float32)


class LegalRaceCarEnv(gym.Env):
    """Headless exact challenge dynamics with legal sensor/API observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_seed: int = 0,
        crash_penalty: float = 250.0,
        wall_crash_penalty: Optional[float] = None,
        reward_scale: float = 1000.0,
        tracked_y: bool = False,
    ) -> None:
        super().__init__()
        _ensure_repo_cwd()
        pygame.init()

        self.base_seed = int(base_seed)
        self.crash_penalty = float(crash_penalty)
        self.wall_crash_penalty = (
            self.crash_penalty
            if wall_crash_penalty is None
            else float(wall_crash_penalty)
        )
        self.reward_scale = float(reward_scale)
        self.use_tracked_y = bool(tracked_y)
        self.lateral_tracker = LateralPositionTracker()
        self.episode_index = 0
        self.episode_seed: Optional[int] = None

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(
                (LEGAL_OBS_SIZE_TRACKED_Y if self.use_tracked_y else LEGAL_OBS_SIZE),
            ),
            dtype=np.float32,
        )

    def _choose_seed(self, options: Optional[Dict[str, Any]]) -> int:
        if options and "sim_seed" in options:
            return int(options["sim_seed"])
        return int(self.np_random.integers(0, 2**31 - 1))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is None and self.episode_index == 0:
            super().reset(seed=self.base_seed)

        self.episode_seed = self._choose_seed(options)
        self.episode_index += 1
        core.initialize_game_state("", self.episode_seed)
        state = core.STATE

        # The external controller can observe sensor state before choosing its
        # next action. Populate the initial frame as well.
        self.lateral_tracker.reset()
        self.lateral_tracker.sync_velocity(float(state.ego.velocity.y))
        update_sensors(state)
        return build_legal_observation(
            state,
            tracked_y=(
                self.lateral_tracker.y if self.use_tracked_y else None
            ),
        ), self._info(state)

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        state = core.STATE
        action_name = ACTIONS[int(action)]

        state.ticks += 1
        # Sync from the legal API field before integrating the action. This is
        # exactly what a deployed controller can do on each observed frame.
        self.lateral_tracker.sync_velocity(float(state.ego.velocity.y))
        core.handle_action(action_name)
        self.lateral_tracker.advance(action_name)
        state.latest_action = action_name

        before = float(state.distance)
        state.distance += state.ego.velocity.x
        core.update_cars()
        core.remove_passed_cars()
        core.place_car()

        # Match game_loop: sensors update after world dynamics/spawning.
        update_sensors(state)

        hit_car = any(
            car is not state.ego and core.intersects(state.ego.rect, car.rect)
            for car in state.cars
        )
        hit_wall = any(
            core.intersects(state.ego.rect, wall.rect)
            for wall in state.road.walls
        )
        state.crashed = bool(hit_car or hit_wall)

        delta_distance = float(state.distance) - before
        reward = delta_distance / self.reward_scale
        if hit_wall:
            reward -= self.wall_crash_penalty
        elif hit_car:
            reward -= self.crash_penalty

        terminated = bool(state.crashed)
        truncated = bool(not state.crashed and state.ticks >= core.MAX_TICKS)

        return (
            build_legal_observation(
                state,
                tracked_y=(
                    self.lateral_tracker.y if self.use_tracked_y else None
                ),
            ),
            float(reward),
            terminated,
            truncated,
            self._info(state),
        )

    def _info(self, state: core.GameState) -> Dict[str, Any]:
        return {
            "distance": float(state.distance),
            "ticks": int(state.ticks),
            "crashed": bool(state.crashed),
            "speed": float(state.ego.velocity.x),
            "lateral_speed": float(state.ego.velocity.y),
            "hit_car": bool(
                any(
                    car is not state.ego
                    and core.intersects(state.ego.rect, car.rect)
                    for car in state.cars
                )
            ),
            "hit_wall": bool(
                any(
                    core.intersects(state.ego.rect, wall.rect)
                    for wall in state.road.walls
                )
            ),
            "tracked_y": float(self.lateral_tracker.y),
            "tracked_vy": float(self.lateral_tracker.vy),
            "sim_seed": self.episode_seed,
        }

    def close(self) -> None:
        pygame.quit()
