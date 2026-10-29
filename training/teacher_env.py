"""Privileged Gymnasium environment for training the PPO teacher.

The teacher sees compact exact simulator state rather than the 16 legal ray
sensors. It directly controls the five primitive challenge actions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Must be set before pygame is imported in each worker process.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

import src.game.core as core


ACTIONS = (
    "NOTHING",
    "ACCELERATE",
    "DECELERATE",
    "STEER_LEFT",
    "STEER_RIGHT",
)

# Observation:
# [ego_vx, ego_y, ego_vy, time_remaining, car_bucket_fraction]
# + 5 lane slots * [occupied, relative_x, relative_vx, opponent_vx]
OBS_SIZE = 5 + core.LANE_COUNT * 4


def _ensure_repo_cwd() -> None:
    """The simulator loads sprites through repo-relative public/assets paths."""
    root = Path(__file__).resolve().parents[1]
    if not (Path.cwd() / "public" / "assets").exists():
        os.chdir(root)


def build_privileged_observation(state: core.GameState) -> np.ndarray:
    """Return a compact normalized exact-state observation.

    No RNG state or future spawn information is exposed. Opponents are encoded
    by lane, which is stable because simulator traffic never changes lanes.
    """
    ego = state.ego
    ego_cx = float(ego.rect.centerx)
    ego_cy = float(ego.rect.centery)
    ego_vx = float(ego.velocity.x)
    ego_vy = float(ego.velocity.y)

    road_mid = (float(state.road.y_start) + float(state.road.y_end)) * 0.5
    road_half = max(1.0, (float(state.road.y_end) - float(state.road.y_start)) * 0.5)

    obs = [
        np.clip(ego_vx / 400.0, 0.0, 2.0),
        np.clip((ego_cy - road_mid) / road_half, -2.0, 2.0),
        np.clip(ego_vy / 20.0, -2.0, 2.0),
        np.clip(1.0 - state.ticks / float(core.MAX_TICKS), 0.0, 1.0),
        np.clip(len(state.car_bucket) / float(core.LANE_COUNT - 1), 0.0, 1.0),
    ]

    slots = [[0.0, 0.0, 0.0, 0.0] for _ in range(core.LANE_COUNT)]
    lane_to_index = {id(lane): i for i, lane in enumerate(state.road.lanes)}

    for car in state.cars:
        if car is ego or car.lane is None:
            continue

        lane_index = lane_to_index.get(id(car.lane))
        if lane_index is None:
            continue

        rel_x = float(car.rect.centerx) - ego_cx
        other_vx = float(car.velocity.x)
        rel_vx = other_vx - ego_vx

        slots[lane_index] = [
            1.0,
            float(np.clip(rel_x / 2000.0, -2.0, 2.0)),
            float(np.clip(rel_vx / 100.0, -4.0, 4.0)),
            float(np.clip(other_vx / 400.0, -1.0, 2.0)),
        ]

    for slot in slots:
        obs.extend(slot)

    return np.asarray(obs, dtype=np.float32)


class PrivilegedRaceCarEnv(gym.Env):
    """Fast headless wrapper around the exact challenge dynamics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_seed: int = 0,
        crash_penalty: float = 25.0,
        reward_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        _ensure_repo_cwd()
        pygame.init()

        self.base_seed = int(base_seed)
        self.crash_penalty = float(crash_penalty)
        self.reward_scale = float(reward_scale)
        self.episode_index = 0
        self.episode_seed: Optional[int] = None

        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-4.0,
            high=4.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

    def _choose_seed(self, options: Optional[Dict[str, Any]]) -> int:
        if options and "sim_seed" in options:
            return int(options["sim_seed"])

        # Different workers get different base_seed values. Gym's RNG gives a
        # deterministic stream when reset(seed=...) is explicitly requested.
        return int(self.np_random.integers(0, 2**31 - 1))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is None and self.episode_index == 0:
            # Make the initial episode deterministic per worker while subsequent
            # resets continue through Gym's seeded RNG stream.
            super().reset(seed=self.base_seed)

        self.episode_seed = self._choose_seed(options)
        self.episode_index += 1

        core.initialize_game_state("", self.episode_seed)
        state = core.STATE

        return build_privileged_observation(state), self._info(state)

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        state = core.STATE
        action_name = ACTIONS[int(action)]

        # Match game_loop ordering, without rendering or ray-sensor updates.
        state.ticks += 1
        core.handle_action(action_name)
        state.latest_action = action_name

        before = float(state.distance)
        state.distance += state.ego.velocity.x
        core.update_cars()
        core.remove_passed_cars()
        core.place_car()

        for car in state.cars:
            if car is not state.ego and core.intersects(state.ego.rect, car.rect):
                state.crashed = True
                break

        if not state.crashed:
            for wall in state.road.walls:
                if core.intersects(state.ego.rect, wall.rect):
                    state.crashed = True
                    break

        delta_distance = float(state.distance) - before
        reward = delta_distance / self.reward_scale
        if state.crashed:
            reward -= self.crash_penalty

        terminated = bool(state.crashed)
        truncated = bool(not state.crashed and state.ticks >= core.MAX_TICKS)

        return (
            build_privileged_observation(state),
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
            "sim_seed": self.episode_seed,
        }

    def close(self) -> None:
        pygame.quit()


def make_env(
    rank: int,
    base_seed: int,
    crash_penalty: float = 25.0,
    reward_scale: float = 1000.0,
):
    """Factory used by Stable-Baselines3 vector environments."""

    def _factory():
        return PrivilegedRaceCarEnv(
            base_seed=base_seed + rank * 100_003,
            crash_penalty=crash_penalty,
            reward_scale=reward_scale,
        )

    return _factory
