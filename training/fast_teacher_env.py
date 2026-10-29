"""Gymnasium wrapper around the Pygame-free numeric simulator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from training.fast_sim import ACTIONS, MAX_TICKS, OBS_SIZE, FastRaceCarSim


class FastPrivilegedRaceCarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_seed: int = 0,
        crash_penalty: float = 25.0,
        reward_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.base_seed = int(base_seed)
        self.crash_penalty = float(crash_penalty)
        self.reward_scale = float(reward_scale)
        self.episode_index = 0
        self.episode_seed: Optional[int] = None
        self.sim = FastRaceCarSim(self.base_seed)

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
        self.sim.reset(self.episode_seed)

        return self.sim.observation(), self.sim.info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        before = self.sim.distance
        self.sim.step(ACTIONS[int(action)])
        delta_distance = self.sim.distance - before

        reward = delta_distance / self.reward_scale
        if self.sim.crashed:
            reward -= self.crash_penalty

        terminated = bool(self.sim.crashed)
        truncated = bool(not self.sim.crashed and self.sim.ticks >= MAX_TICKS)

        return (
            self.sim.observation(),
            float(reward),
            terminated,
            truncated,
            self.sim.info(),
        )


def make_fast_env(
    rank: int,
    base_seed: int,
    crash_penalty: float = 25.0,
    reward_scale: float = 1000.0,
):
    def _factory():
        return FastPrivilegedRaceCarEnv(
            base_seed=base_seed + rank * 100_003,
            crash_penalty=crash_penalty,
            reward_scale=reward_scale,
        )

    return _factory
