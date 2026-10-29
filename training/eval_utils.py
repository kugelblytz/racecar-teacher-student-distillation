"""Evaluation helpers shared by teacher training and CLI evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from training.fast_teacher_env import FastPrivilegedRaceCarEnv
from training.teacher_env import PrivilegedRaceCarEnv


def evaluate_teacher(
    model,
    seeds: Iterable[int],
    crash_penalty: float = 25.0,
    reward_scale: float = 1000.0,
    sim: str = "fast",
) -> Dict[str, object]:
    env_cls = FastPrivilegedRaceCarEnv if sim == "fast" else PrivilegedRaceCarEnv
    env = env_cls(
        base_seed=0,
        crash_penalty=crash_penalty,
        reward_scale=reward_scale,
    )

    distances: List[float] = []
    ticks: List[int] = []
    speeds: List[float] = []
    crashes = 0
    rows = []

    try:
        for sim_seed in seeds:
            obs, info = env.reset(options={"sim_seed": int(sim_seed)})
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(np.asarray(action).item())
                obs, _, terminated, truncated, info = env.step(action_int)

            distance = float(info["distance"])
            crashed = bool(info["crashed"])
            distances.append(distance)
            ticks.append(int(info["ticks"]))
            speeds.append(float(info["speed"]))
            crashes += int(crashed)
            rows.append(
                {
                    "seed": int(sim_seed),
                    "distance": distance,
                    "ticks": int(info["ticks"]),
                    "crashed": crashed,
                    "final_speed": float(info["speed"]),
                }
            )
    finally:
        env.close()

    arr = np.asarray(distances, dtype=np.float64)
    n = max(1, len(distances))

    return {
        "episodes": len(distances),
        "mean_distance": float(arr.mean()) if len(arr) else 0.0,
        "median_distance": float(np.median(arr)) if len(arr) else 0.0,
        "p10_distance": float(np.percentile(arr, 10)) if len(arr) else 0.0,
        "min_distance": float(arr.min()) if len(arr) else 0.0,
        "max_distance": float(arr.max()) if len(arr) else 0.0,
        "crash_rate": float(crashes / n),
        "mean_final_speed": float(np.mean(speeds)) if speeds else 0.0,
        "rows": rows,
    }
