"""Verify the fast simulator against the original challenge dynamics.

Run after changing training/fast_sim.py:
    python verify_fast_sim.py --seeds 50 --max-steps 2000

The test feeds identical primitive actions to both environments and compares the
privileged observation, reward, termination, distance, tick, speed and crash flag
at every step.
"""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from training.fast_teacher_env import FastPrivilegedRaceCarEnv
from training.teacher_env import PrivilegedRaceCarEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=12345)
    p.add_argument("--max-steps", type=int, default=2000)
    return p.parse_args()


def action_for_tick(rng: random.Random, tick: int) -> int:
    """Traffic-heavy action mix with bounded steering excursions.

    Mostly accelerates/nothing so episodes often last long enough to exercise
    opponent retirement/respawn, with short balanced steering pulses mixed in.
    """
    cycle = tick % 240
    if 40 <= cycle < 48:
        return 4  # STEER_RIGHT
    if 48 <= cycle < 56:
        return 3  # STEER_LEFT
    if 140 <= cycle < 148:
        return 3
    if 148 <= cycle < 156:
        return 4

    r = rng.random()
    if r < 0.68:
        return 1  # ACCELERATE
    if r < 0.82:
        return 0  # NOTHING
    if r < 0.91:
        return 2  # DECELERATE
    return 0


def main() -> None:
    args = parse_args()

    original = PrivilegedRaceCarEnv(base_seed=0)
    fast = FastPrivilegedRaceCarEnv(base_seed=0)

    total_steps = 0
    completed = 0

    try:
        for offset in range(args.seeds):
            sim_seed = args.seed_start + offset
            action_rng = random.Random(sim_seed ^ 0x5A17C0DE)

            obs_a, info_a = original.reset(options={"sim_seed": sim_seed})
            obs_b, info_b = fast.reset(options={"sim_seed": sim_seed})

            np.testing.assert_allclose(obs_a, obs_b, rtol=0.0, atol=1e-7)

            for tick in range(1, args.max_steps + 1):
                action = action_for_tick(action_rng, tick)

                out_a = original.step(action)
                out_b = fast.step(action)

                obs_a, reward_a, term_a, trunc_a, info_a = out_a
                obs_b, reward_b, term_b, trunc_b, info_b = out_b

                try:
                    np.testing.assert_allclose(obs_a, obs_b, rtol=0.0, atol=1e-6)
                    if abs(reward_a - reward_b) > 1e-9:
                        raise AssertionError(f"reward {reward_a} != {reward_b}")

                    for key in (
                        "distance",
                        "ticks",
                        "crashed",
                        "speed",
                        "lateral_speed",
                    ):
                        a = info_a[key]
                        b = info_b[key]
                        if isinstance(a, float):
                            if abs(a - b) > 1e-9:
                                raise AssertionError(f"{key}: {a} != {b}")
                        elif a != b:
                            raise AssertionError(f"{key}: {a} != {b}")

                    if term_a != term_b or trunc_a != trunc_b:
                        raise AssertionError(
                            f"termination mismatch: original=({term_a},{trunc_a}) "
                            f"fast=({term_b},{trunc_b})"
                        )
                except Exception as exc:
                    raise AssertionError(
                        f"Parity failure seed={sim_seed} tick={tick} action={action}: {exc}"
                    ) from exc

                total_steps += 1
                if term_a or trunc_a:
                    break

            completed += 1
            print(
                f"seed {sim_seed}: OK through tick {info_a['ticks']} "
                f"distance={info_a['distance']:.1f} crashed={info_a['crashed']}"
            )

    finally:
        original.close()
        fast.close()

    print(
        f"\nPARITY OK: {completed} seeds, {total_steps:,} compared steps, "
        "fast simulator matched original observations and outcomes."
    )


if __name__ == "__main__":
    main()
