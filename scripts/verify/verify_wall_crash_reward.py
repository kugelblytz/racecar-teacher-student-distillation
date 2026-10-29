"""Verify separate car and wall crash penalties in exact and GPU sims."""

from __future__ import annotations

import argparse
import contextlib
import io
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import torch

import src.game.core as core
from src.mathematics.vector import Vector
from training.gpu_sim import (
    ACTIONS,
    EGO_INITIAL_Y,
    EGO_X,
    GpuRaceCarBatch,
)
from training.legal_env import LegalRaceCarEnv


CAR_PENALTY = 50.0
WALL_PENALTY = 100.0
REWARD_SCALE = 1000.0
NOTHING = ACTIONS.index("NOTHING")


def _assert_reward(actual: float, expected: float, label: str) -> None:
    if abs(float(actual) - float(expected)) > 2e-5:
        raise AssertionError(
            f"{label}: reward={actual!r}, expected={expected!r}"
        )


def verify_original() -> None:
    env = LegalRaceCarEnv(
        base_seed=1,
        crash_penalty=CAR_PENALTY,
        wall_crash_penalty=WALL_PENALTY,
        reward_scale=REWARD_SCALE,
        tracked_y=True,
    )
    try:
        # Wall-only crash.
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset(options={"sim_seed": 111})
        state = core.STATE
        state.cars = [state.ego]
        state.ego.y = 40.0
        state.ego.velocity = Vector(10.0, -0.1)
        env.lateral_tracker.y = 40.0
        env.lateral_tracker.vy = -0.1

        _, reward, terminated, _, info = env.step(NOTHING)
        expected = 10.0 / REWARD_SCALE - WALL_PENALTY
        if not terminated or not info["hit_wall"]:
            raise AssertionError("original wall case did not terminate as wall crash")
        if info["hit_car"]:
            raise AssertionError("original wall case unexpectedly hit a car")
        _assert_reward(reward, expected, "original wall")
        print(f"PASS original wall: reward={reward:.6f}")

        # Car-only crash: place one opponent exactly on ego, away from walls.
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset(options={"sim_seed": 222})
        state = core.STATE
        ego = state.ego
        pool = list(state.car_bucket)
        if not pool:
            raise AssertionError("expected opponent in car bucket")
        car = pool[0]
        state.car_bucket = pool[1:]
        car.x = float(ego.x)
        car.y = float(ego.y)
        car.velocity = Vector(float(ego.velocity.x), 0.0)
        car.lane = state.road.lanes[2]
        state.cars = [ego, car]

        _, reward, terminated, _, info = env.step(NOTHING)
        expected = 10.0 / REWARD_SCALE - CAR_PENALTY
        if not terminated or not info["hit_car"]:
            raise AssertionError("original car case did not terminate as car crash")
        if info["hit_wall"]:
            raise AssertionError("original car case unexpectedly hit a wall")
        _assert_reward(reward, expected, "original car")
        print(f"PASS original car:  reward={reward:.6f}")
    finally:
        env.close()


def verify_gpu(device: torch.device) -> None:
    sim = GpuRaceCarBatch(
        1,
        device=device,
        seed=333,
        crash_penalty=CAR_PENALTY,
        wall_crash_penalty=WALL_PENALTY,
        reward_scale=REWARD_SCALE,
        opponent_count=1,
    )
    action = torch.tensor([NOTHING], dtype=torch.int64, device=device)

    # Wall-only crash.
    sim.reset()
    sim.car_active.zero_()
    sim.car_lane.fill_(-1)
    sim.ego_y[0] = 40.0
    sim.ego_vy[0] = -0.1
    sim.tracked_y[0] = 40.0
    sim.tracked_vy[0] = -0.1

    _, reward, done, episode = sim.step(
        action, observation_kind="none", tracked_y=True
    )
    expected = 10.0 / REWARD_SCALE - WALL_PENALTY
    if not bool(done[0].item()) or not bool(episode.hit_wall[0].item()):
        raise AssertionError("GPU wall case did not terminate as wall crash")
    if bool(episode.hit_car[0].item()):
        raise AssertionError("GPU wall case unexpectedly hit a car")
    _assert_reward(float(reward[0].item()), expected, "GPU wall")
    print(f"PASS GPU wall ({device}): reward={float(reward[0]):.6f}")

    # Car-only crash.
    sim.reset()
    sim.car_active.zero_()
    sim.car_lane.fill_(-1)
    slot = 3  # one-opponent curriculum enables the top-of-stack final slot
    sim.car_active[0, slot] = True
    sim.car_lane[0, slot] = 2
    sim.car_x[0, slot] = EGO_X
    sim.car_y[0, slot] = EGO_INITIAL_Y
    sim.car_vx[0, slot] = 10.0

    _, reward, done, episode = sim.step(
        action, observation_kind="none", tracked_y=True
    )
    expected = 10.0 / REWARD_SCALE - CAR_PENALTY
    if not bool(done[0].item()) or not bool(episode.hit_car[0].item()):
        raise AssertionError("GPU car case did not terminate as car crash")
    if bool(episode.hit_wall[0].item()):
        raise AssertionError("GPU car case unexpectedly hit a wall")
    _assert_reward(float(reward[0].item()), expected, "GPU car")
    print(f"PASS GPU car  ({device}): reward={float(reward[0]):.6f}")

    # Backward compatibility: shared teacher callers that omit a wall penalty
    # must retain the old single-penalty behavior.
    fallback = GpuRaceCarBatch(
        1,
        device=device,
        seed=444,
        crash_penalty=123.0,
        wall_crash_penalty=None,
        opponent_count=1,
    )
    if fallback.wall_crash_penalty != 123.0:
        raise AssertionError("wall penalty fallback no longer matches crash_penalty")
    print("PASS backward compatibility: omitted wall penalty falls back to car penalty")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print(
        f"Testing rewards: distance=vx/{REWARD_SCALE:g}, "
        f"car=-{CAR_PENALTY:g}, wall=-{WALL_PENALTY:g}"
    )
    verify_original()
    verify_gpu(device)
    print("\nWALL CRASH REWARD VERIFICATION PASSED")


if __name__ == "__main__":
    main()
