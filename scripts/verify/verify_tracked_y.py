"""Verify controller-tracked lateral position against simulator truth.

Checks:
1. Original Pygame simulator: controller-only tracker matches ego y/vy over
   thousands of real steps and resets.
2. A neighboring car can occlude the wall-facing side sensor without changing
   tracked-y accuracy.
3. GPU simulator's independent tracked tensors match private ego y/vy exactly.
4. 41->42 GRU migration preserves logits, values, and recurrent state to
   floating-point precision.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import random

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

import src.game.core as core
from lateral_tracker import LateralPositionTracker, normalize_lateral_y
from src.mathematics.vector import Vector
from training.gpu_recurrent_ppo import (
    RecurrentActorCritic,
    expand_recurrent_obs_state_dict,
)
from training.gpu_sim import (
    ACTIONS,
    LEGAL_OBS_SIZE,
    LEGAL_OBS_SIZE_TRACKED_Y,
    SENSOR_NAMES,
    GpuRaceCarBatch,
)
from training.legal_env import LegalRaceCarEnv, build_legal_observation, update_sensors


def verify_original(target_steps: int) -> None:
    env = LegalRaceCarEnv(base_seed=777, tracked_y=True)
    rng = random.Random(1234567)

    total = 0
    episodes = 0
    max_y_error = 0.0
    max_vy_error = 0.0
    max_obs_error = 0.0

    try:
        while total < target_steps:
            seed = 900_000 + episodes
            with contextlib.redirect_stdout(io.StringIO()):
                obs, _ = env.reset(options={"sim_seed": seed})
            episodes += 1

            if obs.shape != (LEGAL_OBS_SIZE_TRACKED_Y,):
                raise AssertionError(
                    f"V2 original obs shape={obs.shape}, "
                    f"expected {(LEGAL_OBS_SIZE_TRACKED_Y,)}"
                )

            while total < target_steps:
                # Exercise all five primitives. Episodes ending in either traffic
                # or wall collisions are useful too: accuracy must hold through
                # the terminal transition.
                action = rng.randrange(len(ACTIONS))
                obs, _, terminated, truncated, _ = env.step(action)

                actual_y = float(core.STATE.ego.y)
                actual_vy = float(core.STATE.ego.velocity.y)
                tracked_y = float(env.lateral_tracker.y)
                tracked_vy = float(env.lateral_tracker.vy)

                y_error = abs(tracked_y - actual_y)
                vy_error = abs(tracked_vy - actual_vy)
                obs_expected = np.float32(normalize_lateral_y(actual_y))
                obs_error = abs(float(obs[-1]) - float(obs_expected))

                max_y_error = max(max_y_error, y_error)
                max_vy_error = max(max_vy_error, vy_error)
                max_obs_error = max(max_obs_error, obs_error)
                total += 1

                if y_error != 0.0 or vy_error != 0.0:
                    raise AssertionError(
                        f"original mismatch at step {total}: "
                        f"actual_y={actual_y!r} tracked_y={tracked_y!r} "
                        f"actual_vy={actual_vy!r} tracked_vy={tracked_vy!r}"
                    )
                if obs_error > 1e-7:
                    raise AssertionError(
                        f"tracked-y observation mismatch at step {total}: "
                        f"obs={obs[-1]} expected={obs_expected}"
                    )

                if terminated or truncated:
                    break
    finally:
        env.close()

    print(
        f"PASS original: {total:,} steps across {episodes:,} episodes; "
        f"max_y_error={max_y_error:g} max_vy_error={max_vy_error:g} "
        f"max_obs_error={max_obs_error:g}"
    )


def verify_sensor_occlusion() -> None:
    env = LegalRaceCarEnv(base_seed=778, tracked_y=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset(options={"sim_seed": 12345})
        state = core.STATE
        ego = state.ego

        # Place one opponent directly above ego, separated vertically so there is
        # no collision. The upward/left_side ray now hits that car before the wall.
        pool = list(state.car_bucket)
        if not pool:
            raise AssertionError("expected opponent car bucket after reset")
        car = pool[0]
        state.car_bucket = pool[1:]
        car.x = float(ego.x)
        car.y = 286.0
        car.velocity = Vector(float(ego.velocity.x), 0.0)
        car.lane = state.road.lanes[1]
        state.cars = [ego, car]

        update_sensors(state)
        by_name = {sensor.name: sensor.reading for sensor in state.sensors}
        side_reading = by_name["left_side"]
        if side_reading is None:
            raise AssertionError("expected adjacent car to occlude left_side ray")
        if not float(side_reading) < 400.0:
            raise AssertionError(
                f"left_side did not look car-occluded: reading={side_reading}"
            )

        obs = build_legal_observation(
            state,
            tracked_y=env.lateral_tracker.y,
        )
        expected_y = np.float32(normalize_lateral_y(float(ego.y)))
        if abs(float(obs[-1]) - float(expected_y)) > 1e-7:
            raise AssertionError(
                "tracked-y changed or became inaccurate under sensor occlusion"
            )

        print(
            "PASS occlusion: neighboring car intercepts left_side sensor "
            f"(reading={float(side_reading):.3f}) while tracked-y remains exact"
        )
    finally:
        env.close()


def verify_gpu(device: torch.device, steps: int, envs: int) -> None:
    sim = GpuRaceCarBatch(envs, device=device, seed=24680)
    generator = torch.Generator(device=device)
    generator.manual_seed(13579)

    max_y_error = 0.0
    max_vy_error = 0.0
    for t in range(steps):
        actions = torch.randint(
            0,
            len(ACTIONS),
            (envs,),
            generator=generator,
            device=device,
        )
        sim.step(actions, observation_kind="none", tracked_y=True)

        # Checking every tick intentionally makes this a strict verifier rather
        # than a throughput benchmark.
        y_error = float((sim.tracked_y - sim.ego_y).abs().max().item())
        vy_error = float((sim.tracked_vy - sim.ego_vy).abs().max().item())
        max_y_error = max(max_y_error, y_error)
        max_vy_error = max(max_vy_error, vy_error)

        if y_error != 0.0 or vy_error != 0.0:
            raise AssertionError(
                f"GPU tracker mismatch at tick {t + 1}: "
                f"max_y_error={y_error} max_vy_error={vy_error}"
            )

    obs = sim.legal_observation(tracked_y=True)
    if obs.shape != (envs, LEGAL_OBS_SIZE_TRACKED_Y):
        raise AssertionError(
            f"GPU V2 obs shape={tuple(obs.shape)}, expected "
            f"{(envs, LEGAL_OBS_SIZE_TRACKED_Y)}"
        )

    midpoint = (40.0 + (1160.0 - 179.0)) * 0.5
    half_span = ((1160.0 - 179.0) - 40.0) * 0.5
    expected = ((sim.tracked_y - midpoint) / half_span).to(torch.float32)
    obs_error = float((obs[:, -1] - expected).abs().max().item())
    if obs_error != 0.0:
        raise AssertionError(f"GPU tracked-y obs max error={obs_error}")

    print(
        f"PASS GPU ({device}): {envs:,} envs x {steps:,} steps; "
        f"max_y_error={max_y_error:g} max_vy_error={max_vy_error:g} "
        f"obs_error={obs_error:g}"
    )


def verify_gru_migration() -> None:
    torch.manual_seed(4242)
    old = RecurrentActorCritic(
        LEGAL_OBS_SIZE,
        len(ACTIONS),
        hidden_dim=128,
        safe_init=False,
    )
    migrated_state, old_dim = expand_recurrent_obs_state_dict(
        old.state_dict(),
        LEGAL_OBS_SIZE_TRACKED_Y,
    )
    if old_dim != LEGAL_OBS_SIZE:
        raise AssertionError(f"unexpected source obs dim {old_dim}")

    new = RecurrentActorCritic(
        LEGAL_OBS_SIZE_TRACKED_Y,
        len(ACTIONS),
        hidden_dim=128,
        safe_init=False,
    )
    new.load_state_dict(migrated_state)

    obs41 = torch.randn(1024, LEGAL_OBS_SIZE)
    arbitrary_tracked_y = torch.randn(1024, 1)
    obs42 = torch.cat((obs41, arbitrary_tracked_y), dim=1)
    hidden = torch.randn(1024, 128)

    with torch.no_grad():
        old_logits, old_values, old_h = old.step(obs41, hidden)
        new_logits, new_values, new_h = new.step(obs42, hidden)

    logits_diff = float((old_logits - new_logits).abs().max().item())
    values_diff = float((old_values - new_values).abs().max().item())
    hidden_diff = float((old_h - new_h).abs().max().item())
    tolerance = 1e-6
    if (
        logits_diff > tolerance
        or values_diff > tolerance
        or hidden_diff > tolerance
    ):
        raise AssertionError(
            "41->42 migration changed behavior: "
            f"logits={logits_diff} values={values_diff} hidden={hidden_diff}"
        )

    print(
        "PASS GRU migration: 41->42 zero-column migration preserves "
        f"behavior within {tolerance:g} "
        f"(logits={logits_diff:g}, values={values_diff:g}, hidden={hidden_diff:g})"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--original-steps", type=int, default=10_000)
    p.add_argument("--gpu-steps", type=int, default=500)
    p.add_argument("--gpu-envs", type=int, default=256)
    args = p.parse_args()

    if args.original_steps <= 0 or args.gpu_steps <= 0 or args.gpu_envs <= 0:
        raise ValueError("step/env counts must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print(
        f"V1 obs={LEGAL_OBS_SIZE}, V2 tracked-y obs={LEGAL_OBS_SIZE_TRACKED_Y}"
    )
    verify_original(args.original_steps)
    verify_sensor_occlusion()
    verify_gpu(device, args.gpu_steps, args.gpu_envs)
    verify_gru_migration()
    print("\nTRACKED-Y VERIFICATION PASSED")


if __name__ == "__main__":
    main()
