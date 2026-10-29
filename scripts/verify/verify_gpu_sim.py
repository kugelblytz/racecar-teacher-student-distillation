"""Validate CUDA simulator fidelity against the original headless simulator.

The CUDA trainer intentionally uses torch RNG instead of Python random.Random,
so traffic samples are not expected to be bit-for-bit identical for the same
seed. This verifier therefore combines:

1. exact deterministic parity checks for ego dynamics and privileged encoding,
2. exact pygame collision-boundary checks,
3. statistical parity checks for spawn distributions.

Run before trusting a long GPU training job:
    python verify_gpu_sim.py --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from typing import Dict, Tuple

import numpy as np
import torch

import src.game.core as core
from training.gpu_sim import (
    ACTIONS,
    CAR_HEIGHT,
    CAR_WIDTH,
    EGO_X,
    GpuRaceCarBatch,
    LANE_COUNT,
)
from training.teacher_env import PrivilegedRaceCarEnv, build_privileged_observation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--original-samples", type=int, default=1000)
    p.add_argument("--gpu-samples", type=int, default=131072)
    p.add_argument("--seed-start", type=int, default=30000)
    return p.parse_args()


def quiet_reset(env: PrivilegedRaceCarEnv, sim_seed: int):
    # Upstream randomizer.seed() prints on every reset. Suppress that noise in
    # this validation utility without changing challenge code.
    with contextlib.redirect_stdout(io.StringIO()):
        return env.reset(options={"sim_seed": int(sim_seed)})


def assert_close(name: str, a, b, atol: float = 1e-6) -> None:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if not np.allclose(aa, bb, rtol=0.0, atol=atol):
        diff = float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
        raise AssertionError(f"{name}: max abs diff {diff} > {atol}\n{aa}\n!=\n{bb}")


def exact_ego_dynamics(device: torch.device) -> None:
    env = PrivilegedRaceCarEnv(base_seed=0)
    gpu = GpuRaceCarBatch(1, device=device, seed=123)
    quiet_reset(env, 123456)
    gpu.reset()

    # Balanced steering pulses keep us away from walls and traffic cannot reach
    # the ego during this short horizon. Traffic RNG may differ; ego dynamics do
    # not depend on it.
    actions = (
        [1] * 12
        + [4] * 5
        + [3] * 5
        + [0] * 4
        + [2] * 7
        + [3] * 4
        + [4] * 4
        + [1] * 12
    )

    try:
        for tick, action in enumerate(actions, start=1):
            obs_o, _, term_o, trunc_o, info_o = env.step(action)
            _, _, done_g, _ = gpu.step(
                torch.tensor([action], dtype=torch.int64, device=device)
            )

            if term_o or trunc_o or bool(done_g.item()):
                raise AssertionError(
                    f"unexpected termination during ego parity test at tick {tick}"
                )

            assert_close("ego vx", info_o["speed"], gpu.ego_vx[0].item(), 2e-5)
            assert_close(
                "ego vy", info_o["lateral_speed"], gpu.ego_vy[0].item(), 2e-5
            )
            assert_close("ego y", core.STATE.ego.y, gpu.ego_y[0].item(), 2e-5)
            assert_close("distance", info_o["distance"], gpu.distance[0].item(), 2e-4)

            # First four privileged fields are pure ego/time values and must be
            # exact regardless of traffic RNG. Bucket fraction also follows the
            # same one-car-per-tick spawn schedule.
            obs_g = gpu.observation()[0].detach().cpu().numpy()
            assert_close("ego observation", obs_o[:5], obs_g[:5], 2e-6)
    finally:
        env.close()

    print("PASS exact ego/action/integration parity")


def exact_controlled_observation(device: torch.device) -> None:
    env = PrivilegedRaceCarEnv(base_seed=0)
    gpu = GpuRaceCarBatch(1, device=device, seed=99)
    quiet_reset(env, 654321)
    gpu.reset()

    state = core.STATE
    ego = state.ego

    # Build identical hand-authored traffic state in four lanes.
    cars = list(state.car_bucket)
    state.car_bucket = []
    state.cars = [ego] + cars

    lanes = [0, 1, 3, 4]
    xs = [-100.75, 250.25, 1300.9, 2100.4]
    vxs = [7.25, 12.5, 4.0, 17.75]

    for car, lane_i, x, vx in zip(cars, lanes, xs, vxs):
        lane = state.road.lanes[lane_i]
        car.lane = lane
        car.x = x
        car.y = int((lane.y_start + lane.y_end) / 2 - car.sprite.get_height() / 2)
        car.velocity.x = vx
        car.velocity.y = 0.0

    gpu.car_active.zero_()
    gpu.car_lane.fill_(-1)
    for slot, (lane_i, x, vx) in enumerate(zip(lanes, xs, vxs)):
        gpu.car_active[0, slot] = True
        gpu.car_lane[0, slot] = lane_i
        gpu.car_x[0, slot] = x
        gpu.car_vx[0, slot] = vx
        lane = state.road.lanes[lane_i]
        gpu.car_y[0, slot] = int(
            (lane.y_start + lane.y_end) / 2 - CAR_HEIGHT / 2
        )

    obs_o = build_privileged_observation(state)
    obs_g = gpu.observation()[0].detach().cpu().numpy()

    try:
        assert_close("controlled privileged observation", obs_o, obs_g, 2e-6)
    finally:
        env.close()

    print("PASS exact privileged-observation encoding parity")


def _configure_original_collision(
    env: PrivilegedRaceCarEnv,
    *,
    ego_y: float,
    car_x: float | None,
) -> bool:
    quiet_reset(env, 777)
    state = core.STATE
    state.ego.y = ego_y
    state.ego.velocity.y = 0.0

    if car_x is None:
        state.cars = [state.ego]
        state.car_bucket = []
    else:
        car = state.car_bucket.pop()
        car.lane = state.road.middle_lane()
        car.x = car_x
        car.y = state.ego.y
        car.velocity.x = state.ego.velocity.x
        car.velocity.y = 0.0
        state.cars = [state.ego, car]
        state.car_bucket = []

    _, _, terminated, _, _ = env.step(0)
    return bool(terminated)


def _configure_gpu_collision(
    gpu: GpuRaceCarBatch,
    *,
    ego_y: float,
    car_x: float | None,
) -> bool:
    gpu.reset()
    gpu.ego_y[0] = ego_y
    gpu.ego_vy[0] = 0.0
    gpu.car_active.zero_()
    gpu.car_lane.fill_(-1)

    if car_x is not None:
        gpu.car_active[0, 0] = True
        gpu.car_lane[0, 0] = 2
        gpu.car_x[0, 0] = car_x
        gpu.car_y[0, 0] = ego_y
        gpu.car_vx[0, 0] = gpu.ego_vx[0]

    _, _, done, _ = gpu.step(
        torch.tensor([0], dtype=torch.int64, device=gpu.device)
    )
    return bool(done.item())


def exact_collision_boundaries(device: torch.device) -> None:
    env = PrivilegedRaceCarEnv(base_seed=0)
    gpu = GpuRaceCarBatch(1, device=device, seed=777)

    # pygame.Rect uses half-open boundaries: exact edge contact is not collision.
    cases = [
        ("car edge touch", 510.0, EGO_X + CAR_WIDTH, False),
        ("car 1px overlap", 510.0, EGO_X + CAR_WIDTH - 1, True),
        ("top wall edge", 40.0, None, False),
        ("top wall overlap", 39.0, None, True),
        ("bottom wall edge", 1160.0 - CAR_HEIGHT, None, False),
        ("bottom wall overlap", 1160.0 - CAR_HEIGHT + 1, None, True),
    ]

    try:
        for name, ego_y, car_x, expected in cases:
            orig = _configure_original_collision(
                env, ego_y=ego_y, car_x=car_x
            )
            gp = _configure_gpu_collision(gpu, ego_y=ego_y, car_x=car_x)
            if orig != expected:
                raise AssertionError(
                    f"{name}: original returned {orig}, expected {expected}"
                )
            if gp != orig:
                raise AssertionError(
                    f"{name}: GPU returned {gp}, original returned {orig}"
                )
    finally:
        env.close()

    print("PASS exact pygame collision-boundary parity")


def original_spawn_stats(samples: int, seed_start: int) -> Dict[str, np.ndarray | float]:
    env = PrivilegedRaceCarEnv(base_seed=0)
    lane_counts = np.zeros(LANE_COUNT, dtype=np.int64)
    behind = 0
    coeffs = []

    try:
        for i in range(samples):
            quiet_reset(env, seed_start + i)
            env.step(0)
            state = core.STATE
            opponents = [c for c in state.cars if c is not state.ego]
            if len(opponents) != 1:
                raise AssertionError(
                    f"original first step spawned {len(opponents)} opponents"
                )
            car = opponents[0]
            lane_i = next(
                j for j, lane in enumerate(state.road.lanes) if car.lane is lane
            )
            lane_counts[lane_i] += 1
            behind += int(car.x < 0)
            coeffs.append(abs(float(car.velocity.x) - float(state.ego.velocity.x)))
    finally:
        env.close()

    coeff = np.asarray(coeffs, dtype=np.float64)
    return {
        "lane_freq": lane_counts / max(1, samples),
        "behind": behind / max(1, samples),
        "coeff_mean": float(coeff.mean()),
        "coeff_std": float(coeff.std()),
    }


@torch.no_grad()
def gpu_spawn_stats(
    samples: int, device: torch.device, seed: int
) -> Dict[str, np.ndarray | float]:
    gpu = GpuRaceCarBatch(samples, device=device, seed=seed)
    gpu.reset()
    gpu.step(torch.zeros(samples, dtype=torch.int64, device=device))

    active_count = gpu.car_active.sum(dim=1)
    if not bool(torch.all(active_count == 1).item()):
        raise AssertionError("GPU first step did not spawn exactly one car per env")

    slot = gpu.car_active.int().argmax(dim=1, keepdim=True)
    lane = gpu.car_lane.gather(1, slot).squeeze(1)
    x = gpu.car_x.gather(1, slot).squeeze(1)
    vx = gpu.car_vx.gather(1, slot).squeeze(1)

    counts = torch.bincount(lane, minlength=LANE_COUNT).float() / samples
    coeff = torch.abs(vx - gpu.ego_vx)

    return {
        "lane_freq": counts.cpu().numpy(),
        "behind": float((x < 0).float().mean().cpu()),
        "coeff_mean": float(coeff.mean().cpu()),
        "coeff_std": float(coeff.std(unbiased=False).cpu()),
    }


def statistical_spawn_parity(
    device: torch.device,
    original_samples: int,
    gpu_samples: int,
    seed_start: int,
) -> None:
    orig = original_spawn_stats(original_samples, seed_start)
    gp = gpu_spawn_stats(gpu_samples, device, seed_start)

    expected_lane = np.full(LANE_COUNT, 1.0 / LANE_COUNT)
    expected_behind = 0.5
    expected_mean = 2.5
    expected_std = 5.0 / math.sqrt(12.0)

    print("Spawn distribution:")
    print(f"  original lanes: {np.round(orig['lane_freq'], 4)}")
    print(f"  gpu lanes:      {np.round(gp['lane_freq'], 4)}")
    print(
        f"  behind: original={orig['behind']:.4f} gpu={gp['behind']:.4f} "
        f"expected={expected_behind:.4f}"
    )
    print(
        f"  speed coeff mean: original={orig['coeff_mean']:.4f} "
        f"gpu={gp['coeff_mean']:.4f} expected={expected_mean:.4f}"
    )
    print(
        f"  speed coeff std:  original={orig['coeff_std']:.4f} "
        f"gpu={gp['coeff_std']:.4f} expected={expected_std:.4f}"
    )

    # Tolerances are deliberately loose enough for the smaller original sample
    # but tight enough to catch biased lane/front-back/speed generation.
    if np.max(np.abs(np.asarray(orig["lane_freq"]) - expected_lane)) > 0.07:
        raise AssertionError("original lane sample unexpectedly far from uniform")
    if np.max(np.abs(np.asarray(gp["lane_freq"]) - expected_lane)) > 0.01:
        raise AssertionError("GPU lane sampling is not uniform")
    if abs(float(orig["behind"]) - expected_behind) > 0.06:
        raise AssertionError("original front/back sample unexpectedly biased")
    if abs(float(gp["behind"]) - expected_behind) > 0.01:
        raise AssertionError("GPU front/back sampling is biased")
    if abs(float(orig["coeff_mean"]) - expected_mean) > 0.20:
        raise AssertionError("original speed coefficient mean is unexpected")
    if abs(float(gp["coeff_mean"]) - expected_mean) > 0.03:
        raise AssertionError("GPU speed coefficient mean is wrong")
    if abs(float(orig["coeff_std"]) - expected_std) > 0.20:
        raise AssertionError("original speed coefficient std is unexpected")
    if abs(float(gp["coeff_std"]) - expected_std) > 0.03:
        raise AssertionError("GPU speed coefficient std is wrong")

    print("PASS traffic spawn distribution parity")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print(f"Verification device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Static geometry constants first.
    if core.SCREEN_WIDTH != 1600 or core.SCREEN_HEIGHT != 1200:
        raise AssertionError("screen geometry differs from CUDA constants")
    if core.LANE_COUNT != LANE_COUNT:
        raise AssertionError("lane count differs")
    print(
        f"Geometry: car={CAR_WIDTH}x{CAR_HEIGHT}, ego_x={EGO_X:.0f}, "
        f"lanes={LANE_COUNT}"
    )

    exact_ego_dynamics(device)
    exact_controlled_observation(device)
    exact_collision_boundaries(device)
    statistical_spawn_parity(
        device,
        original_samples=args.original_samples,
        gpu_samples=args.gpu_samples,
        seed_start=args.seed_start,
    )

    print(
        "\nGPU SIM VERIFICATION PASSED\n"
        "Deterministic physics/observation/collision checks match the original, "
        "and CUDA traffic generation matches the original distributions."
    )


if __name__ == "__main__":
    main()
