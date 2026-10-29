"""Verify CUDA legal sensor observations against the original ray caster."""

from __future__ import annotations

import argparse
import contextlib
import io
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

import src.game.core as core
from src.mathematics.vector import Vector
from training.gpu_sim import (
    CAR_HEIGHT,
    LEGAL_OBS_SIZE,
    SENSOR_NAMES,
    SENSOR_RANGE,
    GpuRaceCarBatch,
)
from training.legal_env import build_legal_observation, update_sensors


CASES = [
    {
        "ego_y": 510.0,
        "ego_vx": 34.2,
        "ego_vy": 0.0,
        "ticks": 217,
        "distance": 12345.6,
        "cars": [
            (950.4, 510.0, 31.0, 2),
            (115.7, 286.0, 39.0, 1),
            (1450.2, 734.0, 30.0, 3),
            (-250.9, 958.0, 41.0, 4),
        ],
    },
    {
        "ego_y": 72.6,
        "ego_vx": 81.3,
        "ego_vy": -0.4,
        "ticks": 1700,
        "distance": 99500.25,
        "cars": [
            (710.2, 62.8, 79.5, 0),
            (1250.8, 286.0, 76.0, 1),
            (-430.7, 510.0, 88.0, 2),
        ],
    },
    {
        "ego_y": 875.9,
        "ego_vx": 57.7,
        "ego_vy": 0.3,
        "ticks": 3111,
        "distance": 211234.75,
        "cars": [
            (1230.9, 958.4, 55.0, 4),
            (430.2, 734.8, 61.0, 3),
            (1850.6, 510.2, 53.0, 2),
            (-980.0, 286.6, 62.0, 1),
        ],
    },
]


def original_case(case):
    with contextlib.redirect_stdout(io.StringIO()):
        core.initialize_game_state("", 123456)
    state = core.STATE
    ego = state.ego
    ego.y = float(case["ego_y"])
    ego.velocity = Vector(float(case["ego_vx"]), float(case["ego_vy"]))
    state.ticks = int(case["ticks"])
    state.distance = float(case["distance"])

    pool = list(state.car_bucket)
    state.car_bucket = []
    active = []
    for i, (x, y, vx, lane_idx) in enumerate(case["cars"]):
        car = pool[i]
        car.x = float(x)
        car.y = float(y)
        car.velocity = Vector(float(vx), 0.0)
        car.lane = state.road.lanes[int(lane_idx)]
        active.append(car)
    state.car_bucket.extend(pool[len(active):])
    state.cars = [ego] + active

    update_sensors(state)
    readings = []
    hits = []
    by_name = {sensor.name: sensor.reading for sensor in state.sensors}
    for name in SENSOR_NAMES:
        value = by_name[name]
        readings.append(SENSOR_RANGE if value is None else float(value))
        hits.append(value is not None)

    return state, np.asarray(readings), np.asarray(hits), build_legal_observation(state)


def load_gpu_case(gpu: GpuRaceCarBatch, case) -> None:
    gpu.reset()
    gpu.ego_y[0] = float(case["ego_y"])
    gpu.ego_vx[0] = float(case["ego_vx"])
    gpu.ego_vy[0] = float(case["ego_vy"])
    gpu.ticks[0] = int(case["ticks"])
    gpu.distance[0] = float(case["distance"])

    gpu.car_active.zero_()
    gpu.car_lane.fill_(-1)
    gpu.car_x.zero_()
    gpu.car_y.zero_()
    gpu.car_vx.fill_(8.0)

    for i, (x, y, vx, lane_idx) in enumerate(case["cars"]):
        gpu.car_active[0, i] = True
        gpu.car_lane[0, i] = int(lane_idx)
        gpu.car_x[0, i] = float(x)
        gpu.car_y[0, i] = float(y)
        gpu.car_vx[0, i] = float(vx)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    gpu = GpuRaceCarBatch(1, device=device, seed=999)

    print(f"Verification device: {device}")
    print(f"legal observation size: {LEGAL_OBS_SIZE}")

    for case_i, case in enumerate(CASES):
        _, expected_readings, expected_hits, expected_obs = original_case(case)
        load_gpu_case(gpu, case)

        actual_readings_t, actual_hits_t = gpu.sensor_readings()
        actual_readings = actual_readings_t[0].cpu().numpy()
        actual_hits = actual_hits_t[0].cpu().numpy()
        actual_obs = gpu.legal_observation()[0].cpu().numpy()

        if not np.array_equal(expected_hits, actual_hits):
            bad = [
                SENSOR_NAMES[i]
                for i in range(len(SENSOR_NAMES))
                if expected_hits[i] != actual_hits[i]
            ]
            raise AssertionError(
                f"case {case_i}: sensor hit-mask mismatch: {bad}\n"
                f"original={expected_hits}\ngpu={actual_hits}"
            )

        diff = np.abs(expected_readings - actual_readings)
        if float(diff.max()) > 2e-6:
            i = int(diff.argmax())
            raise AssertionError(
                f"case {case_i}: sensor {SENSOR_NAMES[i]} diff={diff[i]} "
                f"original={expected_readings[i]} gpu={actual_readings[i]}"
            )

        obs_diff = np.abs(expected_obs - actual_obs)
        if float(obs_diff.max()) > 2e-6:
            i = int(obs_diff.argmax())
            raise AssertionError(
                f"case {case_i}: legal obs index {i} diff={obs_diff[i]} "
                f"original={expected_obs[i]} gpu={actual_obs[i]}"
            )

        print(
            f"PASS case {case_i}: raw 16-ray readings, Optional hit masks, "
            "and normalized legal observation"
        )

    print("\nLEGAL SENSOR VERIFICATION PASSED")


if __name__ == "__main__":
    main()
