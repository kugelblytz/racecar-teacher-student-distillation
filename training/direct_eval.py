"""Parallel evaluation of direct legal-observation policies on the original sim."""

from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch

from training.gpu_ppo import ActorCritic
from training.gpu_recurrent_ppo import (
    RecurrentActorCritic,
    ResidualRecurrentActorCritic,
)
from training.gpu_sim import (
    ACTIONS,
    CAR_HEIGHT,
    LANE_CENTERS,
    LEGAL_OBS_SIZE,
    LEGAL_OBS_SIZE_TRACKED_Y,
)
from training.legal_env import LegalRaceCarEnv


PolicyMode = Literal["deterministic", "stochastic", "scheduler", "lane_options"]
Arch = Literal["mlp", "gru", "residual_gru"]


def _summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    distances = np.asarray([float(r["distance"]) for r in rows], dtype=np.float64)
    speeds = np.asarray([float(r["final_speed"]) for r in rows], dtype=np.float64)
    crashes = np.asarray([bool(r["crashed"]) for r in rows], dtype=np.bool_)
    car_crashes = np.asarray([bool(r["hit_car"]) for r in rows], dtype=np.bool_)
    wall_crashes = np.asarray([bool(r["hit_wall"]) for r in rows], dtype=np.bool_)
    lateral_speeds = np.asarray(
        [float(r["final_lateral_speed"]) for r in rows], dtype=np.float64
    )
    crash_or_short = crashes | (distances < 25_000)
    action_counts = (
        np.asarray([r["action_counts"] for r in rows], dtype=np.int64).sum(axis=0)
        if rows
        else np.zeros(len(ACTIONS), dtype=np.int64)
    )
    action_total = max(1, int(action_counts.sum()))

    return {
        "episodes": len(rows),
        "mean_distance": float(distances.mean()) if len(rows) else 0.0,
        "median_distance": float(np.median(distances)) if len(rows) else 0.0,
        "p10_distance": float(np.percentile(distances, 10)) if len(rows) else 0.0,
        "min_distance": float(distances.min()) if len(rows) else 0.0,
        "max_distance": float(distances.max()) if len(rows) else 0.0,
        "crash_rate": float(crashes.mean()) if len(rows) else 0.0,
        "car_crash_rate": float(car_crashes.mean()) if len(rows) else 0.0,
        "wall_crash_rate": float(wall_crashes.mean()) if len(rows) else 0.0,
        "crash_or_below_25k_rate": (
            float(crash_or_short.mean()) if len(rows) else 0.0
        ),
        "mean_final_speed": float(speeds.mean()) if len(rows) else 0.0,
        "mean_abs_final_lateral_speed": (
            float(np.abs(lateral_speeds).mean()) if len(rows) else 0.0
        ),
        "action_counts": [int(value) for value in action_counts.tolist()],
        "action_fractions": [
            float(value) / float(action_total) for value in action_counts.tolist()
        ],
        "below_10k_rate": float((distances < 10_000).mean()) if len(rows) else 0.0,
        "below_25k_rate": float((distances < 25_000).mean()) if len(rows) else 0.0,
        "below_50k_rate": float((distances < 50_000).mean()) if len(rows) else 0.0,
        "below_100k_rate": float((distances < 100_000).mean()) if len(rows) else 0.0,
        "rows": rows,
    }


_WORKER_MODEL = None
_WORKER_ENV: LegalRaceCarEnv | None = None
_WORKER_ARCH: Arch | None = None
_WORKER_ACTION_REPEAT: int = 1
_WORKER_LANE_OPTION_HEAD = None
_WORKER_LANE_OPTION_THRESHOLD: float = 0.5


def _lane_option_primitive(
    proposed: int,
    target_lane: int | None,
    *,
    y: float,
    vy: float,
) -> Tuple[int, int | None]:
    centers = [float(value) for value in LANE_CENTERS.tolist()]
    if proposed < 0 or proposed >= len(ACTIONS):
        raise ValueError(f"invalid proposed action: {proposed}")
    if target_lane is not None and not 0 <= target_lane < len(centers):
        raise ValueError(f"invalid target lane: {target_lane}")
    if not math.isfinite(float(y)) or not math.isfinite(float(vy)):
        raise ValueError("lane-option state must be finite")
    if target_lane is None and proposed in (3, 4):
        ego_center = float(y) + float(CAR_HEIGHT) * 0.5
        nearest = min(range(len(centers)), key=lambda i: abs(ego_center - centers[i]))
        candidate = nearest + (-1 if proposed == 3 else 1)
        if 0 <= candidate < len(centers):
            target_lane = candidate

    if target_lane is None:
        return 0, None
    target_y = centers[target_lane] - float(CAR_HEIGHT) * 0.5
    error = target_y - float(y)
    if abs(error) <= 4.0 and abs(float(vy)) <= 0.2:
        return 0, None
    if error * float(vy) < 0.0:
        action = 4 if error > 0.0 else 3
    else:
        stopping_distance = float(vy) * float(vy) / 0.2
        if abs(float(vy)) > 0.15 and abs(error) <= stopping_distance + 2.0:
            action = 3 if vy > 0.0 else 4
        else:
            action = 4 if error > 0.0 else 3
    return action, target_lane


def _worker_init(
    arch: Arch,
    model_state: Dict[str, torch.Tensor],
    hidden_dim: int,
    crash_penalty: float,
    wall_crash_penalty: float,
    reward_scale: float,
    action_repeat: int,
    obs_dim: int,
    tracked_y: bool,
    lane_option_state: Dict[str, torch.Tensor] | None,
    lane_option_threshold: float,
) -> None:
    global _WORKER_MODEL, _WORKER_ENV, _WORKER_ARCH, _WORKER_ACTION_REPEAT
    global _WORKER_LANE_OPTION_HEAD, _WORKER_LANE_OPTION_THRESHOLD

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if arch == "mlp":
        model = ActorCritic(obs_dim, len(ACTIONS), safe_init=False).cpu()
    elif arch == "gru":
        model = RecurrentActorCritic(
            obs_dim,
            len(ACTIONS),
            hidden_dim=hidden_dim,
            safe_init=False,
        ).cpu()
    elif arch == "residual_gru":
        model = ResidualRecurrentActorCritic(
            obs_dim,
            len(ACTIONS),
            hidden_dim=hidden_dim,
        ).cpu()
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.load_state_dict(model_state)
    model.eval()

    _WORKER_MODEL = model
    _WORKER_ARCH = arch
    _WORKER_ACTION_REPEAT = int(action_repeat)
    _WORKER_LANE_OPTION_HEAD = None
    _WORKER_LANE_OPTION_THRESHOLD = float(lane_option_threshold)
    if lane_option_state is not None:
        if arch != "gru":
            raise ValueError("lane options require a GRU policy")
        head = torch.nn.Linear(hidden_dim, 3).cpu()
        head.load_state_dict(lane_option_state)
        head.eval()
        _WORKER_LANE_OPTION_HEAD = head
    if _WORKER_ACTION_REPEAT <= 0:
        raise ValueError("action_repeat must be positive")
    _WORKER_ENV = LegalRaceCarEnv(
        base_seed=0,
        crash_penalty=crash_penalty,
        wall_crash_penalty=wall_crash_penalty,
        reward_scale=reward_scale,
        tracked_y=tracked_y,
    )


def _worker_task(
    task: Tuple[PolicyMode, int, int],
) -> Tuple[PolicyMode, Dict[str, object]]:
    mode, sim_seed, action_seed = task
    if _WORKER_MODEL is None or _WORKER_ENV is None or _WORKER_ARCH is None:
        raise RuntimeError("Direct-eval worker not initialized")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(action_seed))

    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = _WORKER_ENV.reset(options={"sim_seed": int(sim_seed)})

    if _WORKER_ARCH in ("gru", "residual_gru"):
        hidden = _WORKER_MODEL.initial_state(1, device="cpu")
    else:
        hidden = None

    terminated = False
    truncated = False
    info = {}
    scheduler_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)
    action_counts = [0] * len(ACTIONS)
    target_lane = None

    while not (terminated or truncated):
        obs_t = torch.from_numpy(obs).unsqueeze(0)

        with torch.no_grad():
            if _WORKER_ARCH in ("gru", "residual_gru"):
                logits, _, hidden = _WORKER_MODEL.step(obs_t, hidden)
            else:
                logits, _ = _WORKER_MODEL(obs_t)

        if mode in ("deterministic", "lane_options"):
            action = int(logits.argmax(dim=-1).item())
        elif mode == "stochastic":
            probs = torch.softmax(logits, dim=-1)
            action = int(
                torch.multinomial(probs, 1, generator=generator).item()
            )
        elif mode == "scheduler":
            scheduler_credit.add_(torch.softmax(logits.squeeze(0), dim=-1))
            action = int(scheduler_credit.argmax().item())
            scheduler_credit[action] -= 1.0
        else:
            raise ValueError(f"unknown policy mode: {mode}")
        if mode == "lane_options":
            if _WORKER_LANE_OPTION_HEAD is None or hidden is None:
                raise ValueError("lane_options mode requires a lane-option head")
            if target_lane is None:
                # Steering is exclusively controlled by the option head in this
                # mode; the primitive head continues longitudinal control.
                if action in (3, 4):
                    action = 0
                option_probs = torch.softmax(
                    _WORKER_LANE_OPTION_HEAD(hidden).squeeze(0), dim=-1
                )
                direction_prob, direction = option_probs[1:].max(dim=0)
                if float(direction_prob.item()) >= _WORKER_LANE_OPTION_THRESHOLD:
                    action = 3 + int(direction.item())
            tracker = _WORKER_ENV.lateral_tracker
            if target_lane is not None or action in (3, 4):
                action, target_lane = _lane_option_primitive(
                    action,
                    target_lane,
                    y=float(tracker.y),
                    vy=float(tracker.vy),
                )
        action_counts[action] += 1

        for _ in range(_WORKER_ACTION_REPEAT):
            obs, _, terminated, truncated, info = _WORKER_ENV.step(action)
            if terminated or truncated:
                break

    return mode, {
        "seed": int(sim_seed),
        "distance": float(info["distance"]),
        "ticks": int(info["ticks"]),
        "crashed": bool(info["crashed"]),
        "final_speed": float(info["speed"]),
        "final_lateral_speed": float(info["lateral_speed"]),
        "final_tracked_y": float(info["tracked_y"]),
        "hit_car": bool(info["hit_car"]),
        "hit_wall": bool(info["hit_wall"]),
        "action_counts": action_counts,
    }


def evaluate_direct_original(
    model,
    *,
    arch: Arch,
    episodes: int,
    seed_start: int,
    crash_penalty: float = 250.0,
    wall_crash_penalty: float = 350.0,
    reward_scale: float = 1000.0,
    action_seed_start: int = 1_000_000,
    workers: int = 0,
    hidden_dim: int = 128,
    action_repeat: int = 1,
    obs_dim: int = LEGAL_OBS_SIZE,
    tracked_y: bool = False,
    policy_modes: Tuple[PolicyMode, ...] = ("deterministic", "stochastic"),
    lane_option_state: Dict[str, torch.Tensor] | None = None,
    lane_option_threshold: float = 0.5,
) -> Dict[str, Dict[str, object]]:
    """Evaluate requested policy decoders on identical original-simulator seeds."""
    if arch not in ("mlp", "gru", "residual_gru"):
        raise ValueError(f"Unknown arch: {arch}")
    expected_obs_dim = LEGAL_OBS_SIZE_TRACKED_Y if tracked_y else LEGAL_OBS_SIZE
    if int(obs_dim) != expected_obs_dim:
        raise ValueError(
            f"obs_dim={obs_dim} does not match tracked_y={tracked_y} "
            f"(expected {expected_obs_dim})"
        )

    count = int(episodes)
    if count <= 0:
        raise ValueError("episodes must be positive")
    modes = tuple(policy_modes)
    if not modes:
        raise ValueError("policy_modes must not be empty")
    if len(set(modes)) != len(modes):
        raise ValueError("policy_modes must not contain duplicates")
    valid_modes = {"deterministic", "stochastic", "scheduler", "lane_options"}
    if any(mode not in valid_modes for mode in modes):
        raise ValueError(f"policy_modes must be selected from {sorted(valid_modes)}")
    if "lane_options" in modes and lane_option_state is None:
        raise ValueError("lane_options mode requires lane_option_state")
    if not math.isfinite(float(lane_option_threshold)) or not (
        0.0 < float(lane_option_threshold) <= 1.0
    ):
        raise ValueError("lane_option_threshold must be finite and in (0, 1]")

    tasks = [
        (mode, int(seed_start) + i, int(action_seed_start) + i)
        for mode in modes
        for i in range(count)
    ]

    if workers > 0:
        worker_count = min(int(workers), len(tasks))
    else:
        worker_count = min(os.cpu_count() or 1, len(tasks), 24)
    worker_count = max(1, worker_count)

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    grouped: Dict[str, List[Dict[str, object]]] = {mode: [] for mode in modes}

    if worker_count == 1:
        _worker_init(
            arch,
            state,
            hidden_dim,
            crash_penalty,
            wall_crash_penalty,
            reward_scale,
            action_repeat,
            int(obs_dim),
            bool(tracked_y),
            lane_option_state,
            float(lane_option_threshold),
        )
        for task in tasks:
            mode, row = _worker_task(task)
            grouped[mode].append(row)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                arch,
                state,
                int(hidden_dim),
                float(crash_penalty),
                float(wall_crash_penalty),
                float(reward_scale),
                int(action_repeat),
                int(obs_dim),
                bool(tracked_y),
                lane_option_state,
                float(lane_option_threshold),
            ),
        ) as executor:
            for mode, row in executor.map(_worker_task, tasks, chunksize=1):
                grouped[mode].append(row)

    result: Dict[str, Dict[str, object]] = {}
    for mode, rows in grouped.items():
        rows.sort(key=lambda x: int(x["seed"]))
        stats = _summarize(rows)
        stats["arch"] = arch
        stats["policy_mode"] = mode
        stats["workers"] = worker_count
        stats["action_repeat"] = int(action_repeat)
        stats["obs_dim"] = int(obs_dim)
        stats["tracked_y"] = bool(tracked_y)
        result[mode] = stats

    return result
