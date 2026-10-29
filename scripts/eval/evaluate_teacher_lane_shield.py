"""Evaluate a privileged lane-change initiation safety shield.

The shield is teacher-only: it sees exact simulator state, projects a canonical
48-steer/48-counter-steer maneuver, and vetoes only *initiating* steering
actions whose projected trajectory collides with a current car or wall.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

import src.game.core as core
from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, OBS_SIZE
from training.teacher_env import PrivilegedRaceCarEnv
from training.teacher_lane_safety import (
    ACCELERATE,
    DECELERATE,
    NOTHING,
    current_lane_ttc,
    is_lane_change_initiation,
    project_lane_change,
)


_WORKER_MODEL: ActorCritic | None = None
_WORKER_ENV: PrivilegedRaceCarEnv | None = None
_WORKER_TEMPERATURE = 1.0
_WORKER_X_MARGIN = 0.0
_WORKER_Y_MARGIN = 0.0


def _fallback_action(state) -> int:
    """Choose longitudinal fallback when an unsafe lane change is vetoed."""
    ttc = current_lane_ttc(state)
    front = ttc["front"]
    rear = ttc["rear"]

    # If the current-lane danger is primarily in front, create space by
    # braking. If the primary danger is behind, accelerate away. With no
    # current-lane TTC threat, defer one tick with NOTHING.
    if front is not None and (rear is None or float(front) <= float(rear)):
        return DECELERATE
    if rear is not None:
        return ACCELERATE
    return NOTHING


def _run_episode(
    sim_seed: int,
    action_seed: int,
    shielded: bool,
) -> Dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")

    model = _WORKER_MODEL
    env = _WORKER_ENV

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(action_seed))
    scheduler_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)

    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(options={"sim_seed": int(sim_seed)})

    terminated = False
    truncated = False
    proposed_counts = Counter()
    executed_counts = Counter()
    vetoes = 0
    unsafe_initiations = 0
    fallback_counts = Counter()

    while not (terminated or truncated):
        state = core.STATE
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(
                logits.squeeze(0) / float(_WORKER_TEMPERATURE), dim=-1
            )

        scheduler_credit.add_(probs)
        proposed = int(scheduler_credit.argmax().item())
        scheduler_credit[proposed] -= 1.0
        proposed_counts[ACTIONS[proposed]] += 1

        executed = proposed
        if shielded:
            initiated, _, _, direction = is_lane_change_initiation(
                state, proposed
            )
            if initiated:
                projection = project_lane_change(
                    state,
                    direction,
                    steer_ticks=48,
                    counter_ticks=48,
                    x_margin=_WORKER_X_MARGIN,
                    y_margin=_WORKER_Y_MARGIN,
                )
                if bool(projection["unsafe_at_start"]):
                    unsafe_initiations += 1
                    executed = _fallback_action(state)
                    vetoes += 1
                    fallback_counts[ACTIONS[executed]] += 1

                    # Scheduler accounting must reflect the action that was
                    # actually executed, not the vetoed proposal.
                    scheduler_credit[proposed] += 1.0
                    scheduler_credit[executed] -= 1.0

        executed_counts[ACTIONS[executed]] += 1
        obs, _, terminated, truncated, info = env.step(executed)

    ticks = int(info["ticks"])
    return {
        "seed": int(sim_seed),
        "shielded": bool(shielded),
        "distance": float(info["distance"]),
        "ticks": ticks,
        "crashed": bool(info["crashed"]),
        "final_speed": float(info["speed"]),
        "average_speed": (
            float(info["distance"]) / ticks if ticks > 0 else 0.0
        ),
        "vetoes": int(vetoes),
        "unsafe_initiations": int(unsafe_initiations),
        "veto_tick_rate": float(vetoes / ticks) if ticks > 0 else 0.0,
        "fallback_counts": dict(fallback_counts),
        "proposed_counts": dict(proposed_counts),
        "executed_counts": dict(executed_counts),
    }


def _worker_init(
    model_state: Dict[str, torch.Tensor],
    crash_penalty: float,
    reward_scale: float,
    temperature: float,
    x_margin: float,
    y_margin: float,
) -> None:
    global _WORKER_MODEL, _WORKER_ENV
    global _WORKER_TEMPERATURE, _WORKER_X_MARGIN, _WORKER_Y_MARGIN

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False).cpu()
    model.load_state_dict(model_state)
    model.eval()

    _WORKER_MODEL = model
    _WORKER_ENV = PrivilegedRaceCarEnv(
        base_seed=0,
        crash_penalty=float(crash_penalty),
        reward_scale=float(reward_scale),
    )
    _WORKER_TEMPERATURE = float(temperature)
    _WORKER_X_MARGIN = float(x_margin)
    _WORKER_Y_MARGIN = float(y_margin)


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    distances = np.asarray(
        [float(row["distance"]) for row in rows], dtype=np.float64
    )
    speeds = np.asarray(
        [float(row["average_speed"]) for row in rows], dtype=np.float64
    )
    crashes = np.asarray(
        [bool(row["crashed"]) for row in rows], dtype=np.bool_
    )
    vetoes = sum(int(row["vetoes"]) for row in rows)
    total_ticks = sum(int(row["ticks"]) for row in rows)
    fallback = Counter()
    for row in rows:
        fallback.update(row["fallback_counts"])

    return {
        "episodes": len(rows),
        "mean_distance": float(distances.mean()),
        "median_distance": float(np.median(distances)),
        "p10_distance": float(np.percentile(distances, 10)),
        "crash_rate": float(crashes.mean()),
        "below_10k_rate": float((distances < 10_000).mean()),
        "below_25k_rate": float((distances < 25_000).mean()),
        "below_50k_rate": float((distances < 50_000).mean()),
        "below_100k_rate": float((distances < 100_000).mean()),
        "mean_episode_speed": float(speeds.mean()),
        "vetoes": int(vetoes),
        "veto_tick_rate": (
            float(vetoes / total_ticks) if total_ticks > 0 else 0.0
        ),
        "fallback_counts": dict(fallback),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--seed-start", type=int, default=140_000)
    p.add_argument("--action-seed-start", type=int, default=14_000_000)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--x-margin",
        type=float,
        default=0.0,
        help="Extra longitudinal collision margin in the projected maneuver.",
    )
    p.add_argument(
        "--y-margin",
        type=float,
        default=0.0,
        help="Extra lateral collision/wall margin in the projected maneuver.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/teacher_lane_shield_eval.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.x_margin < 0 or args.y_margin < 0:
        raise ValueError("margins must be non-negative")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    state_dict = {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }

    config = payload.get("config", {})
    crash_penalty = float(config.get("crash_penalty", 250.0))
    reward_scale = float(config.get("reward_scale", 1000.0))

    # Run baseline and shielded variants over the exact same simulator seeds.
    tasks: List[Tuple[int, int, bool]] = []
    for shielded in (False, True):
        for i in range(args.episodes):
            tasks.append(
                (
                    args.seed_start + i,
                    args.action_seed_start + i,
                    shielded,
                )
            )

    workers = max(1, min(int(args.workers), len(tasks)))
    if workers == 1:
        _worker_init(
            state_dict,
            crash_penalty,
            reward_scale,
            args.temperature,
            args.x_margin,
            args.y_margin,
        )
        rows = [_run_episode(*task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                state_dict,
                crash_penalty,
                reward_scale,
                args.temperature,
                args.x_margin,
                args.y_margin,
            ),
        ) as executor:
            rows = list(
                executor.map(
                    _run_episode,
                    [task[0] for task in tasks],
                    [task[1] for task in tasks],
                    [task[2] for task in tasks],
                    chunksize=1,
                )
            )

    baseline_rows = sorted(
        [row for row in rows if not row["shielded"]],
        key=lambda row: int(row["seed"]),
    )
    shield_rows = sorted(
        [row for row in rows if row["shielded"]],
        key=lambda row: int(row["seed"]),
    )

    baseline = _summarize(baseline_rows)
    shielded = _summarize(shield_rows)

    print(
        "\nExact-original privileged lane-change shield "
        f"(scheduler T={args.temperature:g}, n={args.episodes}, "
        f"margin=({args.x_margin:g},{args.y_margin:g}))"
    )
    print(
        f"{'variant':<12} {'mean':>10} {'median':>10} {'p10':>10} "
        f"{'crash':>8} {'<25k':>8} {'avg_v':>8} {'veto%':>8}"
    )
    print("-" * 82)
    for label, result in (
        ("baseline", baseline),
        ("shielded", shielded),
    ):
        print(
            f"{label:<12} "
            f"{result['mean_distance']:10,.0f} "
            f"{result['median_distance']:10,.0f} "
            f"{result['p10_distance']:10,.0f} "
            f"{result['crash_rate']:8.1%} "
            f"{result['below_25k_rate']:8.1%} "
            f"{result['mean_episode_speed']:8.1f} "
            f"{result['veto_tick_rate']:8.2%}"
        )

    delta_crash = (
        float(shielded["crash_rate"]) - float(baseline["crash_rate"])
    )
    delta_mean = (
        float(shielded["mean_distance"]) - float(baseline["mean_distance"])
    )
    print(
        "\nShield delta: "
        f"crash={delta_crash:+.2%}, mean_distance={delta_mean:+,.0f}, "
        f"vetoes={shielded['vetoes']:,}, "
        f"fallbacks={shielded['fallback_counts']}"
    )

    output = {
        "checkpoint": str(args.checkpoint),
        "seed_start": int(args.seed_start),
        "episodes": int(args.episodes),
        "temperature": float(args.temperature),
        "x_margin": float(args.x_margin),
        "y_margin": float(args.y_margin),
        "baseline": baseline,
        "shielded": shielded,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
