"""Benchmark sequence-level privileged receding-horizon MPC around the PPO teacher.

The normal T=1 probability-credit scheduler remains the nominal controller. A
cheap exact-state trigger invokes sequence planning only near relevant traffic.
The planner compares meaningful multi-tick maneuvers and executes only the first
primitive of the selected plan before replanning on the next real simulator tick.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
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
from training.teacher_mpc import (
    choose_confirmed_mpc_action,
    mpc_planning_decision,
)


_WORKER_MODEL: ActorCritic | None = None
_WORKER_ENV: PrivilegedRaceCarEnv | None = None
_WORKER_TEMPERATURE = 1.0
_WORKER_X_MARGIN = 0.0
_WORKER_Y_MARGIN = 0.0
_WORKER_TRIGGER_X = 900.0
_WORKER_TRIGGER_LATERAL = 300.0
_WORKER_TRIGGER_IMMEDIATE = 30.0
_WORKER_CLEARANCE_THRESHOLD = 8.0
_WORKER_WALL_THRESHOLD = 4.0
_WORKER_OPPONENT_UNCERTAINTY_SIGMA = 3.0
_WORKER_STRATEGIC_HORIZON = 0
_WORKER_STRATEGIC_TRIGGER_LIMIT = 0


def _crash_kind() -> str:
    state = core.STATE
    ego = state.ego
    for car in state.cars:
        if car is ego:
            continue
        if core.intersects(ego.rect, car.rect):
            return "car"
    for wall in state.road.walls:
        if core.intersects(ego.rect, wall.rect):
            return "wall"
    return "unknown"


def _scheduler_proposal(
    probs: torch.Tensor,
    scheduler_credit: torch.Tensor,
) -> int:
    scheduler_credit.add_(probs)
    return int(scheduler_credit.argmax().item())


def _run_episode(
    sim_seed: int,
    action_seed: int,
    horizon: int,
    burst_ticks: int,
) -> Dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")

    model = _WORKER_MODEL
    env = _WORKER_ENV
    _ = int(action_seed)  # scheduler path is deterministic

    started = time.perf_counter()
    scheduler_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)

    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(options={"sim_seed": int(sim_seed)})

    terminated = False
    truncated = False

    proposed_counts = Counter()
    executed_counts = Counter()
    trigger_proposals = Counter()
    override_counts = Counter()
    fallback_counts = Counter()
    selected_plans = Counter()
    teacher_collision_kind = Counter()
    teacher_collision_side = Counter()
    trigger_reasons = Counter()

    triggers = 0
    overrides = 0
    all_unsafe = 0
    all_unhealthy = 0
    plans_evaluated = 0
    zero_speed_ticks = 0
    zero_speed_triggers = 0
    zero_speed_overrides = 0
    lane_straddling_ticks = 0
    strategic_triggers = 0
    strategic_candidates = 0

    while not (terminated or truncated):
        state = core.STATE
        ego = state.ego
        zero_speed = float(state.ego.velocity.x) <= 1e-9
        if zero_speed:
            zero_speed_ticks += 1
        ego_top = float(ego.y)
        ego_bottom = ego_top + float(ego.sprite.get_height())
        if any(
            ego_top < float(lane.y_end) < ego_bottom
            for lane in state.road.lanes[:-1]
        ):
            lane_straddling_ticks += 1

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(
                logits.squeeze(0) / float(_WORKER_TEMPERATURE),
                dim=-1,
            )

        proposed = _scheduler_proposal(probs, scheduler_credit)
        proposed_counts[ACTIONS[proposed]] += 1

        if horizon <= 0:
            action = proposed
            diagnostic = None
            scheduler_credit[proposed] -= 1.0
        else:
            trigger_reason, planning_horizon = mpc_planning_decision(
                state,
                proposed_action=proposed,
                horizon=int(horizon),
                strategic_horizon=(
                    _WORKER_STRATEGIC_HORIZON
                    if _WORKER_STRATEGIC_TRIGGER_LIMIT <= 0
                    or strategic_triggers < _WORKER_STRATEGIC_TRIGGER_LIMIT
                    else 0
                ),
                trigger_x=_WORKER_TRIGGER_X,
                trigger_lateral=_WORKER_TRIGGER_LATERAL,
                trigger_immediate=_WORKER_TRIGGER_IMMEDIATE,
                opponent_uncertainty_sigma=_WORKER_OPPONENT_UNCERTAINTY_SIGMA,
            )
            if trigger_reason is None:
                action = proposed
                diagnostic = None
                scheduler_credit[proposed] -= 1.0
            else:
                triggers += 1
                trigger_proposals[ACTIONS[proposed]] += 1
                if zero_speed:
                    zero_speed_triggers += 1
                if planning_horizon > int(horizon):
                    strategic_candidates += 1

                action, diagnostic, strategic_used = choose_confirmed_mpc_action(
                    state,
                    proposed,
                    horizon=int(horizon),
                    planning_horizon=planning_horizon,
                    burst_ticks=int(burst_ticks),
                    x_margin=_WORKER_X_MARGIN,
                    y_margin=_WORKER_Y_MARGIN,
                    clearance_threshold=_WORKER_CLEARANCE_THRESHOLD,
                    wall_threshold=_WORKER_WALL_THRESHOLD,
                    opponent_uncertainty_sigma=_WORKER_OPPONENT_UNCERTAINTY_SIGMA,
                )
                if trigger_reason == "strategic_closing" and not strategic_used:
                    trigger_reason = "lateral"
                if trigger_reason == "strategic_closing":
                    strategic_triggers += 1
                trigger_reasons[str(trigger_reason)] += 1

                plans_evaluated += int(diagnostic["plans_evaluated"])
                selected_plans[str(diagnostic["selected_plan"])] += 1
                all_unsafe += int(bool(diagnostic["all_unsafe"]))
                all_unhealthy += int(bool(diagnostic["all_unhealthy"]))

                teacher_projection = diagnostic["teacher_projection"] or {}
                if not bool(teacher_projection.get("safe", True)):
                    teacher_collision_kind[
                        str(teacher_projection.get("collision_kind", "unknown"))
                    ] += 1
                    teacher_collision_side[
                        str(teacher_projection.get("collision_side", "none"))
                    ] += 1

                # Consume the scheduler request whether or not MPC substitutes a
                # different primitive. Do not debit the safety replacement:
                # otherwise rejected teacher demand can build into a feedback
                # loop with the external controller.
                scheduler_credit[proposed] -= 1.0

                if bool(diagnostic["overridden"]):
                    overrides += 1
                    override_counts[ACTIONS[proposed]] += 1
                    fallback_counts[ACTIONS[int(action)]] += 1
                    if zero_speed:
                        zero_speed_overrides += 1

        executed_counts[ACTIONS[int(action)]] += 1
        obs, _, terminated, truncated, info = env.step(int(action))

    ticks = int(info["ticks"])
    crashed = bool(info["crashed"])
    return {
        "seed": int(sim_seed),
        "horizon": int(horizon),
        "burst_ticks": int(burst_ticks),
        "distance": float(info["distance"]),
        "ticks": ticks,
        "crashed": crashed,
        "crash_kind": _crash_kind() if crashed else "none",
        "final_speed": float(info["speed"]),
        "final_lateral_speed": float(info["lateral_speed"]),
        "average_speed": (
            float(info["distance"]) / ticks if ticks > 0 else 0.0
        ),
        "triggers": int(triggers),
        "trigger_tick_rate": float(triggers / ticks) if ticks > 0 else 0.0,
        "overrides": int(overrides),
        "override_tick_rate": float(overrides / ticks) if ticks > 0 else 0.0,
        "all_unsafe": int(all_unsafe),
        "all_unhealthy": int(all_unhealthy),
        "plans_evaluated": int(plans_evaluated),
        "zero_speed_ticks": int(zero_speed_ticks),
        "zero_speed_triggers": int(zero_speed_triggers),
        "zero_speed_overrides": int(zero_speed_overrides),
        "lane_straddling_ticks": int(lane_straddling_ticks),
        "strategic_triggers": int(strategic_triggers),
        "strategic_candidates": int(strategic_candidates),
        "elapsed_seconds": float(time.perf_counter() - started),
        "proposed_counts": dict(proposed_counts),
        "executed_counts": dict(executed_counts),
        "trigger_proposals": dict(trigger_proposals),
        "override_counts": dict(override_counts),
        "fallback_counts": dict(fallback_counts),
        "selected_plans": dict(selected_plans),
        "teacher_collision_kind": dict(teacher_collision_kind),
        "teacher_collision_side": dict(teacher_collision_side),
        "trigger_reasons": dict(trigger_reasons),
    }


def _worker_init(
    model_state: Dict[str, torch.Tensor],
    crash_penalty: float,
    reward_scale: float,
    temperature: float,
    x_margin: float,
    y_margin: float,
    trigger_x: float,
    trigger_lateral: float,
    trigger_immediate: float,
    clearance_threshold: float,
    wall_threshold: float,
    opponent_uncertainty_sigma: float,
    strategic_horizon: int,
    strategic_trigger_limit: int,
) -> None:
    global _WORKER_MODEL, _WORKER_ENV, _WORKER_TEMPERATURE
    global _WORKER_X_MARGIN, _WORKER_Y_MARGIN
    global _WORKER_TRIGGER_X, _WORKER_TRIGGER_LATERAL
    global _WORKER_TRIGGER_IMMEDIATE
    global _WORKER_CLEARANCE_THRESHOLD, _WORKER_WALL_THRESHOLD
    global _WORKER_OPPONENT_UNCERTAINTY_SIGMA
    global _WORKER_STRATEGIC_HORIZON
    global _WORKER_STRATEGIC_TRIGGER_LIMIT

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
    _WORKER_TRIGGER_X = float(trigger_x)
    _WORKER_TRIGGER_LATERAL = float(trigger_lateral)
    _WORKER_TRIGGER_IMMEDIATE = float(trigger_immediate)
    _WORKER_CLEARANCE_THRESHOLD = float(clearance_threshold)
    _WORKER_WALL_THRESHOLD = float(wall_threshold)
    _WORKER_OPPONENT_UNCERTAINTY_SIGMA = float(opponent_uncertainty_sigma)
    _WORKER_STRATEGIC_HORIZON = int(strategic_horizon)
    _WORKER_STRATEGIC_TRIGGER_LIMIT = int(strategic_trigger_limit)


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    distances = np.asarray(
        [float(row["distance"]) for row in rows],
        dtype=np.float64,
    )
    ticks = np.asarray(
        [int(row["ticks"]) for row in rows],
        dtype=np.float64,
    )
    crashes = np.asarray(
        [bool(row["crashed"]) for row in rows],
        dtype=np.bool_,
    )
    avg_speeds = np.asarray(
        [float(row["average_speed"]) for row in rows],
        dtype=np.float64,
    )

    total_ticks = float(ticks.sum())
    triggers = sum(int(row["triggers"]) for row in rows)
    overrides = sum(int(row["overrides"]) for row in rows)
    all_unsafe = sum(int(row["all_unsafe"]) for row in rows)
    all_unhealthy = sum(int(row["all_unhealthy"]) for row in rows)
    plans = sum(int(row["plans_evaluated"]) for row in rows)
    zero_speed = sum(int(row["zero_speed_ticks"]) for row in rows)
    lane_straddling = sum(int(row["lane_straddling_ticks"]) for row in rows)
    worker_seconds = sum(float(row["elapsed_seconds"]) for row in rows)
    strategic_triggers = sum(int(row["strategic_triggers"]) for row in rows)
    strategic_candidates = sum(int(row["strategic_candidates"]) for row in rows)

    counters = {
        "crash_kind": Counter(),
        "proposed_counts": Counter(),
        "executed_counts": Counter(),
        "trigger_proposals": Counter(),
        "override_counts": Counter(),
        "fallback_counts": Counter(),
        "selected_plans": Counter(),
        "teacher_collision_kind": Counter(),
        "teacher_collision_side": Counter(),
        "trigger_reasons": Counter(),
    }
    for row in rows:
        if bool(row["crashed"]):
            counters["crash_kind"][str(row["crash_kind"])] += 1
        for key in counters:
            if key == "crash_kind":
                continue
            counters[key].update(row[key])

    return {
        "episodes": len(rows),
        "mean_distance": float(distances.mean()),
        "median_distance": float(np.median(distances)),
        "p10_distance": float(np.percentile(distances, 10)),
        "min_distance": float(distances.min()),
        "max_distance": float(distances.max()),
        "crash_rate": float(crashes.mean()),
        "below_10k_rate": float((distances < 10_000).mean()),
        "below_25k_rate": float((distances < 25_000).mean()),
        "below_50k_rate": float((distances < 50_000).mean()),
        "below_100k_rate": float((distances < 100_000).mean()),
        "mean_episode_speed": float(avg_speeds.mean()),
        "trigger_tick_rate": (
            float(triggers / total_ticks) if total_ticks > 0 else 0.0
        ),
        "override_tick_rate": (
            float(overrides / total_ticks) if total_ticks > 0 else 0.0
        ),
        "override_given_trigger_rate": (
            float(overrides / triggers) if triggers > 0 else 0.0
        ),
        "all_unsafe_given_trigger_rate": (
            float(all_unsafe / triggers) if triggers > 0 else 0.0
        ),
        "all_unhealthy_given_trigger_rate": (
            float(all_unhealthy / triggers) if triggers > 0 else 0.0
        ),
        "mean_plans_per_trigger": (
            float(plans / triggers) if triggers > 0 else 0.0
        ),
        "zero_speed_tick_rate": (
            float(zero_speed / total_ticks) if total_ticks > 0 else 0.0
        ),
        "lane_straddling_tick_rate": (
            float(lane_straddling / total_ticks) if total_ticks > 0 else 0.0
        ),
        "worker_seconds": float(worker_seconds),
        "ticks_per_worker_second": (
            float(total_ticks / worker_seconds) if worker_seconds > 0 else 0.0
        ),
        "triggers": int(triggers),
        "overrides": int(overrides),
        "all_unsafe": int(all_unsafe),
        "all_unhealthy": int(all_unhealthy),
        "strategic_triggers": int(strategic_triggers),
        "strategic_candidates": int(strategic_candidates),
        **{key: dict(counter) for key, counter in counters.items()},
        "rows": rows,
    }


def _percentile_interval(values: List[float]) -> List[float]:
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    ]


def _paired_comparison(
    baseline_rows: List[Dict[str, Any]],
    variant_rows: List[Dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    baseline = {int(row["seed"]): row for row in baseline_rows}
    variant = {int(row["seed"]): row for row in variant_rows}
    if baseline.keys() != variant.keys():
        raise ValueError("paired variants must contain identical seed sets")

    seeds = sorted(baseline)
    base_crash = np.asarray(
        [bool(baseline[seed]["crashed"]) for seed in seeds], dtype=np.float64
    )
    variant_crash = np.asarray(
        [bool(variant[seed]["crashed"]) for seed in seeds], dtype=np.float64
    )
    base_distance = np.asarray(
        [float(baseline[seed]["distance"]) for seed in seeds], dtype=np.float64
    )
    variant_distance = np.asarray(
        [float(variant[seed]["distance"]) for seed in seeds], dtype=np.float64
    )
    distance_delta = variant_distance - base_distance

    crash_bootstrap: List[float] = []
    mean_bootstrap: List[float] = []
    p10_bootstrap: List[float] = []
    if bootstrap_samples > 0:
        rng = np.random.default_rng(int(bootstrap_seed))
        for _ in range(int(bootstrap_samples)):
            indices = rng.integers(0, len(seeds), size=len(seeds))
            crash_bootstrap.append(
                float((variant_crash[indices] - base_crash[indices]).mean())
            )
            mean_bootstrap.append(float(distance_delta[indices].mean()))
            p10_bootstrap.append(
                float(
                    np.percentile(variant_distance[indices], 10)
                    - np.percentile(base_distance[indices], 10)
                )
            )

    rescued = int(((base_crash == 1.0) & (variant_crash == 0.0)).sum())
    introduced = int(((base_crash == 0.0) & (variant_crash == 1.0)).sum())
    both = int(((base_crash == 1.0) & (variant_crash == 1.0)).sum())
    return {
        "episodes": len(seeds),
        "crashes_rescued": rescued,
        "new_crashes_introduced": introduced,
        "both_crashed": both,
        "distance_wins": int((distance_delta > 0.0).sum()),
        "distance_losses": int((distance_delta < 0.0).sum()),
        "distance_ties": int((distance_delta == 0.0).sum()),
        "crash_rate_delta": float((variant_crash - base_crash).mean()),
        "mean_distance_delta": float(distance_delta.mean()),
        "p10_distance_delta": float(
            np.percentile(variant_distance, 10) - np.percentile(base_distance, 10)
        ),
        "crash_rate_delta_ci95": (
            _percentile_interval(crash_bootstrap) if crash_bootstrap else None
        ),
        "mean_distance_delta_ci95": (
            _percentile_interval(mean_bootstrap) if mean_bootstrap else None
        ),
        "p10_distance_delta_ci95": (
            _percentile_interval(p10_bootstrap) if p10_bootstrap else None
        ),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata() -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        dirty = None
    return {"revision": revision, "dirty": dirty}


def _dependency_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    for package in ("pygame", "gymnasium", "stable-baselines3"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[72, 96, 120],
        help="Maneuver horizons to compare; baseline is included automatically.",
    )
    p.add_argument(
        "--strategic-trigger-limit",
        type=int,
        default=1,
        help="Maximum accepted strategic lookaheads per episode; 0 is unlimited.",
    )
    p.add_argument("--burst-ticks", type=int, default=12)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--seed-start", type=int, default=200_000)
    p.add_argument("--action-seed-start", type=int, default=20_000_000)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--x-margin", type=float, default=0.0)
    p.add_argument("--y-margin", type=float, default=0.0)
    p.add_argument("--trigger-x", type=float, default=900.0)
    p.add_argument("--trigger-lateral", type=float, default=300.0)
    p.add_argument("--trigger-immediate", type=float, default=30.0)
    p.add_argument("--clearance-threshold", type=float, default=8.0)
    p.add_argument("--wall-threshold", type=float, default=4.0)
    p.add_argument("--opponent-uncertainty-sigma", type=float, default=3.0)
    p.add_argument(
        "--strategic-horizon",
        type=int,
        default=119,
        help=(
            "Use this longer horizon only when the normal trigger is lateral "
            "but traffic becomes longitudinally closing at the longer horizon; "
            "0 disables the confirmation."
        ),
    )
    p.add_argument("--bootstrap-samples", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=73_001)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/teacher_mpc_sweep.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_started = time.perf_counter()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.burst_ticks <= 0:
        raise ValueError("--burst-ticks must be positive")
    if args.x_margin < 0 or args.y_margin < 0:
        raise ValueError("margins must be non-negative")
    if args.trigger_x <= 0 or args.trigger_lateral <= 0:
        raise ValueError("trigger windows must be positive")
    if args.trigger_immediate < 0:
        raise ValueError("--trigger-immediate must be non-negative")
    if args.clearance_threshold < 0 or args.wall_threshold < 0:
        raise ValueError("clearance thresholds must be non-negative")
    if not np.isfinite(args.opponent_uncertainty_sigma) or args.opponent_uncertainty_sigma < 0:
        raise ValueError("--opponent-uncertainty-sigma must be finite and non-negative")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples must be non-negative")
    if args.strategic_horizon < 0:
        raise ValueError("--strategic-horizon must be non-negative")
    if args.strategic_trigger_limit < 0:
        raise ValueError("--strategic-trigger-limit must be non-negative")

    horizons: List[int] = []
    for horizon in args.horizons:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("all horizons must be positive")
        if horizon not in horizons:
            horizons.append(horizon)

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    model_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    config = payload.get("config", {})
    crash_penalty = float(config.get("crash_penalty", 250.0))
    reward_scale = float(config.get("reward_scale", 1000.0))

    variants = [0] + horizons
    tasks: List[Tuple[int, int, int, int]] = []
    for horizon in variants:
        for i in range(args.episodes):
            tasks.append(
                (
                    args.seed_start + i,
                    args.action_seed_start + i,
                    horizon,
                    args.burst_ticks,
                )
            )

    workers = max(1, min(int(args.workers), len(tasks)))
    if workers == 1:
        _worker_init(
            model_state,
            crash_penalty,
            reward_scale,
            args.temperature,
            args.x_margin,
            args.y_margin,
            args.trigger_x,
            args.trigger_lateral,
            args.trigger_immediate,
            args.clearance_threshold,
            args.wall_threshold,
            args.opponent_uncertainty_sigma,
            args.strategic_horizon,
            args.strategic_trigger_limit,
        )
        rows = [_run_episode(*task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                model_state,
                crash_penalty,
                reward_scale,
                args.temperature,
                args.x_margin,
                args.y_margin,
                args.trigger_x,
                args.trigger_lateral,
                args.trigger_immediate,
                args.clearance_threshold,
                args.wall_threshold,
                args.opponent_uncertainty_sigma,
                args.strategic_horizon,
                args.strategic_trigger_limit,
            ),
        ) as executor:
            rows = list(
                executor.map(
                    _run_episode,
                    [task[0] for task in tasks],
                    [task[1] for task in tasks],
                    [task[2] for task in tasks],
                    [task[3] for task in tasks],
                    chunksize=1,
                )
            )

    results: Dict[str, Dict[str, Any]] = {}
    for horizon in variants:
        variant_rows = sorted(
            [row for row in rows if int(row["horizon"]) == int(horizon)],
            key=lambda row: int(row["seed"]),
        )
        label = "baseline" if horizon == 0 else f"horizon_{horizon}"
        results[label] = _summarize(variant_rows)

    print(
        "\nExact-original sequence-level privileged MPC "
        f"(scheduler T={args.temperature:g}, n={args.episodes}, "
        f"burst={args.burst_ticks}, margin=({args.x_margin:g},{args.y_margin:g}))"
    )
    print(
        f"{'variant':<13} {'mean':>10} {'median':>10} {'p10':>10} "
        f"{'crash':>8} {'<25k':>8} {'avg_v':>8} "
        f"{'trigger':>8} {'override':>9}"
    )
    print("-" * 93)

    for label, result in results.items():
        print(
            f"{label:<13} "
            f"{result['mean_distance']:10,.0f} "
            f"{result['median_distance']:10,.0f} "
            f"{result['p10_distance']:10,.0f} "
            f"{result['crash_rate']:8.1%} "
            f"{result['below_25k_rate']:8.1%} "
            f"{result['mean_episode_speed']:8.1f} "
            f"{result['trigger_tick_rate']:8.2%} "
            f"{result['override_tick_rate']:9.2%}"
        )

    baseline = results["baseline"]
    paired_comparisons: Dict[str, Dict[str, Any]] = {}
    print("\nDeltas and planner diagnostics")
    for horizon in horizons:
        result = results[f"horizon_{horizon}"]
        comparison = _paired_comparison(
            baseline["rows"],
            result["rows"],
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_seed=int(args.bootstrap_seed) + int(horizon),
        )
        result["paired_vs_baseline"] = comparison
        paired_comparisons[f"horizon_{horizon}"] = comparison
        print(
            f"h{horizon}: "
            f"crash={float(result['crash_rate']) - float(baseline['crash_rate']):+.2%}, "
            f"mean={float(result['mean_distance']) - float(baseline['mean_distance']):+,.0f}, "
            f"p10={float(result['p10_distance']) - float(baseline['p10_distance']):+,.0f}, "
            f"override|trigger={result['override_given_trigger_rate']:.1%}, "
            f"allunsafe|trigger={result['all_unsafe_given_trigger_rate']:.1%}, "
            f"plans/trigger={result['mean_plans_per_trigger']:.2f}, "
            f"plans={result['selected_plans']}, "
            f"overrides={result['override_counts']}, "
            f"fallbacks={result['fallback_counts']}, "
            f"rescued/new={comparison['crashes_rescued']}/"
            f"{comparison['new_crashes_introduced']}"
        )

    recommended_variant = min(
        (f"horizon_{horizon}" for horizon in horizons),
        key=lambda label: (
            int(round(float(results[label]["crash_rate"]) * args.episodes)),
            int(results[label]["paired_vs_baseline"]["new_crashes_introduced"]),
            -float(results[label]["p10_distance"]),
            float(results[label]["worker_seconds"]),
        ),
    )

    benchmark_elapsed = time.perf_counter() - benchmark_started
    total_ticks = sum(
        int(row["ticks"])
        for result in results.values()
        for row in result["rows"]
    )
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    output = {
        "schema_version": 3,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "git": _git_metadata(),
        "dependency_versions": _dependency_versions(),
        "command": [sys.executable, *sys.argv],
        "arguments": serialized_args,
        "episodes": int(args.episodes),
        "seed_start": int(args.seed_start),
        "action_seed_start": int(args.action_seed_start),
        "temperature": float(args.temperature),
        "burst_ticks": int(args.burst_ticks),
        "x_margin": float(args.x_margin),
        "y_margin": float(args.y_margin),
        "trigger_x": float(args.trigger_x),
        "trigger_lateral": float(args.trigger_lateral),
        "trigger_immediate": float(args.trigger_immediate),
        "clearance_threshold": float(args.clearance_threshold),
        "wall_threshold": float(args.wall_threshold),
        "opponent_uncertainty_sigma": float(args.opponent_uncertainty_sigma),
        "strategic_horizon": int(args.strategic_horizon),
        "strategic_trigger_limit": int(args.strategic_trigger_limit),
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "benchmark_elapsed_seconds": float(benchmark_elapsed),
        "ticks_per_wall_second": (
            float(total_ticks / benchmark_elapsed) if benchmark_elapsed > 0 else 0.0
        ),
        "recommended_variant": recommended_variant,
        "paired_comparisons": paired_comparisons,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
