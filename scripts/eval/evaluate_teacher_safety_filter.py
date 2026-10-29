"""Benchmark a short-horizon privileged safety filter for the PPO teacher.

Unlike the failed lane-initiation shield, this filter replans every simulator
tick. It first projects the scheduler's proposed primitive action for a short
horizon. If that action remains recoverable, it is executed unchanged. Only
when it projects a collision does the filter evaluate the other primitives and
substitute the teacher's best safe alternative (or the least-bad action if all
five project unsafe).

The filter is privileged teacher machinery only. It is not legal deployment
code and does not use future RNG draws or future spawn information.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
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
    project_primitive_action,
    safest_projected_action,
)


_WORKER_MODEL: ActorCritic | None = None
_WORKER_ENV: PrivilegedRaceCarEnv | None = None
_WORKER_TEMPERATURE = 1.0
_WORKER_X_MARGIN = 0.0
_WORKER_Y_MARGIN = 0.0


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


def _filtered_action(
    state,
    probs: torch.Tensor,
    scheduler_credit: torch.Tensor,
    *,
    horizon: int,
) -> Tuple[int, Dict[str, Any]]:
    """Return executed action plus one-tick filter diagnostics.

    Important scheduler accounting rule: if the scheduler proposes an unsafe
    action, that request is *consumed* from credit even though a safety fallback
    executes instead. The fallback is not debited from scheduler credit. This
    prevents the blocked action from accumulating debt and being proposed again
    forever, which was the pathological loop in the initiation-only shield.
    """
    proposed = _scheduler_proposal(probs, scheduler_credit)

    proposed_projection = project_primitive_action(
        state,
        proposed,
        horizon=horizon,
        x_margin=_WORKER_X_MARGIN,
        y_margin=_WORKER_Y_MARGIN,
    )

    if bool(proposed_projection["safe"]):
        scheduler_credit[proposed] -= 1.0
        return proposed, {
            "proposed": proposed,
            "executed": proposed,
            "overridden": False,
            "all_unsafe": False,
            "proposed_projection": proposed_projection,
            "fallback_projection": None,
            "projected_count": 1,
        }

    # Consume/drop this unsafe scheduler request so it cannot accumulate into a
    # permanent blocked-action debt.
    scheduler_credit[proposed] -= 1.0

    projections: List[Dict[str, Any] | None] = [None] * len(ACTIONS)
    projections[proposed] = proposed_projection

    safe_alternatives: List[int] = []
    for action in range(len(ACTIONS)):
        if action == proposed:
            continue
        projection = project_primitive_action(
            state,
            action,
            horizon=horizon,
            x_margin=_WORKER_X_MARGIN,
            y_margin=_WORKER_Y_MARGIN,
        )
        projections[action] = projection
        if bool(projection["safe"]):
            safe_alternatives.append(action)

    if safe_alternatives:
        # Preserve the teacher as much as possible: among candidates that the
        # short projection deems recoverable, use the highest instantaneous
        # teacher probability. Scheduler credit is intentionally untouched by
        # the substituted safety action.
        executed = int(
            max(
                safe_alternatives,
                key=lambda action: float(probs[action].item()),
            )
        )
        all_unsafe = False
    else:
        all_unsafe = True
        complete = [item for item in projections if item is not None]
        if len(complete) != len(ACTIONS):
            raise RuntimeError("missing primitive projection")
        executed = safest_projected_action(
            complete,  # type: ignore[arg-type]
            preference=[float(x) for x in probs.tolist()],
        )

    fallback_projection = projections[executed]
    return executed, {
        "proposed": proposed,
        "executed": executed,
        "overridden": True,
        "all_unsafe": all_unsafe,
        "proposed_projection": proposed_projection,
        "fallback_projection": fallback_projection,
        "projected_count": len(ACTIONS),
    }


def _run_episode(
    sim_seed: int,
    action_seed: int,
    horizon: int,
) -> Dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")

    model = _WORKER_MODEL
    env = _WORKER_ENV

    # Scheduler is deterministic, but retain a distinct action seed in output so
    # this evaluator lines up with the other teacher evaluation tools.
    _ = int(action_seed)
    scheduler_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)

    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(options={"sim_seed": int(sim_seed)})

    terminated = False
    truncated = False

    proposed_counts = Counter()
    executed_counts = Counter()
    override_counts = Counter()
    proposed_collision_kind = Counter()
    proposed_collision_side = Counter()
    fallback_counts = Counter()

    overrides = 0
    all_unsafe = 0
    projected_candidates = 0
    zero_speed_ticks = 0
    zero_speed_overrides = 0

    while not (terminated or truncated):
        state = core.STATE
        if float(state.ego.velocity.x) <= 1e-9:
            zero_speed_ticks += 1

        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(
                logits.squeeze(0) / float(_WORKER_TEMPERATURE),
                dim=-1,
            )

        if horizon <= 0:
            proposed = _scheduler_proposal(probs, scheduler_credit)
            scheduler_credit[proposed] -= 1.0
            action = proposed
            diagnostic = {
                "proposed": proposed,
                "executed": proposed,
                "overridden": False,
                "all_unsafe": False,
                "proposed_projection": None,
                "fallback_projection": None,
                "projected_count": 0,
            }
        else:
            action, diagnostic = _filtered_action(
                state,
                probs,
                scheduler_credit,
                horizon=int(horizon),
            )

        proposed = int(diagnostic["proposed"])
        proposed_counts[ACTIONS[proposed]] += 1
        executed_counts[ACTIONS[int(action)]] += 1
        projected_candidates += int(diagnostic["projected_count"])

        if bool(diagnostic["overridden"]):
            overrides += 1
            override_counts[ACTIONS[proposed]] += 1
            fallback_counts[ACTIONS[int(action)]] += 1
            if float(state.ego.velocity.x) <= 1e-9:
                zero_speed_overrides += 1

            projection = diagnostic["proposed_projection"] or {}
            proposed_collision_kind[
                str(projection.get("collision_kind", "unknown"))
            ] += 1
            proposed_collision_side[
                str(projection.get("collision_side", "none"))
            ] += 1

        if bool(diagnostic["all_unsafe"]):
            all_unsafe += 1

        obs, _, terminated, truncated, info = env.step(int(action))

    ticks = int(info["ticks"])
    crashed = bool(info["crashed"])
    return {
        "seed": int(sim_seed),
        "horizon": int(horizon),
        "distance": float(info["distance"]),
        "ticks": ticks,
        "crashed": crashed,
        "crash_kind": _crash_kind() if crashed else "none",
        "final_speed": float(info["speed"]),
        "final_lateral_speed": float(info["lateral_speed"]),
        "average_speed": (
            float(info["distance"]) / ticks if ticks > 0 else 0.0
        ),
        "overrides": int(overrides),
        "override_tick_rate": (
            float(overrides / ticks) if ticks > 0 else 0.0
        ),
        "all_unsafe_ticks": int(all_unsafe),
        "all_unsafe_tick_rate": (
            float(all_unsafe / ticks) if ticks > 0 else 0.0
        ),
        "projected_candidates": int(projected_candidates),
        "zero_speed_ticks": int(zero_speed_ticks),
        "zero_speed_tick_rate": (
            float(zero_speed_ticks / ticks) if ticks > 0 else 0.0
        ),
        "zero_speed_overrides": int(zero_speed_overrides),
        "proposed_counts": dict(proposed_counts),
        "executed_counts": dict(executed_counts),
        "override_counts": dict(override_counts),
        "fallback_counts": dict(fallback_counts),
        "proposed_collision_kind": dict(proposed_collision_kind),
        "proposed_collision_side": dict(proposed_collision_side),
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
    speeds = np.asarray(
        [float(row["average_speed"]) for row in rows],
        dtype=np.float64,
    )

    overrides = sum(int(row["overrides"]) for row in rows)
    all_unsafe = sum(int(row["all_unsafe_ticks"]) for row in rows)
    projected = sum(int(row["projected_candidates"]) for row in rows)
    zero_speed = sum(int(row["zero_speed_ticks"]) for row in rows)
    total_ticks = float(ticks.sum())

    crash_kind = Counter()
    override_counts = Counter()
    fallback_counts = Counter()
    collision_kind = Counter()
    collision_side = Counter()
    proposed_counts = Counter()
    executed_counts = Counter()

    for row in rows:
        if bool(row["crashed"]):
            crash_kind[str(row["crash_kind"])] += 1
        override_counts.update(row["override_counts"])
        fallback_counts.update(row["fallback_counts"])
        collision_kind.update(row["proposed_collision_kind"])
        collision_side.update(row["proposed_collision_side"])
        proposed_counts.update(row["proposed_counts"])
        executed_counts.update(row["executed_counts"])

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
        "mean_episode_speed": float(speeds.mean()),
        "overrides": int(overrides),
        "override_tick_rate": (
            float(overrides / total_ticks) if total_ticks > 0 else 0.0
        ),
        "all_unsafe_ticks": int(all_unsafe),
        "all_unsafe_tick_rate": (
            float(all_unsafe / total_ticks) if total_ticks > 0 else 0.0
        ),
        "projected_candidates": int(projected),
        "mean_projections_per_tick": (
            float(projected / total_ticks) if total_ticks > 0 else 0.0
        ),
        "zero_speed_tick_rate": (
            float(zero_speed / total_ticks) if total_ticks > 0 else 0.0
        ),
        "crash_kind": dict(crash_kind),
        "override_counts": dict(override_counts),
        "fallback_counts": dict(fallback_counts),
        "proposed_collision_kind": dict(collision_kind),
        "proposed_collision_side": dict(collision_side),
        "proposed_counts": dict(proposed_counts),
        "executed_counts": dict(executed_counts),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[24],
        help=(
            "Short projection horizons to benchmark. Use e.g. 12 24 36. "
            "Baseline horizon 0 is included automatically."
        ),
    )
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--seed-start", type=int, default=160_000)
    p.add_argument("--action-seed-start", type=int, default=16_000_000)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--x-margin", type=float, default=0.0)
    p.add_argument("--y-margin", type=float, default=0.0)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/teacher_safety_filter_eval.json"),
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

    horizons: List[int] = []
    for horizon in args.horizons:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("all --horizons values must be positive")
        if horizon not in horizons:
            horizons.append(horizon)

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    config = payload.get("config", {})
    crash_penalty = float(config.get("crash_penalty", 250.0))
    reward_scale = float(config.get("reward_scale", 1000.0))

    variants = [0] + horizons
    tasks: List[Tuple[int, int, int]] = []
    for horizon in variants:
        for i in range(args.episodes):
            tasks.append(
                (
                    args.seed_start + i,
                    args.action_seed_start + i,
                    horizon,
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

    results: Dict[str, Dict[str, Any]] = {}
    for horizon in variants:
        variant_rows = sorted(
            [row for row in rows if int(row["horizon"]) == horizon],
            key=lambda row: int(row["seed"]),
        )
        label = "baseline" if horizon == 0 else f"horizon_{horizon}"
        results[label] = _summarize(variant_rows)

    print(
        "\nExact-original per-tick privileged safety filter "
        f"(scheduler T={args.temperature:g}, n={args.episodes}, "
        f"margin=({args.x_margin:g},{args.y_margin:g}))"
    )
    print(
        f"{'variant':<13} {'mean':>10} {'median':>10} {'p10':>10} "
        f"{'crash':>8} {'<25k':>8} {'avg_v':>8} "
        f"{'override':>9} {'allbad':>8}"
    )
    print("-" * 94)

    for label, result in results.items():
        print(
            f"{label:<13} "
            f"{result['mean_distance']:10,.0f} "
            f"{result['median_distance']:10,.0f} "
            f"{result['p10_distance']:10,.0f} "
            f"{result['crash_rate']:8.1%} "
            f"{result['below_25k_rate']:8.1%} "
            f"{result['mean_episode_speed']:8.1f} "
            f"{result['override_tick_rate']:9.2%} "
            f"{result['all_unsafe_tick_rate']:8.2%}"
        )

    baseline = results["baseline"]
    print("\nDeltas vs baseline")
    for horizon in horizons:
        label = f"horizon_{horizon}"
        result = results[label]
        print(
            f"{label}: "
            f"crash={float(result['crash_rate']) - float(baseline['crash_rate']):+.2%}, "
            f"mean={float(result['mean_distance']) - float(baseline['mean_distance']):+,.0f}, "
            f"p10={float(result['p10_distance']) - float(baseline['p10_distance']):+,.0f}, "
            f"overrides={result['overrides']:,}, "
            f"override_actions={result['override_counts']}, "
            f"fallbacks={result['fallback_counts']}"
        )

    output = {
        "checkpoint": str(args.checkpoint),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed_start),
        "temperature": float(args.temperature),
        "x_margin": float(args.x_margin),
        "y_margin": float(args.y_margin),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
