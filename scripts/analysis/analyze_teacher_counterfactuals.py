"""Offline causal replay of privileged-teacher crash seeds.

This script starts from crashes already identified by analyze_teacher_crashes.py,
replays each exact simulator seed with the same T=1 scheduler teacher, and then
changes one mechanism only inside a short window around the baseline fatal event.

The intervention start tick is taken from the *baseline failed trajectory*
(fatal lane-change start when available, otherwise TTC<=120 onset, otherwise
120 ticks before crash). Therefore this is an offline causal diagnostic, not a
deployable controller: the trigger itself comes from knowledge of the baseline
failure.

Interventions:
  freeze_noise      Consume opponent RNG draws but force velocity perturbation
                    to zero during the window.
  no_spawns         Suppress recycled-car placement during the window while
                    still consuming the normal placement RNG draws.
  speed_floor       Replace DECELERATE with NOTHING when it would cross a
                    configurable vx floor.
  lateral_sign      Preserve the teacher's steering *opportunities* but replace
                    steering direction with target-lane feedback for the same
                    baseline target lane.
  lateral_full      Strong execution test: drive toward the same baseline target
                    lane every tick until settled; this overrides longitudinal
                    actions while active.
  safer_target      At intervention start, choose the safest of settle/left/right
                    under a privileged frozen-vx maneuver projection, then remap
                    only teacher steering opportunities toward that chosen lane.

Baseline replay is included automatically and checked against the source crash
tick so any simulator mismatch is visible.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

import src.game.core as core
import src.mathematics.randomizer as randomizer
from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, OBS_SIZE
from training.teacher_env import (
    PrivilegedRaceCarEnv,
    build_privileged_observation,
)
from training.teacher_lane_safety import (
    DECELERATE,
    NOTHING,
    STEER_LEFT,
    STEER_RIGHT,
    lane_centers,
)
from training.teacher_mpc import (
    lane_settled,
    project_maneuver,
    steer_toward_lane_action,
)


VARIANTS = (
    "baseline",
    "freeze_noise",
    "no_spawns",
    "speed_floor",
    "lateral_sign",
    "lateral_full",
    "safer_target",
)

_WORKER_MODEL: ActorCritic | None = None
_WORKER_ENV: PrivilegedRaceCarEnv | None = None
_WORKER_TEMPERATURE = 1.0
_WORKER_WINDOW = 160
_WORKER_SPEED_FLOOR = 5.0
_WORKER_LANE_HORIZON = 96


def _scenario_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    maneuver = row.get("fatal_maneuver") or {}
    onset = row.get("risk_onset_120") or {}

    trigger_source = "fatal_maneuver"
    trigger_tick = maneuver.get("start_tick")
    if trigger_tick is None:
        trigger_source = "ttc120"
        trigger_tick = onset.get("tick")
    if trigger_tick is None:
        trigger_source = "crash_minus_120"
        trigger_tick = max(0, int(row["ticks"]) - 120)

    target_lane = maneuver.get("target_lane")
    source_lane = maneuver.get("source_lane")

    return {
        "seed": int(row["seed"]),
        "source_crash_tick": int(row["ticks"]),
        "source_distance": float(row["distance"]),
        "source_category": str(row.get("category", "unknown")),
        "trigger_tick": int(trigger_tick),
        "trigger_source": trigger_source,
        "baseline_source_lane": (
            None if source_lane is None else int(source_lane)
        ),
        "baseline_target_lane": (
            None if target_lane is None else int(target_lane)
        ),
    }


def _scheduler_action(
    model: ActorCritic,
    obs: np.ndarray,
    scheduler_credit: torch.Tensor,
) -> int:
    obs_t = torch.from_numpy(obs).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs_t)
        probs = torch.softmax(
            logits.squeeze(0) / float(_WORKER_TEMPERATURE),
            dim=-1,
        )
    scheduler_credit.add_(probs)
    action = int(scheduler_credit.argmax().item())
    scheduler_credit[action] -= 1.0
    return action


def _update_cars_exact(
    state,
    *,
    freeze_noise: bool,
) -> None:
    """Match core.update_cars while optionally zeroing opponent vx perturbation.

    Even while frozen, consume the exact random_number draw each active opponent
    would normally consume so the underlying RNG stream stays aligned.
    """
    ego = state.ego
    for car in state.cars:
        if car is ego:
            car.y += car.velocity.y
            continue

        car.x += car.velocity.x - ego.velocity.x
        car.y += car.velocity.y

        rn = randomizer.random_number()
        if not freeze_noise:
            car.velocity.x += (rn - 0.5) / 5.0


def _place_or_consume(
    state,
    *,
    suppress_spawn: bool,
) -> None:
    if not suppress_spawn:
        core.place_car()
        return

    # Match place_car's RNG consumption/order without recycling a bucket car.
    if len(state.cars) > core.LANE_COUNT:
        return

    open_lanes = [
        lane
        for lane in state.road.lanes
        if not any(
            car.lane == lane
            for car in state.cars
            if car is not state.ego
        )
    ]
    randomizer.random_choice(open_lanes)
    randomizer.random_choice([-0.5, 1.5])
    randomizer.random_number()


def _collision_kind(state) -> str:
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


def _step_counterfactual(
    action: int,
    *,
    freeze_noise: bool,
    suppress_spawn: bool,
) -> Tuple[np.ndarray, bool, bool, Dict[str, Any]]:
    state = core.STATE
    action_name = ACTIONS[int(action)]

    state.ticks += 1
    core.handle_action(action_name)
    state.latest_action = action_name
    state.distance += state.ego.velocity.x

    _update_cars_exact(state, freeze_noise=freeze_noise)
    core.remove_passed_cars()
    _place_or_consume(state, suppress_spawn=suppress_spawn)

    for car in state.cars:
        if car is not state.ego and core.intersects(state.ego.rect, car.rect):
            state.crashed = True
            break

    if not state.crashed:
        for wall in state.road.walls:
            if core.intersects(state.ego.rect, wall.rect):
                state.crashed = True
                break

    terminated = bool(state.crashed)
    truncated = bool(not state.crashed and state.ticks >= core.MAX_TICKS)
    info = {
        "distance": float(state.distance),
        "ticks": int(state.ticks),
        "crashed": bool(state.crashed),
        "speed": float(state.ego.velocity.x),
        "lateral_speed": float(state.ego.velocity.y),
    }
    return (
        build_privileged_observation(state),
        terminated,
        truncated,
        info,
    )


def _nearest_lane(state) -> int:
    centers = lane_centers(state)
    ego = state.ego
    center_y = float(ego.y) + float(ego.sprite.get_height()) * 0.5
    return int(
        min(
            range(len(centers)),
            key=lambda i: abs(center_y - centers[i]),
        )
    )


def _choose_safer_target(
    state,
    proposed_action: int,
) -> Tuple[int, Dict[str, Any]]:
    """Choose among settle/left/right using only exact current state."""
    current_lane = _nearest_lane(state)
    candidates: List[Tuple[int, Dict[str, Any]]] = []

    for plan, target_lane in (
        ("settle", current_lane),
        ("left", current_lane - 1),
        ("right", current_lane + 1),
    ):
        if not 0 <= target_lane < len(state.road.lanes):
            continue
        projection = project_maneuver(
            state,
            plan,
            proposed_action=int(proposed_action),
            horizon=int(_WORKER_LANE_HORIZON),
            burst_ticks=12,
            x_margin=0.0,
            y_margin=0.0,
        )
        candidates.append((int(target_lane), projection))

    def rank(item: Tuple[int, Dict[str, Any]]):
        target_lane, projection = item
        collision_tick = projection.get("collision_tick")
        survived = (
            int(_WORKER_LANE_HORIZON) + 1
            if collision_tick is None
            else int(collision_tick)
        )
        clearance = min(
            float(projection["min_car_clearance"]),
            float(projection["min_wall_clearance"]),
        )
        # Safe/later collision dominates. Then prefer clearance and progress;
        # use smaller residual lateral speed as a final tie-break.
        return (
            int(bool(projection["safe"])),
            survived,
            clearance,
            float(projection["progress"]),
            -abs(float(projection["final_vy"])),
            int(target_lane != current_lane),
        )

    target_lane, projection = max(candidates, key=rank)
    return int(target_lane), {
        "chosen_target_lane": int(target_lane),
        "current_lane": int(current_lane),
        "chosen_plan": str(projection["plan"]),
        "projection_safe": bool(projection["safe"]),
        "projection_collision_tick": projection["collision_tick"],
        "projection_clearance": float(
            min(
                projection["min_car_clearance"],
                projection["min_wall_clearance"],
            )
        ),
    }


def _run_counterfactual(
    scenario: Dict[str, Any],
    variant: str,
) -> Dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")

    model = _WORKER_MODEL
    env = _WORKER_ENV

    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(options={"sim_seed": int(scenario["seed"])})

    scheduler_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)
    terminated = False
    truncated = False

    target_lane = scenario.get("baseline_target_lane")
    chosen_target_meta: Dict[str, Any] | None = None
    intervention_actions = 0
    intervention_ticks = 0
    intervention_started = False
    intervention_eligible = True

    if variant in ("lateral_sign", "lateral_full") and target_lane is None:
        intervention_eligible = False

    while not (terminated or truncated):
        state = core.STATE
        proposed = _scheduler_action(model, obs, scheduler_credit)

        tick = int(state.ticks)
        active = bool(
            tick >= int(scenario["trigger_tick"])
            and tick < int(scenario["trigger_tick"]) + int(_WORKER_WINDOW)
        )

        if active:
            intervention_started = True
            intervention_ticks += 1

        if (
            variant == "safer_target"
            and active
            and chosen_target_meta is None
        ):
            target_lane, chosen_target_meta = _choose_safer_target(
                state,
                proposed,
            )

        action = int(proposed)
        freeze_noise = bool(active and variant == "freeze_noise")
        suppress_spawn = bool(active and variant == "no_spawns")

        if active and variant == "speed_floor":
            vx = float(state.ego.velocity.x)
            if (
                proposed == DECELERATE
                and vx - 0.1 < float(_WORKER_SPEED_FLOOR) - 1e-9
            ):
                action = NOTHING

        elif (
            active
            and variant in ("lateral_sign", "safer_target")
            and target_lane is not None
            and proposed in (STEER_LEFT, STEER_RIGHT)
        ):
            action = steer_toward_lane_action(state, int(target_lane))

        elif (
            active
            and variant == "lateral_full"
            and target_lane is not None
            and not lane_settled(state, int(target_lane))
        ):
            action = steer_toward_lane_action(state, int(target_lane))

        if action != proposed:
            intervention_actions += 1

        obs, terminated, truncated, info = _step_counterfactual(
            action,
            freeze_noise=freeze_noise,
            suppress_spawn=suppress_spawn,
        )

    state = core.STATE
    crashed = bool(info["crashed"])
    return {
        **scenario,
        "variant": variant,
        "crashed": crashed,
        "ticks": int(info["ticks"]),
        "distance": float(info["distance"]),
        "final_speed": float(info["speed"]),
        "final_lateral_speed": float(info["lateral_speed"]),
        "crash_kind": _collision_kind(state) if crashed else "none",
        "intervention_started": bool(intervention_started),
        "intervention_eligible": bool(intervention_eligible),
        "intervention_ticks": int(intervention_ticks),
        "intervention_actions": int(intervention_actions),
        "chosen_target": chosen_target_meta,
    }


def _worker_init(
    model_state: Dict[str, torch.Tensor],
    crash_penalty: float,
    reward_scale: float,
    temperature: float,
    window: int,
    speed_floor: float,
    lane_horizon: int,
) -> None:
    global _WORKER_MODEL, _WORKER_ENV, _WORKER_TEMPERATURE
    global _WORKER_WINDOW, _WORKER_SPEED_FLOOR, _WORKER_LANE_HORIZON

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
    _WORKER_WINDOW = int(window)
    _WORKER_SPEED_FLOOR = float(speed_floor)
    _WORKER_LANE_HORIZON = int(lane_horizon)


def _aggregate(
    scenarios: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    variants: List[str],
) -> Dict[str, Any]:
    by_variant: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for variant in variants:
        by_variant[variant] = {
            int(row["seed"]): row
            for row in rows
            if row["variant"] == variant
        }

    baseline = by_variant["baseline"]
    source_by_seed = {int(row["seed"]): row for row in scenarios}

    exact_tick_matches = 0
    baseline_crashes = 0
    for seed, row in baseline.items():
        source = source_by_seed[seed]
        baseline_crashes += int(bool(row["crashed"]))
        exact_tick_matches += int(
            bool(row["crashed"])
            and int(row["ticks"]) == int(source["source_crash_tick"])
        )

    summary: Dict[str, Any] = {
        "source_crashes": len(scenarios),
        "baseline_replay_crashes": int(baseline_crashes),
        "baseline_exact_crash_tick_matches": int(exact_tick_matches),
        "baseline_exact_crash_tick_match_share": (
            exact_tick_matches / max(1, len(scenarios))
        ),
        "variants": {},
    }

    fix_sets = Counter()

    for seed in source_by_seed:
        fixed_by = []
        for variant in variants:
            if variant == "baseline":
                continue
            row = by_variant[variant][seed]
            if baseline[seed]["crashed"] and not row["crashed"]:
                fixed_by.append(variant)
        fix_sets["+".join(sorted(fixed_by)) if fixed_by else "none"] += 1

    summary["fix_sets"] = dict(fix_sets)

    for variant in variants:
        variant_rows = list(by_variant[variant].values())
        fixed = []
        earlier = 0
        later_crash_deltas: List[float] = []
        distance_gains: List[float] = []
        eligible = 0
        started = 0
        action_changes = 0

        by_category: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "fixed": 0}
        )
        by_trigger: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "fixed": 0}
        )

        for row in variant_rows:
            seed = int(row["seed"])
            b = baseline[seed]
            source = source_by_seed[seed]
            category = str(source["source_category"])
            trigger = str(source["trigger_source"])

            by_category[category]["total"] += 1
            by_trigger[trigger]["total"] += 1
            eligible += int(bool(row["intervention_eligible"]))
            started += int(bool(row["intervention_started"]))
            action_changes += int(row["intervention_actions"])
            distance_gains.append(float(row["distance"]) - float(b["distance"]))

            is_fixed = bool(b["crashed"] and not row["crashed"])
            if is_fixed:
                fixed.append(seed)
                by_category[category]["fixed"] += 1
                by_trigger[trigger]["fixed"] += 1
            elif b["crashed"] and row["crashed"]:
                delta = float(row["ticks"]) - float(b["ticks"])
                later_crash_deltas.append(delta)
                earlier += int(delta < 0)

        summary["variants"][variant] = {
            "episodes": len(variant_rows),
            "fixed": len(fixed),
            "fixed_share": len(fixed) / max(1, len(variant_rows)),
            "fixed_seeds": sorted(fixed),
            "crashed_earlier": int(earlier),
            "mean_distance_gain_vs_baseline": float(
                np.mean(distance_gains) if distance_gains else 0.0
            ),
            "median_crash_tick_delta_if_still_crashed": float(
                np.median(later_crash_deltas)
                if later_crash_deltas
                else 0.0
            ),
            "eligible_share": eligible / max(1, len(variant_rows)),
            "intervention_started_share": started / max(1, len(variant_rows)),
            "action_changes": int(action_changes),
            "by_category": dict(by_category),
            "by_trigger_source": dict(by_trigger),
        }

    return summary


def _print_summary(summary: Dict[str, Any], variants: List[str]) -> None:
    print("\nTeacher crash causal replay")
    print(
        f"source_crashes={summary['source_crashes']} "
        f"baseline_replay_crashes={summary['baseline_replay_crashes']} "
        f"exact_tick_match="
        f"{summary['baseline_exact_crash_tick_matches']}/"
        f"{summary['source_crashes']} "
        f"({summary['baseline_exact_crash_tick_match_share']:.1%})"
    )

    print("\nCounterfactual rescue rates")
    print(
        f"{'variant':<18} {'fixed':>8} {'share':>8} "
        f"{'earlier':>8} {'dist_gain':>11} {'tick_delta':>11} "
        f"{'actions':>10}"
    )
    print("-" * 82)
    for variant in variants:
        row = summary["variants"][variant]
        print(
            f"{variant:<18} "
            f"{row['fixed']:>8d} "
            f"{row['fixed_share']:>7.1%} "
            f"{row['crashed_earlier']:>8d} "
            f"{row['mean_distance_gain_vs_baseline']:>11,.0f} "
            f"{row['median_crash_tick_delta_if_still_crashed']:>11.1f} "
            f"{row['action_changes']:>10,d}"
        )

    print("\nFix-set overlap")
    for key, count in sorted(
        summary["fix_sets"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{key:<60} {count:>5d}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--crash-analysis",
        type=Path,
        default=Path("models/teacher_crash_analysis_v2.json"),
        help="Full JSON emitted by analyze_teacher_crashes.py.",
    )
    p.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS[1:],
        default=list(VARIANTS[1:]),
        help="Counterfactuals to run; baseline is always included.",
    )
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--window",
        type=int,
        default=160,
        help="Ticks after the baseline trigger during which intervention is active.",
    )
    p.add_argument("--speed-floor", type=float, default=5.0)
    p.add_argument("--lane-horizon", type=int, default=96)
    p.add_argument(
        "--max-crashes",
        type=int,
        default=0,
        help="Optional deterministic prefix limit for a quick smoke run.",
    )
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("models/teacher_counterfactuals"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.window <= 0:
        raise ValueError("--window must be positive")
    if args.speed_floor < 0:
        raise ValueError("--speed-floor must be non-negative")
    if args.lane_horizon <= 0:
        raise ValueError("--lane-horizon must be positive")

    source_rows = json.loads(args.crash_analysis.read_text(encoding="utf-8"))
    crash_rows = [row for row in source_rows if bool(row.get("crashed", False))]
    crash_rows.sort(key=lambda row: int(row["seed"]))
    if args.max_crashes > 0:
        crash_rows = crash_rows[: int(args.max_crashes)]
    if not crash_rows:
        raise ValueError("crash-analysis JSON contains no crashes")

    scenarios = [_scenario_from_row(row) for row in crash_rows]

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

    variants = ["baseline"] + list(dict.fromkeys(args.variants))
    tasks: List[Tuple[Dict[str, Any], str]] = [
        (scenario, variant)
        for variant in variants
        for scenario in scenarios
    ]

    workers = max(1, min(int(args.workers), len(tasks)))
    if workers == 1:
        _worker_init(
            model_state,
            crash_penalty,
            reward_scale,
            args.temperature,
            args.window,
            args.speed_floor,
            args.lane_horizon,
        )
        rows = [_run_counterfactual(*task) for task in tasks]
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
                args.window,
                args.speed_floor,
                args.lane_horizon,
            ),
        ) as executor:
            rows = list(
                executor.map(
                    _run_counterfactual,
                    [task[0] for task in tasks],
                    [task[1] for task in tasks],
                    chunksize=1,
                )
            )

    rows.sort(key=lambda row: (str(row["variant"]), int(row["seed"])))
    summary = _aggregate(scenarios, rows, variants)
    _print_summary(summary, variants)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.parent / f"{prefix.name}.json"
    summary_path = prefix.parent / f"{prefix.name}_summary.json"
    csv_path = prefix.parent / f"{prefix.name}.csv"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    flat_rows = []
    for row in rows:
        chosen = row.get("chosen_target") or {}
        flat_rows.append(
            {
                "seed": row["seed"],
                "variant": row["variant"],
                "category": row["source_category"],
                "trigger_tick": row["trigger_tick"],
                "trigger_source": row["trigger_source"],
                "baseline_target_lane": row["baseline_target_lane"],
                "crashed": row["crashed"],
                "ticks": row["ticks"],
                "distance": row["distance"],
                "crash_kind": row["crash_kind"],
                "intervention_eligible": row["intervention_eligible"],
                "intervention_ticks": row["intervention_ticks"],
                "intervention_actions": row["intervention_actions"],
                "chosen_target_lane": chosen.get("chosen_target_lane"),
                "chosen_plan": chosen.get("chosen_plan"),
                "chosen_projection_safe": chosen.get("projection_safe"),
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
