"""Deterministic tick-level replay for privileged teacher MPC failures."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import torch

import src.game.core as core
from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, OBS_SIZE
from training.teacher_env import PrivilegedRaceCarEnv
from training.teacher_lane_safety import lane_centers, lane_index
from training.teacher_mpc import (
    choose_confirmed_mpc_action,
    mpc_planning_decision,
)


def _scheduler_proposal(probs: torch.Tensor, credit: torch.Tensor) -> int:
    credit.add_(probs)
    return int(credit.argmax().item())


def _run_length_actions(actions: Iterable[int]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for action in actions:
        name = ACTIONS[int(action)]
        if runs and runs[-1]["action"] == name:
            runs[-1]["ticks"] += 1
        else:
            runs.append({"action": name, "ticks": 1})
    return runs


def _projection_summary(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "plan": str(item["plan"]),
        "first_action": str(item["first_action_name"]),
        "safe": bool(item["safe"]),
        "healthy": bool(item.get("healthy", False)),
        "collision_tick": item["collision_tick"],
        "collision_kind": item["collision_kind"],
        "collision_side": item["collision_side"],
        "min_car_clearance": float(item["min_car_clearance"]),
        "min_wall_clearance": float(item["min_wall_clearance"]),
        "progress": float(item["progress"]),
        "zero_speed_ticks": int(item["zero_speed_ticks"]),
        "final_vx": float(item["final_vx"]),
        "final_vy": float(item["final_vy"]),
        "final_lane_offset": float(item["final_lane_offset"]),
        "action_runs": _run_length_actions(item["actions"]),
    }


def _state_snapshot(state) -> Dict[str, Any]:
    ego = state.ego
    ego_cx = float(ego.x) + float(ego.sprite.get_width()) * 0.5
    ego_cy = float(ego.y) + float(ego.sprite.get_height()) * 0.5
    centers = lane_centers(state)
    nearest = min(range(len(centers)), key=lambda i: abs(ego_cy - centers[i]))
    traffic = []
    for car in state.cars:
        if car is ego:
            continue
        car_cx = float(car.x) + float(car.sprite.get_width()) * 0.5
        car_cy = float(car.y) + float(car.sprite.get_height()) * 0.5
        traffic.append(
            {
                "lane": int(lane_index(state, car.lane)),
                "x": float(car.x),
                "y": float(car.y),
                "vx": float(car.velocity.x),
                "rel_x": car_cx - ego_cx,
                "rel_y": car_cy - ego_cy,
            }
        )
    return {
        "tick": int(state.ticks),
        "distance": float(state.distance),
        "ego_x": float(ego.x),
        "ego_y": float(ego.y),
        "ego_vx": float(ego.velocity.x),
        "ego_vy": float(ego.velocity.y),
        "ego_center_y": ego_cy,
        "nearest_lane": int(nearest),
        "lane_offset": float(ego_cy - centers[nearest]),
        "traffic": traffic,
    }


def _collision_snapshot(state) -> Dict[str, Any]:
    ego = state.ego
    cars = []
    for car in state.cars:
        if car is ego or not core.intersects(ego.rect, car.rect):
            continue
        cars.append(
            {
                "lane": int(lane_index(state, car.lane)),
                "rel_x": float(car.rect.centerx - ego.rect.centerx),
                "rel_y": float(car.rect.centery - ego.rect.centery),
                "other_vx": float(car.velocity.x),
            }
        )
    walls = [
        int(i)
        for i, wall in enumerate(state.road.walls)
        if core.intersects(ego.rect, wall.rect)
    ]
    return {"cars": cars, "walls": walls}


def _run_seed(
    model: ActorCritic,
    env: PrivilegedRaceCarEnv,
    seed: int,
    *,
    horizon: int,
    burst_ticks: int,
    temperature: float,
    uncertainty_sigma: float,
    trigger_immediate: float,
    history_ticks: int,
    use_mpc: bool,
    strategic_horizon: int,
    strategic_trigger_limit: int,
) -> Dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(options={"sim_seed": int(seed)})
    credit = torch.zeros(len(ACTIONS), dtype=torch.float32)
    history = deque(maxlen=int(history_ticks))
    trigger_counts = Counter()
    selected_counts = Counter()
    overrides = 0
    strategic_triggers = 0
    strategic_candidates = 0

    terminated = truncated = False
    while not (terminated or truncated):
        state = core.STATE
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(logits.squeeze(0) / float(temperature), dim=-1)

        proposed = _scheduler_proposal(probs, credit)
        reason = None
        diagnostic = None
        executed = proposed
        if use_mpc:
            reason, planning_horizon = mpc_planning_decision(
                state,
                proposed_action=proposed,
                horizon=horizon,
                strategic_horizon=(
                    strategic_horizon
                    if strategic_trigger_limit <= 0
                    or strategic_triggers < strategic_trigger_limit
                    else 0
                ),
                trigger_immediate=trigger_immediate,
                opponent_uncertainty_sigma=uncertainty_sigma,
            )
            if reason is not None:
                strategic_candidates += int(planning_horizon > horizon)
                executed, diagnostic, strategic_used = choose_confirmed_mpc_action(
                    state,
                    proposed,
                    horizon=horizon,
                    planning_horizon=planning_horizon,
                    burst_ticks=burst_ticks,
                    opponent_uncertainty_sigma=uncertainty_sigma,
                )
                if reason == "strategic_closing" and not strategic_used:
                    reason = "lateral"
                if reason == "strategic_closing":
                    strategic_triggers += 1
                trigger_counts[str(reason)] += 1
                selected_counts[str(diagnostic["selected_plan"])] += 1
                overrides += int(bool(diagnostic["overridden"]))

        # Consume the teacher proposal even if the supervisor substitutes it.
        credit[proposed] -= 1.0
        frame = {
            **_state_snapshot(state),
            "proposed_action": ACTIONS[proposed],
            "executed_action": ACTIONS[int(executed)],
            "overridden": int(executed) != proposed,
            "trigger_reason": reason,
            "policy_probs": [float(value) for value in probs.tolist()],
            "scheduler_credit": [float(value) for value in credit.tolist()],
            "selected_plan": (
                None if diagnostic is None else str(diagnostic["selected_plan"])
            ),
            "all_unsafe": (
                False if diagnostic is None else bool(diagnostic["all_unsafe"])
            ),
            "all_unhealthy": (
                False if diagnostic is None else bool(diagnostic["all_unhealthy"])
            ),
            "projections": (
                []
                if diagnostic is None
                else [
                    _projection_summary(item)
                    for item in diagnostic.get(
                        "projections", [diagnostic["selected_projection"]]
                    )
                ]
            ),
        }
        history.append(frame)
        obs, _, terminated, truncated, info = env.step(int(executed))

    return {
        "seed": int(seed),
        "mode": "mpc" if use_mpc else "baseline",
        "distance": float(info["distance"]),
        "ticks": int(info["ticks"]),
        "crashed": bool(info["crashed"]),
        "final_speed": float(info["speed"]),
        "final_lateral_speed": float(info["lateral_speed"]),
        "trigger_counts": dict(trigger_counts),
        "selected_plans": dict(selected_counts),
        "overrides": int(overrides),
        "strategic_triggers": int(strategic_triggers),
        "strategic_candidates": int(strategic_candidates),
        "collision": _collision_snapshot(core.STATE),
        "history": list(history),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[300397, 300417])
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--burst-ticks", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opponent-uncertainty-sigma", type=float, default=3.0)
    parser.add_argument("--trigger-immediate", type=float, default=30.0)
    parser.add_argument("--strategic-horizon", type=int, default=119)
    parser.add_argument("--strategic-trigger-limit", type=int, default=1)
    parser.add_argument("--history-ticks", type=int, default=240)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/teacher_mpc_failure_replays.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0 or args.burst_ticks <= 0 or args.history_ticks <= 0:
        raise ValueError("horizon, burst ticks, and history ticks must be positive")
    if args.strategic_horizon < 0:
        raise ValueError("strategic horizon must be non-negative")
    if args.strategic_trigger_limit < 0:
        raise ValueError("strategic trigger limit must be non-negative")
    if args.temperature <= 0 or args.opponent_uncertainty_sigma < 0:
        raise ValueError("temperature must be positive and sigma non-negative")
    if args.trigger_immediate < 0:
        raise ValueError("trigger immediate must be non-negative")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False).cpu()
    model.load_state_dict(payload["model_state"])
    model.eval()
    config = payload.get("config", {})
    env = PrivilegedRaceCarEnv(
        base_seed=0,
        crash_penalty=float(config.get("crash_penalty", 250.0)),
        reward_scale=float(config.get("reward_scale", 1000.0)),
    )

    rows = []
    try:
        for seed in args.seeds:
            if not args.skip_baseline:
                rows.append(
                    _run_seed(
                        model,
                        env,
                        seed,
                        horizon=args.horizon,
                        burst_ticks=args.burst_ticks,
                        temperature=args.temperature,
                        uncertainty_sigma=args.opponent_uncertainty_sigma,
                        trigger_immediate=args.trigger_immediate,
                        history_ticks=args.history_ticks,
                        use_mpc=False,
                        strategic_horizon=args.strategic_horizon,
                        strategic_trigger_limit=args.strategic_trigger_limit,
                    )
                )
            rows.append(
                _run_seed(
                    model,
                    env,
                    seed,
                    horizon=args.horizon,
                    burst_ticks=args.burst_ticks,
                    temperature=args.temperature,
                    uncertainty_sigma=args.opponent_uncertainty_sigma,
                    trigger_immediate=args.trigger_immediate,
                    history_ticks=args.history_ticks,
                    use_mpc=True,
                    strategic_horizon=args.strategic_horizon,
                    strategic_trigger_limit=args.strategic_trigger_limit,
                )
            )
    finally:
        env.close()

    output = {
        "checkpoint": str(args.checkpoint),
        "horizon": int(args.horizon),
        "burst_ticks": int(args.burst_ticks),
        "temperature": float(args.temperature),
        "opponent_uncertainty_sigma": float(args.opponent_uncertainty_sigma),
        "trigger_immediate": float(args.trigger_immediate),
        "strategic_horizon": int(args.strategic_horizon),
        "strategic_trigger_limit": int(args.strategic_trigger_limit),
        "history_ticks": int(args.history_ticks),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"seed={row['seed']} mode={row['mode']} crashed={row['crashed']} "
            f"ticks={row['ticks']} distance={row['distance']:.1f} "
            f"overrides={row['overrides']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
