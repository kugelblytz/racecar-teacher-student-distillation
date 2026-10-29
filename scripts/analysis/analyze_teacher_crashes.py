"""Exact-original privileged-teacher crash forensics.

Runs the teacher on many fixed seeds and records the final history window before
each crash. The goal is to distinguish late front-car reactions, rear hits,
lane-change collisions, and wall/lateral-control failures before changing the
reward or architecture again.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import multiprocessing as mp
import os
from collections import Counter, deque
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
    ego_lane_geometry as projected_ego_lane_geometry,
    is_lane_change_initiation,
    project_lane_change,
)


_WORKER_MODEL: ActorCritic | None = None
_WORKER_ENV: PrivilegedRaceCarEnv | None = None
_WORKER_HISTORY = 180
_WORKER_TEMPERATURE = 1.0


def _lane_index(state, lane) -> int:
    if lane is None:
        return -1
    for i, candidate in enumerate(state.road.lanes):
        if candidate is lane:
            return i
    return -1


def _ego_lane_geometry(state) -> Tuple[int, float, float]:
    ego_cy = float(state.ego.rect.centery)
    centers = [
        (float(lane.y_start) + float(lane.y_end)) * 0.5
        for lane in state.road.lanes
    ]
    nearest = min(range(len(centers)), key=lambda i: abs(ego_cy - centers[i]))
    return nearest, ego_cy - centers[nearest], ego_cy


def _traffic_snapshot(state) -> Dict[str, Any]:
    ego = state.ego
    ego_cx = float(ego.rect.centerx)
    ego_left = float(ego.rect.left)
    ego_right = float(ego.rect.right)
    ego_top = float(ego.rect.top)
    ego_bottom = float(ego.rect.bottom)
    ego_vx = float(ego.velocity.x)

    lane_rows: List[Dict[str, Any] | None] = [None] * len(state.road.lanes)
    min_front_ttc = float("inf")
    min_rear_ttc = float("inf")
    min_any_ttc = float("inf")
    min_ttc_side = "none"
    min_ttc_lane = -1

    for car in state.cars:
        if car is ego:
            continue

        lane = _lane_index(state, car.lane)
        rel_x = float(car.rect.centerx) - ego_cx
        rel_vx = float(car.velocity.x) - ego_vx

        lateral_overlap = not (
            float(car.rect.bottom) <= ego_top
            or float(car.rect.top) >= ego_bottom
        )

        if rel_x >= 0:
            gap = float(car.rect.left) - ego_right
            closing = ego_vx - float(car.velocity.x)
            side = "front"
        else:
            gap = ego_left - float(car.rect.right)
            closing = float(car.velocity.x) - ego_vx
            side = "rear"

        ttc = float("inf")
        if closing > 1e-9:
            ttc = max(0.0, gap) / closing

        row = {
            "lane": lane,
            "rel_x": rel_x,
            "rel_vx": rel_vx,
            "other_vx": float(car.velocity.x),
            "gap": gap,
            "closing": max(0.0, closing),
            "ttc": ttc,
            "lateral_overlap": lateral_overlap,
        }
        if 0 <= lane < len(lane_rows):
            lane_rows[lane] = row

        if lateral_overlap and closing > 1e-9:
            if side == "front":
                min_front_ttc = min(min_front_ttc, ttc)
            else:
                min_rear_ttc = min(min_rear_ttc, ttc)
            if ttc < min_any_ttc:
                min_any_ttc = ttc
                min_ttc_side = side
                min_ttc_lane = lane

    return {
        "lanes": lane_rows,
        "min_front_ttc": min_front_ttc,
        "min_rear_ttc": min_rear_ttc,
        "min_any_ttc": min_any_ttc,
        "min_ttc_side": min_ttc_side,
        "min_ttc_lane": min_ttc_lane,
    }


def _frame(state, action: int, probs: torch.Tensor) -> Dict[str, Any]:
    lane, lane_offset, ego_cy = _ego_lane_geometry(state)
    traffic = _traffic_snapshot(state)
    probs_list = [float(x) for x in probs.tolist()]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs_list)

    return {
        "tick": int(state.ticks),
        "distance": float(state.distance),
        "action": int(action),
        "action_name": ACTIONS[int(action)],
        "policy_argmax": int(np.argmax(probs_list)),
        "policy_argmax_name": ACTIONS[int(np.argmax(probs_list))],
        "policy_probs": probs_list,
        "policy_entropy": entropy,
        "ego_vx": float(state.ego.velocity.x),
        "ego_vy": float(state.ego.velocity.y),
        "ego_cy": ego_cy,
        "nearest_lane": lane,
        "lane_offset": lane_offset,
        **traffic,
    }


def _colliders(state) -> List[Dict[str, Any]]:
    ego = state.ego
    rows = []
    for car in state.cars:
        if car is ego:
            continue
        if core.intersects(ego.rect, car.rect):
            rows.append(
                {
                    "lane": _lane_index(state, car.lane),
                    "rel_x": float(car.rect.centerx) - float(ego.rect.centerx),
                    "rel_y": float(car.rect.centery) - float(ego.rect.centery),
                    "other_vx": float(car.velocity.x),
                    "rel_vx": float(car.velocity.x) - float(ego.velocity.x),
                }
            )
    return rows


def _wall_hit(state) -> bool:
    return any(core.intersects(state.ego.rect, wall.rect) for wall in state.road.walls)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _classify_crash(
    history: List[Dict[str, Any]],
    colliders: List[Dict[str, Any]],
    wall_hit: bool,
) -> str:
    if wall_hit and not colliders:
        return "wall"
    if not colliders:
        return "unknown"

    last = history[-1]
    collider = min(colliders, key=lambda row: abs(float(row["rel_x"])))
    direction = "front" if float(collider["rel_x"]) >= 0 else "rear"

    recent = history[-96:]
    steering_recent = sum(
        frame["action"] in (3, 4) for frame in recent
    )
    moving_laterally = abs(float(last["ego_vy"])) >= 0.5
    between_lanes = abs(float(last["lane_offset"])) >= 55.0
    collider_other_lane = (
        int(collider["lane"]) >= 0
        and int(collider["lane"]) != int(last["nearest_lane"])
    )

    if moving_laterally or between_lanes or collider_other_lane or steering_recent >= 8:
        return f"lane_change_{direction}"
    return f"same_lane_{direction}"


def _risk_onset(history: List[Dict[str, Any]], threshold: float) -> Dict[str, Any] | None:
    if not history:
        return None
    crash_tick = int(history[-1]["tick"]) + 1
    for frame in history:
        ttc = float(frame["min_any_ttc"])
        if math.isfinite(ttc) and ttc <= threshold:
            return {
                "ticks_before_crash": crash_tick - int(frame["tick"]),
                "tick": int(frame["tick"]),
                "ttc": ttc,
                "side": frame["min_ttc_side"],
                "lane": int(frame["min_ttc_lane"]),
                "action": frame["action_name"],
                "policy_argmax": frame["policy_argmax_name"],
                "ego_vx": float(frame["ego_vx"]),
                "ego_vy": float(frame["ego_vy"]),
                "nearest_lane": int(frame["nearest_lane"]),
                "lane_offset": float(frame["lane_offset"]),
                "policy_probs": frame["policy_probs"],
            }
    return None


def _summarize_recent(history: List[Dict[str, Any]], ticks: int) -> Dict[str, Any]:
    recent = history[-int(ticks):]
    counts = Counter(frame["action_name"] for frame in recent)
    total = max(1, len(recent))
    return {
        "window": len(recent),
        "nothing": counts["NOTHING"] / total,
        "accelerate": counts["ACCELERATE"] / total,
        "decelerate": counts["DECELERATE"] / total,
        "steer_left": counts["STEER_LEFT"] / total,
        "steer_right": counts["STEER_RIGHT"] / total,
    }


def _run_episode(sim_seed: int, action_seed: int) -> Dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_ENV is None:
        raise RuntimeError("worker not initialized")

    model = _WORKER_MODEL
    env = _WORKER_ENV

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(action_seed))
    scheduler_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)
    history = deque(maxlen=_WORKER_HISTORY)
    maneuvers: List[Dict[str, Any]] = []
    active_maneuver: Dict[str, Any] | None = None

    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(options={"sim_seed": int(sim_seed)})

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state = core.STATE
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            probs = torch.softmax(
                logits.squeeze(0) / float(_WORKER_TEMPERATURE), dim=-1
            )

        scheduler_credit.add_(probs)
        action = int(scheduler_credit.argmax().item())
        scheduler_credit[action] -= 1.0

        frame = _frame(state, action, probs)

        # A maneuver begins when a steering primitive is issued from a settled
        # lane-centred state. Once started, do not mistake counter-steering for a
        # second maneuver. The exact start state is projected through the
        # canonical 48-steer/48-counter-steer trajectory with current opponent
        # velocities frozen. This answers whether the maneuver was already
        # geometrically unsafe when the teacher initiated it.
        if active_maneuver is None:
            initiated, source_lane, target_lane, direction = (
                is_lane_change_initiation(state, action)
            )
            if initiated:
                prior = list(history)[-120:]
                prior_counts = Counter(x["action_name"] for x in prior)
                projection = project_lane_change(
                    state,
                    direction,
                    steer_ticks=48,
                    counter_ticks=48,
                    x_margin=0.0,
                    y_margin=0.0,
                )
                active_maneuver = {
                    "start_tick": int(state.ticks),
                    "source_lane": int(source_lane),
                    "target_lane": int(target_lane),
                    "direction": int(direction),
                    "direction_name": "left" if direction < 0 else "right",
                    "ego_vx_start": float(state.ego.velocity.x),
                    "ego_vy_start": float(state.ego.velocity.y),
                    "lane_offset_start": float(
                        projected_ego_lane_geometry(state)[1]
                    ),
                    "action_start": ACTIONS[int(action)],
                    "policy_probs_start": [
                        float(x) for x in probs.tolist()
                    ],
                    "prior_120_ticks": len(prior),
                    "prior_120_decelerate": int(
                        prior_counts["DECELERATE"]
                    ),
                    "prior_120_accelerate": int(
                        prior_counts["ACCELERATE"]
                    ),
                    "projection": projection,
                    "status": "active",
                    "end_tick": None,
                }
                maneuvers.append(active_maneuver)

        history.append(frame)
        obs, _, terminated, truncated, info = env.step(action)

        if active_maneuver is not None and not (terminated or truncated):
            post_state = core.STATE
            post_lane, post_offset, _ = projected_ego_lane_geometry(post_state)
            age = int(post_state.ticks) - int(active_maneuver["start_tick"])
            settled_target = (
                int(post_lane) == int(active_maneuver["target_lane"])
                and abs(float(post_offset)) <= 55.0
                and abs(float(post_state.ego.velocity.y)) < 0.35
            )
            if settled_target:
                active_maneuver["status"] = "completed"
                active_maneuver["end_tick"] = int(post_state.ticks)
                active_maneuver = None
            elif age > 160:
                active_maneuver["status"] = "timeout"
                active_maneuver["end_tick"] = int(post_state.ticks)
                active_maneuver = None

    state = core.STATE
    crashed = bool(info["crashed"])
    hist = list(history)

    if not crashed:
        return {
            "seed": int(sim_seed),
            "crashed": False,
            "ticks": int(info["ticks"]),
            "distance": float(info["distance"]),
            "final_speed": float(info["speed"]),
            "category": "completed",
        }

    colliders = _colliders(state)
    wall = _wall_hit(state)
    category = _classify_crash(hist, colliders, wall)
    last = hist[-1] if hist else {}

    onset_120 = _risk_onset(hist, 120.0)
    onset_60 = _risk_onset(hist, 60.0)

    collider = (
        min(colliders, key=lambda row: abs(float(row["rel_x"])))
        if colliders
        else None
    )

    fatal_maneuver = active_maneuver
    if fatal_maneuver is None:
        crash_tick = int(info["ticks"])
        for maneuver in reversed(maneuvers):
            if crash_tick - int(maneuver["start_tick"]) <= 140:
                fatal_maneuver = maneuver
                break

    if fatal_maneuver is not None and crashed:
        fatal_maneuver["status_at_episode_end"] = fatal_maneuver["status"]
        fatal_maneuver["ticks_start_to_crash"] = (
            int(info["ticks"]) - int(fatal_maneuver["start_tick"])
        )

    return {
        "seed": int(sim_seed),
        "crashed": True,
        "ticks": int(info["ticks"]),
        "distance": float(info["distance"]),
        "final_speed": float(info["speed"]),
        "category": category,
        "wall_hit": bool(wall),
        "collider": collider,
        "ego_vx_pre": float(last.get("ego_vx", float("nan"))),
        "ego_vy_pre": float(last.get("ego_vy", float("nan"))),
        "nearest_lane_pre": int(last.get("nearest_lane", -1)),
        "lane_offset_pre": float(last.get("lane_offset", float("nan"))),
        "action_pre": last.get("action_name"),
        "argmax_pre": last.get("policy_argmax_name"),
        "entropy_pre": float(last.get("policy_entropy", float("nan"))),
        "front_ttc_pre": _finite_or_none(
            float(last.get("min_front_ttc", float("inf")))
        ),
        "rear_ttc_pre": _finite_or_none(
            float(last.get("min_rear_ttc", float("inf")))
        ),
        "risk_onset_120": onset_120,
        "risk_onset_60": onset_60,
        "actions_last_30": _summarize_recent(hist, 30),
        "actions_last_96": _summarize_recent(hist, 96),
        "fatal_maneuver": fatal_maneuver,
        "maneuvers": maneuvers,
        "history": hist,
    }


def _worker_init(
    model_state: Dict[str, torch.Tensor],
    crash_penalty: float,
    reward_scale: float,
    history_ticks: int,
    temperature: float,
) -> None:
    global _WORKER_MODEL, _WORKER_ENV, _WORKER_HISTORY, _WORKER_TEMPERATURE

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
    _WORKER_HISTORY = int(history_ticks)
    _WORKER_TEMPERATURE = float(temperature)


def _flat_row(row: Dict[str, Any]) -> Dict[str, Any]:
    collider = row.get("collider") or {}
    onset120 = row.get("risk_onset_120") or {}
    onset60 = row.get("risk_onset_60") or {}
    a30 = row.get("actions_last_30") or {}
    maneuver = row.get("fatal_maneuver") or {}
    projection = maneuver.get("projection") or {}
    target = projection.get("target_start") or {}
    return {
        "seed": row["seed"],
        "crashed": row["crashed"],
        "category": row["category"],
        "ticks": row["ticks"],
        "distance": row["distance"],
        "final_speed": row["final_speed"],
        "ego_vx_pre": row.get("ego_vx_pre"),
        "ego_vy_pre": row.get("ego_vy_pre"),
        "nearest_lane_pre": row.get("nearest_lane_pre"),
        "lane_offset_pre": row.get("lane_offset_pre"),
        "action_pre": row.get("action_pre"),
        "argmax_pre": row.get("argmax_pre"),
        "front_ttc_pre": row.get("front_ttc_pre"),
        "rear_ttc_pre": row.get("rear_ttc_pre"),
        "collider_lane": collider.get("lane"),
        "collider_rel_x": collider.get("rel_x"),
        "collider_rel_vx": collider.get("rel_vx"),
        "risk120_ticks_before": onset120.get("ticks_before_crash"),
        "risk120_side": onset120.get("side"),
        "risk120_ttc": onset120.get("ttc"),
        "risk120_action": onset120.get("action"),
        "risk60_ticks_before": onset60.get("ticks_before_crash"),
        "risk60_side": onset60.get("side"),
        "risk60_ttc": onset60.get("ttc"),
        "risk60_action": onset60.get("action"),
        "last30_nothing": a30.get("nothing"),
        "last30_accelerate": a30.get("accelerate"),
        "last30_decelerate": a30.get("decelerate"),
        "last30_steer_left": a30.get("steer_left"),
        "last30_steer_right": a30.get("steer_right"),
        "maneuver_start_tick": maneuver.get("start_tick"),
        "maneuver_ticks_to_crash": maneuver.get("ticks_start_to_crash"),
        "maneuver_source_lane": maneuver.get("source_lane"),
        "maneuver_target_lane": maneuver.get("target_lane"),
        "maneuver_direction": maneuver.get("direction_name"),
        "maneuver_ego_vx_start": maneuver.get("ego_vx_start"),
        "maneuver_ego_vy_start": maneuver.get("ego_vy_start"),
        "maneuver_prior120_decelerate": maneuver.get(
            "prior_120_decelerate"
        ),
        "maneuver_prior120_accelerate": maneuver.get(
            "prior_120_accelerate"
        ),
        "unsafe_at_start": projection.get("unsafe_at_start"),
        "predicted_collision_tick": projection.get("collision_tick"),
        "predicted_collision_kind": projection.get("collision_kind"),
        "predicted_collision_lane": projection.get("collision_lane"),
        "predicted_collision_side": projection.get("collision_side"),
        "target_occupied_at_start": target.get("occupied"),
        "target_side_at_start": target.get("side"),
        "target_gap_at_start": target.get("gap"),
        "target_rel_vx_at_start": target.get("rel_vx"),
        "target_ttc_at_start": target.get("ttc"),
        "target_rel_x_t48": projection.get("target_rel_x_t48"),
        "target_clearance_t48": projection.get("target_clearance_t48"),
        "target_rel_x_t96": projection.get("target_rel_x_t96"),
        "target_clearance_t96": projection.get("target_clearance_t96"),
    }


def _percentile(values: List[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    crashes = [row for row in rows if row["crashed"]]
    completed = [row for row in rows if not row["crashed"]]
    categories = Counter(row["category"] for row in crashes)

    crash_ticks = [float(row["ticks"]) for row in crashes]
    crash_speeds = [float(row["ego_vx_pre"]) for row in crashes]
    abs_vy = [abs(float(row["ego_vy_pre"])) for row in crashes]

    risk120 = [
        float(row["risk_onset_120"]["ticks_before_crash"])
        for row in crashes
        if row.get("risk_onset_120") is not None
    ]
    risk60 = [
        float(row["risk_onset_60"]["ticks_before_crash"])
        for row in crashes
        if row.get("risk_onset_60") is not None
    ]

    lane_change_crashes = [
        row for row in crashes if str(row["category"]).startswith("lane_change_")
    ]
    maneuver_linked = [
        row for row in lane_change_crashes
        if row.get("fatal_maneuver") is not None
    ]
    unsafe_starts = [
        row for row in maneuver_linked
        if bool(
            (row["fatal_maneuver"].get("projection") or {}).get(
                "unsafe_at_start", False
            )
        )
    ]
    target_predicted = [
        row for row in unsafe_starts
        if (
            (row["fatal_maneuver"].get("projection") or {}).get(
                "collision_lane"
            )
            == row["fatal_maneuver"].get("target_lane")
        )
    ]
    safe_start_failed = [
        row for row in maneuver_linked if row not in unsafe_starts
    ]
    maneuver_start_vx = [
        float(row["fatal_maneuver"]["ego_vx_start"])
        for row in maneuver_linked
    ]
    maneuver_lead = [
        float(row["fatal_maneuver"].get("ticks_start_to_crash", 0))
        for row in maneuver_linked
    ]
    prior_decel_frac = []
    target_sides = Counter()
    for row in maneuver_linked:
        maneuver = row["fatal_maneuver"]
        window = int(maneuver.get("prior_120_ticks", 0))
        if window > 0:
            prior_decel_frac.append(
                float(maneuver.get("prior_120_decelerate", 0)) / window
            )
        projection = maneuver.get("projection") or {}
        target = projection.get("target_start") or {}
        target_sides[str(target.get("side", "none"))] += 1

    n = max(1, len(rows))
    nc = max(1, len(crashes))
    return {
        "episodes": len(rows),
        "completed": len(completed),
        "crashes": len(crashes),
        "crash_rate": len(crashes) / n,
        "category_counts": dict(categories),
        "category_rates_all": {
            key: value / n for key, value in categories.items()
        },
        "category_share_of_crashes": {
            key: value / nc for key, value in categories.items()
        },
        "crash_tick": {
            "median": _percentile(crash_ticks, 50),
            "p10": _percentile(crash_ticks, 10),
            "p90": _percentile(crash_ticks, 90),
        },
        "crash_ego_vx": {
            "median": _percentile(crash_speeds, 50),
            "p10": _percentile(crash_speeds, 10),
            "p90": _percentile(crash_speeds, 90),
        },
        "crash_abs_vy": {
            "median": _percentile(abs_vy, 50),
            "p90": _percentile(abs_vy, 90),
        },
        "risk_120_detected_share": len(risk120) / nc,
        "risk_120_lead_ticks": {
            "median": _percentile(risk120, 50),
            "p10": _percentile(risk120, 10),
            "p90": _percentile(risk120, 90),
        },
        "risk_60_detected_share": len(risk60) / nc,
        "risk_60_lead_ticks": {
            "median": _percentile(risk60, 50),
            "p10": _percentile(risk60, 10),
            "p90": _percentile(risk60, 90),
        },
        "lane_change_initiation": {
            "lane_change_crashes": len(lane_change_crashes),
            "linked_to_start": len(maneuver_linked),
            "linked_share": (
                len(maneuver_linked) / max(1, len(lane_change_crashes))
            ),
            "unsafe_at_start": len(unsafe_starts),
            "unsafe_at_start_share": (
                len(unsafe_starts) / max(1, len(maneuver_linked))
            ),
            "predicted_target_lane_collision": len(target_predicted),
            "predicted_target_lane_collision_share": (
                len(target_predicted) / max(1, len(maneuver_linked))
            ),
            "safe_at_start_but_failed": len(safe_start_failed),
            "safe_at_start_but_failed_share": (
                len(safe_start_failed) / max(1, len(maneuver_linked))
            ),
            "target_side_at_start": dict(target_sides),
            "start_ego_vx": {
                "median": _percentile(maneuver_start_vx, 50),
                "p10": _percentile(maneuver_start_vx, 10),
                "p90": _percentile(maneuver_start_vx, 90),
            },
            "start_to_crash_ticks": {
                "median": _percentile(maneuver_lead, 50),
                "p10": _percentile(maneuver_lead, 10),
                "p90": _percentile(maneuver_lead, 90),
            },
            "prior_120_decelerate_fraction": {
                "median": _percentile(prior_decel_frac, 50),
                "p90": _percentile(prior_decel_frac, 90),
            },
        },
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\nTeacher crash forensics")
    print(
        f"episodes={summary['episodes']:,} "
        f"crashes={summary['crashes']:,} "
        f"crash_rate={summary['crash_rate']:.2%}"
    )

    print("\nCrash categories")
    print(f"{'category':<24} {'count':>8} {'all':>9} {'of crashes':>12}")
    print("-" * 57)
    counts = summary["category_counts"]
    for category, count in sorted(
        counts.items(), key=lambda item: item[1], reverse=True
    ):
        print(
            f"{category:<24} {count:>8,d} "
            f"{summary['category_rates_all'][category]:>8.2%} "
            f"{summary['category_share_of_crashes'][category]:>11.2%}"
        )

    tick = summary["crash_tick"]
    speed = summary["crash_ego_vx"]
    vy = summary["crash_abs_vy"]
    print(
        "\nCrash timing: "
        f"p10={tick['p10']:.0f} median={tick['median']:.0f} "
        f"p90={tick['p90']:.0f} ticks"
    )
    print(
        "Ego vx at crash: "
        f"p10={speed['p10']:.1f} median={speed['median']:.1f} "
        f"p90={speed['p90']:.1f}"
    )
    print(
        "Abs lateral speed at crash: "
        f"median={vy['median']:.2f} p90={vy['p90']:.2f}"
    )

    r120 = summary["risk_120_lead_ticks"]
    r60 = summary["risk_60_lead_ticks"]
    print(
        "\nTTC<=120 was visible in the retained history for "
        f"{summary['risk_120_detected_share']:.1%} of crashes; "
        f"median warning={r120['median'] if r120['median'] is not None else 'n/a'} ticks."
    )
    print(
        "TTC<=60 was visible in the retained history for "
        f"{summary['risk_60_detected_share']:.1%} of crashes; "
        f"median warning={r60['median'] if r60['median'] is not None else 'n/a'} ticks."
    )

    initiation = summary["lane_change_initiation"]
    print("\nFatal lane-change initiation analysis")
    print(
        f"linked starts: {initiation['linked_to_start']}/"
        f"{initiation['lane_change_crashes']} "
        f"({initiation['linked_share']:.1%})"
    )
    print(
        "unsafe at initiation by canonical 96-tick projection: "
        f"{initiation['unsafe_at_start']}/"
        f"{initiation['linked_to_start']} "
        f"({initiation['unsafe_at_start_share']:.1%})"
    )
    print(
        "predicted collision specifically with target-lane car: "
        f"{initiation['predicted_target_lane_collision']} "
        f"({initiation['predicted_target_lane_collision_share']:.1%})"
    )
    print(
        "safe at initiation but later failed: "
        f"{initiation['safe_at_start_but_failed']} "
        f"({initiation['safe_at_start_but_failed_share']:.1%})"
    )
    print(
        "target car side at initiation: "
        f"{initiation['target_side_at_start']}"
    )
    start_v = initiation["start_ego_vx"]
    lead = initiation["start_to_crash_ticks"]
    print(
        "ego vx at lane-change start: "
        f"p10={start_v['p10'] if start_v['p10'] is not None else 'n/a'} "
        f"median={start_v['median'] if start_v['median'] is not None else 'n/a'} "
        f"p90={start_v['p90'] if start_v['p90'] is not None else 'n/a'}"
    )
    print(
        "lane-change start -> crash: "
        f"p10={lead['p10'] if lead['p10'] is not None else 'n/a'} "
        f"median={lead['median'] if lead['median'] is not None else 'n/a'} "
        f"p90={lead['p90'] if lead['p90'] is not None else 'n/a'} ticks"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=2048)
    p.add_argument("--seed-start", type=int, default=120_000)
    p.add_argument("--action-seed-start", type=int, default=12_000_000)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--history-ticks", type=int, default=180)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("models/teacher_crash_analysis"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.history_ticks < 120:
        raise ValueError("--history-ticks should be at least 120")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    config = payload.get("config", {})
    crash_penalty = float(config.get("crash_penalty", 250.0))
    reward_scale = float(config.get("reward_scale", 1000.0))

    tasks = [
        (args.seed_start + i, args.action_seed_start + i)
        for i in range(args.episodes)
    ]

    workers = max(1, min(int(args.workers), int(args.episodes)))
    if workers == 1:
        _worker_init(
            model_state,
            crash_penalty,
            reward_scale,
            args.history_ticks,
            args.temperature,
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
                args.history_ticks,
                args.temperature,
            ),
        ) as executor:
            rows = list(executor.map(
                _run_episode,
                [task[0] for task in tasks],
                [task[1] for task in tasks],
                chunksize=1,
            ))

    rows.sort(key=lambda row: int(row["seed"]))
    summary = _aggregate(rows)
    _print_summary(summary)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = prefix.parent / f"{prefix.name}.json"
    csv_path = prefix.parent / f"{prefix.name}_crashes.csv"
    summary_path = prefix.parent / f"{prefix.name}_summary.json"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    crash_rows = [_flat_row(row) for row in rows if row["crashed"]]
    if crash_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(crash_rows[0].keys()))
            writer.writeheader()
            writer.writerows(crash_rows)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {json_path}")
    if crash_rows:
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
