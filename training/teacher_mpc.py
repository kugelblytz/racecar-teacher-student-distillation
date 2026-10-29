"""Privileged sequence-level receding-horizon controller helpers.

This module is teacher-only. It uses exact simulator state to compare a small set
of meaningful multi-tick maneuver sequences, but it never reads future RNG or
future spawn information. The real controller executes only the first primitive
of the selected plan and replans from exact state on the next simulator tick.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from training.gpu_sim import ACTIONS
from training.teacher_lane_safety import (
    ACCELERATE,
    DECELERATE,
    NOTHING,
    STEER_LEFT,
    STEER_RIGHT,
    lane_centers,
    lane_index,
)


PLAN_NAMES = (
    "teacher",
    "settle",
    "left",
    "right",
    "accelerate",
    "brake",
    "coast",
)

OPPONENT_VELOCITY_STEP_STD = 0.1 / math.sqrt(3.0)


def _validate_action(action: int) -> int:
    action = int(action)
    if not 0 <= action < len(ACTIONS):
        raise ValueError(f"action must be in [0, {len(ACTIONS) - 1}]")
    return action


def opponent_position_uncertainty(tick: int, sigma: float) -> float:
    """Return a causal position envelope for the opponent velocity random walk."""
    tick = int(tick)
    sigma = float(sigma)
    if tick <= 0:
        raise ValueError("tick must be positive")
    if not math.isfinite(sigma) or sigma < 0.0:
        raise ValueError("opponent uncertainty sigma must be finite and non-negative")
    count = tick - 1
    squared_weights = count * (count + 1) * (2 * count + 1) / 6.0
    return float(sigma * OPPONENT_VELOCITY_STEP_STD * math.sqrt(squared_weights))


def _apply_action(action: int, vx: float, vy: float) -> Tuple[float, float]:
    action = int(action)
    if action == ACCELERATE:
        vx += 0.1
    elif action == DECELERATE:
        vx = max(0.0, vx - 0.1)
    elif action == STEER_LEFT:
        vy -= 0.1
    elif action == STEER_RIGHT:
        vy += 0.1
    return float(vx), float(vy)


def _steer_toward_target(
    current_y: float,
    current_vy: float,
    target_y: float,
    *,
    settle_tolerance: float = 3.0,
    vy_tolerance: float = 0.15,
) -> int:
    """Bang-bang lateral controller for the unusual velocity-steering dynamics."""
    error = float(target_y) - float(current_y)
    vy = float(current_vy)

    if abs(error) <= settle_tolerance and abs(vy) <= vy_tolerance:
        return NOTHING

    # If moving away from the target, immediately accelerate back toward it.
    if error * vy < 0.0:
        return STEER_RIGHT if error > 0.0 else STEER_LEFT

    # Discrete stopping-distance approximation under 0.1 lateral acceleration.
    stopping_distance = (vy * vy) / 0.2 if abs(vy) > 1e-12 else 0.0
    if abs(vy) > vy_tolerance and abs(error) <= stopping_distance + 2.0:
        return STEER_LEFT if vy > 0.0 else STEER_RIGHT

    return STEER_RIGHT if error > 0.0 else STEER_LEFT


def steer_toward_lane_action(
    state,
    target_lane: int,
    *,
    settle_tolerance: float = 3.0,
    vy_tolerance: float = 0.15,
) -> int:
    """Return one exact primitive that drives ego toward a chosen lane center."""
    centers = lane_centers(state)
    target_lane = int(target_lane)
    if not 0 <= target_lane < len(centers):
        return NOTHING

    ego = state.ego
    target_y = (
        float(centers[target_lane])
        - float(ego.sprite.get_height()) * 0.5
    )
    return _steer_toward_target(
        float(ego.y),
        float(ego.velocity.y),
        target_y,
        settle_tolerance=settle_tolerance,
        vy_tolerance=vy_tolerance,
    )


def lane_settled(
    state,
    target_lane: int,
    *,
    center_tolerance: float = 4.0,
    vy_tolerance: float = 0.2,
) -> bool:
    """Whether ego is centered and laterally settled in the chosen lane."""
    centers = lane_centers(state)
    target_lane = int(target_lane)
    if not 0 <= target_lane < len(centers):
        return False
    ego = state.ego
    center_y = (
        float(ego.y) + float(ego.sprite.get_height()) * 0.5
    )
    return bool(
        abs(center_y - float(centers[target_lane])) <= center_tolerance
        and abs(float(ego.velocity.y)) <= vy_tolerance
    )


def _plan_target_top_left(
    state,
    plan: str,
    ego_y: float,
    ego_h: int,
) -> float | None:
    centers = lane_centers(state)
    ego_center_y = float(ego_y) + float(ego_h) * 0.5
    nearest = min(
        range(len(centers)),
        key=lambda i: abs(ego_center_y - centers[i]),
    )

    if plan == "settle":
        target_lane = nearest
    elif plan == "left":
        target_lane = nearest - 1
    elif plan == "right":
        target_lane = nearest + 1
    else:
        return None

    if not 0 <= target_lane < len(centers):
        return None
    return float(centers[target_lane]) - float(ego_h) * 0.5


def mpc_trigger_reason(
    state,
    *,
    proposed_action: int | None = None,
    horizon: int = 72,
    trigger_x: float = 900.0,
    trigger_lateral: float = 300.0,
    trigger_extra: float = 120.0,
    trigger_immediate: float = 30.0,
    opponent_uncertainty_sigma: float = 3.0,
) -> str | None:
    """Return why MPC is needed, or ``None`` for a benign current state."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if (
        trigger_x <= 0
        or trigger_lateral <= 0
        or trigger_extra < 0
        or trigger_immediate < 0
    ):
        raise ValueError("trigger windows must be positive and buffers non-negative")
    if proposed_action is not None:
        proposed_action = _validate_action(proposed_action)
    uncertainty = opponent_position_uncertainty(
        int(horizon), float(opponent_uncertainty_sigma)
    )
    ego = state.ego
    ego_x = float(ego.x)
    ego_y = float(ego.y)
    ego_w = int(ego.sprite.get_width())
    ego_h = int(ego.sprite.get_height())
    ego_cx = ego_x + float(ego_w) * 0.5
    ego_cy = ego_y + float(ego_h) * 0.5
    ego_vx = float(ego.velocity.x)

    lateral_unsettled = abs(float(ego.velocity.y)) >= 0.25
    lateral_proposal = proposed_action in (STEER_LEFT, STEER_RIGHT)

    for car in state.cars:
        if car is ego:
            continue

        car_cx = float(car.x) + float(car.sprite.get_width()) * 0.5
        car_h = float(car.sprite.get_height())
        car_cy = float(car.y) + car_h * 0.5
        rel_x = car_cx - ego_cx
        lateral = abs(car_cy - ego_cy)

        if lateral > float(trigger_lateral):
            continue

        # Settled longitudinal motion is only threatened by traffic whose
        # vertical rectangles overlap. Adjacent lanes become relevant while a
        # lateral maneuver is proposed or already under way.
        lateral_gap = lateral - (float(ego_h) + car_h) * 0.5
        if lateral_gap > 0.0 and not (lateral_unsettled or lateral_proposal):
            continue

        if rel_x >= 0.0:
            gap = float(car.x) - (ego_x + float(ego_w))
            closing = ego_vx - float(car.velocity.x)
        else:
            gap = ego_x - (float(car.x) + float(car.sprite.get_width()))
            closing = float(car.velocity.x) - ego_vx

        gap = max(0.0, gap)
        if gap <= float(trigger_immediate):
            return "immediate"

        reach = max(0.0, closing) * float(horizon) + uncertainty
        if closing > 0.0 and gap <= reach + float(trigger_extra):
            return "closing"

        if (lateral_unsettled or lateral_proposal) and gap <= float(trigger_x):
            return "lateral"

    return None


def should_invoke_mpc(
    state,
    *,
    proposed_action: int | None = None,
    horizon: int = 72,
    trigger_x: float = 900.0,
    trigger_lateral: float = 300.0,
    trigger_extra: float = 120.0,
    trigger_immediate: float = 30.0,
    opponent_uncertainty_sigma: float = 3.0,
) -> bool:
    """Cheap causal prefilter for when sequence planning is worth the CPU cost."""
    return mpc_trigger_reason(
        state,
        proposed_action=proposed_action,
        horizon=horizon,
        trigger_x=trigger_x,
        trigger_lateral=trigger_lateral,
        trigger_extra=trigger_extra,
        trigger_immediate=trigger_immediate,
        opponent_uncertainty_sigma=opponent_uncertainty_sigma,
    ) is not None


def mpc_planning_decision(
    state,
    *,
    proposed_action: int | None = None,
    horizon: int = 72,
    strategic_horizon: int = 0,
    trigger_x: float = 900.0,
    trigger_lateral: float = 300.0,
    trigger_extra: float = 120.0,
    trigger_immediate: float = 30.0,
    opponent_uncertainty_sigma: float = 3.0,
) -> Tuple[str | None, int]:
    """Choose the trigger reason and horizon for one controller tick.

    A longer strategic horizon is proposed only when the normal horizon sees
    nearby lateral relevance while the longer reach reveals a closing
    longitudinal trajectory. ``choose_confirmed_mpc_action`` then accepts that
    candidate only for a nominal front collision.
    """
    horizon = int(horizon)
    strategic_horizon = int(strategic_horizon)
    if strategic_horizon < 0:
        raise ValueError("strategic_horizon must be non-negative")

    reason = mpc_trigger_reason(
        state,
        proposed_action=proposed_action,
        horizon=horizon,
        trigger_x=trigger_x,
        trigger_lateral=trigger_lateral,
        trigger_extra=trigger_extra,
        trigger_immediate=trigger_immediate,
        opponent_uncertainty_sigma=opponent_uncertainty_sigma,
    )
    planning_horizon = horizon
    if reason == "lateral" and strategic_horizon > horizon:
        strategic_reason = mpc_trigger_reason(
            state,
            proposed_action=proposed_action,
            horizon=strategic_horizon,
            trigger_x=trigger_x,
            trigger_lateral=trigger_lateral,
            trigger_extra=trigger_extra,
            trigger_immediate=trigger_immediate,
            opponent_uncertainty_sigma=opponent_uncertainty_sigma,
        )
        if strategic_reason == "closing":
            reason = "strategic_closing"
            planning_horizon = strategic_horizon
    return reason, planning_horizon


def _teacher_target_top_left(
    state,
    proposed_action: int,
    *,
    ego_y: float,
    ego_vy: float,
    ego_h: int,
) -> float | None:
    """Infer a plausible lane target for a steering proposal.

    Steering is lateral *acceleration*, so the opposite steering primitive is
    often counter-steering rather than intent to reverse the lane change. When
    the ego already has meaningful lateral velocity and the proposal opposes it,
    estimate where current lateral motion would stop under counter-steering and
    target the nearest lane center to that point. Otherwise interpret the
    proposal as motion toward the adjacent lane in the steering direction.
    """
    action = _validate_action(proposed_action)
    if action not in (STEER_LEFT, STEER_RIGHT):
        return None

    centers = lane_centers(state)
    center_y = float(ego_y) + float(ego_h) * 0.5
    nearest = min(
        range(len(centers)),
        key=lambda i: abs(center_y - centers[i]),
    )
    steer_sign = -1 if action == STEER_LEFT else 1

    # Counter-steering: infer intent from the lateral stopping position. Unlike
    # an ahead-only filter, this remains continuous across small lane overshoots.
    if abs(float(ego_vy)) >= 0.2 and float(ego_vy) * steer_sign < 0.0:
        stopping_offset = math.copysign(
            float(ego_vy) * float(ego_vy) / 0.2,
            float(ego_vy),
        )
        stopping_center = min(
            max(center_y + stopping_offset, float(centers[0])),
            float(centers[-1]),
        )
        target_center = min(centers, key=lambda center: abs(center - stopping_center))
        return float(target_center) - float(ego_h) * 0.5

    target_lane = nearest + steer_sign
    if not 0 <= target_lane < len(centers):
        return None
    return float(centers[target_lane]) - float(ego_h) * 0.5


def _teacher_continuation_action(
    proposed_action: int,
    tick: int,
    *,
    ego_y: float,
    ego_vy: float,
    target_y: float | None,
    burst_ticks: int,
) -> int:
    """Meaningful local continuation for the teacher's currently proposed move."""
    proposed_action = int(proposed_action)

    if proposed_action in (STEER_LEFT, STEER_RIGHT):
        if target_y is None:
            return NOTHING
        return _steer_toward_target(ego_y, ego_vy, target_y)

    if proposed_action == ACCELERATE:
        return ACCELERATE if tick <= burst_ticks else NOTHING
    if proposed_action == DECELERATE:
        return DECELERATE if tick <= burst_ticks else NOTHING
    return NOTHING


def project_maneuver(
    state,
    plan: str,
    *,
    proposed_action: int,
    horizon: int = 72,
    burst_ticks: int = 12,
    x_margin: float = 0.0,
    y_margin: float = 0.0,
    opponent_uncertainty_sigma: float = 3.0,
) -> Dict[str, Any]:
    """Project one maneuver with a causal opponent-position uncertainty envelope."""
    plan = str(plan)
    proposed_action = _validate_action(proposed_action)
    if plan not in PLAN_NAMES:
        raise ValueError(f"unknown plan: {plan}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if burst_ticks <= 0:
        raise ValueError("burst_ticks must be positive")
    if x_margin < 0 or y_margin < 0:
        raise ValueError("margins must be non-negative")
    if not math.isfinite(float(opponent_uncertainty_sigma)) or float(
        opponent_uncertainty_sigma
    ) < 0.0:
        raise ValueError("opponent uncertainty sigma must be finite and non-negative")

    ego = state.ego
    ego_x = float(ego.x)
    ego_y = float(ego.y)
    ego_vx = float(ego.velocity.x)
    ego_vy = float(ego.velocity.y)
    ego_w = int(ego.sprite.get_width())
    ego_h = int(ego.sprite.get_height())
    ego_cx = ego_x + float(ego_w) * 0.5

    centers = lane_centers(state)

    target_y = _plan_target_top_left(state, plan, ego_y, ego_h)
    teacher_target_y = _teacher_target_top_left(
        state,
        int(proposed_action),
        ego_y=ego_y,
        ego_vy=ego_vy,
        ego_h=ego_h,
    )

    # Invalid directional plan: keep it explicit so selection can reject it.
    if plan in ("left", "right") and target_y is None:
        return {
            "plan": plan,
            "first_action": STEER_LEFT if plan == "left" else STEER_RIGHT,
            "first_action_name": (
                ACTIONS[STEER_LEFT] if plan == "left" else ACTIONS[STEER_RIGHT]
            ),
            "safe": False,
            "healthy": False,
            "collision_tick": 0,
            "collision_kind": "road_edge",
            "collision_lane": None,
            "collision_side": None,
            "min_car_clearance": -1e9,
            "min_wall_clearance": -1e9,
            "progress": 0.0,
            "zero_speed_ticks": 0,
            "final_vx": ego_vx,
            "final_vy": ego_vy,
            "final_y": ego_y,
            "final_lane_offset": 1e9,
            "opponent_uncertainty_sigma": float(opponent_uncertainty_sigma),
            "max_opponent_position_uncertainty": 0.0,
            "actions": [],
        }

    cars: List[Dict[str, Any]] = []
    for car in state.cars:
        if car is ego:
            continue
        cars.append(
            {
                "x": float(car.x),
                "y": float(car.y),
                "vx": float(car.velocity.x),
                "lane": lane_index(state, car.lane),
                "w": int(car.sprite.get_width()),
                "h": int(car.sprite.get_height()),
                "active": True,
            }
        )

    top_wall_bottom = max(
        float(wall.rect.bottom)
        for wall in state.road.walls
        if wall.rect.centery < state.road.y_start
    )
    bottom_wall_top = min(
        float(wall.rect.top)
        for wall in state.road.walls
        if wall.rect.centery > state.road.y_end
    )

    progress = 0.0
    zero_speed_ticks = 0
    min_car_clearance = math.inf
    min_wall_clearance = math.inf
    collision_tick: int | None = None
    collision_kind: str | None = None
    collision_lane: int | None = None
    collision_side: str | None = None
    actions: List[int] = []
    max_opponent_position_uncertainty = 0.0

    for tick in range(1, int(horizon) + 1):
        if plan == "teacher":
            action = _teacher_continuation_action(
                int(proposed_action),
                tick,
                ego_y=ego_y,
                ego_vy=ego_vy,
                target_y=teacher_target_y,
                burst_ticks=int(burst_ticks),
            )
            if tick == 1:
                action = int(proposed_action)
        elif plan in ("settle", "left", "right"):
            if target_y is None:
                action = NOTHING
            else:
                action = _steer_toward_target(
                    ego_y,
                    ego_vy,
                    target_y,
                )
        elif plan == "accelerate":
            if tick <= int(burst_ticks):
                action = ACCELERATE
            elif abs(ego_vy) > 0.15:
                action = STEER_LEFT if ego_vy > 0 else STEER_RIGHT
            else:
                action = NOTHING
        elif plan == "brake":
            if tick <= int(burst_ticks):
                action = DECELERATE
            elif abs(ego_vy) > 0.15:
                action = STEER_LEFT if ego_vy > 0 else STEER_RIGHT
            else:
                action = NOTHING
        else:  # coast
            if abs(ego_vy) > 0.15:
                action = STEER_LEFT if ego_vy > 0 else STEER_RIGHT
            else:
                action = NOTHING

        actions.append(int(action))
        ego_vx, ego_vy = _apply_action(action, ego_vx, ego_vy)
        progress += ego_vx
        if ego_vx <= 1e-9:
            zero_speed_ticks += 1
        ego_y += ego_vy

        position_uncertainty = opponent_position_uncertainty(
            tick, float(opponent_uncertainty_sigma)
        )
        max_opponent_position_uncertainty = max(
            max_opponent_position_uncertainty, position_uncertainty
        )

        for car in cars:
            if not bool(car["active"]):
                continue
            car["x"] += float(car["vx"]) - ego_vx
            if (
                float(car["x"]) + position_uncertainty < -1000.0
                or float(car["x"]) - position_uncertainty > 2600.0
            ):
                car["active"] = False

        ego_ix = int(ego_x)
        ego_iy = int(ego_y)
        wall_clearance = min(
            float(ego_iy) - (top_wall_bottom + float(y_margin)),
            (bottom_wall_top - float(y_margin))
            - float(ego_iy + ego_h),
        )
        min_wall_clearance = min(min_wall_clearance, wall_clearance)
        if wall_clearance < 0.0:
            collision_tick = int(tick)
            collision_kind = "wall"
            break

        ego_cy_i = float(ego_iy) + float(ego_h) * 0.5
        for car in cars:
            if not bool(car["active"]):
                continue

            car_ix = int(float(car["x"]))
            car_iy = int(float(car["y"]))
            car_cx = float(car_ix) + float(car["w"]) * 0.5
            car_cy = float(car_iy) + float(car["h"]) * 0.5

            x_sep = (
                abs(car_cx - ego_cx)
                - float(ego_w + int(car["w"])) * 0.5
                - float(x_margin)
                - position_uncertainty
            )
            y_sep = (
                abs(car_cy - ego_cy_i)
                - float(ego_h + int(car["h"])) * 0.5
                - float(y_margin)
            )
            clearance = max(x_sep, y_sep)
            min_car_clearance = min(min_car_clearance, clearance)

            overlap = (
                float(ego_ix)
                < float(car_ix + int(car["w"]))
                + float(x_margin)
                + position_uncertainty
                and float(ego_ix + ego_w)
                > float(car_ix) - float(x_margin) - position_uncertainty
                and float(ego_iy) < float(car_iy + int(car["h"])) + float(y_margin)
                and float(ego_iy + ego_h) > float(car_iy) - float(y_margin)
            )
            if not overlap:
                continue

            collision_tick = int(tick)
            collision_kind = "car"
            collision_lane = int(car["lane"])
            collision_side = "front" if car_cx - ego_cx >= 0.0 else "rear"
            break

        if collision_tick is not None:
            break

    if not math.isfinite(min_car_clearance):
        min_car_clearance = 1e9
    if not math.isfinite(min_wall_clearance):
        min_wall_clearance = 1e9

    final_center_y = float(ego_y) + float(ego_h) * 0.5
    nearest_final = min(
        range(len(centers)),
        key=lambda i: abs(final_center_y - centers[i]),
    )
    final_lane_offset = final_center_y - centers[nearest_final]

    return {
        "plan": plan,
        "first_action": int(actions[0]) if actions else NOTHING,
        "first_action_name": ACTIONS[int(actions[0])] if actions else ACTIONS[NOTHING],
        "safe": collision_tick is None,
        "collision_tick": collision_tick,
        "collision_kind": collision_kind,
        "collision_lane": collision_lane,
        "collision_side": collision_side,
        "min_car_clearance": float(min_car_clearance),
        "min_wall_clearance": float(min_wall_clearance),
        "progress": float(progress),
        "zero_speed_ticks": int(zero_speed_ticks),
        "final_vx": float(ego_vx),
        "final_vy": float(ego_vy),
        "final_y": float(ego_y),
        "final_lane_offset": float(final_lane_offset),
        "opponent_uncertainty_sigma": float(opponent_uncertainty_sigma),
        "max_opponent_position_uncertainty": float(
            max_opponent_position_uncertainty
        ),
        "actions": actions,
    }


def _healthy(
    projection: Dict[str, Any],
    *,
    clearance_threshold: float,
    wall_threshold: float,
) -> bool:
    return bool(
        projection["safe"]
        and float(projection["min_car_clearance"]) >= float(clearance_threshold)
        and float(projection["min_wall_clearance"]) >= float(wall_threshold)
    )


def _safe_rank(
    projection: Dict[str, Any],
    *,
    proposed_action: int,
    horizon: int,
    lane_half_width: float,
    clearance_threshold: float,
    wall_threshold: float,
    clearance_cap: float,
) -> Tuple[float, float, float, float, float, int]:
    """Safety-first lexicographic rank for collision-free projections."""
    risk_margin = min(
        float(projection["min_car_clearance"]) - float(clearance_threshold),
        float(projection["min_wall_clearance"]) - float(wall_threshold),
    )
    stopped_fraction = float(projection["zero_speed_ticks"]) / float(horizon)
    lane_offset = abs(float(projection["final_lane_offset"])) / max(
        float(lane_half_width), 1.0
    )
    return (
        min(risk_margin, float(clearance_cap)),
        -stopped_fraction,
        -lane_offset,
        -abs(float(projection["final_vy"])),
        float(projection["progress"]) / float(horizon),
        int(int(projection["first_action"]) == int(proposed_action)),
    )


def choose_mpc_action(
    state,
    proposed_action: int,
    *,
    horizon: int = 72,
    burst_ticks: int = 12,
    x_margin: float = 0.0,
    y_margin: float = 0.0,
    clearance_threshold: float = 8.0,
    wall_threshold: float = 4.0,
    clearance_cap: float = 120.0,
    opponent_uncertainty_sigma: float = 3.0,
    # Retained for call compatibility. Selection is now lexicographic rather
    # than a scale-sensitive weighted sum.
    clearance_weight: float = 2.0,
    wall_weight: float = 0.25,
    zero_speed_penalty: float = 4.0,
    lateral_penalty: float = 20.0,
    teacher_bonus: float = 40.0,
) -> Tuple[int, Dict[str, Any]]:
    """Receding-horizon maneuver selection, preserving a healthy teacher plan."""
    proposed_action = _validate_action(proposed_action)
    numeric_non_negative = {
        "x_margin": x_margin,
        "y_margin": y_margin,
        "clearance_threshold": clearance_threshold,
        "wall_threshold": wall_threshold,
        "opponent_uncertainty_sigma": opponent_uncertainty_sigma,
        "clearance_weight": clearance_weight,
        "wall_weight": wall_weight,
        "zero_speed_penalty": zero_speed_penalty,
        "lateral_penalty": lateral_penalty,
        "teacher_bonus": teacher_bonus,
    }
    for name, value in numeric_non_negative.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if horizon <= 0 or burst_ticks <= 0:
        raise ValueError("horizon and burst_ticks must be positive")
    if not math.isfinite(float(clearance_cap)) or clearance_cap <= 0:
        raise ValueError("clearance_cap must be finite and positive")

    centers = lane_centers(state)
    if len(centers) >= 2:
        lane_half_width = abs(float(centers[1]) - float(centers[0])) * 0.5
    else:
        lane_half_width = 1.0

    teacher = project_maneuver(
        state,
        "teacher",
        proposed_action=proposed_action,
        horizon=horizon,
        burst_ticks=burst_ticks,
        x_margin=x_margin,
        y_margin=y_margin,
        opponent_uncertainty_sigma=opponent_uncertainty_sigma,
    )
    teacher["healthy"] = _healthy(
        teacher,
        clearance_threshold=clearance_threshold,
        wall_threshold=wall_threshold,
    )

    # Nominal-controller preservation: if the teacher's meaningful local
    # continuation is collision-free with a small positive clearance buffer,
    # do not interfere at all.
    if bool(teacher["healthy"]):
        return proposed_action, {
            "overridden": False,
            "proposed_action": proposed_action,
            "selected_action": proposed_action,
            "selected_plan": "teacher",
            "teacher_projection": teacher,
            "selected_projection": teacher,
            "plans_evaluated": 1,
            "all_unsafe": False,
            "all_unhealthy": False,
            "opponent_uncertainty_sigma": float(opponent_uncertainty_sigma),
            "projections": [teacher],
        }

    projections: List[Dict[str, Any]] = [teacher]
    for plan in ("settle", "left", "right", "accelerate", "brake", "coast"):
        item = project_maneuver(
            state,
            plan,
            proposed_action=proposed_action,
            horizon=horizon,
            burst_ticks=burst_ticks,
            x_margin=x_margin,
            y_margin=y_margin,
            opponent_uncertainty_sigma=opponent_uncertainty_sigma,
        )
        item["healthy"] = _healthy(
            item,
            clearance_threshold=clearance_threshold,
            wall_threshold=wall_threshold,
        )
        projections.append(item)

    healthy = [item for item in projections if bool(item["healthy"])]
    safe = [item for item in projections if bool(item["safe"])]

    if healthy:
        selected = max(
            healthy,
            key=lambda item: _safe_rank(
                item,
                proposed_action=proposed_action,
                horizon=horizon,
                lane_half_width=lane_half_width,
                clearance_threshold=clearance_threshold,
                wall_threshold=wall_threshold,
                clearance_cap=clearance_cap,
            ),
        )
        all_unsafe = False
        all_unhealthy = False
    elif safe:
        # No plan clears the comfort threshold, but at least one avoids an
        # actual overlap. Prefer the best progress/clearance tradeoff.
        selected = max(
            safe,
            key=lambda item: _safe_rank(
                item,
                proposed_action=proposed_action,
                horizon=horizon,
                lane_half_width=lane_half_width,
                clearance_threshold=clearance_threshold,
                wall_threshold=wall_threshold,
                clearance_cap=clearance_cap,
            ),
        )
        all_unsafe = False
        all_unhealthy = True
    else:
        # If every modeled maneuver collides, maximize time-to-collision first,
        # then raw clearance and progress. This avoids rediscovering "always
        # brake" merely because braking has low immediate velocity.
        def unsafe_rank(item: Dict[str, Any]):
            tick = item["collision_tick"]
            survived = int(horizon) + 1 if tick is None else int(tick)
            clearance = min(
                float(item["min_car_clearance"]),
                float(item["min_wall_clearance"]),
            )
            return (
                survived,
                clearance,
                -abs(float(item["final_lane_offset"])),
                -abs(float(item["final_vy"])),
                float(item["progress"]) / float(horizon),
                int(item["first_action"] == proposed_action),
            )

        selected = max(projections, key=unsafe_rank)
        all_unsafe = True
        all_unhealthy = True

    selected_action = int(selected["first_action"])
    return selected_action, {
        "overridden": selected_action != proposed_action,
        "proposed_action": proposed_action,
        "selected_action": selected_action,
        "selected_plan": str(selected["plan"]),
        "teacher_projection": teacher,
        "selected_projection": selected,
        "plans_evaluated": len(projections),
        "all_unsafe": bool(all_unsafe),
        "all_unhealthy": bool(all_unhealthy),
        "opponent_uncertainty_sigma": float(opponent_uncertainty_sigma),
        "projections": projections,
    }


def choose_confirmed_mpc_action(
    state,
    proposed_action: int,
    *,
    horizon: int = 72,
    planning_horizon: int | None = None,
    **kwargs: Any,
) -> Tuple[int, Dict[str, Any], bool]:
    """Apply a longer strategic projection only for a nominal front collision.

    Returns ``(action, diagnostic, strategic_used)``. Rear collisions predicted
    only after a proposed lane move are evaluated again at the normal horizon;
    this prevents uncertain adjacent traffic from causing premature evasions.
    """
    horizon = int(horizon)
    planning_horizon = (
        horizon if planning_horizon is None else int(planning_horizon)
    )
    if horizon <= 0 or planning_horizon < horizon:
        raise ValueError("planning_horizon must be at least the positive horizon")

    action, diagnostic = choose_mpc_action(
        state,
        proposed_action,
        horizon=planning_horizon,
        **kwargs,
    )
    if planning_horizon == horizon:
        return action, diagnostic, False

    teacher = diagnostic["teacher_projection"]
    strategic_used = bool(
        not teacher["safe"]
        and teacher.get("collision_kind") == "car"
        and teacher.get("collision_side") == "front"
    )
    if strategic_used:
        return action, diagnostic, True

    action, diagnostic = choose_mpc_action(
        state,
        proposed_action,
        horizon=horizon,
        **kwargs,
    )
    return action, diagnostic, False
