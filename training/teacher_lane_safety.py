"""Privileged lane-change safety diagnostics and projection helpers.

These helpers are intentionally teacher-only. They inspect exact simulator state
and must never be used by the legal competition policy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from training.gpu_sim import ACTIONS


STEER_LEFT = ACTIONS.index("STEER_LEFT")
STEER_RIGHT = ACTIONS.index("STEER_RIGHT")
NOTHING = ACTIONS.index("NOTHING")
ACCELERATE = ACTIONS.index("ACCELERATE")
DECELERATE = ACTIONS.index("DECELERATE")


def lane_index(state, lane) -> int:
    if lane is None:
        return -1
    for i, candidate in enumerate(state.road.lanes):
        if candidate is lane:
            return i
    return -1


def lane_centers(state) -> List[float]:
    return [
        (float(lane.y_start) + float(lane.y_end)) * 0.5
        for lane in state.road.lanes
    ]


def ego_lane_geometry(state) -> Tuple[int, float, float]:
    """Return nearest lane, signed center offset, and geometric ego center-y."""
    ego = state.ego
    ego_cy = float(ego.y) + float(ego.sprite.get_height()) * 0.5
    centers = lane_centers(state)
    nearest = min(range(len(centers)), key=lambda i: abs(ego_cy - centers[i]))
    return nearest, ego_cy - centers[nearest], ego_cy


def steering_direction(action: int) -> int:
    """Map primitive action to lane direction: left=-1, right=+1, else 0."""
    action = int(action)
    if action == STEER_LEFT:
        return -1
    if action == STEER_RIGHT:
        return 1
    return 0


def target_lane_snapshot(state, target_lane: int) -> Dict[str, Any]:
    """Describe the current opponent in a target lane, if one exists."""
    ego = state.ego
    ego_cx = float(ego.x) + float(ego.sprite.get_width()) * 0.5
    ego_left = float(ego.x)
    ego_right = ego_left + float(ego.sprite.get_width())
    ego_vx = float(ego.velocity.x)

    for car in state.cars:
        if car is ego or lane_index(state, car.lane) != int(target_lane):
            continue

        car_cx = float(car.x) + float(car.sprite.get_width()) * 0.5
        rel_x = car_cx - ego_cx
        rel_vx = float(car.velocity.x) - ego_vx

        if rel_x >= 0.0:
            side = "front"
            gap = float(car.x) - ego_right
            closing = ego_vx - float(car.velocity.x)
        else:
            side = "rear"
            gap = ego_left - (
                float(car.x) + float(car.sprite.get_width())
            )
            closing = float(car.velocity.x) - ego_vx

        ttc = None
        if closing > 1e-9:
            ttc = max(0.0, gap) / closing

        return {
            "occupied": True,
            "lane": int(target_lane),
            "side": side,
            "rel_x": float(rel_x),
            "rel_vx": float(rel_vx),
            "other_vx": float(car.velocity.x),
            "gap": float(gap),
            "closing": float(max(0.0, closing)),
            "ttc": None if ttc is None else float(ttc),
        }

    return {
        "occupied": False,
        "lane": int(target_lane),
        "side": "none",
        "rel_x": None,
        "rel_vx": None,
        "other_vx": None,
        "gap": None,
        "closing": 0.0,
        "ttc": None,
    }


def _rect_overlap(
    ax: int,
    ay: int,
    aw: int,
    ah: int,
    bx: int,
    by: int,
    bw: int,
    bh: int,
    *,
    x_margin: float = 0.0,
    y_margin: float = 0.0,
) -> bool:
    # With zero margins this is pygame.Rect.colliderect semantics.
    return (
        float(ax) < float(bx + bw) + x_margin
        and float(ax + aw) > float(bx) - x_margin
        and float(ay) < float(by + bh) + y_margin
        and float(ay + ah) > float(by) - y_margin
    )


def project_lane_change(
    state,
    direction: int,
    *,
    steer_ticks: int = 48,
    counter_ticks: int = 48,
    x_margin: float = 0.0,
    y_margin: float = 0.0,
) -> Dict[str, Any]:
    """Project a canonical steer/counter-steer lane change from exact state.

    Opponent longitudinal velocities are frozen at their current values. This is
    deliberately a *start-state feasibility test*, not an oracle for future RNG.
    Ego vx is constant because every projected primitive is a steering action.

    The returned collision tick is 1-indexed from the proposed initiation.
    """
    direction = int(direction)
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 (left) or +1 (right)")
    if steer_ticks <= 0 or counter_ticks <= 0:
        raise ValueError("steer_ticks and counter_ticks must be positive")

    source_lane, source_offset, _ = ego_lane_geometry(state)
    target_lane = source_lane + direction
    lane_count = len(state.road.lanes)

    result: Dict[str, Any] = {
        "source_lane": int(source_lane),
        "target_lane": int(target_lane),
        "direction": int(direction),
        "valid_target": bool(0 <= target_lane < lane_count),
        "source_lane_offset": float(source_offset),
        "steer_ticks": int(steer_ticks),
        "counter_ticks": int(counter_ticks),
        "horizon": int(steer_ticks + counter_ticks),
        "x_margin": float(x_margin),
        "y_margin": float(y_margin),
        "target_start": None,
        "target_rel_x_t48": None,
        "target_clearance_t48": None,
        "target_rel_x_t96": None,
        "target_clearance_t96": None,
        "predicted_collision": False,
        "collision_tick": None,
        "collision_kind": None,
        "collision_lane": None,
        "collision_side": None,
        "collision_rel_x": None,
        "collision_rel_vx": None,
        "unsafe_at_start": False,
    }

    if not result["valid_target"]:
        result["collision_kind"] = "road_edge"
        result["unsafe_at_start"] = True
        return result

    result["target_start"] = target_lane_snapshot(state, target_lane)

    ego = state.ego
    ego_x = float(ego.x)
    ego_y = float(ego.y)
    ego_vx = float(ego.velocity.x)
    ego_vy = float(ego.velocity.y)
    ego_w = int(ego.sprite.get_width())
    ego_h = int(ego.sprite.get_height())
    ego_cx = ego_x + ego_w * 0.5

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
            }
        )

    # Exact wall vertical boundaries from the original simulator.
    top_wall_bottom = max(float(wall.rect.bottom) for wall in state.road.walls if wall.rect.centery < state.road.y_start)
    bottom_wall_top = min(float(wall.rect.top) for wall in state.road.walls if wall.rect.centery > state.road.y_end)

    target_car = next(
        (car for car in cars if int(car["lane"]) == int(target_lane)),
        None,
    )

    horizon = int(steer_ticks + counter_ticks)
    for tick in range(1, horizon + 1):
        if tick <= steer_ticks:
            ego_vy += 0.1 * direction
        else:
            ego_vy -= 0.1 * direction
        ego_y += ego_vy

        for car in cars:
            car["x"] += float(car["vx"]) - ego_vx

        ego_ix = int(ego_x)
        ego_iy = int(ego_y)

        if (
            float(ego_iy) < top_wall_bottom + float(y_margin)
            or float(ego_iy + ego_h) > bottom_wall_top - float(y_margin)
        ):
            result.update(
                {
                    "predicted_collision": True,
                    "collision_tick": int(tick),
                    "collision_kind": "wall",
                    "unsafe_at_start": True,
                }
            )
            break

        for car in cars:
            car_ix = int(float(car["x"]))
            car_iy = int(float(car["y"]))
            if not _rect_overlap(
                ego_ix,
                ego_iy,
                ego_w,
                ego_h,
                car_ix,
                car_iy,
                int(car["w"]),
                int(car["h"]),
                x_margin=x_margin,
                y_margin=y_margin,
            ):
                continue

            car_cx = float(car["x"]) + float(car["w"]) * 0.5
            rel_x = car_cx - ego_cx
            rel_vx = float(car["vx"]) - ego_vx
            result.update(
                {
                    "predicted_collision": True,
                    "collision_tick": int(tick),
                    "collision_kind": "car",
                    "collision_lane": int(car["lane"]),
                    "collision_side": "front" if rel_x >= 0 else "rear",
                    "collision_rel_x": float(rel_x),
                    "collision_rel_vx": float(rel_vx),
                    "unsafe_at_start": True,
                }
            )
            break

        if result["predicted_collision"]:
            break

        if target_car is not None and tick in (48, 96):
            target_cx = float(target_car["x"]) + float(target_car["w"]) * 0.5
            rel_x = target_cx - ego_cx
            clearance = abs(rel_x) - (
                float(ego_w + int(target_car["w"])) * 0.5
            )
            key = "t48" if tick == 48 else "t96"
            result[f"target_rel_x_{key}"] = float(rel_x)
            result[f"target_clearance_{key}"] = float(clearance)

    # If a collision happened before t48/t96, still report the frozen-velocity
    # target-lane longitudinal projections at those canonical times.
    if target_car is not None:
        initial = result["target_start"]
        rel0 = float(initial["rel_x"])
        rel_vx = float(initial["rel_vx"])
        half_sum = float(ego_w + int(target_car["w"])) * 0.5
        if result["target_rel_x_t48"] is None:
            rel48 = rel0 + rel_vx * 48.0
            result["target_rel_x_t48"] = float(rel48)
            result["target_clearance_t48"] = float(abs(rel48) - half_sum)
        if result["target_rel_x_t96"] is None:
            rel96 = rel0 + rel_vx * 96.0
            result["target_rel_x_t96"] = float(rel96)
            result["target_clearance_t96"] = float(abs(rel96) - half_sum)

    return result


def is_lane_change_initiation(
    state,
    action: int,
    *,
    lane_snap_tolerance: float = 55.0,
    vy_tolerance: float = 0.35,
) -> Tuple[bool, int, int, int]:
    """Detect a steering action that begins from a settled lane state."""
    direction = steering_direction(action)
    source_lane, lane_offset, _ = ego_lane_geometry(state)
    target_lane = source_lane + direction

    if direction == 0:
        return False, source_lane, target_lane, direction

    settled = (
        abs(float(lane_offset)) <= float(lane_snap_tolerance)
        and abs(float(state.ego.velocity.y)) <= float(vy_tolerance)
    )
    valid = 0 <= target_lane < len(state.road.lanes)
    return bool(settled and valid), source_lane, target_lane, direction


def current_lane_ttc(state) -> Dict[str, float | None]:
    """Nearest front/rear TTC among cars laterally overlapping the ego."""
    ego = state.ego
    ego_left = float(ego.x)
    ego_right = ego_left + float(ego.sprite.get_width())
    ego_top = float(ego.y)
    ego_bottom = ego_top + float(ego.sprite.get_height())
    ego_vx = float(ego.velocity.x)

    front = math.inf
    rear = math.inf
    for car in state.cars:
        if car is ego:
            continue
        car_top = float(car.y)
        car_bottom = car_top + float(car.sprite.get_height())
        if car_bottom <= ego_top or car_top >= ego_bottom:
            continue

        car_left = float(car.x)
        car_right = car_left + float(car.sprite.get_width())
        car_cx = (car_left + car_right) * 0.5
        ego_cx = (ego_left + ego_right) * 0.5

        if car_cx >= ego_cx:
            closing = ego_vx - float(car.velocity.x)
            if closing > 1e-9:
                front = min(front, max(0.0, car_left - ego_right) / closing)
        else:
            closing = float(car.velocity.x) - ego_vx
            if closing > 1e-9:
                rear = min(rear, max(0.0, ego_left - car_right) / closing)

    return {
        "front": None if not math.isfinite(front) else float(front),
        "rear": None if not math.isfinite(rear) else float(rear),
    }


def _apply_primitive(
    action: int,
    vx: float,
    vy: float,
) -> Tuple[float, float]:
    """Apply one primitive velocity update using exact challenge increments."""
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


def _projected_emergency_action(
    *,
    ego_x: float,
    ego_y: float,
    ego_w: int,
    ego_h: int,
    ego_vx: float,
    ego_vy: float,
    cars: List[Dict[str, Any]],
    remaining_ticks: int,
) -> int:
    """Simple recovery controller used after a candidate's first projected tick.

    Lateral velocity is damped first because steering and longitudinal control
    cannot be combined in one challenge tick. Once laterally settled, brake for
    an imminent front threat or accelerate away from an imminent rear threat.
    Otherwise coast. This is deliberately conservative and causal: it uses only
    current projected state and frozen current opponent velocities.
    """
    if ego_vy < -0.05:
        return STEER_RIGHT
    if ego_vy > 0.05:
        return STEER_LEFT

    ego_left = float(ego_x)
    ego_right = ego_left + float(ego_w)
    ego_top = float(ego_y)
    ego_bottom = ego_top + float(ego_h)
    ego_cx = ego_left + float(ego_w) * 0.5

    front_ttc = math.inf
    rear_ttc = math.inf

    for car in cars:
        if not bool(car.get("active", True)):
            continue

        car_top = float(car["y"])
        car_bottom = car_top + float(car["h"])
        if car_bottom <= ego_top or car_top >= ego_bottom:
            continue

        car_left = float(car["x"])
        car_right = car_left + float(car["w"])
        car_cx = car_left + float(car["w"]) * 0.5

        if car_cx >= ego_cx:
            closing = ego_vx - float(car["vx"])
            if closing > 1e-9:
                gap = max(0.0, car_left - ego_right)
                front_ttc = min(front_ttc, gap / closing)
        else:
            closing = float(car["vx"]) - ego_vx
            if closing > 1e-9:
                gap = max(0.0, ego_left - car_right)
                rear_ttc = min(rear_ttc, gap / closing)

    # Add a few ticks of buffer so the recovery begins before the predicted
    # overlap becomes unavoidable. This is not a hard TTC reward; it is only the
    # continuation policy inside the short safety projection.
    trigger = float(max(1, int(remaining_ticks)) + 4)
    front_urgent = front_ttc <= trigger
    rear_urgent = rear_ttc <= trigger

    if front_urgent and (not rear_urgent or front_ttc <= rear_ttc):
        return DECELERATE
    if rear_urgent:
        return ACCELERATE
    return NOTHING


def project_primitive_action(
    state,
    action: int,
    *,
    horizon: int = 24,
    x_margin: float = 0.0,
    y_margin: float = 0.0,
) -> Dict[str, Any]:
    """Short-horizon recoverability projection for one primitive action.

    The supplied action is executed on projected tick 1. On later projected
    ticks a small emergency controller tries to damp lateral velocity first,
    then handles imminent front/rear longitudinal threats. Existing opponents
    keep their *current* longitudinal velocity; future random-walk draws and
    future spawns are intentionally unknown.

    This is designed for a receding-horizon safety filter, not for predicting a
    whole lane change. The real filter recomputes the projection from exact state
    on the next simulator tick.
    """
    action = int(action)
    horizon = int(horizon)
    if not 0 <= action < len(ACTIONS):
        raise ValueError(f"invalid action index: {action}")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if x_margin < 0 or y_margin < 0:
        raise ValueError("margins must be non-negative")

    ego = state.ego
    ego_x = float(ego.x)
    ego_y = float(ego.y)
    ego_vx = float(ego.velocity.x)
    ego_vy = float(ego.velocity.y)
    ego_w = int(ego.sprite.get_width())
    ego_h = int(ego.sprite.get_height())
    ego_cx = ego_x + float(ego_w) * 0.5

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

    min_car_clearance = math.inf
    min_wall_clearance = math.inf
    collision_tick: int | None = None
    collision_kind: str | None = None
    collision_lane: int | None = None
    collision_side: str | None = None
    recovery_actions: List[int] = []

    for tick in range(1, horizon + 1):
        if tick == 1:
            step_action = action
        else:
            step_action = _projected_emergency_action(
                ego_x=ego_x,
                ego_y=ego_y,
                ego_w=ego_w,
                ego_h=ego_h,
                ego_vx=ego_vx,
                ego_vy=ego_vy,
                cars=cars,
                remaining_ticks=horizon - tick + 1,
            )
            recovery_actions.append(int(step_action))

        ego_vx, ego_vy = _apply_primitive(
            step_action,
            ego_vx,
            ego_vy,
        )
        ego_y += ego_vy

        # Existing opponents use frozen current vx. Their screen-relative x
        # changes by other_vx - ego_vx exactly as in the challenge update.
        for car in cars:
            if not bool(car["active"]):
                continue
            car["x"] += float(car["vx"]) - ego_vx
            if float(car["x"]) < -1000.0 or float(car["x"]) > 2600.0:
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

        for car in cars:
            if not bool(car["active"]):
                continue

            car_ix = int(float(car["x"]))
            car_iy = int(float(car["y"]))

            ego_center_y = float(ego_iy) + float(ego_h) * 0.5
            car_center_x = float(car_ix) + float(car["w"]) * 0.5
            car_center_y = float(car_iy) + float(car["h"]) * 0.5

            x_sep = (
                abs(car_center_x - ego_cx)
                - float(ego_w + int(car["w"])) * 0.5
                - float(x_margin)
            )
            y_sep = (
                abs(car_center_y - ego_center_y)
                - float(ego_h + int(car["h"])) * 0.5
                - float(y_margin)
            )
            # For AABBs, positive max(x_sep, y_sep) means at least one axis is
            # separated. <=0 means overlap on both axes.
            clearance = max(x_sep, y_sep)
            min_car_clearance = min(min_car_clearance, clearance)

            if not _rect_overlap(
                ego_ix,
                ego_iy,
                ego_w,
                ego_h,
                car_ix,
                car_iy,
                int(car["w"]),
                int(car["h"]),
                x_margin=x_margin,
                y_margin=y_margin,
            ):
                continue

            collision_tick = int(tick)
            collision_kind = "car"
            collision_lane = int(car["lane"])
            collision_side = (
                "front" if car_center_x - ego_cx >= 0.0 else "rear"
            )
            break

        if collision_tick is not None:
            break

    if not math.isfinite(min_car_clearance):
        min_car_clearance = 1e9
    if not math.isfinite(min_wall_clearance):
        min_wall_clearance = 1e9

    return {
        "action": int(action),
        "action_name": ACTIONS[int(action)],
        "horizon": int(horizon),
        "safe": collision_tick is None,
        "collision_tick": collision_tick,
        "collision_kind": collision_kind,
        "collision_lane": collision_lane,
        "collision_side": collision_side,
        "min_car_clearance": float(min_car_clearance),
        "min_wall_clearance": float(min_wall_clearance),
        "final_vx": float(ego_vx),
        "final_vy": float(ego_vy),
        "final_y": float(ego_y),
        "recovery_actions": recovery_actions,
    }


def primitive_safety_mask(
    state,
    *,
    horizon: int = 24,
    x_margin: float = 0.0,
    y_margin: float = 0.0,
) -> Tuple[List[bool], List[Dict[str, Any]]]:
    """Project all five primitive actions and return their short-horizon mask."""
    projections = [
        project_primitive_action(
            state,
            action,
            horizon=horizon,
            x_margin=x_margin,
            y_margin=y_margin,
        )
        for action in range(len(ACTIONS))
    ]
    return [bool(item["safe"]) for item in projections], projections


def safest_projected_action(
    projections: List[Dict[str, Any]],
    preference: List[float] | None = None,
) -> int:
    """Choose the least-bad candidate when every action projects unsafe."""
    if len(projections) != len(ACTIONS):
        raise ValueError("expected one projection per primitive action")
    if preference is None:
        preference = [0.0] * len(ACTIONS)
    if len(preference) != len(ACTIONS):
        raise ValueError("preference must match action count")

    def rank(index: int):
        item = projections[index]
        collision_tick = item["collision_tick"]
        survived = (
            int(item["horizon"]) + 1
            if collision_tick is None
            else int(collision_tick)
        )
        clearance = min(
            float(item["min_car_clearance"]),
            float(item["min_wall_clearance"]),
        )
        # Later failure dominates; then prefer more clearance, lower residual
        # lateral speed, and finally the teacher's own preference.
        return (
            survived,
            clearance,
            -abs(float(item["final_vy"])),
            float(preference[index]),
        )

    return int(max(range(len(ACTIONS)), key=rank))
