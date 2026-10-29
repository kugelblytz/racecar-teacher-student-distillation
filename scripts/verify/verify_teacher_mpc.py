"""Focused deterministic checks for the privileged teacher MPC."""

from __future__ import annotations

import contextlib
import io
import math
import os
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import src.game.core as core
import src.elements.car as car_module
from scripts.eval.evaluate_teacher_mpc import _paired_comparison
from src.mathematics.vector import Vector
from training.gpu_sim import ACTIONS
from training.teacher_lane_safety import NOTHING, STEER_LEFT, STEER_RIGHT, lane_centers
from training.teacher_mpc import (
    _safe_rank,
    _teacher_target_top_left,
    choose_confirmed_mpc_action,
    choose_mpc_action,
    mpc_planning_decision,
    mpc_trigger_reason,
    opponent_position_uncertainty,
    project_maneuver,
    should_invoke_mpc,
)


def _reset(seed: int = 1234):
    with contextlib.redirect_stdout(io.StringIO()):
        core.initialize_game_state("", seed)
    state = core.STATE
    state.cars = [state.ego]
    state.car_bucket = []
    return state


def _add_car(state, *, lane: int, x: float, vx: float):
    car = car_module.Car(
        "blue",
        Vector(float(vx), 0.0),
        lane=state.road.lanes[int(lane)],
        target_height=int(state.road.lane_height * 0.8),
    )
    car.x = float(x)
    car.y = float(lane_centers(state)[lane]) - car.sprite.get_height() * 0.5
    state.cars.append(car)
    return car


def verify_counter_steering() -> None:
    state = _reset()
    ego_h = state.ego.sprite.get_height()
    centers = lane_centers(state)
    target = centers[3]
    for center_y in (target - 3.0, target, target + 2.0, target + 3.0, target + 6.0):
        top = _teacher_target_top_left(
            state,
            STEER_LEFT,
            ego_y=center_y - ego_h * 0.5,
            ego_vy=0.5,
            ego_h=ego_h,
        )
        assert top is not None
        assert math.isclose(top + ego_h * 0.5, target)

    for center_y in (centers[1] - 6.0, centers[1] - 3.0, centers[1] + 3.0):
        top = _teacher_target_top_left(
            state,
            STEER_RIGHT,
            ego_y=center_y - ego_h * 0.5,
            ego_vy=-0.5,
            ego_h=ego_h,
        )
        assert top is not None
        assert math.isclose(top + ego_h * 0.5, centers[1])
    print("PASS counter-steering target continuity")


def _replay_projection(state, projection):
    progress = 0.0
    collision_tick = None
    original_random = car_module.random_number
    car_module.random_number = lambda: 0.5
    try:
        for tick, action in enumerate(projection["actions"], 1):
            core.handle_action(ACTIONS[int(action)])
            progress += float(state.ego.velocity.x)
            core.update_cars()
            core.remove_passed_cars()
            for car in state.cars:
                if car is not state.ego and core.intersects(state.ego.rect, car.rect):
                    collision_tick = tick
                    break
            if collision_tick is None:
                for wall in state.road.walls:
                    if core.intersects(state.ego.rect, wall.rect):
                        collision_tick = tick
                        break
            if collision_tick is not None:
                break
    finally:
        car_module.random_number = original_random
    return progress, collision_tick


def verify_projection_parity() -> None:
    state = _reset(2001)
    projection = project_maneuver(
        state,
        "left",
        proposed_action=STEER_LEFT,
        horizon=24,
        opponent_uncertainty_sigma=0.0,
    )
    progress, collision_tick = _replay_projection(state, projection)
    assert collision_tick == projection["collision_tick"]
    assert math.isclose(progress, projection["progress"], abs_tol=1e-9)
    assert math.isclose(state.ego.y, projection["final_y"], abs_tol=1e-9)
    assert math.isclose(state.ego.velocity.x, projection["final_vx"], abs_tol=1e-9)
    assert math.isclose(state.ego.velocity.y, projection["final_vy"], abs_tol=1e-9)

    state = _reset(2002)
    ego = state.ego
    lane = 2
    _add_car(
        state,
        lane=lane,
        x=float(ego.x) + float(ego.sprite.get_width()) + 240.0,
        vx=float(ego.velocity.x) - 2.0,
    )
    projection = project_maneuver(
        state,
        "teacher",
        proposed_action=NOTHING,
        horizon=160,
        opponent_uncertainty_sigma=0.0,
    )
    _, collision_tick = _replay_projection(state, projection)
    assert projection["collision_kind"] == "car"
    assert collision_tick == projection["collision_tick"]
    print("PASS projection parity against deterministic original simulator")


def verify_uncertainty() -> None:
    state = _reset(3001)
    ego = state.ego
    _add_car(
        state,
        lane=2,
        x=float(ego.x) + float(ego.sprite.get_width()) + 500.0,
        vx=float(ego.velocity.x) - 1.0,
    )
    projections = [
        project_maneuver(
            state,
            "teacher",
            proposed_action=NOTHING,
            horizon=96,
            opponent_uncertainty_sigma=sigma,
        )
        for sigma in (0.0, 2.0, 3.0)
    ]
    clearances = [float(item["min_car_clearance"]) for item in projections]
    assert clearances[0] >= clearances[1] >= clearances[2]
    assert opponent_position_uncertainty(1, 3.0) == 0.0
    assert opponent_position_uncertainty(96, 3.0) > 90.0
    print("PASS conservative opponent uncertainty envelope")


def verify_triggering() -> None:
    state = _reset(4001)
    ego = state.ego
    car = _add_car(
        state,
        lane=2,
        x=float(ego.x) + float(ego.sprite.get_width()) + 500.0,
        vx=float(ego.velocity.x) + 5.0,
    )
    assert mpc_trigger_reason(state, proposed_action=NOTHING) is None

    car.velocity.x = float(ego.velocity.x) - 5.0
    assert mpc_trigger_reason(state, proposed_action=NOTHING) == "closing"

    car.x = float(ego.x) + float(ego.sprite.get_width()) + 10.0
    car.velocity.x = float(ego.velocity.x) + 5.0
    assert mpc_trigger_reason(state, proposed_action=NOTHING) == "immediate"

    car.x = float(ego.x) + float(ego.sprite.get_width()) + 500.0
    ego.velocity.y = 0.3
    assert mpc_trigger_reason(state, proposed_action=NOTHING) == "lateral"
    ego.velocity.y = 0.0
    car.lane = state.road.lanes[1]
    car.y = lane_centers(state)[1] - car.sprite.get_height() * 0.5
    assert mpc_trigger_reason(state, proposed_action=NOTHING) is None
    assert mpc_trigger_reason(state, proposed_action=STEER_LEFT) == "lateral"
    assert should_invoke_mpc(state, proposed_action=STEER_LEFT)

    car.x = float(ego.x) + float(ego.sprite.get_width()) + 700.0
    car.velocity.x = float(ego.velocity.x) - 5.0
    reason, planning_horizon = mpc_planning_decision(
        state,
        proposed_action=STEER_LEFT,
        horizon=72,
        strategic_horizon=120,
    )
    assert reason == "strategic_closing"
    assert planning_horizon == 120
    reason, planning_horizon = mpc_planning_decision(
        state,
        proposed_action=STEER_LEFT,
        horizon=72,
        strategic_horizon=0,
    )
    assert reason == "lateral"
    assert planning_horizon == 72
    print("PASS causal trigger cases")


def _projection_template(**changes):
    item = {
        "plan": "teacher",
        "first_action": NOTHING,
        "first_action_name": ACTIONS[NOTHING],
        "safe": True,
        "healthy": True,
        "collision_tick": None,
        "collision_kind": None,
        "collision_lane": None,
        "collision_side": None,
        "min_car_clearance": 20.0,
        "min_wall_clearance": 20.0,
        "progress": 1000.0,
        "zero_speed_ticks": 0,
        "final_vx": 10.0,
        "final_vy": 0.0,
        "final_y": 0.0,
        "final_lane_offset": 0.0,
        "opponent_uncertainty_sigma": 3.0,
        "max_opponent_position_uncertainty": 0.0,
        "actions": [NOTHING],
    }
    item.update(changes)
    return item


def verify_selection() -> None:
    safe_centered = _projection_template()
    unsafe_progress = _projection_template(
        min_car_clearance=10.0,
        progress=100_000.0,
        final_lane_offset=80.0,
    )
    rank_args = {
        "proposed_action": NOTHING,
        "horizon": 96,
        "lane_half_width": 112.0,
        "clearance_threshold": 8.0,
        "wall_threshold": 4.0,
        "clearance_cap": 120.0,
    }
    assert _safe_rank(safe_centered, **rank_args) > _safe_rank(
        unsafe_progress, **rank_args
    )
    offset = _projection_template(final_lane_offset=40.0)
    assert _safe_rank(safe_centered, **rank_args) > _safe_rank(offset, **rank_args)

    state = _reset(5001)
    action_by_plan = {
        "teacher": NOTHING,
        "settle": NOTHING,
        "left": STEER_LEFT,
        "right": STEER_RIGHT,
        "accelerate": 0,
        "brake": 1,
        "coast": NOTHING,
    }

    def all_unsafe(_state, plan, **_kwargs):
        tick = 20 if plan == "right" else 10
        return _projection_template(
            plan=plan,
            first_action=action_by_plan[plan],
            safe=False,
            healthy=False,
            collision_tick=tick,
            collision_kind="car",
            min_car_clearance=-2.0,
        )

    with patch("training.teacher_mpc.project_maneuver", side_effect=all_unsafe):
        action, diagnostic = choose_mpc_action(state, NOTHING, horizon=32)
    assert action == STEER_RIGHT
    assert diagnostic["selected_plan"] == "right"
    assert diagnostic["all_unsafe"] is True
    required = {
        "overridden",
        "proposed_action",
        "selected_action",
        "selected_plan",
        "teacher_projection",
        "selected_projection",
        "plans_evaluated",
        "all_unsafe",
        "all_unhealthy",
        "opponent_uncertainty_sigma",
        "projections",
    }
    assert required <= diagnostic.keys()

    state = _reset(5002)
    action, diagnostic = choose_mpc_action(state, NOTHING, horizon=12)
    assert action == NOTHING
    assert diagnostic["plans_evaluated"] == 1
    assert required <= diagnostic.keys()
    print("PASS safety-first and all-unsafe selection")


def verify_strategic_confirmation() -> None:
    state = _reset(5501)
    horizons = []

    def rear_then_base(_state, _action, *, horizon, **_kwargs):
        horizons.append(horizon)
        if horizon == 119:
            teacher = _projection_template(
                safe=False,
                healthy=False,
                collision_tick=100,
                collision_kind="car",
                collision_side="rear",
            )
            return STEER_RIGHT, {"teacher_projection": teacher}
        return NOTHING, {"teacher_projection": _projection_template()}

    with patch(
        "training.teacher_mpc.choose_mpc_action", side_effect=rear_then_base
    ):
        action, diagnostic, strategic_used = choose_confirmed_mpc_action(
            state,
            NOTHING,
            horizon=72,
            planning_horizon=119,
        )
    assert action == NOTHING
    assert diagnostic["teacher_projection"]["safe"] is True
    assert strategic_used is False
    assert horizons == [119, 72]

    front = _projection_template(
        safe=False,
        healthy=False,
        collision_tick=100,
        collision_kind="car",
        collision_side="front",
    )
    with patch(
        "training.teacher_mpc.choose_mpc_action",
        return_value=(STEER_RIGHT, {"teacher_projection": front}),
    ) as mocked:
        action, _, strategic_used = choose_confirmed_mpc_action(
            state,
            NOTHING,
            horizon=72,
            planning_horizon=119,
        )
    assert action == STEER_RIGHT
    assert strategic_used is True
    assert mocked.call_count == 1
    print("PASS front-only strategic confirmation")


def verify_validation_and_reporting() -> None:
    state = _reset(6001)
    invalid_calls = (
        lambda: project_maneuver(state, "teacher", proposed_action=-1),
        lambda: project_maneuver(
            state, "teacher", proposed_action=NOTHING, opponent_uncertainty_sigma=-1
        ),
        lambda: mpc_trigger_reason(state, horizon=0),
        lambda: choose_mpc_action(state, NOTHING, clearance_cap=0),
        lambda: mpc_planning_decision(state, strategic_horizon=-1),
        lambda: choose_confirmed_mpc_action(
            state, NOTHING, horizon=72, planning_horizon=48
        ),
    )
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    base = [
        {"seed": 1, "crashed": True, "distance": 10.0},
        {"seed": 2, "crashed": False, "distance": 20.0},
    ]
    variant = [
        {"seed": 1, "crashed": False, "distance": 30.0},
        {"seed": 2, "crashed": True, "distance": 5.0},
    ]
    comparison = _paired_comparison(
        base, variant, bootstrap_samples=100, bootstrap_seed=99
    )
    assert comparison["crashes_rescued"] == 1
    assert comparison["new_crashes_introduced"] == 1
    assert len(comparison["crash_rate_delta_ci95"]) == 2
    print("PASS validation and paired-report shape")


def main() -> None:
    verify_counter_steering()
    verify_projection_parity()
    verify_uncertainty()
    verify_triggering()
    verify_selection()
    verify_strategic_confirmation()
    verify_validation_and_reporting()
    print("PASS teacher MPC verifier")


if __name__ == "__main__":
    main()
