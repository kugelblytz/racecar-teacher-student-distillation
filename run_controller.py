"""Run the deterministic controller directly inside the local pygame simulator."""

import pygame

import src.game.core as game_core
from controller import FullThrottleLaneController, PredictState
from src.game.core import game_loop, initialize_game_state


controller = FullThrottleLaneController()
action_queue = []


def state_to_request() -> PredictState:
    sensors = {sensor.name: sensor.reading for sensor in game_core.STATE.sensors}
    return PredictState(
        did_crash=game_core.STATE.crashed,
        elapsed_ticks=game_core.STATE.ticks,
        distance=game_core.STATE.distance,
        velocity={
            "x": float(game_core.STATE.ego.velocity.x),
            "y": float(game_core.STATE.ego.velocity.y),
        },
        sensors=sensors,
    )


def privileged_lane_guard(direction: int) -> bool:
    """Privileged local-only rollout guard for validating the architecture.

    Simulates the exact deterministic ego lateral macro against all current cars
    and both walls. Other-car longitudinal velocity is frozen at its current
    value, with inflated collision margins to absorb random-walk uncertainty.
    """
    state = game_core.STATE
    ego = state.ego
    current = controller.current_lane
    target_index = current + direction
    if not (0 <= target_index < len(state.road.lanes)):
        return False

    steer_ticks = controller.cfg.steer_ticks
    counter_ticks = controller.cfg.countersteer_ticks
    horizon = steer_ticks + counter_ticks

    ego_w = float(ego.sprite.get_width())
    ego_h = float(ego.sprite.get_height())
    ego_cx = float(ego.rect.centerx)
    y = float(ego.rect.centery)
    vy = float(ego.velocity.y)
    ego_vx = float(ego.velocity.x)

    # Inflate both axes a little because opponent vx performs a random walk.
    x_margin = 55.0
    y_margin = 18.0

    cars = []
    for car in state.cars:
        if car is ego:
            continue
        cars.append(
            {
                "cx": float(car.rect.centerx),
                "cy": float(car.rect.centery),
                "vx": float(car.velocity.x),
                "w": float(car.sprite.get_width()),
                "h": float(car.sprite.get_height()),
            }
        )

    top_wall = float(state.road.walls[0].rect.bottom)
    bottom_wall = float(state.road.walls[1].rect.top)

    for t in range(1, horizon + 1):
        # Exact ego lateral macro ordering: turn first, then y += vy.
        if t <= steer_ticks:
            vy += 0.1 * direction
        else:
            vy -= 0.1 * direction
        y += vy

        ego_top = y - ego_h / 2.0
        ego_bottom = y + ego_h / 2.0
        if ego_top <= top_wall + y_margin or ego_bottom >= bottom_wall - y_margin:
            return False

        for car in cars:
            rel_cx = (car["cx"] - ego_cx) + (car["vx"] - ego_vx) * t
            horizontal_overlap = abs(rel_cx) < ((ego_w + car["w"]) / 2.0 + x_margin)
            vertical_overlap = abs(car["cy"] - y) < ((ego_h + car["h"]) / 2.0 + y_margin)

            if horizontal_overlap and vertical_overlap:
                return False

    return True


def privileged_brake_guard() -> bool:
    """Exact-state local braking trigger for the current lane.

    Starts braking when the longitudinal gap to any laterally-overlapping car
    ahead is within relative stopping distance plus a conservative buffer.
    """
    state = game_core.STATE
    ego = state.ego

    ego_left = float(ego.rect.left)
    ego_right = float(ego.rect.right)
    ego_top = float(ego.rect.top)
    ego_bottom = float(ego.rect.bottom)
    ego_vx = float(ego.velocity.x)

    # Extra room covers opponent random-walk speed changes, 4-tick action
    # batching, and sensor/controller discretization.
    safety_buffer = 220.0
    reaction_ticks = controller.cfg.batch_size

    for car in state.cars:
        if car is ego:
            continue

        # Only cars whose rectangles overlap the ego's lateral corridor matter
        # for straight-line braking.
        if float(car.rect.bottom) <= ego_top or float(car.rect.top) >= ego_bottom:
            continue

        gap = float(car.rect.left) - ego_right
        if gap < 0:
            # Already longitudinally overlapping while laterally overlapping.
            return True

        closing = ego_vx - float(car.velocity.x)
        if closing <= 0:
            continue

        reaction_distance = closing * reaction_ticks
        stopping_distance = (closing * closing) / (2.0 * controller.cfg.brake_deceleration)

        if gap <= reaction_distance + stopping_distance + safety_buffer:
            return True

    return False


def _longitudinal_rollout(action: str, horizon: int = 140):
    """Return (collision_tick, min_clearance) for repeated longitudinal action."""
    state = game_core.STATE
    ego = state.ego

    ego_w = float(ego.sprite.get_width())
    ego_h = float(ego.sprite.get_height())
    ego_cx = float(ego.rect.centerx)
    ego_cy = float(ego.rect.centery)
    ego_vx = float(ego.velocity.x)

    cars = []
    for car in state.cars:
        if car is ego:
            continue

        # Only cars that overlap the ego's current lateral corridor matter while
        # we are evaluating straight-line accelerate/brake actions.
        vertical_overlap = abs(float(car.rect.centery) - ego_cy) < (
            (ego_h + float(car.sprite.get_height())) / 2.0
        )
        if not vertical_overlap:
            continue

        cars.append(
            {
                "rel_x": float(car.rect.centerx) - ego_cx,
                "vx": float(car.velocity.x),
                "half_width_sum": (ego_w + float(car.sprite.get_width())) / 2.0,
            }
        )

    if not cars:
        return None, float("inf")

    rel = [car["rel_x"] for car in cars]
    min_clearance = float("inf")

    for t in range(1, horizon + 1):
        if action == "ACCELERATE":
            ego_vx += 0.1
        elif action == "DECELERATE":
            ego_vx = max(0.0, ego_vx - 0.1)

        for i, car in enumerate(cars):
            # Matches simulator ordering closely: relative position changes by
            # other_vx - ego_vx after the ego action is applied.
            rel[i] += car["vx"] - ego_vx
            clearance = abs(rel[i]) - car["half_width_sum"]
            min_clearance = min(min_clearance, clearance)
            if clearance <= 0:
                return t, min_clearance

    return None, min_clearance


def privileged_longitudinal_advisor():
    """Choose accel vs brake when lane-change escape is unavailable.

    This handles the important case where braking for a slower front car would
    cause a faster rear car to hit us. The choice is receding-horizon: every
    controller batch we roll out both longitudinal actions again from exact state.
    """
    state = game_core.STATE
    ego = state.ego

    # If a full lane-change trajectory is safe, leave control to the normal lane
    # planner. This advisor is specifically for boxed-in longitudinal conflicts.
    left_safe = controller.current_lane > 0 and privileged_lane_guard(-1)
    right_safe = (
        controller.current_lane < controller.cfg.lane_count - 1
        and privileged_lane_guard(+1)
    )
    if left_safe or right_safe:
        return None

    ego_cy = float(ego.rect.centery)
    ego_h = float(ego.sprite.get_height())
    front_threat = False
    rear_threat = False

    for car in state.cars:
        if car is ego:
            continue

        if abs(float(car.rect.centery) - ego_cy) >= (
            (ego_h + float(car.sprite.get_height())) / 2.0
        ):
            continue

        dx = float(car.rect.centerx) - float(ego.rect.centerx)
        if dx > 0 and ego.velocity.x > car.velocity.x:
            front_threat = True
        elif dx < 0 and car.velocity.x > ego.velocity.x:
            rear_threat = True

    if not (front_threat or rear_threat):
        return None

    accel_hit, accel_clear = _longitudinal_rollout("ACCELERATE")
    brake_hit, brake_clear = _longitudinal_rollout("DECELERATE")

    # No predicted collision: keep the full-throttle bias.
    if accel_hit is None and brake_hit is None:
        return "ACCELERATE"

    # One action survives the horizon and the other does not.
    if accel_hit is None:
        return "ACCELERATE"
    if brake_hit is None:
        return "DECELERATE"

    # Both eventually collide: choose the action that buys the most time.
    if accel_hit > brake_hit:
        return "ACCELERATE"
    if brake_hit > accel_hit:
        return "DECELERATE"

    # Tie-break on larger minimum clearance, then preserve full-throttle bias.
    if brake_clear > accel_clear:
        return "DECELERATE"
    return "ACCELERATE"


def controller_action():
    global action_queue
    if not action_queue:
        action_queue = controller.predict(state_to_request())
    return action_queue.pop(0)


def describe_collision():
    state = game_core.STATE
    ego = state.ego
    hits = []

    for car in state.cars:
        if car is ego:
            continue
        if ego.rect.colliderect(car.rect):
            hits.append(
                f"CAR lane={car.lane.name if car.lane else '?'} "
                f"dx={car.rect.centerx - ego.rect.centerx:.1f} "
                f"ego_vx={ego.velocity.x:.2f} other_vx={car.velocity.x:.2f}"
            )

    for i, wall in enumerate(state.road.walls):
        if ego.rect.colliderect(wall.rect):
            hits.append(f"WALL {'top' if i == 0 else 'bottom'} y={ego.rect.centery}")

    if hits:
        print("\n=== COLLISION DIAGNOSTIC ===")
        print(f"tick={state.ticks} distance={state.distance:.1f}")
        print(f"ego lane estimate={controller.current_lane} target={controller.target_lane}")
        print(f"ego x={ego.rect.centerx} y={ego.rect.centery} vx={ego.velocity.x:.2f} vy={ego.velocity.y:.2f}")
        for hit in hits:
            print(hit)
        print("============================\n")


if __name__ == "__main__":
    game_core.get_action = lambda: [controller_action()]
    controller.lane_guard = privileged_lane_guard
    controller.brake_guard = privileged_brake_guard
    controller.longitudinal_advisor = privileged_longitudinal_advisor

    # Patch collision test only to print diagnostics; behavior remains unchanged.
    original_intersects = game_core.intersects

    def diagnostic_intersects(a, b):
        hit = original_intersects(a, b)
        if hit and game_core.STATE is not None and a == game_core.STATE.ego.rect:
            # game_loop will set crashed immediately after this returns.
            pass
        return hit

    game_core.intersects = diagnostic_intersects

    pygame.init()
    initialize_game_state("", seed_value=None)

    try:
        game_loop(verbose=True, log_actions=False)
    finally:
        describe_collision()
        pygame.quit()
