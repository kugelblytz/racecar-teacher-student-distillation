from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class LaneDecision(str, Enum):
    LEFT = "LEFT"
    STAY = "STAY"
    RIGHT = "RIGHT"


@dataclass
class ControllerConfig:
    # Environment geometry from the starter simulator.
    lane_count: int = 5
    road_top: float = 40.0
    road_bottom: float = 1160.0
    initial_lane: int = 2
    sensor_range: float = 1000.0
    ego_half_length: float = 180.0  # ~360 px wide after scaling

    # One clean lane change in the public hand-written controllers is ~48 ticks
    # steering into the lane, followed by ~48 ticks counter-steering.
    steer_ticks: int = 48
    countersteer_ticks: int = 48

    # Keep batches short: reaction latency matters more than HTTP overhead here.
    batch_size: int = 4

    # High-level trigger / safety settings.
    plan_front_distance: float = 900.0
    hard_front_distance: float = 260.0
    min_side_clearance: float = 330.0
    min_diag_clearance: float = 430.0
    min_rear_clearance: float = 330.0
    ttc_plan_ticks: float = 105.0
    ttc_emergency_ticks: float = 68.0
    brake_deceleration: float = 0.1
    brake_distance_margin: float = 120.0

    # Lane-centre tracking. The API does not expose y, so we integrate the known
    # simulator lateral dynamics, starting from the known middle lane.
    lane_snap_tolerance: float = 55.0

    @property
    def lane_height(self) -> float:
        return (self.road_bottom - self.road_top) / self.lane_count

    @property
    def lane_centers(self) -> List[float]:
        h = self.lane_height
        return [self.road_top + (i + 0.5) * h for i in range(self.lane_count)]


@dataclass
class PredictState:
    did_crash: bool
    elapsed_ticks: int
    distance: float
    velocity: Dict[str, float]
    sensors: Dict[str, Optional[float]]


@dataclass
class FullThrottleLaneController:
    """Stateful L/STAY/R controller with deterministic lane-change execution.

    The high-level `choose_lane()` method is deliberately isolated so it can be
    replaced later by a learned student policy without changing safety or
    low-level steering.
    """

    cfg: ControllerConfig = field(default_factory=ControllerConfig)

    estimated_y: float = field(init=False)
    estimated_vy: float = 0.0
    current_lane: int = field(init=False)
    target_lane: int = field(init=False)

    phase: str = "IDLE"  # IDLE | STEER | COUNTER
    phase_remaining: int = 0
    lane_change_dir: int = 0  # -1 left, +1 right

    last_elapsed_ticks: Optional[int] = None
    last_front: Optional[float] = None
    last_front_tick: Optional[int] = None
    filtered_closing_speed: float = 0.0
    # Optional local-simulator safety hook. When set, it receives the proposed
    # lane direction (-1 left, +1 right) and can veto a lane change using
    # privileged simulator state. The competition API leaves this as None.
    lane_guard: object = None
    # Optional local-simulator hook returning whether braking is already urgent
    # from privileged exact state. Competition/API mode leaves this as None.
    brake_guard: object = None
    # Optional local-only privileged longitudinal advisor. It may return
    # "ACCELERATE", "DECELERATE", or None while the controller is idle.
    longitudinal_advisor: object = None

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.current_lane = self.cfg.initial_lane
        self.target_lane = self.cfg.initial_lane
        self.estimated_y = self.cfg.lane_centers[self.cfg.initial_lane]
        self.estimated_vy = 0.0
        self.phase = "IDLE"
        self.phase_remaining = 0
        self.lane_change_dir = 0
        self.last_elapsed_ticks = None
        self.last_front = None
        self.last_front_tick = None
        self.filtered_closing_speed = 0.0

    @staticmethod
    def _sensor(sensors: Dict[str, Optional[float]], name: str) -> float:
        value = sensors.get(name)
        return 1000.0 if value is None else float(value)

    def _sync_episode(self, state: PredictState) -> None:
        # A lower tick means a new simulator episode. Crash also terminates an
        # episode, so the next request should begin from a clean controller.
        if self.last_elapsed_ticks is not None and state.elapsed_ticks < self.last_elapsed_ticks:
            self.reset()
        self.last_elapsed_ticks = state.elapsed_ticks

        # velocity.y is legal observation and lets us keep lateral velocity in
        # sync even if the caller changes batch size.
        self.estimated_vy = float(state.velocity.get("y", self.estimated_vy))

        # Snap our integrated y to the nearest lane when we are nearly settled.
        nearest = min(range(self.cfg.lane_count), key=lambda i: abs(self.estimated_y - self.cfg.lane_centers[i]))
        if abs(self.estimated_y - self.cfg.lane_centers[nearest]) <= self.cfg.lane_snap_tolerance and abs(self.estimated_vy) < 0.35:
            self.current_lane = nearest
            if self.phase == "IDLE":
                self.target_lane = nearest

    def _update_front_closing_speed(self, state: PredictState) -> None:
        front = self._sensor(state.sensors, "front")
        tick = int(state.elapsed_ticks)

        if self.last_front is not None and self.last_front_tick is not None and tick > self.last_front_tick:
            dt = tick - self.last_front_tick
            raw = (self.last_front - front) / dt
            # Ignore disappearance/reappearance spikes and smooth noisy random walk.
            if -5.0 <= raw <= 25.0:
                alpha = 0.45
                self.filtered_closing_speed = alpha * max(0.0, raw) + (1.0 - alpha) * self.filtered_closing_speed

        self.last_front = front
        self.last_front_tick = tick

    def _ttc_ticks(self, state: PredictState) -> float:
        front = self._sensor(state.sensors, "front")
        clearance = max(0.0, front - self.cfg.ego_half_length)
        c = self.filtered_closing_speed
        if c < 1e-3:
            return float("inf")
        return clearance / c

    def _lane_side_clear(self, sensors: Dict[str, Optional[float]], direction: int) -> bool:
        if direction < 0:
            vals = {
                "side": self._sensor(sensors, "left_side"),
                "front": self._sensor(sensors, "left_side_front"),
                "diag_front": self._sensor(sensors, "front_left_front"),
                "back": self._sensor(sensors, "left_side_back"),
                "diag_back": self._sensor(sensors, "back_left_back"),
                "back2": self._sensor(sensors, "left_back"),
            }
        else:
            vals = {
                "side": self._sensor(sensors, "right_side"),
                "front": self._sensor(sensors, "right_side_front"),
                "diag_front": self._sensor(sensors, "front_right_front"),
                "back": self._sensor(sensors, "right_side_back"),
                "diag_back": self._sensor(sensors, "back_right_back"),
                "back2": self._sensor(sensors, "right_back"),
            }

        return (
            vals["side"] >= self.cfg.min_side_clearance
            and vals["front"] >= self.cfg.min_side_clearance
            and vals["back"] >= self.cfg.min_side_clearance
            and vals["diag_front"] >= self.cfg.min_diag_clearance
            and vals["diag_back"] >= self.cfg.min_rear_clearance
            and vals["back2"] >= self.cfg.min_rear_clearance
        )

    def _corridor_score(self, sensors: Dict[str, Optional[float]], direction: int) -> float:
        """Score a neighbouring corridor using front, side and rear clearance."""
        if direction < 0:
            names = [
                ("front_left_front", 1.4),
                ("left_front", 1.2),
                ("left_side_front", 1.1),
                ("left_side", 0.9),
                ("left_side_back", 0.7),
                ("left_back", 0.6),
                ("back_left_back", 0.5),
            ]
        else:
            names = [
                ("front_right_front", 1.4),
                ("right_front", 1.2),
                ("right_side_front", 1.1),
                ("right_side", 0.9),
                ("right_side_back", 0.7),
                ("right_back", 0.6),
                ("back_right_back", 0.5),
            ]
        return sum(weight * self._sensor(sensors, name) for name, weight in names)

    def choose_lane(self, state: PredictState) -> LaneDecision:
        """Replace this method with the learned L/STAY/R student later."""
        sensors = state.sensors
        front = self._sensor(sensors, "front")
        ttc = self._ttc_ticks(state)

        should_plan = front < self.cfg.plan_front_distance or ttc < self.cfg.ttc_plan_ticks
        if not should_plan:
            return LaneDecision.STAY

        candidates = []
        left_guard_ok = self.lane_guard is None or bool(self.lane_guard(-1))
        right_guard_ok = self.lane_guard is None or bool(self.lane_guard(+1))

        if self.current_lane > 0 and left_guard_ok and self._lane_side_clear(sensors, -1):
            candidates.append((self._corridor_score(sensors, -1), LaneDecision.LEFT))
        if self.current_lane < self.cfg.lane_count - 1 and right_guard_ok and self._lane_side_clear(sensors, +1):
            candidates.append((self._corridor_score(sensors, +1), LaneDecision.RIGHT))

        if not candidates:
            return LaneDecision.STAY
        return max(candidates, key=lambda x: x[0])[1]

    def _start_lane_change(self, decision: LaneDecision) -> None:
        direction = -1 if decision == LaneDecision.LEFT else 1
        target = self.current_lane + direction
        if not (0 <= target < self.cfg.lane_count):
            return
        self.target_lane = target
        self.lane_change_dir = direction
        self.phase = "STEER"
        self.phase_remaining = self.cfg.steer_ticks

    def _macro_action(self) -> Optional[str]:
        if self.phase == "IDLE":
            return None

        if self.phase == "STEER":
            action = "STEER_LEFT" if self.lane_change_dir < 0 else "STEER_RIGHT"
            self.phase_remaining -= 1
            if self.phase_remaining <= 0:
                self.phase = "COUNTER"
                self.phase_remaining = self.cfg.countersteer_ticks
            return action

        # Counter-steer to kill lateral velocity.
        action = "STEER_RIGHT" if self.lane_change_dir < 0 else "STEER_LEFT"
        self.phase_remaining -= 1
        if self.phase_remaining <= 0:
            self.phase = "IDLE"
            self.phase_remaining = 0
            self.current_lane = self.target_lane
            self.lane_change_dir = 0
        return action

    def _advance_internal_dynamics(self, action: str) -> None:
        # Mirrors Car.turn() and Car.update() for the ego vehicle.
        if action == "STEER_LEFT":
            self.estimated_vy -= 0.1
        elif action == "STEER_RIGHT":
            self.estimated_vy += 0.1
        self.estimated_y += self.estimated_vy

    def _emergency_brake_needed(self, state: PredictState) -> bool:
        if self.phase != "IDLE":
            # Once committed to a lane change, finishing it is usually safer
            # than alternating brake/steer. A future version can use rollout MPC.
            return False

        front = self._sensor(state.sensors, "front")
        ttc = self._ttc_ticks(state)
        clearance = max(0.0, front - self.cfg.ego_half_length)
        closing = max(0.0, self.filtered_closing_speed)

        # Relative stopping distance if the front car roughly maintains speed.
        # DECELERATE changes ego vx by only 0.1 px/tick, so at high closing
        # speeds this can be much larger than a fixed TTC threshold suggests.
        braking_distance = (
            (closing * closing) / (2.0 * self.cfg.brake_deceleration)
            if closing > 0.0
            else 0.0
        )
        braking_urgent = clearance <= braking_distance + self.cfg.brake_distance_margin
        if self.brake_guard is not None:
            braking_urgent = bool(self.brake_guard())

        if (
            front >= self.cfg.hard_front_distance
            and ttc >= self.cfg.ttc_emergency_ticks
            and not braking_urgent
        ):
            return False

        left_guard_ok = self.lane_guard is None or bool(self.lane_guard(-1))
        right_guard_ok = self.lane_guard is None or bool(self.lane_guard(+1))
        left_ok = self.current_lane > 0 and left_guard_ok and self._lane_side_clear(state.sensors, -1)
        right_ok = self.current_lane < self.cfg.lane_count - 1 and right_guard_ok and self._lane_side_clear(state.sensors, +1)
        return not (left_ok or right_ok)

    def predict(self, state: PredictState) -> List[str]:
        self._sync_episode(state)
        self._update_front_closing_speed(state)

        if state.did_crash:
            return ["NOTHING"]

        actions: List[str] = []
        for _ in range(self.cfg.batch_size):
            action = self._macro_action()

            if action is None:
                advised = self.longitudinal_advisor() if self.longitudinal_advisor is not None else None
                if advised in ("ACCELERATE", "DECELERATE"):
                    action = advised
                elif self._emergency_brake_needed(state):
                    action = "DECELERATE"
                else:
                    decision = self.choose_lane(state)
                    if decision != LaneDecision.STAY:
                        self._start_lane_change(decision)
                        action = self._macro_action()
                    else:
                        # Full-throttle policy: accelerate whenever not steering
                        # and not forced to brake by the safety shield.
                        action = "ACCELERATE"

            assert action is not None
            actions.append(action)
            self._advance_internal_dynamics(action)

        return actions
