"""GPU-native batched race-car simulator for privileged PPO training.

This simulator matches the public challenge dynamics and traffic distribution,
but intentionally does not reproduce Python random.Random bit-for-bit. Training
uses independent CUDA RNG streams; evaluation should be performed on the original
headless simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
OPPONENT_COUNT = 4
MAX_TICKS = 3600

ROAD_TOP = 40.0
ROAD_BOTTOM = 1160.0
LANE_HEIGHT = (ROAD_BOTTOM - ROAD_TOP) / LANE_COUNT
LANE_CENTERS = torch.tensor(
    [ROAD_TOP + (i + 0.5) * LANE_HEIGHT for i in range(LANE_COUNT)],
    dtype=torch.float64,
)

# All public car PNGs are 800x397 and are scaled to target height 179.
CAR_HEIGHT = int(LANE_HEIGHT * 0.8)  # 179
CAR_WIDTH = int((800.0 / 397.0) * CAR_HEIGHT)  # 360
EGO_X = float((SCREEN_WIDTH // 2) - (CAR_WIDTH // 2))  # top-left
EGO_CENTER_X = EGO_X + CAR_WIDTH / 2.0
EGO_INITIAL_Y = float(int(600.0 - CAR_HEIGHT / 2.0))
EGO_INITIAL_VX = 10.0

RETIRE_MIN_X = -1000.0
RETIRE_MAX_X = SCREEN_WIDTH + 1000.0

ACTIONS = (
    "NOTHING",
    "ACCELERATE",
    "DECELERATE",
    "STEER_LEFT",
    "STEER_RIGHT",
)

OBS_SIZE = 5 + LANE_COUNT * 4

# Exact public sensor ordering. Vector(0, -1).rotate(degrees) gives the ray
# direction (sin(theta), -cos(theta)) in screen coordinates.
SENSOR_SPECS = (
    (90.0, "front"),
    (135.0, "right_front"),
    (180.0, "right_side"),
    (225.0, "right_back"),
    (270.0, "back"),
    (315.0, "left_back"),
    (0.0, "left_side"),
    (45.0, "left_front"),
    (22.5, "left_side_front"),
    (67.5, "front_left_front"),
    (112.5, "front_right_front"),
    (157.5, "right_side_front"),
    (202.5, "right_side_back"),
    (247.5, "back_right_back"),
    (292.5, "back_left_back"),
    (337.5, "left_side_back"),
)
SENSOR_NAMES = tuple(name for _, name in SENSOR_SPECS)
SENSOR_COUNT = len(SENSOR_SPECS)
SENSOR_RANGE = 1000.0

# Legal API observation V1:
# 16 normalized sensor distances (None -> 1.0)
# 16 hit masks (preserves Optional[float] semantics)
# ego vx, ego vy, time remaining, normalized distance
# + previous primitive action one-hot (controller-known legal history)
LEGAL_OBS_SIZE = SENSOR_COUNT * 2 + 4 + len(ACTIONS)  # 41
# V2 appends one controller-owned normalized lateral-position feature.
LEGAL_OBS_SIZE_TRACKED_Y = LEGAL_OBS_SIZE + 1  # 42

TRACKED_Y_UPPER_BOUNDARY = ROAD_TOP
TRACKED_Y_LOWER_BOUNDARY = ROAD_BOTTOM - CAR_HEIGHT
TRACKED_Y_MIDPOINT = (
    TRACKED_Y_UPPER_BOUNDARY + TRACKED_Y_LOWER_BOUNDARY
) * 0.5
TRACKED_Y_HALF_SPAN = (
    TRACKED_Y_LOWER_BOUNDARY - TRACKED_Y_UPPER_BOUNDARY
) * 0.5


@dataclass
class EpisodeBatch:
    done: torch.Tensor
    crashed: torch.Tensor
    hit_car: torch.Tensor
    hit_wall: torch.Tensor
    distance: torch.Tensor
    ticks: torch.Tensor
    final_speed: torch.Tensor


class GpuRaceCarBatch:
    """Thousands of independent race-car environments in CUDA tensors."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cuda",
        seed: int = 2025,
        crash_penalty: float = 250.0,
        wall_crash_penalty: Optional[float] = None,
        reward_scale: float = 1000.0,
        opponent_count: int = OPPONENT_COUNT,
        survival_bonus: float = 0.0,
        ttc_risk_penalty: float = 0.0,
        ttc_risk_horizon: float = 120.0,
        ttc_lateral_margin: float = 24.0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.crash_penalty = float(crash_penalty)
        self.wall_crash_penalty = (
            self.crash_penalty
            if wall_crash_penalty is None
            else float(wall_crash_penalty)
        )
        self.reward_scale = float(reward_scale)
        self.survival_bonus = float(survival_bonus)
        self.ttc_risk_penalty = float(ttc_risk_penalty)
        self.ttc_risk_horizon = float(ttc_risk_horizon)
        self.ttc_lateral_margin = float(ttc_lateral_margin)
        if self.survival_bonus < 0:
            raise ValueError("survival_bonus must be non-negative")
        if self.ttc_risk_penalty < 0:
            raise ValueError("ttc_risk_penalty must be non-negative")
        if self.ttc_risk_horizon <= 0:
            raise ValueError("ttc_risk_horizon must be positive")
        if self.ttc_lateral_margin < 0:
            raise ValueError("ttc_lateral_margin must be non-negative")
        self.opponent_count = int(opponent_count)
        if not 1 <= self.opponent_count <= OPPONENT_COUNT:
            raise ValueError(
                f"opponent_count must be in [1, {OPPONENT_COUNT}]"
            )

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

        n = self.num_envs
        d = self.device

        self.ticks = torch.zeros(n, dtype=torch.int32, device=d)
        self.last_action = torch.zeros(n, dtype=torch.int64, device=d)
        # The public simulator stores physics in Python floats (IEEE float64).
        # Keep physical state in float64 so long rollouts and int-truncated
        # collision boundaries track the original closely. Observations remain
        # float32 for the neural network.
        self.distance = torch.zeros(n, dtype=torch.float64, device=d)
        self.ego_vx = torch.full((n,), EGO_INITIAL_VX, dtype=torch.float64, device=d)
        self.ego_vy = torch.zeros(n, dtype=torch.float64, device=d)
        self.ego_y = torch.full((n,), EGO_INITIAL_Y, dtype=torch.float64, device=d)

        # Controller-owned reconstruction. These tensors are updated only from
        # issued actions; they deliberately do not read ego_y. Keeping them
        # separate lets us prove the new legal feature is not privileged state.
        self.tracked_vy = torch.zeros(n, dtype=torch.float64, device=d)
        self.tracked_y = torch.full(
            (n,), EGO_INITIAL_Y, dtype=torch.float64, device=d
        )

        self.car_x = torch.zeros((n, OPPONENT_COUNT), dtype=torch.float64, device=d)
        self.car_y = torch.zeros((n, OPPONENT_COUNT), dtype=torch.float64, device=d)
        self.car_vx = torch.full(
            (n, OPPONENT_COUNT), 8.0, dtype=torch.float64, device=d
        )
        self.car_lane = torch.full(
            (n, OPPONENT_COUNT), -1, dtype=torch.int64, device=d
        )
        self.car_active = torch.zeros(
            (n, OPPONENT_COUNT), dtype=torch.bool, device=d
        )

        self._lane_centers = LANE_CENTERS.to(d)
        self._row_index = torch.arange(n, device=d)
        self._slot_ids = torch.arange(OPPONENT_COUNT, device=d)
        self._enabled_slots = torch.zeros(
            OPPONENT_COUNT, dtype=torch.bool, device=d
        )
        self._refresh_enabled_slots()

    def _refresh_enabled_slots(self) -> None:
        """Enable the top-of-stack slots used by the public car bucket."""
        self._enabled_slots.zero_()
        self._enabled_slots[OPPONENT_COUNT - self.opponent_count :] = True

    @torch.no_grad()
    def set_opponent_count(self, opponent_count: int) -> None:
        """Change traffic density for training curricula.

        The competition setting is always four opponents. Lower values are a
        training-only curriculum; increasing the count lets newly enabled cars
        spawn through the normal one-car-per-tick placement logic.
        """
        count = int(opponent_count)
        if not 1 <= count <= OPPONENT_COUNT:
            raise ValueError(
                f"opponent_count must be in [1, {OPPONENT_COUNT}]"
            )
        self.opponent_count = count
        self._refresh_enabled_slots()

        disabled = ~self._enabled_slots
        if bool(disabled.any()):
            self.car_active[:, disabled] = False
            self.car_lane[:, disabled] = -1

    @torch.no_grad()
    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.ticks.masked_fill_(mask, 0)
        self.last_action.masked_fill_(mask, 0)
        self.distance.masked_fill_(mask, 0.0)
        self.ego_vx.masked_fill_(mask, EGO_INITIAL_VX)
        self.ego_vy.masked_fill_(mask, 0.0)
        self.ego_y.masked_fill_(mask, EGO_INITIAL_Y)
        self.tracked_vy.masked_fill_(mask, 0.0)
        self.tracked_y.masked_fill_(mask, EGO_INITIAL_Y)

        mask2 = mask[:, None]
        self.car_x.masked_fill_(mask2, 0.0)
        self.car_y.masked_fill_(mask2, 0.0)
        self.car_vx.masked_fill_(mask2, 8.0)
        self.car_lane.masked_fill_(mask2, -1)
        self.car_active.masked_fill_(mask2, False)

    @torch.no_grad()
    def observation(self) -> torch.Tensor:
        n = self.num_envs

        ego_center_y = torch.trunc(self.ego_y) + CAR_HEIGHT // 2
        road_mid = (ROAD_TOP + ROAD_BOTTOM) * 0.5
        road_half = (ROAD_BOTTOM - ROAD_TOP) * 0.5

        base = torch.stack(
            (
                torch.clamp(self.ego_vx / 400.0, 0.0, 2.0),
                torch.clamp((ego_center_y - road_mid) / road_half, -2.0, 2.0),
                torch.clamp(self.ego_vy / 20.0, -2.0, 2.0),
                torch.clamp(
                    1.0 - self.ticks.to(torch.float64) / MAX_TICKS, 0.0, 1.0
                ),
                (~self.car_active).sum(dim=1).to(torch.float64) / OPPONENT_COUNT,
            ),
            dim=1,
        ).to(torch.float32)

        lane_idx = self.car_lane.clamp_min(0)
        active_f = self.car_active.float()

        car_center_x = torch.trunc(self.car_x) + CAR_WIDTH // 2
        rel_x = torch.clamp(
            (car_center_x - EGO_CENTER_X) / 2000.0, -2.0, 2.0
        ).to(torch.float32)
        rel_vx = torch.clamp(
            (self.car_vx - self.ego_vx[:, None]) / 100.0, -4.0, 4.0
        ).to(torch.float32)
        other_vx = torch.clamp(
            self.car_vx / 400.0, -1.0, 2.0
        ).to(torch.float32)

        presence = torch.zeros((n, LANE_COUNT), dtype=torch.float32, device=self.device)
        rel_x_slots = torch.zeros_like(presence)
        rel_vx_slots = torch.zeros_like(presence)
        other_vx_slots = torch.zeros_like(presence)

        presence.scatter_add_(1, lane_idx, active_f)
        rel_x_slots.scatter_add_(1, lane_idx, rel_x * active_f)
        rel_vx_slots.scatter_add_(1, lane_idx, rel_vx * active_f)
        other_vx_slots.scatter_add_(1, lane_idx, other_vx * active_f)

        slots = torch.stack(
            (presence, rel_x_slots, rel_vx_slots, other_vx_slots), dim=2
        ).reshape(n, LANE_COUNT * 4)

        return torch.cat((base, slots), dim=1)

    @torch.no_grad()
    def sensor_readings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return exact-style 16-ray readings and hit masks.

        Geometry mirrors Sensor.update(): rays originate from pygame.Rect.center,
        opponent/wall rectangles use integer-truncated positions, and the closest
        line/rectangle intersection within 1000 px is returned.
        """
        n = self.num_envs
        d = self.device
        dtype = torch.float64

        angles = torch.tensor(
            [degrees for degrees, _ in SENSOR_SPECS],
            dtype=dtype,
            device=d,
        )
        radians = torch.deg2rad(angles)
        ray_dx = torch.sin(radians).view(1, SENSOR_COUNT, 1)
        ray_dy = (-torch.cos(radians)).view(1, SENSOR_COUNT, 1)

        origin_x = torch.full(
            (n, 1, 1), EGO_CENTER_X, dtype=dtype, device=d
        )
        origin_y = (
            torch.trunc(self.ego_y) + CAR_HEIGHT // 2
        ).view(n, 1, 1)

        car_x0 = torch.trunc(self.car_x)
        car_y0 = torch.trunc(self.car_y)
        car_x1 = car_x0 + CAR_WIDTH
        car_y1 = car_y0 + CAR_HEIGHT

        wall_x0 = torch.tensor([0.0, 0.0], dtype=dtype, device=d).expand(n, -1)
        wall_x1 = torch.tensor(
            [float(SCREEN_WIDTH), float(SCREEN_WIDTH)],
            dtype=dtype,
            device=d,
        ).expand(n, -1)
        wall_y0 = torch.tensor(
            [0.0, ROAD_BOTTOM], dtype=dtype, device=d
        ).expand(n, -1)
        wall_y1 = torch.tensor(
            [ROAD_TOP, float(SCREEN_HEIGHT)], dtype=dtype, device=d
        ).expand(n, -1)

        x0 = torch.cat((car_x0, wall_x0), dim=1).unsqueeze(1)
        y0 = torch.cat((car_y0, wall_y0), dim=1).unsqueeze(1)
        x1 = torch.cat((car_x1, wall_x1), dim=1).unsqueeze(1)
        y1 = torch.cat((car_y1, wall_y1), dim=1).unsqueeze(1)

        active = torch.cat(
            (
                self.car_active,
                torch.ones((n, 2), dtype=torch.bool, device=d),
            ),
            dim=1,
        ).unsqueeze(1)

        eps = 1e-12
        inf = torch.tensor(float("inf"), dtype=dtype, device=d)
        neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=d)

        x_parallel = ray_dx.abs() <= eps
        y_parallel = ray_dy.abs() <= eps
        safe_dx = torch.where(x_parallel, torch.ones_like(ray_dx), ray_dx)
        safe_dy = torch.where(y_parallel, torch.ones_like(ray_dy), ray_dy)

        tx_a = (x0 - origin_x) / safe_dx
        tx_b = (x1 - origin_x) / safe_dx
        ty_a = (y0 - origin_y) / safe_dy
        ty_b = (y1 - origin_y) / safe_dy

        tx_min = torch.minimum(tx_a, tx_b)
        tx_max = torch.maximum(tx_a, tx_b)
        ty_min = torch.minimum(ty_a, ty_b)
        ty_max = torch.maximum(ty_a, ty_b)

        x_inside = (origin_x >= x0) & (origin_x <= x1)
        y_inside = (origin_y >= y0) & (origin_y <= y1)

        tx_min = torch.where(x_parallel, neg_inf, tx_min)
        tx_max = torch.where(x_parallel, inf, tx_max)
        ty_min = torch.where(y_parallel, neg_inf, ty_min)
        ty_max = torch.where(y_parallel, inf, ty_max)

        slab_ok = (
            (~x_parallel | x_inside)
            & (~y_parallel | y_inside)
        )
        t_enter = torch.maximum(tx_min, ty_min)
        t_exit = torch.minimum(tx_max, ty_max)
        distance = torch.where(t_enter >= 0.0, t_enter, t_exit)

        valid = (
            active
            & slab_ok
            & (t_exit >= torch.maximum(t_enter, torch.zeros_like(t_enter)))
            & (distance >= 0.0)
            & (distance <= SENSOR_RANGE)
        )
        candidates = torch.where(valid, distance, inf)
        readings = candidates.min(dim=2).values
        hits = torch.isfinite(readings)
        readings = torch.where(
            hits,
            readings,
            torch.full_like(readings, SENSOR_RANGE),
        )
        return readings, hits

    @torch.no_grad()
    def legal_observation(self, *, tracked_y: bool = False) -> torch.Tensor:
        """Observation made only from API fields plus optional controller history.

        tracked_y=True appends a controller-owned lateral-position reconstruction
        as feature 42. It is integrated solely from issued actions, not read from
        simulator-private ego_y.
        """
        readings, hits = self.sensor_readings()

        sensor_dist = torch.clamp(
            readings / SENSOR_RANGE, 0.0, 1.0
        ).to(torch.float32)
        sensor_hits = hits.to(torch.float32)

        extras = torch.stack(
            (
                torch.clamp(self.ego_vx / 400.0, 0.0, 2.0),
                torch.clamp(self.ego_vy / 20.0, -2.0, 2.0),
                torch.clamp(
                    1.0 - self.ticks.to(torch.float64) / MAX_TICKS,
                    0.0,
                    1.0,
                ),
                torch.clamp(self.distance / 400_000.0, 0.0, 2.0),
            ),
            dim=1,
        ).to(torch.float32)

        previous_action = F.one_hot(
            self.last_action, num_classes=len(ACTIONS)
        ).to(torch.float32)

        base_obs = torch.cat(
            (sensor_dist, sensor_hits, extras, previous_action),
            dim=1,
        )
        if not tracked_y:
            return base_obs

        tracked_y_norm = (
            (self.tracked_y - TRACKED_Y_MIDPOINT) / TRACKED_Y_HALF_SPAN
        ).to(torch.float32).unsqueeze(1)
        return torch.cat((base_obs, tracked_y_norm), dim=1)

    @torch.no_grad()
    def step(
        self,
        actions: torch.Tensor,
        observation_kind: str = "privileged",
        *,
        tracked_y: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, EpisodeBatch]:
        actions = actions.to(device=self.device, dtype=torch.int64)
        self.last_action.copy_(actions)

        # Primitive ego action. The masks are converted to float64 before the
        # update so 0.1 has the same precision as Python-float dynamics.
        physics_dtype = self.ego_vx.dtype
        self.ego_vx.add_((actions == 1).to(physics_dtype), alpha=0.1)
        self.ego_vx.add_((actions == 2).to(physics_dtype), alpha=-0.1)
        self.ego_vx.clamp_(min=0.0)
        self.ego_vy.add_((actions == 3).to(physics_dtype), alpha=-0.1)
        self.ego_vy.add_((actions == 4).to(physics_dtype), alpha=0.1)

        # Mirror the exact legal controller-side integrator independently.
        self.tracked_vy.add_((actions == 3).to(physics_dtype), alpha=-0.1)
        self.tracked_vy.add_((actions == 4).to(physics_dtype), alpha=0.1)

        self.ticks += 1
        self.distance += self.ego_vx

        # Exact public ego lateral update and independent legal reconstruction.
        self.ego_y += self.ego_vy
        self.tracked_y += self.tracked_vy

        active_f = self.car_active.to(self.car_vx.dtype)
        self.car_x += (self.car_vx - self.ego_vx[:, None]) * active_f

        # Opponent vx random walk: U(-0.1, +0.1) per active car per tick.
        noise = (
            torch.rand(
                self.car_vx.shape,
                generator=self.generator,
                device=self.device,
                dtype=self.car_vx.dtype,
            )
            - 0.5
        ) / 5.0
        self.car_vx += noise * active_f

        # Retire traffic outside the challenge's x bounds.
        retired = self.car_active & (
            (self.car_x < RETIRE_MIN_X) | (self.car_x > RETIRE_MAX_X)
        )
        self.car_active &= ~retired
        self.car_lane.masked_fill_(retired, -1)

        # Spawn at most one enabled inactive opponent per environment each tick.
        enabled = self._enabled_slots[None, :]
        eligible_inactive = (~self.car_active) & enabled
        spawn_mask = eligible_inactive.any(dim=1)

        # Public implementation uses a stack; choosing the highest inactive
        # slot reproduces that lifecycle while remaining vectorizable. This is
        # intentionally branch-free to avoid a CUDA->CPU synchronization.
        inactive_slots = torch.where(
            eligible_inactive,
            self._slot_ids[None, :],
            torch.full_like(self._slot_ids[None, :], -1),
        )
        spawn_slot = inactive_slots.max(dim=1).values

        # Find open lanes. At most one active opponent occupies each lane.
        lane_one_hot = F.one_hot(
            self.car_lane.clamp_min(0), num_classes=LANE_COUNT
        ).bool()
        occupied = (lane_one_hot & self.car_active[..., None]).any(dim=1)

        lane_scores = torch.rand(
            (self.num_envs, LANE_COUNT),
            generator=self.generator,
            device=self.device,
        )
        lane_scores.masked_fill_(occupied, -1.0)
        spawn_lane = lane_scores.argmax(dim=1)

        behind = (
            torch.rand(
                self.num_envs, generator=self.generator, device=self.device
            )
            < 0.5
        )
        speed_coeff = (
            torch.rand(
                self.num_envs,
                generator=self.generator,
                device=self.device,
                dtype=self.ego_vx.dtype,
            )
            * 5.0
        )

        # Avoid boolean indexing with variable output sizes: it can force CUDA
        # synchronization. Scatter one candidate per environment and make the
        # write a no-op when spawn_mask is false.
        slot = spawn_slot.clamp_min(0)
        slot_col = slot[:, None]

        spawn_vx = torch.where(
            behind,
            self.ego_vx + speed_coeff,
            self.ego_vx - speed_coeff,
        )
        spawn_x = torch.where(
            behind,
            torch.full_like(spawn_vx, -980.0),
            torch.full_like(spawn_vx, 2220.0),
        )
        spawn_y = torch.trunc(
            self._lane_centers[spawn_lane] - CAR_HEIGHT / 2.0
        )

        old_x = self.car_x.gather(1, slot_col).squeeze(1)
        old_y = self.car_y.gather(1, slot_col).squeeze(1)
        old_vx = self.car_vx.gather(1, slot_col).squeeze(1)
        old_lane = self.car_lane.gather(1, slot_col).squeeze(1)
        old_active = self.car_active.gather(1, slot_col).squeeze(1)

        self.car_x.scatter_(
            1, slot_col, torch.where(spawn_mask, spawn_x, old_x)[:, None]
        )
        self.car_y.scatter_(
            1, slot_col, torch.where(spawn_mask, spawn_y, old_y)[:, None]
        )
        self.car_vx.scatter_(
            1, slot_col, torch.where(spawn_mask, spawn_vx, old_vx)[:, None]
        )
        self.car_lane.scatter_(
            1, slot_col, torch.where(spawn_mask, spawn_lane, old_lane)[:, None]
        )
        self.car_active.scatter_(
            1,
            slot_col,
            torch.where(
                spawn_mask,
                torch.ones_like(old_active),
                old_active,
            )[:, None],
        )

        # Collision semantics mirror pygame.Rect(int(...), ...).colliderect(...).
        ego_y_i = torch.trunc(self.ego_y)
        car_x_i = torch.trunc(self.car_x)
        car_y_i = torch.trunc(self.car_y)

        x_overlap = (
            (EGO_X < car_x_i + CAR_WIDTH)
            & (EGO_X + CAR_WIDTH > car_x_i)
        )
        y_overlap = (
            (ego_y_i[:, None] < car_y_i + CAR_HEIGHT)
            & (ego_y_i[:, None] + CAR_HEIGHT > car_y_i)
        )

        hit_car = (self.car_active & x_overlap & y_overlap).any(dim=1)
        hit_wall = (ego_y_i < 40.0) | (ego_y_i + CAR_HEIGHT > 1160.0)
        crashed = hit_car | hit_wall
        timed_out = self.ticks >= MAX_TICKS
        done = crashed | timed_out

        reward = (self.ego_vx / self.reward_scale).to(torch.float32)

        # Optional teacher-only safety shaping. Defaults are exactly zero, so
        # direct-policy PPO and parity tests retain the old reward. The TTC term
        # gives dense warning before a longitudinally closing car enters the
        # ego's lateral corridor; this improves credit assignment for avoiding
        # traffic collisions without exposing any extra feature to the policy.
        if self.survival_bonus:
            reward = reward + (~crashed).to(torch.float32) * self.survival_bonus

        if self.ttc_risk_penalty:
            rel_x = self.car_x - EGO_X
            clearance = (rel_x.abs() - CAR_WIDTH).clamp_min(0.0)
            closing = torch.where(
                rel_x >= 0.0,
                self.ego_vx[:, None] - self.car_vx,
                self.car_vx - self.ego_vx[:, None],
            ).clamp_min(0.0)
            inf = torch.full_like(clearance, float("inf"))
            ttc = torch.where(
                closing > 1e-9,
                clearance / closing.clamp_min(1e-9),
                inf,
            )
            lateral_near = (
                (self.car_y - self.ego_y[:, None]).abs()
                < (CAR_HEIGHT + self.ttc_lateral_margin)
            )
            risk = (
                (self.ttc_risk_horizon - ttc) / self.ttc_risk_horizon
            ).clamp(0.0, 1.0)
            risk = torch.where(
                self.car_active & lateral_near,
                risk,
                torch.zeros_like(risk),
            )
            max_risk = risk.max(dim=1).values.to(torch.float32)
            reward = reward - max_risk * self.ttc_risk_penalty

        # Wall crashes are separately identifiable from deterministic ego-y.
        # If wall and car overlap on the same tick, apply one penalty only and
        # give wall precedence.
        crash_cost = torch.where(
            hit_wall,
            torch.full_like(reward, self.wall_crash_penalty),
            torch.where(
                hit_car,
                torch.full_like(reward, self.crash_penalty),
                torch.zeros_like(reward),
            ),
        )
        reward = reward - crash_cost

        episode = EpisodeBatch(
            done=done.clone(),
            crashed=crashed.clone(),
            hit_car=(done & hit_car).clone(),
            hit_wall=(done & hit_wall).clone(),
            distance=torch.where(done, self.distance, torch.zeros_like(self.distance)),
            ticks=torch.where(done, self.ticks, torch.zeros_like(self.ticks)),
            final_speed=torch.where(done, self.ego_vx, torch.zeros_like(self.ego_vx)),
        )

        # Auto-reset completed environments. PPO sees done=True for the transition
        # and the next observation is the first state of a fresh episode.
        # Masked reset is safe even when no environments are done and avoids
        # a per-tick CUDA synchronization.
        self.reset(done)

        if observation_kind == "privileged":
            next_obs = self.observation()
        elif observation_kind == "legal":
            next_obs = self.legal_observation(tracked_y=tracked_y)
        elif observation_kind == "none":
            next_obs = torch.empty((0,), dtype=torch.float32, device=self.device)
        else:
            raise ValueError(
                "observation_kind must be 'privileged', 'legal', or 'none', "
                f"got {observation_kind!r}"
            )

        return next_obs, reward, done, episode

    @torch.no_grad()
    def macro_step(
        self,
        actions: torch.Tensor,
        *,
        repeat: int,
        discount: float,
        observation_kind: str = "legal",
        tracked_y: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, EpisodeBatch]:
        """Hold one primitive action for several simulator ticks.

        The policy still selects only one of the five legal primitive actions,
        but the controller commits to it for repeat ticks before observing again.
        Rewards inside the macro are discounted per primitive tick.
        """
        repeat = int(repeat)
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        discount = float(discount)
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")

        actions = actions.to(device=self.device, dtype=torch.int64)
        n = self.num_envs
        d = self.device

        alive = torch.ones(n, dtype=torch.bool, device=d)
        finished = torch.zeros(n, dtype=torch.bool, device=d)
        total_reward = torch.zeros(n, dtype=torch.float32, device=d)

        ep_crashed = torch.zeros(n, dtype=torch.bool, device=d)
        ep_hit_car = torch.zeros(n, dtype=torch.bool, device=d)
        ep_hit_wall = torch.zeros(n, dtype=torch.bool, device=d)
        ep_distance = torch.zeros(n, dtype=torch.float64, device=d)
        ep_ticks = torch.zeros(n, dtype=torch.int32, device=d)
        ep_speed = torch.zeros(n, dtype=torch.float64, device=d)

        weight = 1.0
        for _ in range(repeat):
            step_actions = torch.where(alive, actions, torch.zeros_like(actions))
            _, reward, done, episode = self.step(
                step_actions, observation_kind="none", tracked_y=tracked_y
            )

            total_reward += reward * alive.to(torch.float32) * weight

            first_done = alive & done
            ep_crashed |= first_done & episode.crashed
            ep_hit_car |= first_done & episode.hit_car
            ep_hit_wall |= first_done & episode.hit_wall
            ep_distance = torch.where(first_done, episode.distance, ep_distance)
            ep_ticks = torch.where(first_done, episode.ticks, ep_ticks)
            ep_speed = torch.where(first_done, episode.final_speed, ep_speed)

            alive &= ~done
            finished |= first_done
            weight *= discount

        # Terminal envs auto-reset inside step(). Reset them again here because
        # throwaway ticks after their first terminal transition may have advanced
        # the fresh episode during the remainder of this macro.
        self.reset(finished)

        if observation_kind == "legal":
            next_obs = self.legal_observation(tracked_y=tracked_y)
        elif observation_kind == "privileged":
            next_obs = self.observation()
        else:
            raise ValueError(
                "macro_step observation_kind must be 'legal' or 'privileged'"
            )

        episode = EpisodeBatch(
            done=finished.clone(),
            crashed=ep_crashed,
            hit_car=ep_hit_car,
            hit_wall=ep_hit_wall,
            distance=ep_distance,
            ticks=ep_ticks,
            final_speed=ep_speed,
        )
        return next_obs, total_reward, finished, episode
