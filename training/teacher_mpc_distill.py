"""Exact-simulator label collection for MPC teacher distillation.

The privileged PPO proposal and sequence-level MPC require simulator geometry
that is intentionally absent from the deployable observation.  This module
keeps that state inside CPU worker processes and returns only legal observations
and supervised action targets to the GPU learner.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Mapping

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
import torch.nn.functional as F

import src.game.core as core
from training.gpu_ppo import ActorCritic
from training.gpu_recurrent_ppo import RecurrentActorCritic
from training.gpu_sim import ACTIONS, LEGAL_OBS_SIZE_TRACKED_Y, OBS_SIZE
from training.legal_env import LegalRaceCarEnv
from training.teacher_env import build_privileged_observation
from training.teacher_mpc import (
    choose_confirmed_mpc_action,
    mpc_planning_decision,
)


@dataclass(frozen=True)
class MpcLabelConfig:
    horizon: int = 72
    strategic_horizon: int = 119
    strategic_trigger_limit: int = 1
    burst_ticks: int = 12
    temperature: float = 1.0
    x_margin: float = 0.0
    y_margin: float = 0.0
    trigger_x: float = 900.0
    trigger_lateral: float = 300.0
    trigger_extra: float = 120.0
    trigger_immediate: float = 30.0
    clearance_threshold: float = 8.0
    wall_threshold: float = 4.0
    clearance_cap: float = 120.0
    opponent_uncertainty_sigma: float = 3.0
    override_soft_target: float = 1.0

    def validate(self) -> None:
        if self.horizon <= 0 or self.burst_ticks <= 0:
            raise ValueError("horizon and burst_ticks must be positive")
        if self.strategic_horizon < 0:
            raise ValueError("strategic_horizon must be non-negative")
        if self.strategic_horizon and self.strategic_horizon < self.horizon:
            raise ValueError("strategic_horizon must be zero or at least horizon")
        if self.strategic_trigger_limit < 0:
            raise ValueError("strategic_trigger_limit must be non-negative")
        positive = {
            "temperature": self.temperature,
            "trigger_x": self.trigger_x,
            "trigger_lateral": self.trigger_lateral,
            "clearance_cap": self.clearance_cap,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        non_negative = {
            "x_margin": self.x_margin,
            "y_margin": self.y_margin,
            "trigger_extra": self.trigger_extra,
            "trigger_immediate": self.trigger_immediate,
            "clearance_threshold": self.clearance_threshold,
            "wall_threshold": self.wall_threshold,
            "opponent_uncertainty_sigma": self.opponent_uncertainty_sigma,
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= float(self.override_soft_target) <= 1.0:
            raise ValueError("override_soft_target must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _new_stats() -> Dict[str, Any]:
    return {
        "ticks": 0,
        "episodes": 0,
        "crashes": 0,
        "car_crashes": 0,
        "wall_crashes": 0,
        "completed_distance": 0.0,
        "triggers": 0,
        "overrides": 0,
        "all_unsafe": 0,
        "all_unhealthy": 0,
        "plans_evaluated": 0,
        "strategic_candidates": 0,
        "strategic_accepted": 0,
        "lane_straddling_ticks": 0,
        "proposed_counts": [0] * len(ACTIONS),
        "executed_counts": [0] * len(ACTIONS),
        "label_counts": [0] * len(ACTIONS),
        "override_action_counts": [0] * len(ACTIONS),
        "lane_option_counts": [0, 0, 0],
        "student_behavior_ticks": 0,
        "trigger_reasons": {},
        "selected_plans": {},
    }


class ExactMpcLabelCollector:
    """One persistent exact simulator stream and privileged MPC teacher."""

    def __init__(
        self,
        teacher_state: Mapping[str, torch.Tensor],
        *,
        worker_rank: int,
        worker_count: int,
        seed_start: int,
        config: MpcLabelConfig,
        behavior_student_state: Mapping[str, torch.Tensor] | None = None,
        behavior_hidden_dim: int = 128,
        behavior_mode: str = "stochastic",
        proposal_student_state: Mapping[str, torch.Tensor] | None = None,
        proposal_obs_dim: int = LEGAL_OBS_SIZE_TRACKED_Y,
        proposal_hidden_dim: int = 128,
        proposal_mode: str = "deterministic",
    ) -> None:
        config.validate()
        if worker_rank < 0 or worker_count <= 0 or worker_rank >= worker_count:
            raise ValueError("worker_rank must identify a valid worker")

        self.config = config
        self.worker_rank = int(worker_rank)
        self.worker_count = int(worker_count)
        self.seed_start = int(seed_start)
        self.episode_index = 0
        if behavior_mode not in ("deterministic", "stochastic"):
            raise ValueError("behavior_mode must be deterministic or stochastic")
        self.behavior_mode = str(behavior_mode)
        self.behavior_generator = torch.Generator(device="cpu")
        self.behavior_generator.manual_seed(
            self.seed_start + self.worker_rank + 91_000_003
        )
        if proposal_mode not in ("deterministic", "scheduler"):
            raise ValueError("proposal_mode must be deterministic or scheduler")
        if proposal_obs_dim not in (
            LEGAL_OBS_SIZE_TRACKED_Y - 1,
            LEGAL_OBS_SIZE_TRACKED_Y,
        ):
            raise ValueError("proposal_obs_dim must be 41 or 42")
        self.proposal_mode = str(proposal_mode)
        self.proposal_obs_dim = int(proposal_obs_dim)

        self.teacher = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False).cpu()
        self.teacher.load_state_dict(teacher_state)
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

        self.proposal_student = None
        self.proposal_hidden = None
        self.proposal_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)
        if proposal_student_state is not None:
            self.proposal_student = RecurrentActorCritic(
                self.proposal_obs_dim,
                len(ACTIONS),
                hidden_dim=int(proposal_hidden_dim),
                safe_init=False,
            ).cpu()
            self.proposal_student.load_state_dict(proposal_student_state)
            self.proposal_student.eval()
            for parameter in self.proposal_student.parameters():
                parameter.requires_grad_(False)
            self.proposal_hidden = self.proposal_student.initial_state(
                1, device="cpu"
            )

        self.behavior_student = None
        self.behavior_hidden = None
        if behavior_student_state is not None:
            self.behavior_student = RecurrentActorCritic(
                LEGAL_OBS_SIZE_TRACKED_Y,
                len(ACTIONS),
                hidden_dim=int(behavior_hidden_dim),
                safe_init=False,
            ).cpu()
            self.behavior_student.load_state_dict(behavior_student_state)
            self.behavior_student.eval()
            for parameter in self.behavior_student.parameters():
                parameter.requires_grad_(False)
            self.behavior_hidden = self.behavior_student.initial_state(
                1, device="cpu"
            )

        self.env = LegalRaceCarEnv(
            base_seed=self.seed_start + self.worker_rank,
            crash_penalty=50.0,
            wall_crash_penalty=100.0,
            reward_scale=1000.0,
            tracked_y=True,
        )
        self.scheduler_credit = torch.zeros(len(ACTIONS), dtype=torch.float32)
        self.strategic_accepted = 0
        self.obs = self._reset()

    def _episode_seed(self) -> int:
        return (
            self.seed_start
            + self.worker_rank
            + self.episode_index * self.worker_count
        )

    def _reset(self) -> np.ndarray:
        seed = self._episode_seed()
        self.episode_index += 1
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = self.env.reset(options={"sim_seed": seed})
        self.scheduler_credit.zero_()
        self.proposal_credit.zero_()
        self.strategic_accepted = 0
        if self.proposal_hidden is not None:
            self.proposal_hidden.zero_()
        if self.behavior_hidden is not None:
            self.behavior_hidden.zero_()
        return obs

    @torch.no_grad()
    def collect(
        self,
        steps: int,
        *,
        student_behavior_fraction: float = 0.0,
    ) -> Dict[str, Any]:
        steps = int(steps)
        if steps <= 0:
            raise ValueError("steps must be positive")
        fraction = float(student_behavior_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("student_behavior_fraction must be in [0, 1]")
        if fraction > 0.0 and self.behavior_student is None:
            raise ValueError("student behavior requested without a student model")

        observations = np.empty(
            (steps, LEGAL_OBS_SIZE_TRACKED_Y), dtype=np.float32
        )
        actions = np.empty(steps, dtype=np.int64)
        probabilities = np.empty((steps, len(ACTIONS)), dtype=np.float32)
        dones = np.empty(steps, dtype=np.bool_)
        trigger_mask = np.empty(steps, dtype=np.bool_)
        override_mask = np.empty(steps, dtype=np.bool_)
        lane_options = np.empty(steps, dtype=np.int64)
        behavior_actions = np.empty(steps, dtype=np.int64)
        student_behavior_mask = np.empty(steps, dtype=np.bool_)
        stats = _new_stats()
        trigger_reasons: Counter[str] = Counter()
        selected_plans: Counter[str] = Counter()

        cfg = self.config
        for tick in range(steps):
            state = core.STATE
            observations[tick] = self.obs

            ego_top = float(state.ego.y)
            ego_bottom = ego_top + float(state.ego.sprite.get_height())
            if any(
                ego_top < float(lane.y_end) < ego_bottom
                for lane in state.road.lanes[:-1]
            ):
                stats["lane_straddling_ticks"] += 1

            privileged = torch.from_numpy(
                build_privileged_observation(state)
            ).unsqueeze(0)
            if self.proposal_student is None:
                logits, _ = self.teacher(privileged)
                probs = torch.softmax(
                    logits.squeeze(0) / float(cfg.temperature), dim=-1
                )
                self.scheduler_credit.add_(probs)
                proposed = int(self.scheduler_credit.argmax().item())
                self.scheduler_credit[proposed] -= 1.0
            else:
                proposal_obs = torch.from_numpy(
                    self.obs[: self.proposal_obs_dim]
                ).unsqueeze(0)
                logits, _, self.proposal_hidden = self.proposal_student.step(
                    proposal_obs, self.proposal_hidden
                )
                probs = torch.softmax(logits.squeeze(0), dim=-1)
                if self.proposal_mode == "scheduler":
                    self.proposal_credit.add_(probs)
                    proposed = int(self.proposal_credit.argmax().item())
                    self.proposal_credit[proposed] -= 1.0
                else:
                    proposed = int(probs.argmax().item())
            stats["proposed_counts"][proposed] += 1

            strategic_horizon = (
                cfg.strategic_horizon
                if cfg.strategic_trigger_limit <= 0
                or self.strategic_accepted < cfg.strategic_trigger_limit
                else 0
            )
            reason, planning_horizon = mpc_planning_decision(
                state,
                proposed_action=proposed,
                horizon=cfg.horizon,
                strategic_horizon=strategic_horizon,
                trigger_x=cfg.trigger_x,
                trigger_lateral=cfg.trigger_lateral,
                trigger_extra=cfg.trigger_extra,
                trigger_immediate=cfg.trigger_immediate,
                opponent_uncertainty_sigma=cfg.opponent_uncertainty_sigma,
            )

            action = proposed
            diagnostic = None
            lane_option = 0
            triggered = reason is not None
            if triggered:
                stats["triggers"] += 1
                if planning_horizon > cfg.horizon:
                    stats["strategic_candidates"] += 1
                action, diagnostic, strategic_used = choose_confirmed_mpc_action(
                    state,
                    proposed,
                    horizon=cfg.horizon,
                    planning_horizon=planning_horizon,
                    burst_ticks=cfg.burst_ticks,
                    x_margin=cfg.x_margin,
                    y_margin=cfg.y_margin,
                    clearance_threshold=cfg.clearance_threshold,
                    wall_threshold=cfg.wall_threshold,
                    clearance_cap=cfg.clearance_cap,
                    opponent_uncertainty_sigma=cfg.opponent_uncertainty_sigma,
                )
                if reason == "strategic_closing" and not strategic_used:
                    reason = "lateral"
                if strategic_used:
                    self.strategic_accepted += 1
                    stats["strategic_accepted"] += 1
                trigger_reasons[str(reason)] += 1
                selected_plans[str(diagnostic["selected_plan"])] += 1
                stats["plans_evaluated"] += int(diagnostic["plans_evaluated"])
                stats["all_unsafe"] += int(bool(diagnostic["all_unsafe"]))
                stats["all_unhealthy"] += int(bool(diagnostic["all_unhealthy"]))
                stats["overrides"] += int(bool(diagnostic["overridden"]))
                if diagnostic["selected_plan"] == "left":
                    lane_option = 1
                elif diagnostic["selected_plan"] == "right":
                    lane_option = 2

            target_probs = probs
            if action != proposed and cfg.override_soft_target > 0.0:
                one_hot = F.one_hot(
                    torch.tensor(action), num_classes=len(ACTIONS)
                ).to(torch.float32)
                blend = float(cfg.override_soft_target)
                target_probs = (1.0 - blend) * probs + blend * one_hot

            label_action = int(action)
            actions[tick] = label_action
            probabilities[tick] = target_probs.numpy()
            trigger_mask[tick] = triggered
            override_mask[tick] = label_action != proposed
            stats["label_counts"][label_action] += 1
            lane_options[tick] = lane_option
            stats["lane_option_counts"][lane_option] += 1
            if label_action != proposed:
                stats["override_action_counts"][label_action] += 1

            behavior_action = label_action
            use_student = False
            if self.behavior_student is not None:
                student_obs = torch.from_numpy(self.obs).unsqueeze(0)
                student_logits, _, self.behavior_hidden = self.behavior_student.step(
                    student_obs, self.behavior_hidden
                )
                use_student = bool(
                    fraction > 0.0
                    and torch.rand((), generator=self.behavior_generator).item()
                    < fraction
                )
                if use_student:
                    if self.behavior_mode == "deterministic":
                        behavior_action = int(student_logits.argmax(dim=-1).item())
                    else:
                        student_probs = torch.softmax(student_logits, dim=-1)
                        behavior_action = int(
                            torch.multinomial(
                                student_probs,
                                1,
                                generator=self.behavior_generator,
                            ).item()
                        )
            behavior_actions[tick] = behavior_action
            student_behavior_mask[tick] = use_student
            stats["student_behavior_ticks"] += int(use_student)
            stats["executed_counts"][behavior_action] += 1

            self.obs, _, terminated, truncated, info = self.env.step(behavior_action)
            done = bool(terminated or truncated)
            dones[tick] = done
            stats["ticks"] += 1
            if done:
                if self.behavior_hidden is not None:
                    self.behavior_hidden.zero_()
                if self.proposal_hidden is not None:
                    self.proposal_hidden.zero_()
                stats["episodes"] += 1
                stats["crashes"] += int(bool(info["crashed"]))
                stats["car_crashes"] += int(bool(info["hit_car"]))
                stats["wall_crashes"] += int(bool(info["hit_wall"]))
                stats["completed_distance"] += float(info["distance"])
                self.obs = self._reset()

        stats["trigger_reasons"] = dict(trigger_reasons)
        stats["selected_plans"] = dict(selected_plans)
        return {
            "observations": observations,
            "actions": actions,
            "probabilities": probabilities,
            "dones": dones,
            "triggers": trigger_mask,
            "overrides": override_mask,
            "lane_options": lane_options,
            "behavior_actions": behavior_actions,
            "student_behavior": student_behavior_mask,
            "stats": stats,
        }

    def close(self) -> None:
        self.env.close()


def merge_collection_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine worker-local counters without losing diagnostic categories."""
    result = _new_stats()
    reason_counts: Counter[str] = Counter()
    plan_counts: Counter[str] = Counter()
    for item in items:
        for key in (
            "ticks",
            "episodes",
            "crashes",
            "car_crashes",
            "wall_crashes",
            "completed_distance",
            "triggers",
            "overrides",
            "all_unsafe",
            "all_unhealthy",
            "plans_evaluated",
            "strategic_candidates",
            "strategic_accepted",
            "lane_straddling_ticks",
            "student_behavior_ticks",
        ):
            result[key] += item[key]
        for key in (
            "proposed_counts",
            "label_counts",
            "override_action_counts",
            "lane_option_counts",
            "executed_counts",
        ):
            result[key] = [
                int(a) + int(b) for a, b in zip(result[key], item[key])
            ]
        reason_counts.update(item["trigger_reasons"])
        plan_counts.update(item["selected_plans"])
    result["trigger_reasons"] = dict(reason_counts)
    result["selected_plans"] = dict(plan_counts)
    return result


def collector_worker(
    connection: Connection,
    teacher_state: Mapping[str, torch.Tensor],
    worker_rank: int,
    worker_count: int,
    seed_start: int,
    config_values: Dict[str, Any],
    behavior_student_state: Mapping[str, torch.Tensor] | None = None,
    behavior_hidden_dim: int = 128,
    behavior_mode: str = "stochastic",
    proposal_student_state: Mapping[str, torch.Tensor] | None = None,
    proposal_obs_dim: int = LEGAL_OBS_SIZE_TRACKED_Y,
    proposal_hidden_dim: int = 128,
    proposal_mode: str = "deterministic",
) -> None:
    """Process entry point used by the training script."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    collector = None
    try:
        collector = ExactMpcLabelCollector(
            teacher_state,
            worker_rank=worker_rank,
            worker_count=worker_count,
            seed_start=seed_start,
            config=MpcLabelConfig(**config_values),
            behavior_student_state=behavior_student_state,
            behavior_hidden_dim=behavior_hidden_dim,
            behavior_mode=behavior_mode,
            proposal_student_state=proposal_student_state,
            proposal_obs_dim=proposal_obs_dim,
            proposal_hidden_dim=proposal_hidden_dim,
            proposal_mode=proposal_mode,
        )
        connection.send({"ready": True})
        while True:
            command = connection.recv()
            if command[0] == "collect":
                fraction = float(command[2]) if len(command) > 2 else 0.0
                connection.send(
                    collector.collect(
                        int(command[1]), student_behavior_fraction=fraction
                    )
                )
            elif command[0] == "close":
                break
            else:
                raise ValueError(f"unknown collector command: {command[0]}")
    except BaseException:
        try:
            connection.send({"error": traceback.format_exc()})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if collector is not None:
            collector.close()
        connection.close()
