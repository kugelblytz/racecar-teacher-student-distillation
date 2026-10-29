"""Distill the validated exact-state MPC teacher into a legal 42-D GRU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from training.direct_eval import evaluate_direct_original
from training.gpu_distill import distill_update_recurrent
from training.gpu_ppo import ActorCritic
from training.gpu_recurrent_ppo import RecurrentActorCritic
from training.gpu_recurrent_ppo import (
    expand_recurrent_obs_state_dict,
    recurrent_obs_dim_from_state_dict,
)
from training.gpu_sim import (
    ACTIONS,
    LEGAL_OBS_SIZE,
    LEGAL_OBS_SIZE_TRACKED_Y,
    OBS_SIZE,
)
from training.teacher_mpc_distill import (
    MpcLabelConfig,
    collector_worker,
    merge_collection_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect exact-simulator labels from the validated privileged "
            "MPC and distill them into a deployable legal-observation GRU."
        )
    )
    parser.add_argument("teacher_checkpoint", type=Path)
    parser.add_argument("--transitions", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--data-seed-start", type=int, default=500_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--env-minibatch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument("--soft-weight", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--lane-option-loss-weight",
        type=float,
        default=0.0,
        help="Train a separate stay/left/right MPC-plan head when positive.",
    )
    parser.add_argument(
        "--lane-option-positive-weight",
        type=float,
        default=32.0,
        help="Class weight for left/right lane-option targets.",
    )
    parser.add_argument(
        "--lane-option-threshold",
        type=float,
        default=0.5,
        help="Minimum left/right probability that starts a tracked lane change.",
    )
    parser.add_argument(
        "--trigger-sample-weight",
        type=float,
        default=2.0,
        help="Loss weight for MPC-triggered labels that are not overridden.",
    )
    parser.add_argument(
        "--override-sample-weight",
        type=float,
        default=8.0,
        help="Loss weight for the rare safety action substitutions.",
    )
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--strategic-horizon", type=int, default=119)
    parser.add_argument("--strategic-trigger-limit", type=int, default=1)
    parser.add_argument("--burst-ticks", type=int, default=12)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--x-margin", type=float, default=0.0)
    parser.add_argument("--y-margin", type=float, default=0.0)
    parser.add_argument("--trigger-x", type=float, default=900.0)
    parser.add_argument("--trigger-lateral", type=float, default=300.0)
    parser.add_argument("--trigger-extra", type=float, default=120.0)
    parser.add_argument("--trigger-immediate", type=float, default=30.0)
    parser.add_argument("--clearance-threshold", type=float, default=8.0)
    parser.add_argument("--wall-threshold", type=float, default=4.0)
    parser.add_argument("--clearance-cap", type=float, default=120.0)
    parser.add_argument("--opponent-uncertainty-sigma", type=float, default=3.0)
    parser.add_argument(
        "--override-soft-target",
        type=float,
        default=1.0,
        help=(
            "Blend MPC override actions into soft labels (1 makes an override "
            "one-hot; non-overridden labels retain PPO probabilities)."
        ),
    )

    parser.add_argument("--eval-every", type=int, default=2_000_000)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--eval-seed-start", type=int, default=400_000)
    parser.add_argument("--eval-action-seed-start", type=int, default=40_000_000)
    parser.add_argument("--eval-workers", type=int, default=24)
    parser.add_argument("--save-every", type=int, default=2_000_000)
    parser.add_argument(
        "--output", type=Path, default=Path("models/distilled_gru_mpc")
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-student-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize the legal GRU from a 41-D or 42-D checkpoint. A 41-D "
            "checkpoint is expanded losslessly with a zero tracked-y column."
        ),
    )
    parser.add_argument(
        "--proposal-student-checkpoint",
        type=Path,
        default=None,
        help=(
            "Use a fixed legal GRU as the nominal controller and collect only "
            "the exact MPC's safety corrections to its proposals."
        ),
    )
    parser.add_argument(
        "--proposal-mode",
        choices=["deterministic", "scheduler"],
        default="deterministic",
    )
    parser.add_argument(
        "--behavior-student-checkpoint",
        type=Path,
        default=None,
        help="Optional fixed legal GRU used to visit off-teacher states (DAgger).",
    )
    parser.add_argument(
        "--student-behavior-fraction",
        type=float,
        default=0.0,
        help="Fraction of collector actions executed by the behavior student.",
    )
    parser.add_argument(
        "--behavior-mode",
        choices=["deterministic", "stochastic"],
        default="stochastic",
    )
    return parser.parse_args()


def load_teacher_state(path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if state is None:
        raise ValueError(f"{path} does not contain model_state")
    teacher = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    teacher.load_state_dict(state)
    return {key: value.detach().cpu().share_memory_() for key, value in state.items()}


def make_optimizer(
    model: RecurrentActorCritic,
    args: argparse.Namespace,
    lane_option_head: nn.Module | None = None,
):
    kwargs = {"lr": args.learning_rate, "eps": 1e-5}
    parameters = list(model.parameters())
    if lane_option_head is not None:
        parameters.extend(lane_option_head.parameters())
    if torch.device(args.device).type == "cuda":
        try:
            return torch.optim.Adam(parameters, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.Adam(parameters, **kwargs)


def mpc_config(args: argparse.Namespace) -> MpcLabelConfig:
    return MpcLabelConfig(
        horizon=args.horizon,
        strategic_horizon=args.strategic_horizon,
        strategic_trigger_limit=args.strategic_trigger_limit,
        burst_ticks=args.burst_ticks,
        temperature=args.teacher_temperature,
        x_margin=args.x_margin,
        y_margin=args.y_margin,
        trigger_x=args.trigger_x,
        trigger_lateral=args.trigger_lateral,
        trigger_extra=args.trigger_extra,
        trigger_immediate=args.trigger_immediate,
        clearance_threshold=args.clearance_threshold,
        wall_threshold=args.wall_threshold,
        clearance_cap=args.clearance_cap,
        opponent_uncertainty_sigma=args.opponent_uncertainty_sigma,
        override_soft_target=args.override_soft_target,
    )


class CollectorPool:
    def __init__(
        self,
        teacher_state: Dict[str, torch.Tensor],
        *,
        workers: int,
        seed_start: int,
        config: MpcLabelConfig,
        behavior_student_state: Dict[str, torch.Tensor] | None,
        behavior_hidden_dim: int,
        behavior_mode: str,
        proposal_student_state: Dict[str, torch.Tensor] | None,
        proposal_obs_dim: int,
        proposal_hidden_dim: int,
        proposal_mode: str,
    ) -> None:
        context = mp.get_context("spawn")
        self.connections = []
        self.processes = []
        for rank in range(workers):
            parent, child = context.Pipe()
            process = context.Process(
                target=collector_worker,
                args=(
                    child,
                    teacher_state,
                    rank,
                    workers,
                    seed_start,
                    config.to_dict(),
                    behavior_student_state,
                    behavior_hidden_dim,
                    behavior_mode,
                    proposal_student_state,
                    proposal_obs_dim,
                    proposal_hidden_dim,
                    proposal_mode,
                ),
                daemon=True,
            )
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        for connection in self.connections:
            message = connection.recv()
            if not message.get("ready", False):
                raise RuntimeError(message.get("error", "collector failed to start"))

    def collect(
        self, steps: int, *, student_behavior_fraction: float
    ) -> List[Dict[str, Any]]:
        for connection in self.connections:
            connection.send(
                ("collect", int(steps), float(student_behavior_fraction))
            )
        results = []
        for connection in self.connections:
            result = connection.recv()
            if "error" in result:
                raise RuntimeError(result["error"])
            results.append(result)
        return results

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("close",))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for connection in self.connections:
            connection.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _score(stats: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (
        -float(stats["crash_or_below_25k_rate"]),
        -float(stats["crash_rate"]),
        float(stats["p10_distance"]),
        float(stats["mean_distance"]),
    )


def evaluate_student(
    student: RecurrentActorCritic,
    args: argparse.Namespace,
    lane_option_head: nn.Module | None,
):
    modes = ["deterministic", "stochastic", "scheduler"]
    if lane_option_head is not None:
        modes.append("lane_options")
    return evaluate_direct_original(
        student,
        arch="gru",
        episodes=args.eval_episodes,
        seed_start=args.eval_seed_start,
        crash_penalty=50.0,
        wall_crash_penalty=100.0,
        reward_scale=1000.0,
        action_seed_start=args.eval_action_seed_start,
        workers=args.eval_workers,
        hidden_dim=args.hidden_dim,
        action_repeat=1,
        obs_dim=LEGAL_OBS_SIZE_TRACKED_Y,
        tracked_y=True,
        policy_modes=tuple(modes),
        lane_option_state=(
            None
            if lane_option_head is None
            else {
                key: value.detach().cpu()
                for key, value in lane_option_head.state_dict().items()
            }
        ),
        lane_option_threshold=args.lane_option_threshold,
    )


def save_checkpoint(
    path: Path,
    student: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    transitions: int,
    args: argparse.Namespace,
    config: MpcLabelConfig,
    best_scores: Dict[str, Tuple[float, float, float, float]],
    collection_totals: Dict[str, Any],
    lane_option_head: nn.Module | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "gru",
            "obs_kind": "legal",
            "obs_dim": LEGAL_OBS_SIZE_TRACKED_Y,
            "obs_version": 2,
            "hidden_dim": int(args.hidden_dim),
            "distilled": True,
            "teacher_mode": (
                "mpc_corrected_student"
                if args.proposal_student_checkpoint is not None
                else "mpc_scheduler"
            ),
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "teacher_checkpoint_sha256": _sha256(args.teacher_checkpoint),
            "proposal_student_checkpoint": (
                None
                if args.proposal_student_checkpoint is None
                else str(args.proposal_student_checkpoint)
            ),
            "proposal_student_checkpoint_sha256": (
                None
                if args.proposal_student_checkpoint is None
                else _sha256(args.proposal_student_checkpoint)
            ),
            "proposal_mode": str(args.proposal_mode),
            "mpc_config": config.to_dict(),
            "distill_transitions": int(transitions),
            "distill_optimizer_state": optimizer.state_dict(),
            "model_state": {
                key: value.detach().cpu()
                for key, value in student.state_dict().items()
            },
            "lane_option_state": (
                None
                if lane_option_head is None
                else {
                    key: value.detach().cpu()
                    for key, value in lane_option_head.state_dict().items()
                }
            ),
            "lane_option_threshold": float(args.lane_option_threshold),
            "best_scores": {key: list(value) for key, value in best_scores.items()},
            "student_score_version": 2,
            "collection_totals": collection_totals,
            "git_revision": _revision(),
            "config": {
                **_json_args(args),
                "tracked_y": True,
                "action_repeat": 1,
                "crash_penalty": 50.0,
                "wall_crash_penalty": 100.0,
                "reward_scale": 1000.0,
            },
        },
        path,
    )


def _add_totals(total: Dict[str, Any], batch: Dict[str, Any]) -> None:
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
        total[key] += batch[key]
    for key in (
        "proposed_counts",
        "label_counts",
        "override_action_counts",
        "lane_option_counts",
        "executed_counts",
    ):
        total[key] = [a + b for a, b in zip(total[key], batch[key])]
    for key in ("trigger_reasons", "selected_plans"):
        counts = Counter(total[key])
        counts.update(batch[key])
        total[key] = dict(counts)


def _log_collection(stats: Dict[str, Any]) -> str:
    episodes = int(stats["episodes"])
    ticks = max(1, int(stats["ticks"]))
    crash_rate = stats["crashes"] / episodes if episodes else 0.0
    distance = stats["completed_distance"] / episodes if episodes else 0.0
    proposed = "/".join(
        f"{100.0 * value / ticks:.0f}" for value in stats["proposed_counts"]
    )
    executed = "/".join(
        f"{100.0 * value / ticks:.0f}" for value in stats["executed_counts"]
    )
    corrected = "/".join(
        str(int(value)) for value in stats["override_action_counts"]
    )
    return (
        f"teacher_ep={episodes} crash={crash_rate:.1%} dist={distance:,.0f} "
        f"trigger={stats['triggers'] / ticks:.1%} "
        f"override={stats['overrides'] / ticks:.1%} "
        f"strategic={stats['strategic_accepted']}/{stats['strategic_candidates']} "
        f"student_behavior={stats['student_behavior_ticks'] / ticks:.0%} "
        f"proposed[N/A/D/L/R]={proposed} behavior={executed} "
        f"corrected[N/A/D/L/R]={corrected}"
    )


def main() -> None:
    args = parse_args()
    if args.transitions <= 0 or args.workers <= 0 or args.rollout_steps <= 0:
        raise ValueError("transitions, workers, and rollout-steps must be positive")
    if args.epochs <= 0 or args.env_minibatch_size <= 0:
        raise ValueError("epochs and env-minibatch-size must be positive")
    if args.hard_weight < 0 or args.soft_weight < 0:
        raise ValueError("distillation weights must be non-negative")
    if args.hard_weight == 0 and args.soft_weight == 0:
        raise ValueError("at least one distillation weight must be positive")
    if (
        not math.isfinite(args.lane_option_loss_weight)
        or args.lane_option_loss_weight < 0
    ):
        raise ValueError("lane-option-loss-weight must be finite and non-negative")
    if (
        not math.isfinite(args.lane_option_positive_weight)
        or args.lane_option_positive_weight <= 0
    ):
        raise ValueError("lane-option-positive-weight must be finite and positive")
    if (
        not math.isfinite(args.lane_option_threshold)
        or not 0.0 < args.lane_option_threshold <= 1.0
    ):
        raise ValueError("lane-option-threshold must be finite and in (0, 1]")
    if args.trigger_sample_weight <= 0 or args.override_sample_weight <= 0:
        raise ValueError("sample weights must be positive")
    if not 0.0 <= args.student_behavior_fraction <= 1.0:
        raise ValueError("student-behavior-fraction must be in [0, 1]")
    if args.student_behavior_fraction > 0 and args.behavior_student_checkpoint is None:
        raise ValueError(
            "--behavior-student-checkpoint is required for student behavior"
        )
    if args.resume is not None and args.init_student_checkpoint is not None:
        raise ValueError(
            "--resume and --init-student-checkpoint are mutually exclusive"
        )
    config = mpc_config(args)
    config.validate()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    args.output.mkdir(parents=True, exist_ok=True)
    teacher_state = load_teacher_state(args.teacher_checkpoint)

    def load_legal_gru(path: Path) -> tuple[Dict[str, torch.Tensor], int, int]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("arch") != "gru" or "model_state" not in payload:
            raise ValueError(f"{path} must be a legal GRU checkpoint")
        state = payload["model_state"]
        obs_dim = recurrent_obs_dim_from_state_dict(state)
        if obs_dim not in (LEGAL_OBS_SIZE, LEGAL_OBS_SIZE_TRACKED_Y):
            raise ValueError(f"{path} uses unsupported {obs_dim}-D observations")
        hidden_dim = int(payload.get("hidden_dim", state["gru.weight_hh"].shape[1]))
        return dict(state), obs_dim, hidden_dim

    behavior_student_state = None
    behavior_hidden_dim = args.hidden_dim
    if args.behavior_student_checkpoint is not None:
        behavior_state, behavior_obs_dim, behavior_hidden_dim = load_legal_gru(
            args.behavior_student_checkpoint
        )
        if behavior_obs_dim != LEGAL_OBS_SIZE_TRACKED_Y:
            behavior_state, _ = expand_recurrent_obs_state_dict(
                behavior_state, LEGAL_OBS_SIZE_TRACKED_Y
            )
        behavior_student_state = {
            key: value.detach().cpu().share_memory_()
            for key, value in behavior_state.items()
        }
    proposal_student_state = None
    proposal_obs_dim = LEGAL_OBS_SIZE_TRACKED_Y
    proposal_hidden_dim = args.hidden_dim
    if args.proposal_student_checkpoint is not None:
        proposal_state, proposal_obs_dim, proposal_hidden_dim = load_legal_gru(
            args.proposal_student_checkpoint
        )
        proposal_student_state = {
            key: value.detach().cpu().share_memory_()
            for key, value in proposal_state.items()
        }
    student = RecurrentActorCritic(
        LEGAL_OBS_SIZE_TRACKED_Y,
        len(ACTIONS),
        hidden_dim=args.hidden_dim,
        safe_init=False,
    ).to(device)
    with torch.no_grad():
        student.value_head.weight.zero_()
        student.value_head.bias.zero_()
    if args.init_student_checkpoint is not None:
        initial_state, initial_obs_dim, initial_hidden_dim = load_legal_gru(
            args.init_student_checkpoint
        )
        if initial_hidden_dim != args.hidden_dim:
            raise ValueError(
                "initial checkpoint hidden dimension does not match --hidden-dim"
            )
        if initial_obs_dim < LEGAL_OBS_SIZE_TRACKED_Y:
            initial_state, _ = expand_recurrent_obs_state_dict(
                initial_state, LEGAL_OBS_SIZE_TRACKED_Y
            )
        student.load_state_dict(initial_state)
    lane_option_head = None
    if args.lane_option_loss_weight > 0:
        lane_option_head = nn.Linear(args.hidden_dim, 3).to(device)
        with torch.no_grad():
            lane_option_head.weight.zero_()
            lane_option_head.bias.zero_()
    optimizer = make_optimizer(student, args, lane_option_head)

    transitions = 0
    best_scores = {
        "deterministic": (-float("inf"),) * 4,
        "stochastic": (-float("inf"),) * 4,
        "scheduler": (-float("inf"),) * 4,
        "lane_options": (-float("inf"),) * 4,
    }
    collection_totals = merge_collection_stats([])
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if (
            not payload.get("distilled")
            or payload.get("obs_dim") != LEGAL_OBS_SIZE_TRACKED_Y
        ):
            raise ValueError("resume checkpoint must be a 42-D distillation checkpoint")
        student.load_state_dict(payload["model_state"])
        saved_lane_option = payload.get("lane_option_state")
        if lane_option_head is not None and saved_lane_option is not None:
            lane_option_head.load_state_dict(saved_lane_option)
        optimizer_compatible = (lane_option_head is None) == (
            saved_lane_option is None
        )
        if "distill_optimizer_state" in payload and optimizer_compatible:
            optimizer.load_state_dict(payload["distill_optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        elif "distill_optimizer_state" in payload:
            print("Starting fresh optimizer state because parameter groups changed")
        transitions = int(payload.get("distill_transitions", 0))
        if int(payload.get("student_score_version", 1)) == 2:
            for mode, score in payload.get("best_scores", {}).items():
                best_scores[mode] = tuple(float(value) for value in score)
        saved_totals = payload.get("collection_totals")
        if saved_totals:
            migrated_totals = merge_collection_stats([])
            for key in migrated_totals:
                if key in saved_totals:
                    migrated_totals[key] = saved_totals[key]
            # Checkpoints created before DAgger distinguished the MPC label
            # action from the action that drove data collection.
            if "label_counts" not in saved_totals:
                migrated_totals["label_counts"] = list(
                    saved_totals.get("executed_counts", [0] * len(ACTIONS))
                )
            collection_totals = migrated_totals
        print(f"Resumed {args.resume} at {transitions:,} transitions")

    # A resumed job starts beyond all earlier episode seeds. This intentionally
    # sacrifices exact mid-episode continuation to avoid duplicating labels.
    collector_seed_start = args.data_seed_start + transitions
    pool = CollectorPool(
        teacher_state,
        workers=args.workers,
        seed_start=collector_seed_start,
        config=config,
        behavior_student_state=behavior_student_state,
        behavior_hidden_dim=behavior_hidden_dim,
        behavior_mode=args.behavior_mode,
        proposal_student_state=proposal_student_state,
        proposal_obs_dim=proposal_obs_dim,
        proposal_hidden_dim=proposal_hidden_dim,
        proposal_mode=args.proposal_mode,
    )
    hidden = student.initial_state(args.workers, device=device)
    lane_option_class_weights = torch.tensor(
        [1.0, args.lane_option_positive_weight, args.lane_option_positive_weight],
        dtype=torch.float32,
        device=device,
    )

    rollout_batch = args.workers * args.rollout_steps
    next_eval = (
        ((transitions // args.eval_every) + 1) * args.eval_every
        if args.eval_every > 0 else 2**63 - 1
    )
    next_save = (
        ((transitions // args.save_every) + 1) * args.save_every
        if args.save_every > 0 else 2**63 - 1
    )
    print(
        f"Exact MPC -> legal GRU: workers={args.workers} rollout={args.rollout_steps} "
        f"batch={rollout_batch:,} device={device} horizon={config.horizon}/"
        f"{config.strategic_horizon} sigma={config.opponent_uncertainty_sigma:g}"
    )
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")

    started = time.perf_counter()
    iteration = 0
    last_eval = None
    evaluation_history: List[Dict[str, Any]] = []
    try:
        while transitions < args.transitions:
            iteration += 1
            iteration_started = time.perf_counter()
            results = pool.collect(
                args.rollout_steps,
                student_behavior_fraction=args.student_behavior_fraction,
            )
            batch_stats = merge_collection_stats([item["stats"] for item in results])
            _add_totals(collection_totals, batch_stats)

            obs = torch.from_numpy(
                np.stack([item["observations"] for item in results], axis=1)
            ).to(device)
            teacher_actions = torch.from_numpy(
                np.stack([item["actions"] for item in results], axis=1)
            ).to(device)
            teacher_probs = torch.from_numpy(
                np.stack([item["probabilities"] for item in results], axis=1)
            ).to(device)
            dones = torch.from_numpy(
                np.stack([item["dones"] for item in results], axis=1)
            ).to(device)
            trigger_mask = torch.from_numpy(
                np.stack([item["triggers"] for item in results], axis=1)
            ).to(device)
            override_mask = torch.from_numpy(
                np.stack([item["overrides"] for item in results], axis=1)
            ).to(device)
            lane_option_targets = torch.from_numpy(
                np.stack([item["lane_options"] for item in results], axis=1)
            ).to(device)
            sample_weights = torch.ones_like(teacher_actions, dtype=torch.float32)
            sample_weights = torch.where(
                trigger_mask,
                torch.full_like(sample_weights, args.trigger_sample_weight),
                sample_weights,
            )
            sample_weights = torch.where(
                override_mask,
                torch.full_like(sample_weights, args.override_sample_weight),
                sample_weights,
            )

            initial_hidden = hidden.detach().clone()
            student.eval()
            with torch.no_grad():
                for tick in range(args.rollout_steps):
                    _, _, hidden = student.step(obs[tick], hidden)
                    hidden *= (~dones[tick]).to(hidden.dtype).unsqueeze(1)
            student.train()
            stats = distill_update_recurrent(
                student,
                optimizer,
                obs,
                teacher_actions,
                teacher_probs,
                dones,
                initial_hidden,
                sample_weights=sample_weights,
                lane_option_head=lane_option_head,
                lane_option_targets=(
                    lane_option_targets if lane_option_head is not None else None
                ),
                lane_option_class_weights=(
                    lane_option_class_weights if lane_option_head is not None else None
                ),
                lane_option_loss_weight=args.lane_option_loss_weight,
                epochs=args.epochs,
                env_minibatch_size=args.env_minibatch_size,
                hard_weight=args.hard_weight,
                soft_weight=args.soft_weight,
                ent_coef=args.ent_coef,
                max_grad_norm=args.max_grad_norm,
            )
            hidden = hidden.detach()
            transitions += rollout_batch
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - iteration_started
            print(
                f"[{iteration:04d}] transitions={transitions:,} "
                f"fps={rollout_batch / max(elapsed, 1e-9):,.0f} "
                f"{_log_collection(batch_stats)} loss={stats.loss:.4f} "
                f"KL={stats.teacher_kl:.4f} acc={stats.action_accuracy:.1%} "
                f"option_loss={stats.lane_option_loss:.4f} "
                f"option_acc={stats.lane_option_accuracy:.1%}"
            )

            saved_this_iteration = transitions >= next_save
            if saved_this_iteration:
                save_checkpoint(
                    args.output / f"distilled_gru_mpc_{transitions}.pt",
                    student,
                    optimizer,
                    transitions=transitions,
                    args=args,
                    config=config,
                    best_scores=best_scores,
                    collection_totals=collection_totals,
                    lane_option_head=lane_option_head,
                )
                save_checkpoint(
                    args.output / "latest.pt",
                    student,
                    optimizer,
                    transitions=transitions,
                    args=args,
                    config=config,
                    best_scores=best_scores,
                    collection_totals=collection_totals,
                    lane_option_head=lane_option_head,
                )
                while next_save <= transitions:
                    next_save += args.save_every

            if transitions >= next_eval:
                last_eval = evaluate_student(student, args, lane_option_head)
                evaluation_history.append(
                    {"transitions": int(transitions), "results": last_eval}
                )
                for mode in last_eval:
                    result = last_eval[mode]
                    print(
                        f"  ORIGINAL {mode}: mean={result['mean_distance']:,.0f} "
                        f"p10={result['p10_distance']:,.0f} "
                        f"crash={result['crash_rate']:.1%} "
                        f"<25k={result['below_25k_rate']:.1%}"
                    )
                    score = _score(result)
                    if score > best_scores[mode]:
                        best_scores[mode] = score
                        save_checkpoint(
                            args.output / f"best_{mode}.pt",
                            student,
                            optimizer,
                            transitions=transitions,
                            args=args,
                            config=config,
                            best_scores=best_scores,
                            collection_totals=collection_totals,
                            lane_option_head=lane_option_head,
                        )
                (args.output / "latest_eval.json").write_text(
                    json.dumps(last_eval, indent=2), encoding="utf-8"
                )
                # Periodic checkpoints are initially written before evaluation.
                # Refresh their selection metadata so resuming cannot forget a
                # best score discovered at this same transition boundary.
                if saved_this_iteration:
                    save_checkpoint(
                        args.output / "latest.pt",
                        student,
                        optimizer,
                        transitions=transitions,
                        args=args,
                        config=config,
                        best_scores=best_scores,
                        collection_totals=collection_totals,
                        lane_option_head=lane_option_head,
                    )
                    save_checkpoint(
                        args.output / f"distilled_gru_mpc_{transitions}.pt",
                        student,
                        optimizer,
                        transitions=transitions,
                        args=args,
                        config=config,
                        best_scores=best_scores,
                        collection_totals=collection_totals,
                        lane_option_head=lane_option_head,
                    )
                while next_eval <= transitions:
                    next_eval += args.eval_every
    finally:
        pool.close()

    # Always finish with an exact held-out evaluation, including smoke runs
    # where periodic evaluation was disabled.
    if not evaluation_history or evaluation_history[-1]["transitions"] != transitions:
        last_eval = evaluate_student(student, args, lane_option_head)
        evaluation_history.append(
            {"transitions": int(transitions), "results": last_eval}
        )
    assert last_eval is not None
    for mode in last_eval:
        result = last_eval[mode]
        print(
            f"  FINAL ORIGINAL {mode}: mean={result['mean_distance']:,.0f} "
            f"p10={result['p10_distance']:,.0f} "
            f"crash={result['crash_rate']:.1%} "
            f"<25k={result['below_25k_rate']:.1%}"
        )
        score = _score(result)
        if score > best_scores[mode]:
            best_scores[mode] = score
            save_checkpoint(
                args.output / f"best_{mode}.pt",
                student,
                optimizer,
                transitions=transitions,
                args=args,
                config=config,
                best_scores=best_scores,
                collection_totals=collection_totals,
                lane_option_head=lane_option_head,
            )
    save_checkpoint(
        args.output / "final.pt",
        student,
        optimizer,
        transitions=transitions,
        args=args,
        config=config,
        best_scores=best_scores,
        collection_totals=collection_totals,
        lane_option_head=lane_option_head,
    )
    wall_time = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "arguments": _json_args(args),
        "git_revision": _revision(),
        "teacher_checkpoint_sha256": _sha256(args.teacher_checkpoint),
        "proposal_student_checkpoint_sha256": (
            None
            if args.proposal_student_checkpoint is None
            else _sha256(args.proposal_student_checkpoint)
        ),
        "mpc_config": config.to_dict(),
        "transitions": transitions,
        "elapsed_seconds": wall_time,
        "transitions_per_second": transitions / max(wall_time, 1e-9),
        "collection_totals": collection_totals,
        "best_scores": {key: list(value) for key, value in best_scores.items()},
        "evaluation_history": evaluation_history,
        "evaluation": last_eval,
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"Finished {transitions:,} transitions in {wall_time:.1f}s; "
        f"report={args.output / 'training_report.json'}"
    )


if __name__ == "__main__":
    main()
