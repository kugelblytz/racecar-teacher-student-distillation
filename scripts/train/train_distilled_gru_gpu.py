"""GPU-native privileged-teacher -> legal GRU policy distillation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.direct_eval import evaluate_direct_original
from training.gpu_distill import distill_update_recurrent
from training.gpu_ppo import ActorCritic
from training.gpu_recurrent_ppo import RecurrentActorCritic
from training.gpu_sim import (
    ACTIONS,
    LEGAL_OBS_SIZE_TRACKED_Y,
    OBS_SIZE,
    GpuRaceCarBatch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Distill a privileged CUDA teacher into a recurrent legal-observation "
            "student. The teacher drives the simulator; the student sees only the "
            "42-D legal V2 observation."
        )
    )
    p.add_argument("teacher_checkpoint", type=Path)
    p.add_argument("--transitions", type=int, default=200_000_000)
    p.add_argument("--envs", type=int, default=16_384)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--env-minibatch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2)

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--hard-weight",
        type=float,
        default=1.0,
        help="Cross-entropy weight for the teacher's executed action.",
    )
    p.add_argument(
        "--soft-weight",
        type=float,
        default=0.25,
        help="Cross-entropy weight for the teacher's full action distribution.",
    )
    p.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Optional additional student entropy bonus during distillation.",
    )
    p.add_argument(
        "--teacher-mode",
        choices=["scheduler", "deterministic", "stochastic"],
        default="scheduler",
        help=(
            "How the privileged teacher drives data collection. scheduler is "
            "the deterministic probability-credit decoder used by our strongest "
            "teacher evaluation."
        ),
    )
    p.add_argument("--teacher-temperature", type=float, default=1.0)

    p.add_argument("--eval-every", type=int, default=25_000_000)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--eval-seed-start", type=int, default=70_000)
    p.add_argument("--eval-action-seed-start", type=int, default=7_000_000)
    p.add_argument("--eval-workers", type=int, default=24)
    p.add_argument("--save-every", type=int, default=25_000_000)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/distilled_gru"),
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a distillation checkpoint (not a PPO checkpoint).",
    )
    return p.parse_args()


def make_optimizer(model: RecurrentActorCritic, args: argparse.Namespace):
    kwargs = dict(lr=args.learning_rate, eps=1e-5)
    if torch.device(args.device).type == "cuda":
        try:
            return torch.optim.Adam(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.Adam(model.parameters(), **kwargs)


def load_teacher(path: Path, device: torch.device) -> ActorCritic:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if state is None:
        raise ValueError(f"{path} does not contain model_state")

    teacher = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    teacher.load_state_dict(state)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def save_checkpoint(
    path: Path,
    student: RecurrentActorCritic,
    optimizer,
    *,
    transitions: int,
    best_det: float,
    best_stoch: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "tracked_y": True,
            "action_repeat": 1,
            "crash_penalty": 50.0,
            "wall_crash_penalty": 100.0,
            "reward_scale": 1000.0,
        }
    )
    torch.save(
        {
            "arch": "gru",
            "obs_kind": "legal",
            "obs_dim": LEGAL_OBS_SIZE_TRACKED_Y,
            "obs_version": 2,
            "hidden_dim": int(args.hidden_dim),
            "distilled": True,
            "teacher_checkpoint": str(args.teacher_checkpoint),
            "teacher_mode": str(args.teacher_mode),
            "teacher_temperature": float(args.teacher_temperature),
            "distill_transitions": int(transitions),
            # Deliberately not named optimizer_state: PPO fine-tuning should
            # start with fresh Adam moments rather than reuse supervised ones.
            "distill_optimizer_state": optimizer.state_dict(),
            "model_state": {
                k: v.detach().cpu() for k, v in student.state_dict().items()
            },
            "best_deterministic_mean": float(best_det),
            "best_stochastic_mean": float(best_stoch),
            "config": config,
        },
        path,
    )


@torch.no_grad()
def choose_teacher_action(
    logits: torch.Tensor,
    *,
    mode: str,
    temperature: float,
    scheduler_credit: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("teacher temperature must be positive")

    probs = torch.softmax(logits / float(temperature), dim=-1)

    if mode == "deterministic":
        actions = logits.argmax(dim=-1)
    elif mode == "stochastic":
        actions = torch.multinomial(
            probs,
            1,
            generator=generator,
        ).squeeze(1)
    elif mode == "scheduler":
        scheduler_credit.add_(probs)
        actions = scheduler_credit.argmax(dim=-1)
        scheduler_credit.sub_(
            F.one_hot(actions, num_classes=len(ACTIONS)).to(
                scheduler_credit.dtype
            )
        )
    else:
        raise ValueError(f"unknown teacher mode: {mode}")

    return actions, probs


def evaluate_student(
    student: RecurrentActorCritic,
    args: argparse.Namespace,
):
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
    )


def main() -> None:
    args = parse_args()
    if args.transitions <= 0:
        raise ValueError("--transitions must be positive")
    if args.rollout_steps <= 0:
        raise ValueError("--rollout-steps must be positive")
    if args.teacher_temperature <= 0:
        raise ValueError("--teacher-temperature must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    args.output.mkdir(parents=True, exist_ok=True)

    teacher = load_teacher(args.teacher_checkpoint, device)
    student = RecurrentActorCritic(
        LEGAL_OBS_SIZE_TRACKED_Y,
        len(ACTIONS),
        hidden_dim=args.hidden_dim,
        safe_init=False,
    ).to(device)

    # Distillation trains the policy representation only. Start PPO's future
    # value head from a neutral V=0 rather than random predictions.
    with torch.no_grad():
        student.value_head.weight.zero_()
        student.value_head.bias.zero_()

    optimizer = make_optimizer(student, args)

    transitions = 0
    best_det = float("-inf")
    best_stoch = float("-inf")
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if not bool(payload.get("distilled", False)):
            raise ValueError("--resume must be a distillation checkpoint")
        if int(payload.get("obs_dim", LEGAL_OBS_SIZE_TRACKED_Y)) != LEGAL_OBS_SIZE_TRACKED_Y:
            raise ValueError("distillation resume checkpoint is not 42-D V2")
        student.load_state_dict(payload["model_state"])
        if "distill_optimizer_state" in payload:
            optimizer.load_state_dict(payload["distill_optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        transitions = int(payload.get("distill_transitions", 0))
        best_det = float(payload.get("best_deterministic_mean", float("-inf")))
        best_stoch = float(payload.get("best_stochastic_mean", float("-inf")))
        print(f"Resumed distillation from {args.resume} at {transitions:,} transitions")

    sim = GpuRaceCarBatch(
        args.envs,
        device=device,
        seed=args.seed,
        crash_penalty=50.0,
        wall_crash_penalty=100.0,
        reward_scale=1000.0,
        opponent_count=4,
    )

    n = int(args.envs)
    T = int(args.rollout_steps)
    rollout_batch = n * T

    obs_buf = torch.empty(
        (T, n, LEGAL_OBS_SIZE_TRACKED_Y),
        dtype=torch.float32,
        device=device,
    )
    teacher_actions_buf = torch.empty(
        (T, n), dtype=torch.int64, device=device
    )
    teacher_probs_buf = torch.empty(
        (T, n, len(ACTIONS)), dtype=torch.float32, device=device
    )
    dones_buf = torch.empty((T, n), dtype=torch.bool, device=device)

    student_hidden = student.initial_state(n, device=device)
    scheduler_credit = torch.zeros(
        (n, len(ACTIONS)), dtype=torch.float32, device=device
    )
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed + 1_000_000)

    def next_boundary(interval: int) -> int:
        if interval <= 0:
            return 2**63 - 1
        return ((transitions // interval) + 1) * interval

    next_eval = next_boundary(args.eval_every)
    next_save = next_boundary(args.save_every)

    print(
        f"Teacher -> GRU distillation: device={device} envs={n:,} rollout={T} "
        f"batch={rollout_batch:,} legal_obs={LEGAL_OBS_SIZE_TRACKED_Y} "
        f"teacher={args.teacher_mode}@T{args.teacher_temperature:g} "
        f"epochs={args.epochs} hard={args.hard_weight:g} "
        f"soft={args.soft_weight:g} hidden={args.hidden_dim}"
    )
    print(f"Teacher checkpoint: {args.teacher_checkpoint}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    wall_start = time.perf_counter()
    iteration = 0

    while transitions < args.transitions:
        iteration += 1
        rollout_start = time.perf_counter()

        completed = torch.zeros((), dtype=torch.float32, device=device)
        crashes = torch.zeros((), dtype=torch.float32, device=device)
        car_crashes = torch.zeros((), dtype=torch.float32, device=device)
        wall_crashes = torch.zeros((), dtype=torch.float32, device=device)
        completed_distance = torch.zeros((), dtype=torch.float64, device=device)
        teacher_action_counts = torch.zeros(
            len(ACTIONS), dtype=torch.float32, device=device
        )

        initial_hidden = student_hidden.detach().clone()
        student.eval()

        for t in range(T):
            with torch.no_grad():
                legal_obs = sim.legal_observation(tracked_y=True)
                privileged_obs = sim.observation()
                teacher_logits, _ = teacher(privileged_obs)
                teacher_actions, teacher_probs = choose_teacher_action(
                    teacher_logits,
                    mode=args.teacher_mode,
                    temperature=args.teacher_temperature,
                    scheduler_credit=scheduler_credit,
                    generator=action_generator,
                )

                _, _, next_student_hidden = student.step(
                    legal_obs, student_hidden
                )

            obs_buf[t].copy_(legal_obs)
            teacher_actions_buf[t].copy_(teacher_actions)
            teacher_probs_buf[t].copy_(teacher_probs)

            teacher_action_counts += torch.bincount(
                teacher_actions, minlength=len(ACTIONS)
            ).to(torch.float32)

            _, _, dones, episode = sim.step(
                teacher_actions,
                observation_kind="none",
                tracked_y=True,
            )
            dones_buf[t].copy_(dones)

            student_hidden = next_student_hidden * (
                ~dones
            ).to(torch.float32).unsqueeze(1)
            scheduler_credit *= (~dones).to(torch.float32).unsqueeze(1)

            completed += episode.done.float().sum()
            crashes += episode.crashed.float().sum()
            car_crashes += episode.hit_car.float().sum()
            wall_crashes += episode.hit_wall.float().sum()
            completed_distance += episode.distance.sum()

        student.train()
        stats = distill_update_recurrent(
            student,
            optimizer,
            obs_buf,
            teacher_actions_buf,
            teacher_probs_buf,
            dones_buf,
            initial_hidden,
            epochs=args.epochs,
            env_minibatch_size=args.env_minibatch_size,
            hard_weight=args.hard_weight,
            soft_weight=args.soft_weight,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
        )
        student_hidden = student_hidden.detach()

        transitions += rollout_batch
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - rollout_start
        completed_n = float(completed.item())
        crash_rate = (
            float(crashes.item()) / completed_n if completed_n > 0 else 0.0
        )
        car_rate = (
            float(car_crashes.item()) / completed_n if completed_n > 0 else 0.0
        )
        wall_rate = (
            float(wall_crashes.item()) / completed_n if completed_n > 0 else 0.0
        )
        ep_dist = (
            float(completed_distance.item()) / completed_n
            if completed_n > 0
            else 0.0
        )
        action_frac = (
            teacher_action_counts
            / max(float(teacher_action_counts.sum().item()), 1.0)
        ).detach().cpu().tolist()
        action_text = "/".join(f"{100.0 * x:.0f}" for x in action_frac)

        print(
            f"[{iteration:04d}] transitions={transitions:,} "
            f"fps={rollout_batch / max(elapsed, 1e-9):,.0f} "
            f"time={elapsed:.2f}s teacher_ep={int(completed_n):,} "
            f"crash={crash_rate:.1%}(car={car_rate:.1%},wall={wall_rate:.1%}) "
            f"ep_dist={ep_dist:,.0f} "
            f"teacher_act[N/A/D/L/R]={action_text} "
            f"loss={stats.loss:.4f} hard={stats.hard_loss:.4f} "
            f"soft={stats.soft_loss:.4f} KL={stats.teacher_kl:.4f} "
            f"acc={stats.action_accuracy:.1%} ent={stats.entropy:.3f}"
        )

        if transitions >= next_save:
            save_checkpoint(
                args.output / f"distilled_gru_{transitions}.pt",
                student,
                optimizer,
                transitions=transitions,
                best_det=best_det,
                best_stoch=best_stoch,
                args=args,
            )
            save_checkpoint(
                args.output / "latest.pt",
                student,
                optimizer,
                transitions=transitions,
                best_det=best_det,
                best_stoch=best_stoch,
                args=args,
            )
            while next_save <= transitions:
                next_save += args.save_every

        if transitions >= next_eval:
            eval_start = time.perf_counter()
            eval_stats = evaluate_student(student, args)
            for mode in ("deterministic", "stochastic"):
                s = eval_stats[mode]
                print(
                    f"  ORIGINAL {mode:13s}: mean={s['mean_distance']:,.1f} "
                    f"median={s['median_distance']:,.1f} "
                    f"p10={s['p10_distance']:,.1f} crash={s['crash_rate']:.1%} "
                    f"<25k={s['below_25k_rate']:.1%}"
                )
            print(
                f"  eval_time={time.perf_counter() - eval_start:.1f}s "
                f"workers={eval_stats['deterministic']['workers']}"
            )

            (args.output / "latest_eval.json").write_text(
                json.dumps(eval_stats, indent=2),
                encoding="utf-8",
            )

            det_mean = float(eval_stats["deterministic"]["mean_distance"])
            stoch_mean = float(eval_stats["stochastic"]["mean_distance"])
            if det_mean > best_det:
                best_det = det_mean
                save_checkpoint(
                    args.output / "best_deterministic.pt",
                    student,
                    optimizer,
                    transitions=transitions,
                    best_det=best_det,
                    best_stoch=best_stoch,
                    args=args,
                )
            if stoch_mean > best_stoch:
                best_stoch = stoch_mean
                save_checkpoint(
                    args.output / "best_stochastic.pt",
                    student,
                    optimizer,
                    transitions=transitions,
                    best_det=best_det,
                    best_stoch=best_stoch,
                    args=args,
                )

            while next_eval <= transitions:
                next_eval += args.eval_every

    save_checkpoint(
        args.output / "final.pt",
        student,
        optimizer,
        transitions=transitions,
        best_det=best_det,
        best_stoch=best_stoch,
        args=args,
    )
    print(
        f"Finished {transitions:,} distillation transitions in "
        f"{time.perf_counter() - wall_start:.1f}s"
    )


if __name__ == "__main__":
    main()
