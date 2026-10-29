"""CUDA-native privileged PPO teacher training.

Unlike train_teacher.py / Stable-Baselines3, this keeps simulation, observations,
rollout buffers, GAE and PPO optimization on one GPU.

Example RTX 5090 smoke test:
    python train_teacher_gpu.py --timesteps 10000000 --envs 8192
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from training.gpu_eval import evaluate_policy_diagnostics
from training.gpu_ppo import ActorCritic, ppo_update
from training.gpu_sim import ACTIONS, OBS_SIZE, GpuRaceCarBatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--timesteps", type=int, default=50_000_000)
    p.add_argument("--envs", type=int, default=8192)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--minibatch-size", type=int, default=65_536)
    p.add_argument("--epochs", type=int, default=4)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.9995)
    p.add_argument("--gae-lambda", type=float, default=0.97)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03)

    p.add_argument("--crash-penalty", type=float, default=250.0)
    p.add_argument("--reward-scale", type=float, default=1000.0)
    p.add_argument(
        "--survival-bonus",
        type=float,
        default=0.0,
        help=(
            "Optional per-tick bonus on non-crash transitions. Intended for "
            "safety fine-tuning; 0 preserves the historical teacher reward."
        ),
    )
    p.add_argument(
        "--ttc-risk-penalty",
        type=float,
        default=0.0,
        help=(
            "Optional dense penalty for imminent longitudinal collision risk "
            "inside/near the ego lateral corridor."
        ),
    )
    p.add_argument("--ttc-risk-horizon", type=float, default=120.0)
    p.add_argument("--ttc-lateral-margin", type=float, default=24.0)

    p.add_argument(
        "--safe-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bias the initial primitive policy toward ACCELERATE/lane holding.",
    )
    p.add_argument(
        "--fused-adam",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--eval-every", type=int, default=5_000_000)
    p.add_argument("--eval-episodes", type=int, default=32)
    p.add_argument("--eval-seed-start", type=int, default=10_000)
    p.add_argument(
        "--eval-workers",
        type=int,
        default=0,
        help="Original-simulator CPU workers; 0=auto (up to 24), 1=serial.",
    )
    p.add_argument(
        "--eval-action-seed-start",
        type=int,
        default=1_000_000,
        help="Reproducible action RNG seed for stochastic evaluation.",
    )
    p.add_argument(
        "--eval-gpu-seed",
        type=int,
        default=3_000_000,
        help="CUDA simulator RNG seed used by diagnostic evaluation.",
    )
    p.add_argument("--save-every", type=int, default=5_000_000)
    p.add_argument(
        "--selection-mode",
        choices=["scheduler", "deterministic", "stochastic"],
        default="scheduler",
        help="Original-simulator decoder used for best-teacher checkpointing.",
    )
    p.add_argument(
        "--selection-max-crash",
        type=float,
        default=0.05,
        help=(
            "Crash-rate gate for best-teacher selection. Any checkpoint at or "
            "below this rate outranks checkpoints above it."
        ),
    )
    p.add_argument(
        "--selection-crash-weight",
        type=float,
        default=500_000.0,
        help=(
            "Distance-equivalent penalty per unit crash rate while above the "
            "selection crash-rate gate."
        ),
    )

    p.add_argument("--output", type=Path, default=Path("models/gpu_teacher"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--resume-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restore Adam state when resuming. The CLI --learning-rate is always "
            "re-applied after loading. Use --no-resume-optimizer for a fresh Adam."
        ),
    )

    return p.parse_args()


def make_optimizer(model: ActorCritic, args: argparse.Namespace):
    kwargs = dict(lr=args.learning_rate, eps=1e-5)
    if args.fused_adam and torch.device(args.device).type == "cuda":
        try:
            return torch.optim.Adam(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.Adam(model.parameters(), **kwargs)


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    best_mean_distance: float,
    best_selection_score: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "timesteps": int(total_steps),
        "best_mean_distance": float(best_mean_distance),
        "best_selection_score": float(best_selection_score),
        "config": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    load_optimizer: bool,
    learning_rate: float,
):
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])

    optimizer_loaded = False
    if load_optimizer and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
        optimizer_loaded = True

    # torch.optim state_dict restores the old param-group learning rate. For
    # fine-tuning, the command-line learning rate must win.
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)

    return (
        int(payload.get("timesteps", 0)),
        float(payload.get("best_mean_distance", float("-inf"))),
        float(payload.get("best_selection_score", float("-inf"))),
        optimizer_loaded,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    if args.minibatch_size <= 0:
        raise ValueError("--minibatch-size must be positive")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    args.output.mkdir(parents=True, exist_ok=True)

    sim = GpuRaceCarBatch(
        num_envs=args.envs,
        device=device,
        seed=args.seed,
        crash_penalty=args.crash_penalty,
        reward_scale=args.reward_scale,
        survival_bonus=args.survival_bonus,
        ttc_risk_penalty=args.ttc_risk_penalty,
        ttc_risk_horizon=args.ttc_risk_horizon,
        ttc_lateral_margin=args.ttc_lateral_margin,
    )

    model = ActorCritic(
        obs_dim=OBS_SIZE,
        action_dim=len(ACTIONS),
        safe_init=args.safe_init,
    ).to(device)
    optimizer = make_optimizer(model, args)

    total_steps = 0
    best_mean_distance = float("-inf")
    best_selection_score = float("-inf")
    if args.resume is not None:
        (
            total_steps,
            best_mean_distance,
            best_selection_score,
            optimizer_loaded,
        ) = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
            load_optimizer=args.resume_optimizer,
            learning_rate=args.learning_rate,
        )
        optimizer_note = "restored Adam state" if optimizer_loaded else "fresh Adam state"
        print(
            f"Resumed {args.resume} at {total_steps:,} environment steps; "
            f"{optimizer_note}; lr={args.learning_rate:g}"
        )

    n = args.envs
    t_steps = args.rollout_steps
    rollout_batch = n * t_steps

    obs_buf = torch.empty(
        (t_steps, n, OBS_SIZE), dtype=torch.float32, device=device
    )
    actions_buf = torch.empty((t_steps, n), dtype=torch.int64, device=device)
    logprobs_buf = torch.empty((t_steps, n), dtype=torch.float32, device=device)
    rewards_buf = torch.empty((t_steps, n), dtype=torch.float32, device=device)
    dones_buf = torch.empty((t_steps, n), dtype=torch.float32, device=device)
    values_buf = torch.empty((t_steps, n), dtype=torch.float32, device=device)
    advantages_buf = torch.empty_like(rewards_buf)

    obs = sim.observation()

    def next_boundary(interval: int) -> int:
        if interval <= 0:
            return 2**63 - 1
        return ((total_steps // interval) + 1) * interval

    next_eval = next_boundary(args.eval_every)
    next_save = next_boundary(args.save_every)

    print(
        f"GPU teacher: device={device} envs={n:,} rollout={t_steps} "
        f"batch={rollout_batch:,} minibatch={args.minibatch_size:,} "
        f"crash_penalty={args.crash_penalty:g} "
        f"survival_bonus={args.survival_bonus:g} "
        f"ttc_risk={args.ttc_risk_penalty:g}@{args.ttc_risk_horizon:g} "
        f"select={args.selection_mode} max_crash={args.selection_max_crash:.1%}"
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    wall_start = time.perf_counter()
    iteration = 0

    while total_steps < args.timesteps:
        iteration += 1

        completed = torch.zeros((), dtype=torch.float32, device=device)
        crashes = torch.zeros((), dtype=torch.float32, device=device)
        car_crashes = torch.zeros((), dtype=torch.float32, device=device)
        wall_crashes = torch.zeros((), dtype=torch.float32, device=device)
        completed_distance = torch.zeros((), dtype=torch.float32, device=device)
        completed_speed = torch.zeros((), dtype=torch.float32, device=device)

        model.eval()
        rollout_start = time.perf_counter()

        for t in range(t_steps):
            obs_buf[t].copy_(obs)

            with torch.no_grad():
                actions, logprobs, values = model.act(obs)

            actions_buf[t].copy_(actions)
            logprobs_buf[t].copy_(logprobs)
            values_buf[t].copy_(values)

            obs, rewards, dones, episode = sim.step(actions)
            rewards_buf[t].copy_(rewards)
            dones_buf[t].copy_(dones.float())

            done_f = episode.done.float()
            completed += done_f.sum()
            crashes += episode.crashed.float().sum()
            car_crashes += episode.hit_car.float().sum()
            wall_crashes += episode.hit_wall.float().sum()
            completed_distance += episode.distance.sum()
            completed_speed += episode.final_speed.sum()

        with torch.no_grad():
            _, last_value = model(obs)

        # Generalized advantage estimation entirely on-device.
        last_gae = torch.zeros(n, dtype=torch.float32, device=device)
        for t in range(t_steps - 1, -1, -1):
            if t == t_steps - 1:
                next_value = last_value
            else:
                next_value = values_buf[t + 1]

            next_nonterminal = 1.0 - dones_buf[t]
            delta = (
                rewards_buf[t]
                + args.gamma * next_value * next_nonterminal
                - values_buf[t]
            )
            last_gae = (
                delta
                + args.gamma
                * args.gae_lambda
                * next_nonterminal
                * last_gae
            )
            advantages_buf[t] = last_gae

        returns_buf = advantages_buf + values_buf

        flat_obs = obs_buf.reshape(-1, OBS_SIZE)
        flat_actions = actions_buf.reshape(-1)
        flat_logprobs = logprobs_buf.reshape(-1)
        flat_advantages = advantages_buf.reshape(-1)
        flat_returns = returns_buf.reshape(-1)

        model.train()
        stats = ppo_update(
            model,
            optimizer,
            flat_obs,
            flat_actions,
            flat_logprobs,
            flat_advantages,
            flat_returns,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
        )

        total_steps += rollout_batch

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - wall_start
        rollout_elapsed = time.perf_counter() - rollout_start
        fps = total_steps / max(elapsed, 1e-9)

        completed_n = float(completed.item())
        crash_n = float(crashes.item())
        mean_ep_distance = (
            float(completed_distance.item()) / completed_n
            if completed_n > 0
            else 0.0
        )
        mean_ep_speed = (
            float(completed_speed.item()) / completed_n
            if completed_n > 0
            else 0.0
        )
        crash_rate = crash_n / completed_n if completed_n > 0 else 0.0
        car_crash_rate = (
            float(car_crashes.item()) / completed_n if completed_n > 0 else 0.0
        )
        wall_crash_rate = (
            float(wall_crashes.item()) / completed_n if completed_n > 0 else 0.0
        )

        print(
            f"[{iteration:04d}] steps={total_steps:,} fps={fps:,.0f} "
            f"rollout+update={rollout_elapsed:.2f}s "
            f"episodes={int(completed_n):,} crash={crash_rate:.1%} "
            f"(car={car_crash_rate:.1%},wall={wall_crash_rate:.1%}) "
            f"ep_dist={mean_ep_distance:,.0f} ep_v={mean_ep_speed:.1f} "
            f"entropy={stats.entropy:.3f} kl={stats.approx_kl:.4f} "
            f"clip={stats.clip_fraction:.3f} "
            f"pg={stats.policy_loss:.4f} vf={stats.value_loss:.2f}"
        )

        if total_steps >= next_save:
            save_checkpoint(
                args.output / f"teacher_{total_steps}.pt",
                model,
                optimizer,
                total_steps,
                best_mean_distance,
                best_selection_score,
                args,
            )
            save_checkpoint(
                args.output / "latest_teacher.pt",
                model,
                optimizer,
                total_steps,
                best_mean_distance,
                best_selection_score,
                args,
            )
            while next_save <= total_steps:
                next_save += args.save_every

        if total_steps >= next_eval:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            eval_start = time.perf_counter()
            eval_stats = evaluate_policy_diagnostics(
                model,
                episodes=args.eval_episodes,
                seed_start=args.eval_seed_start,
                crash_penalty=args.crash_penalty,
                reward_scale=args.reward_scale,
                action_seed_start=args.eval_action_seed_start,
                gpu_seed=args.eval_gpu_seed,
                original_workers=args.eval_workers,
            )

            eval_elapsed = time.perf_counter() - eval_start

            def _print_eval(label: str, stats: dict) -> None:
                print(
                    f"{label}: "
                    f"mean={stats['mean_distance']:,.1f} "
                    f"median={stats['median_distance']:,.1f} "
                    f"p10={stats['p10_distance']:,.1f} "
                    f"crash={stats['crash_rate']:.1%}"
                )

            print(f"\nPOLICY DIAGNOSTICS @ {total_steps:,} steps")
            _print_eval(
                "  ORIGINAL deterministic",
                eval_stats["original_deterministic"],
            )
            _print_eval(
                "  ORIGINAL stochastic   ",
                eval_stats["original_stochastic"],
            )
            _print_eval(
                "  ORIGINAL scheduler    ",
                eval_stats["original_scheduler"],
            )
            _print_eval(
                "  GPU      deterministic",
                eval_stats["gpu_deterministic"],
            )
            _print_eval(
                "  GPU      stochastic   ",
                eval_stats["gpu_stochastic"],
            )
            _print_eval(
                "  GPU      scheduler    ",
                eval_stats["gpu_scheduler"],
            )
            cpu_workers = eval_stats["original_deterministic"].get("workers", 1)
            print(
                f"  diagnostic evaluation time: {eval_elapsed:.1f}s "
                f"(original CPU workers={cpu_workers})"
            )

            (args.output / "latest_eval.json").write_text(
                json.dumps(eval_stats, indent=2), encoding="utf-8"
            )

            # Select the teacher on the exact original simulator using the
            # deployment decoder we actually intend to distill. Reliability is
            # gated first: any checkpoint at/below selection_max_crash outranks
            # one above the gate. Above the gate, crash rate is converted into a
            # strong distance-equivalent penalty.
            key = f"original_{args.selection_mode}"
            canonical = eval_stats[key]
            mean_distance = float(canonical["mean_distance"])
            crash_rate_eval = float(canonical["crash_rate"])
            safe = crash_rate_eval <= float(args.selection_max_crash)
            selection_score = (
                1_000_000.0 + mean_distance
                if safe
                else mean_distance
                - float(args.selection_crash_weight) * crash_rate_eval
            )

            if selection_score > best_selection_score:
                best_selection_score = selection_score
                best_mean_distance = mean_distance
                save_checkpoint(
                    args.output / "best_teacher.pt",
                    model,
                    optimizer,
                    total_steps,
                    best_mean_distance,
                    best_selection_score,
                    args,
                )
                (args.output / "best_eval.json").write_text(
                    json.dumps(eval_stats, indent=2), encoding="utf-8"
                )
                print(
                    f"New best ORIGINAL {args.selection_mode}: "
                    f"mean={mean_distance:,.1f} crash={crash_rate_eval:.1%} "
                    f"selection_score={selection_score:,.1f}"
                )

            while next_eval <= total_steps:
                next_eval += args.eval_every

    save_checkpoint(
        args.output / "final_teacher.pt",
        model,
        optimizer,
        total_steps,
        best_mean_distance,
        best_selection_score,
        args,
    )

    print(
        f"Finished {total_steps:,} steps in "
        f"{time.perf_counter() - wall_start:.1f}s"
    )


if __name__ == "__main__":
    main()
