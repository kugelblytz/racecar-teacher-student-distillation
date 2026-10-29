"""CUDA-native PPO directly from legal challenge sensor observations (MLP)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from training.direct_eval import evaluate_direct_original
from training.gpu_ppo import ActorCritic, ppo_update
from training.gpu_sim import ACTIONS, LEGAL_OBS_SIZE, GpuRaceCarBatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000_000)
    p.add_argument("--envs", type=int, default=32_768)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help=(
            "Hold each selected legal primitive action for this many simulator "
            "ticks before the next policy decision."
        ),
    )
    p.add_argument("--minibatch-size", type=int, default=65_536)
    p.add_argument("--epochs", type=int, default=4)

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.9995)
    p.add_argument("--gae-lambda", type=float, default=0.97)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03)

    p.add_argument(
        "--crash-penalty",
        type=float,
        default=250.0,
        help="Penalty for crashing into another car.",
    )
    p.add_argument(
        "--wall-crash-penalty",
        type=float,
        default=350.0,
        help="Penalty for crashing into a wall (applied instead of car penalty).",
    )
    p.add_argument("--reward-scale", type=float, default=1000.0)
    p.add_argument(
        "--training-opponents",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="Fixed opponent count for training when curriculum is disabled.",
    )
    p.add_argument(
        "--curriculum-stage-steps",
        type=int,
        default=0,
        help=(
            "Training-only traffic curriculum. 0 disables it. When positive, "
            "train with 1 opponent initially, then add one opponent after each "
            "stage interval until reaching the competition's 4 opponents."
        ),
    )
    p.add_argument(
        "--safe-init",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--fused-adam",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--eval-every", type=int, default=25_000_000)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--eval-seed-start", type=int, default=40_000)
    p.add_argument("--eval-action-seed-start", type=int, default=4_000_000)
    p.add_argument("--eval-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=25_000_000)

    p.add_argument("--output", type=Path, default=Path("models/direct_mlp"))
    p.add_argument("--resume", type=Path, default=None)
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
    optimizer,
    total_steps: int,
    best_det: float,
    best_stoch: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "mlp",
            "obs_kind": "legal",
            "model_state": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "timesteps": int(total_steps),
            "best_deterministic_mean": float(best_det),
            "best_stochastic_mean": float(best_stoch),
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.action_repeat <= 0:
        raise ValueError("--action-repeat must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    args.output.mkdir(parents=True, exist_ok=True)

    initial_opponents = (
        1 if args.curriculum_stage_steps > 0 else int(args.training_opponents)
    )
    sim = GpuRaceCarBatch(
        args.envs,
        device=device,
        seed=args.seed,
        crash_penalty=args.crash_penalty,
        wall_crash_penalty=args.wall_crash_penalty,
        reward_scale=args.reward_scale,
        opponent_count=initial_opponents,
    )
    model = ActorCritic(
        LEGAL_OBS_SIZE,
        len(ACTIONS),
        safe_init=args.safe_init,
    ).to(device)
    optimizer = make_optimizer(model, args)

    total_steps = 0
    best_det = float("-inf")
    best_stoch = float("-inf")
    if args.resume is not None:
        payload = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        total_steps = int(payload.get("timesteps", 0))
        best_det = float(payload.get("best_deterministic_mean", float("-inf")))
        best_stoch = float(payload.get("best_stochastic_mean", float("-inf")))
        print(f"Resumed {args.resume} at {total_steps:,} steps")

    n = args.envs
    T = args.rollout_steps
    rollout_batch = n * T
    gamma_macro = args.gamma ** int(args.action_repeat)
    gae_lambda_macro = args.gae_lambda ** int(args.action_repeat)

    obs_buf = torch.empty((T, n, LEGAL_OBS_SIZE), dtype=torch.float32, device=device)
    actions_buf = torch.empty((T, n), dtype=torch.int64, device=device)
    logprobs_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    rewards_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    dones_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    values_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    advantages_buf = torch.empty_like(rewards_buf)

    obs = sim.legal_observation()

    def next_boundary(interval: int) -> int:
        if interval <= 0:
            return 2**63 - 1
        return ((total_steps // interval) + 1) * interval

    next_eval = next_boundary(args.eval_every)
    next_save = next_boundary(args.save_every)

    print(
        f"Direct MLP PPO: device={device} envs={n:,} rollout={T} "
        f"batch={rollout_batch:,} legal_obs={LEGAL_OBS_SIZE} "
        f"action_repeat={args.action_repeat}"
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    wall_start = time.perf_counter()
    iteration = 0
    current_opponents = initial_opponents

    def desired_opponent_count() -> int:
        if args.curriculum_stage_steps <= 0:
            return int(args.training_opponents)
        return min(
            4,
            1 + total_steps // int(args.curriculum_stage_steps),
        )

    while total_steps < args.timesteps:
        desired = desired_opponent_count()
        if desired != current_opponents:
            sim.set_opponent_count(desired)
            current_opponents = desired
            print(
                f"  CURRICULUM: {current_opponents} opponent(s) "
                f"at {total_steps:,} steps"
            )
        iteration += 1
        completed = torch.zeros((), dtype=torch.float32, device=device)
        crashes = torch.zeros((), dtype=torch.float32, device=device)
        car_crashes = torch.zeros((), dtype=torch.float32, device=device)
        wall_crashes = torch.zeros((), dtype=torch.float32, device=device)
        action_counts = torch.zeros(
            len(ACTIONS), dtype=torch.float32, device=device
        )
        completed_distance = torch.zeros((), dtype=torch.float64, device=device)

        rollout_start = time.perf_counter()
        model.eval()

        for t in range(T):
            obs_buf[t].copy_(obs)
            with torch.no_grad():
                actions, logprobs, values = model.act(obs)

            actions_buf[t].copy_(actions)
            action_counts += torch.bincount(
                actions, minlength=len(ACTIONS)
            ).to(torch.float32)
            logprobs_buf[t].copy_(logprobs)
            values_buf[t].copy_(values)

            if args.action_repeat == 1:
                obs, rewards, dones, episode = sim.step(
                    actions, observation_kind="legal"
                )
            else:
                obs, rewards, dones, episode = sim.macro_step(
                    actions,
                    repeat=args.action_repeat,
                    discount=args.gamma,
                    observation_kind="legal",
                )

            rewards_buf[t].copy_(rewards)
            dones_buf[t].copy_(dones.float())
            completed += episode.done.float().sum()
            crashes += episode.crashed.float().sum()
            car_crashes += episode.hit_car.float().sum()
            wall_crashes += episode.hit_wall.float().sum()
            completed_distance += episode.distance.sum()

        with torch.no_grad():
            _, last_value = model(obs)

        last_gae = torch.zeros(n, dtype=torch.float32, device=device)
        for t in range(T - 1, -1, -1):
            next_value = last_value if t == T - 1 else values_buf[t + 1]
            next_nonterminal = 1.0 - dones_buf[t]
            delta = (
                rewards_buf[t]
                + gamma_macro * next_value * next_nonterminal
                - values_buf[t]
            )
            last_gae = (
                delta
                + gamma_macro * gae_lambda_macro * next_nonterminal * last_gae
            )
            advantages_buf[t] = last_gae

        returns_buf = advantages_buf + values_buf
        model.train()
        stats = ppo_update(
            model,
            optimizer,
            obs_buf.reshape(-1, LEGAL_OBS_SIZE),
            actions_buf.reshape(-1),
            logprobs_buf.reshape(-1),
            advantages_buf.reshape(-1),
            returns_buf.reshape(-1),
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

        rollout_elapsed = time.perf_counter() - rollout_start
        completed_n = float(completed.item())
        crash_n = float(crashes.item())
        ep_dist = (
            float(completed_distance.item()) / completed_n
            if completed_n > 0
            else 0.0
        )
        crash_rate = crash_n / completed_n if completed_n > 0 else 0.0
        car_crash_rate = (
            float(car_crashes.item()) / completed_n
            if completed_n > 0 else 0.0
        )
        wall_crash_rate = (
            float(wall_crashes.item()) / completed_n
            if completed_n > 0 else 0.0
        )
        action_frac = (
            action_counts / max(float(action_counts.sum().item()), 1.0)
        ).detach().cpu().tolist()
        action_text = "/".join(f"{100.0 * x:.0f}" for x in action_frac)

        print(
            f"[{iteration:04d}] steps={total_steps:,} "
            f"traffic={current_opponents}/4 repeat={args.action_repeat} "
            f"train_fps={rollout_batch / max(rollout_elapsed, 1e-9):,.0f} "
            f"rollout+update={rollout_elapsed:.2f}s "
            f"episodes={int(completed_n):,} crash={crash_rate:.1%} "
            f"(car={car_crash_rate:.1%},wall={wall_crash_rate:.1%}) "
            f"ep_dist={ep_dist:,.0f} "
            f"act[N/A/D/L/R]={action_text} "
            f"entropy={stats.entropy:.3f} "
            f"kl={stats.approx_kl:.4f} clip={stats.clip_fraction:.3f}"
        )

        if total_steps >= next_save:
            save_checkpoint(
                args.output / f"direct_mlp_{total_steps}.pt",
                model,
                optimizer,
                total_steps,
                best_det,
                best_stoch,
                args,
            )
            save_checkpoint(
                args.output / "latest.pt",
                model,
                optimizer,
                total_steps,
                best_det,
                best_stoch,
                args,
            )
            while next_save <= total_steps:
                next_save += args.save_every

        if total_steps >= next_eval:
            eval_start = time.perf_counter()
            eval_stats = evaluate_direct_original(
                model,
                arch="mlp",
                episodes=args.eval_episodes,
                seed_start=args.eval_seed_start,
                crash_penalty=args.crash_penalty,
                reward_scale=args.reward_scale,
                action_seed_start=args.eval_action_seed_start,
                workers=args.eval_workers,
                action_repeat=args.action_repeat,
            )
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
                    model,
                    optimizer,
                    total_steps,
                    best_det,
                    best_stoch,
                    args,
                )
            if stoch_mean > best_stoch:
                best_stoch = stoch_mean
                save_checkpoint(
                    args.output / "best_stochastic.pt",
                    model,
                    optimizer,
                    total_steps,
                    best_det,
                    best_stoch,
                    args,
                )

            while next_eval <= total_steps:
                next_eval += args.eval_every

    save_checkpoint(
        args.output / "final.pt",
        model,
        optimizer,
        total_steps,
        best_det,
        best_stoch,
        args,
    )
    print(
        f"Finished {total_steps:,} steps in "
        f"{time.perf_counter() - wall_start:.1f}s"
    )


if __name__ == "__main__":
    main()
