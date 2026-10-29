"""PPO fine-tuning with an MLP-warm-started residual GRU.

The residual branch is initialized to zero, so before the first optimizer step
this policy is exactly the source MLP for every observation. Recurrent memory can
then learn temporal corrections while the proven MLP path moves at a much lower
learning rate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from training.direct_eval import evaluate_direct_original
from training.gpu_ppo import ActorCritic
from training.gpu_recurrent_ppo import (
    ResidualRecurrentActorCritic,
    ppo_update_recurrent,
)
from training.gpu_sim import ACTIONS, LEGAL_OBS_SIZE, GpuRaceCarBatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-from-mlp", type=Path)
    source.add_argument("--resume", type=Path)

    p.add_argument("--timesteps", type=int, default=200_000_000)
    p.add_argument("--envs", type=int, default=16_384)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--env-minibatch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2)

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--base-learning-rate",
        type=float,
        default=0.0,
        help="0 freezes the proven MLP path; unfreeze later with a tiny LR.",
    )
    p.add_argument("--recurrent-learning-rate", type=float, default=5e-5)
    p.add_argument("--gamma", type=float, default=0.9995)
    p.add_argument("--gae-lambda", type=float, default=0.97)
    p.add_argument("--clip-range", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.01)

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
        default=1,
        choices=[1, 2, 3, 4],
    )
    p.add_argument(
        "--action-repeat",
        type=int,
        default=1,
        help="Keep at 1 for the MLP-to-GRU experiment unless testing persistence.",
    )
    p.add_argument(
        "--fused-adam",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--eval-every", type=int, default=25_000_000)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--eval-seed-start", type=int, default=60_000)
    p.add_argument("--eval-action-seed-start", type=int, default=6_000_000)
    p.add_argument("--eval-workers", type=int, default=24)
    p.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--save-every", type=int, default=25_000_000)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/direct_residual_gru"),
    )
    return p.parse_args()


def make_optimizer(model: ResidualRecurrentActorCritic, args: argparse.Namespace):
    groups = [
        {
            "params": list(model.base_parameters()),
            "lr": args.base_learning_rate,
            "name": "base",
        },
        {
            "params": list(model.recurrent_parameters()),
            "lr": args.recurrent_learning_rate,
            "name": "recurrent",
        },
    ]
    kwargs = dict(eps=1e-5)
    if args.fused_adam and torch.device(args.device).type == "cuda":
        try:
            return torch.optim.Adam(groups, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.Adam(groups, **kwargs)


def reapply_lrs(optimizer, args: argparse.Namespace) -> None:
    for group in optimizer.param_groups:
        name = group.get("name")
        if name == "base":
            group["lr"] = args.base_learning_rate
        elif name == "recurrent":
            group["lr"] = args.recurrent_learning_rate


def save_checkpoint(
    path: Path,
    model: ResidualRecurrentActorCritic,
    optimizer,
    residual_steps: int,
    best_det: float,
    best_stoch: float,
    source_checkpoint: str,
    source_mlp_timesteps: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "residual_gru",
            "obs_kind": "legal",
            "hidden_dim": int(args.hidden_dim),
            "model_state": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "timesteps": int(residual_steps),
            "source_checkpoint": source_checkpoint,
            "source_mlp_timesteps": int(source_mlp_timesteps),
            "best_deterministic_mean": float(best_det),
            "best_stochastic_mean": float(best_stoch),
            "config": vars(args),
        },
        path,
    )


def evaluate_and_print(
    model: ResidualRecurrentActorCritic,
    args: argparse.Namespace,
    *,
    label: str,
):
    start = time.perf_counter()
    stats = evaluate_direct_original(
        model,
        arch="residual_gru",
        episodes=args.eval_episodes,
        seed_start=args.eval_seed_start,
        crash_penalty=args.crash_penalty,
        wall_crash_penalty=args.wall_crash_penalty,
        reward_scale=args.reward_scale,
        action_seed_start=args.eval_action_seed_start,
        workers=args.eval_workers,
        hidden_dim=args.hidden_dim,
        action_repeat=args.action_repeat,
    )
    print(f"\n{label}")
    for mode in ("deterministic", "stochastic"):
        s = stats[mode]
        print(
            f"  ORIGINAL {mode:13s}: mean={s['mean_distance']:,.1f} "
            f"median={s['median_distance']:,.1f} "
            f"p10={s['p10_distance']:,.1f} crash={s['crash_rate']:.1%} "
            f"<25k={s['below_25k_rate']:.1%}"
        )
    print(
        f"  eval_time={time.perf_counter() - start:.1f}s "
        f"workers={stats['deterministic']['workers']}"
    )
    return stats


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

    model = ResidualRecurrentActorCritic(
        LEGAL_OBS_SIZE,
        len(ACTIONS),
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = make_optimizer(model, args)

    residual_steps = 0
    best_det = float("-inf")
    best_stoch = float("-inf")
    source_checkpoint = ""
    source_mlp_timesteps = 0

    if args.init_from_mlp is not None:
        payload = torch.load(
            args.init_from_mlp,
            map_location=device,
            weights_only=False,
        )
        if str(payload.get("arch", "mlp")) != "mlp":
            raise ValueError("--init-from-mlp must point to an MLP checkpoint")
        if str(payload.get("obs_kind", "legal")) != "legal":
            raise ValueError("source MLP must use legal observations")

        source_config = payload.get("config", {})
        source_repeat = int(source_config.get("action_repeat", 1))
        if source_repeat != args.action_repeat:
            raise ValueError(
                f"source MLP action_repeat={source_repeat}, but requested "
                f"action_repeat={args.action_repeat}; warm start must preserve "
                "the source policy's action semantics"
            )

        model.load_from_mlp_state_dict(payload["model_state"])
        source_checkpoint = str(args.init_from_mlp)
        source_mlp_timesteps = int(payload.get("timesteps", 0))

        # Prove the residual model is exactly the source MLP before training.
        source_model = ActorCritic(
            LEGAL_OBS_SIZE,
            len(ACTIONS),
            safe_init=False,
        ).to(device)
        source_model.load_state_dict(payload["model_state"])
        source_model.eval()
        model.eval()

        probe = torch.randn(2048, LEGAL_OBS_SIZE, device=device)
        hidden_probe = torch.randn(
            2048, args.hidden_dim, device=device
        )
        with torch.no_grad():
            mlp_logits, mlp_values = source_model(probe)
            residual_logits, residual_values, _ = model.step(
                probe, hidden_probe
            )
        logits_diff = float(
            (mlp_logits - residual_logits).abs().max().item()
        )
        value_diff = float(
            (mlp_values - residual_values).abs().max().item()
        )
        if logits_diff != 0.0 or value_diff != 0.0:
            raise AssertionError(
                "Residual warm start is not exact: "
                f"logits_diff={logits_diff} value_diff={value_diff}"
            )
        del source_model, probe, hidden_probe

        print(
            f"Warm-started exactly from {args.init_from_mlp} "
            f"(MLP steps={source_mlp_timesteps:,}); "
            "max logits/value diff = 0"
        )
    else:
        payload = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        if str(payload.get("arch")) != "residual_gru":
            raise ValueError("--resume must point to a residual_gru checkpoint")
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
            reapply_lrs(optimizer, args)
        residual_steps = int(payload.get("timesteps", 0))
        best_det = float(payload.get("best_deterministic_mean", float("-inf")))
        best_stoch = float(payload.get("best_stochastic_mean", float("-inf")))
        source_checkpoint = str(payload.get("source_checkpoint", ""))
        source_mlp_timesteps = int(payload.get("source_mlp_timesteps", 0))
        print(
            f"Resumed residual GRU from {args.resume} "
            f"at {residual_steps:,} residual PPO steps"
        )

    sim = GpuRaceCarBatch(
        args.envs,
        device=device,
        seed=args.seed,
        crash_penalty=args.crash_penalty,
        wall_crash_penalty=args.wall_crash_penalty,
        reward_scale=args.reward_scale,
        opponent_count=args.training_opponents,
    )

    n = args.envs
    T = args.rollout_steps
    rollout_batch = n * T
    gamma_macro = args.gamma ** int(args.action_repeat)
    gae_lambda_macro = args.gae_lambda ** int(args.action_repeat)

    obs_buf = torch.empty(
        (T, n, LEGAL_OBS_SIZE), dtype=torch.float32, device=device
    )
    actions_buf = torch.empty((T, n), dtype=torch.int64, device=device)
    logprobs_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    rewards_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    dones_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    values_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    advantages_buf = torch.empty_like(rewards_buf)

    obs = sim.legal_observation()
    hidden = model.initial_state(n, device=device)

    def next_boundary(interval: int) -> int:
        if interval <= 0:
            return 2**63 - 1
        return ((residual_steps // interval) + 1) * interval

    next_eval = next_boundary(args.eval_every)
    next_save = next_boundary(args.save_every)

    print(
        f"Residual GRU PPO: device={device} envs={n:,} rollout={T} "
        f"batch={rollout_batch:,} legal_obs={LEGAL_OBS_SIZE} "
        f"hidden={args.hidden_dim} opponents={args.training_opponents}/4 "
        f"base_lr={args.base_learning_rate:g} "
        f"recurrent_lr={args.recurrent_learning_rate:g}"
    )
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    if args.eval_at_start and residual_steps == 0:
        initial_stats = evaluate_and_print(
            model, args, label="INITIAL WARM-START EVAL"
        )
        (args.output / "initial_eval.json").write_text(
            json.dumps(initial_stats, indent=2),
            encoding="utf-8",
        )
        best_det = float(initial_stats["deterministic"]["mean_distance"])
        best_stoch = float(initial_stats["stochastic"]["mean_distance"])
        save_checkpoint(
            args.output / "warm_start.pt",
            model,
            optimizer,
            residual_steps,
            best_det,
            best_stoch,
            source_checkpoint,
            source_mlp_timesteps,
            args,
        )

    wall_start = time.perf_counter()
    iteration = 0

    while residual_steps < args.timesteps:
        iteration += 1
        completed = torch.zeros((), dtype=torch.float32, device=device)
        crashes = torch.zeros((), dtype=torch.float32, device=device)
        car_crashes = torch.zeros((), dtype=torch.float32, device=device)
        wall_crashes = torch.zeros((), dtype=torch.float32, device=device)
        action_counts = torch.zeros(
            len(ACTIONS), dtype=torch.float32, device=device
        )
        completed_distance = torch.zeros(
            (), dtype=torch.float64, device=device
        )

        rollout_start = time.perf_counter()
        model.eval()
        initial_hidden = hidden.detach().clone()

        for t in range(T):
            obs_buf[t].copy_(obs)
            with torch.no_grad():
                actions, logprobs, values, next_hidden = model.act(
                    obs, hidden
                )

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
            hidden = next_hidden * (~dones).float().unsqueeze(1)

            completed += episode.done.float().sum()
            crashes += episode.crashed.float().sum()
            car_crashes += episode.hit_car.float().sum()
            wall_crashes += episode.hit_wall.float().sum()
            completed_distance += episode.distance.sum()

        with torch.no_grad():
            _, last_value, _ = model.step(obs, hidden)

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
                + gamma_macro
                * gae_lambda_macro
                * next_nonterminal
                * last_gae
            )
            advantages_buf[t] = last_gae

        returns_buf = advantages_buf + values_buf

        model.train()
        stats = ppo_update_recurrent(
            model,
            optimizer,
            obs_buf,
            actions_buf,
            logprobs_buf,
            advantages_buf,
            returns_buf,
            dones_buf,
            initial_hidden,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            epochs=args.epochs,
            env_minibatch_size=args.env_minibatch_size,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
        )
        hidden = hidden.detach()

        residual_steps += rollout_batch
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - rollout_start
        completed_n = float(completed.item())
        crash_n = float(crashes.item())
        ep_dist = (
            float(completed_distance.item()) / completed_n
            if completed_n > 0
            else 0.0
        )
        crash_rate = crash_n / completed_n if completed_n > 0 else 0.0
        car_rate = (
            float(car_crashes.item()) / completed_n
            if completed_n > 0
            else 0.0
        )
        wall_rate = (
            float(wall_crashes.item()) / completed_n
            if completed_n > 0
            else 0.0
        )
        action_frac = (
            action_counts / max(float(action_counts.sum().item()), 1.0)
        ).detach().cpu().tolist()
        action_text = "/".join(f"{100.0 * x:.0f}" for x in action_frac)

        with torch.no_grad():
            residual_norm = float(
                model.recurrent_projection.weight.norm().item()
            )

        print(
            f"[{iteration:04d}] residual_steps={residual_steps:,} "
            f"traffic={args.training_opponents}/4 "
            f"train_fps={rollout_batch / max(elapsed, 1e-9):,.0f} "
            f"rollout+update={elapsed:.2f}s "
            f"episodes={int(completed_n):,} crash={crash_rate:.1%} "
            f"(car={car_rate:.1%},wall={wall_rate:.1%}) "
            f"ep_dist={ep_dist:,.0f} "
            f"act[N/A/D/L/R]={action_text} "
            f"residual_norm={residual_norm:.3f} "
            f"entropy={stats.entropy:.3f} kl={stats.approx_kl:.4f} "
            f"clip={stats.clip_fraction:.3f}"
        )

        if residual_steps >= next_save:
            save_checkpoint(
                args.output / f"residual_gru_{residual_steps}.pt",
                model,
                optimizer,
                residual_steps,
                best_det,
                best_stoch,
                source_checkpoint,
                source_mlp_timesteps,
                args,
            )
            save_checkpoint(
                args.output / "latest.pt",
                model,
                optimizer,
                residual_steps,
                best_det,
                best_stoch,
                source_checkpoint,
                source_mlp_timesteps,
                args,
            )
            while next_save <= residual_steps:
                next_save += args.save_every

        if residual_steps >= next_eval:
            eval_stats = evaluate_and_print(
                model,
                args,
                label=f"EVAL @ {residual_steps:,} RESIDUAL STEPS",
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
                    residual_steps,
                    best_det,
                    best_stoch,
                    source_checkpoint,
                    source_mlp_timesteps,
                    args,
                )
            if stoch_mean > best_stoch:
                best_stoch = stoch_mean
                save_checkpoint(
                    args.output / "best_stochastic.pt",
                    model,
                    optimizer,
                    residual_steps,
                    best_det,
                    best_stoch,
                    source_checkpoint,
                    source_mlp_timesteps,
                    args,
                )

            while next_eval <= residual_steps:
                next_eval += args.eval_every

    save_checkpoint(
        args.output / "final.pt",
        model,
        optimizer,
        residual_steps,
        best_det,
        best_stoch,
        source_checkpoint,
        source_mlp_timesteps,
        args,
    )
    print(
        f"Finished {residual_steps:,} residual PPO steps in "
        f"{time.perf_counter() - wall_start:.1f}s"
    )


if __name__ == "__main__":
    main()
