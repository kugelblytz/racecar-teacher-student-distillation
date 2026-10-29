"""CUDA-native recurrent PPO directly from legal challenge observations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from training.direct_eval import evaluate_direct_original
from training.gpu_recurrent_ppo import (
    RecurrentActorCritic,
    expand_recurrent_obs_state_dict,
    recurrent_obs_dim_from_state_dict,
    ppo_update_recurrent,
)
from training.gpu_sim import (
    ACTIONS,
    LEGAL_OBS_SIZE,
    LEGAL_OBS_SIZE_TRACKED_Y,
    GpuRaceCarBatch,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000_000)
    p.add_argument("--envs", type=int, default=16_384)
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
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument(
        "--tracked-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use legal observation V2: append controller-integrated absolute "
            "lateral position. A resumed 41-D GRU is migrated exactly with a "
            "zero-weight new input column and a fresh optimizer."
        ),
    )
    p.add_argument("--env-minibatch-size", type=int, default=512)
    p.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="PPO passes over each recurrent rollout batch.",
    )

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.9995)
    p.add_argument("--gae-lambda", type=float, default=0.97)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient; nonzero discourages early action collapse.",
    )
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.03)

    p.add_argument(
        "--crash-penalty",
        type=float,
        default=50.0,
        help="Penalty for crashing into another car.",
    )
    p.add_argument(
        "--wall-crash-penalty",
        type=float,
        default=100.0,
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

    p.add_argument("--output", type=Path, default=Path("models/direct_gru"))
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def make_optimizer(model, args):
    kwargs = dict(lr=args.learning_rate, eps=1e-5)
    if args.fused_adam and torch.device(args.device).type == "cuda":
        try:
            return torch.optim.Adam(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.Adam(model.parameters(), **kwargs)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    total_steps: int,
    best_det: float,
    best_stoch: float,
    args,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arch": "gru",
            "obs_kind": "legal",
            "hidden_dim": int(args.hidden_dim),
            "obs_dim": int(
                LEGAL_OBS_SIZE_TRACKED_Y if args.tracked_y else LEGAL_OBS_SIZE
            ),
            "obs_version": 2 if args.tracked_y else 1,
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

    obs_dim = (
        LEGAL_OBS_SIZE_TRACKED_Y if args.tracked_y else LEGAL_OBS_SIZE
    )

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
    model = RecurrentActorCritic(
        obs_dim,
        len(ACTIONS),
        hidden_dim=args.hidden_dim,
        safe_init=args.safe_init,
    ).to(device)
    optimizer = make_optimizer(model, args)

    total_steps = 0
    best_det = float("-inf")
    best_stoch = float("-inf")
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        source_state = payload["model_state"]
        source_obs_dim = recurrent_obs_dim_from_state_dict(source_state)
        if source_obs_dim > obs_dim:
            raise ValueError(
                f"checkpoint uses {source_obs_dim}-D observations but this run "
                f"requests {obs_dim}-D; use --tracked-y for a V2 checkpoint"
            )

        migrated = source_obs_dim < obs_dim
        if migrated:
            source_state, old_dim = expand_recurrent_obs_state_dict(
                source_state, obs_dim
            )
            print(
                f"Migrating GRU observation {old_dim} -> {obs_dim}: "
                "old input weights copied exactly, new tracked-y column = 0"
            )

        model.load_state_dict(source_state)

        # Adam tensors for the old first-layer shape cannot be reused after
        # 41->42 migration. A fresh optimizer also gives the new feature a clean
        # learning signal while preserving network behavior exactly at step 0.
        if not migrated and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        elif migrated:
            print("Starting a fresh optimizer after observation migration")
        elif bool(payload.get("distilled", False)):
            print(
                "Starting a fresh PPO optimizer from distilled policy "
                f"({int(payload.get('distill_transitions', 0)):,} "
                "teacher-driven transitions)"
            )

        total_steps = int(payload.get("timesteps", 0))
        if migrated:
            # Start V2 checkpoint selection fresh; source V1 best metrics remain
            # in the source checkpoint and should not block saving V2 bests.
            best_det = float("-inf")
            best_stoch = float("-inf")
        else:
            best_det = float(
                payload.get("best_deterministic_mean", float("-inf"))
            )
            best_stoch = float(
                payload.get("best_stochastic_mean", float("-inf"))
            )
        print(f"Resumed {args.resume} at {total_steps:,} steps")

    n = args.envs
    T = args.rollout_steps
    rollout_batch = n * T
    gamma_macro = args.gamma ** int(args.action_repeat)
    gae_lambda_macro = args.gae_lambda ** int(args.action_repeat)

    obs_buf = torch.empty((T, n, obs_dim), dtype=torch.float32, device=device)
    actions_buf = torch.empty((T, n), dtype=torch.int64, device=device)
    logprobs_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    rewards_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    dones_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    values_buf = torch.empty((T, n), dtype=torch.float32, device=device)
    advantages_buf = torch.empty_like(rewards_buf)

    obs = sim.legal_observation(tracked_y=args.tracked_y)
    hidden = model.initial_state(n, device=device)

    def next_boundary(interval: int) -> int:
        if interval <= 0:
            return 2**63 - 1
        return ((total_steps // interval) + 1) * interval

    next_eval = next_boundary(args.eval_every)
    next_save = next_boundary(args.save_every)

    print(
        f"Direct GRU PPO: device={device} envs={n:,} rollout={T} "
        f"batch={rollout_batch:,} legal_obs={obs_dim} "
        f"tracked_y={args.tracked_y} "
        f"hidden={args.hidden_dim} env_mb={args.env_minibatch_size} "
        f"epochs={args.epochs} ent_coef={args.ent_coef:g} "
        f"car_penalty={args.crash_penalty:g} "
        f"wall_penalty={args.wall_crash_penalty:g}"
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
        initial_hidden = hidden.detach().clone()

        for t in range(T):
            obs_buf[t].copy_(obs)
            with torch.no_grad():
                actions, logprobs, values, next_hidden = model.act(obs, hidden)

            actions_buf[t].copy_(actions)
            action_counts += torch.bincount(
                actions, minlength=len(ACTIONS)
            ).to(torch.float32)
            logprobs_buf[t].copy_(logprobs)
            values_buf[t].copy_(values)

            if args.action_repeat == 1:
                obs, rewards, dones, episode = sim.step(
                    actions,
                    observation_kind="legal",
                    tracked_y=args.tracked_y,
                )
            else:
                obs, rewards, dones, episode = sim.macro_step(
                    actions,
                    repeat=args.action_repeat,
                    discount=args.gamma,
                    observation_kind="legal",
                    tracked_y=args.tracked_y,
                )

            rewards_buf[t].copy_(rewards)
            dones_buf[t].copy_(dones.float())

            # A completed environment has already auto-reset in the simulator,
            # so recurrent memory must reset before its new episode.
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
                + gamma_macro * gae_lambda_macro * next_nonterminal * last_gae
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

        # Hidden is behavior-policy state carried across rollout boundaries. It is
        # intentionally detached; PPO updates never backpropagate across rollouts.
        hidden = hidden.detach()

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
                args.output / f"direct_gru_{total_steps}.pt",
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
                arch="gru",
                episodes=args.eval_episodes,
                seed_start=args.eval_seed_start,
                crash_penalty=args.crash_penalty,
                wall_crash_penalty=args.wall_crash_penalty,
                reward_scale=args.reward_scale,
                action_seed_start=args.eval_action_seed_start,
                workers=args.eval_workers,
                hidden_dim=args.hidden_dim,
                action_repeat=args.action_repeat,
                obs_dim=obs_dim,
                tracked_y=args.tracked_y,
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
