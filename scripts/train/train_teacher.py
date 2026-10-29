"""Train the privileged PPO teacher.

Example:
    python train_teacher.py --timesteps 5000000 --envs 8
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from training.eval_utils import evaluate_teacher
from training.fast_teacher_env import make_fast_env
from training.teacher_env import make_env


class FixedSeedEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_every: int,
        eval_episodes: int,
        eval_seed_start: int,
        output_dir: Path,
        crash_penalty: float,
        reward_scale: float,
        sim: str,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.eval_seed_start = int(eval_seed_start)
        self.output_dir = output_dir
        self.crash_penalty = float(crash_penalty)
        self.reward_scale = float(reward_scale)
        self.sim = sim
        self.next_eval = self.eval_every
        self.best_mean_distance = float("-inf")

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval:
            return True

        while self.next_eval <= self.num_timesteps:
            self.next_eval += self.eval_every

        seeds = range(
            self.eval_seed_start,
            self.eval_seed_start + self.eval_episodes,
        )
        stats = evaluate_teacher(
            self.model,
            seeds,
            crash_penalty=self.crash_penalty,
            reward_scale=self.reward_scale,
            sim=self.sim,
        )

        mean_distance = float(stats["mean_distance"])
        crash_rate = float(stats["crash_rate"])

        self.logger.record("eval/mean_distance", mean_distance)
        self.logger.record("eval/median_distance", float(stats["median_distance"]))
        self.logger.record("eval/p10_distance", float(stats["p10_distance"]))
        self.logger.record("eval/crash_rate", crash_rate)
        self.logger.record("eval/mean_final_speed", float(stats["mean_final_speed"]))

        if self.verbose:
            print(
                "\n"
                f"[teacher eval @ {self.num_timesteps:,} steps] "
                f"mean={mean_distance:,.1f} "
                f"median={float(stats['median_distance']):,.1f} "
                f"p10={float(stats['p10_distance']):,.1f} "
                f"crash_rate={crash_rate:.1%}"
            )

        latest_path = self.output_dir / "latest_eval.json"
        latest_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        if mean_distance > self.best_mean_distance:
            self.best_mean_distance = mean_distance
            self.model.save(str(self.output_dir / "best_teacher"))
            (self.output_dir / "best_eval.json").write_text(
                json.dumps(stats, indent=2),
                encoding="utf-8",
            )
            if self.verbose:
                print(f"New best teacher: {mean_distance:,.1f} mean distance")

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--crash-penalty", type=float, default=25.0)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--sim",
        choices=("fast", "original"),
        default="original",
        help="Training simulator. original is currently faster; fast is the numeric parity/reference implementation.",
    )
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-seed-start", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=Path("models/teacher"))
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional .zip PPO checkpoint to continue training.",
    )
    return parser.parse_args()


def build_vec_env(args: argparse.Namespace):
    factory_builder = make_fast_env if args.sim == "fast" else make_env
    factories = [
        factory_builder(
            rank=i,
            base_seed=args.seed,
            crash_penalty=args.crash_penalty,
            reward_scale=args.reward_scale,
        )
        for i in range(args.envs)
    ]

    if args.envs == 1:
        return DummyVecEnv(factories)

    # spawn is the safest choice on Windows and keeps each worker's global
    # simulator STATE isolated in a separate process.
    return SubprocVecEnv(factories, start_method="spawn")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(parents=True, exist_ok=True)

    env = build_vec_env(args)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(
            pi=[256, 256, 128],
            vf=[256, 256, 128],
        ),
    )

    if args.resume is not None:
        model = PPO.load(
            str(args.resume),
            env=env,
            device=args.device,
        )
        reset_num_timesteps = False
        print(f"Resuming from {args.resume}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=1024,
            n_epochs=10,
            gamma=0.9995,
            gae_lambda=0.97,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(args.output / "tensorboard"),
            verbose=1,
            seed=args.seed,
            device=args.device,
        )
        reset_num_timesteps = True

    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // max(1, args.envs)),
        save_path=str(args.output / "checkpoints"),
        name_prefix="teacher",
    )

    evaluator = FixedSeedEvalCallback(
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        eval_seed_start=args.eval_seed_start,
        output_dir=args.output,
        crash_penalty=args.crash_penalty,
        reward_scale=args.reward_scale,
        sim=args.sim,
    )

    callbacks = CallbackList([checkpoint, evaluator])

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(str(args.output / "final_teacher"))
    finally:
        env.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
