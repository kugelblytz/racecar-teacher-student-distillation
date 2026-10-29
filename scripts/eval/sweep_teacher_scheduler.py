"""Compare deterministic argmax, stochastic PPO, and probability scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.gpu_eval import evaluate_original_scheduler_sweep
from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, OBS_SIZE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.7, 0.85, 1.0, 1.15, 1.3],
    )
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--seed-start", type=int, default=30_000)
    p.add_argument("--action-seed-start", type=int, default=3_000_000)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    config = payload.get("config", {})
    results = evaluate_original_scheduler_sweep(
        model,
        temperatures=args.temperatures,
        episodes=args.episodes,
        seed_start=args.seed_start,
        crash_penalty=float(config.get("crash_penalty", 250.0)),
        reward_scale=float(config.get("reward_scale", 1000.0)),
        action_seed_start=args.action_seed_start,
        workers=args.workers,
        include_stochastic_t1=True,
    )

    print("\nORIGINAL SIM SCHEDULER SWEEP")
    print(
        f"{'mode':>18} {'mean':>11} {'median':>11} {'p10':>11} "
        f"{'crash':>8} {'<10k':>8} {'<25k':>8} {'<50k':>8} {'speed':>8}"
    )
    for name, stats in results.items():
        print(
            f"{name:>18} "
            f"{stats['mean_distance']:>11,.0f} "
            f"{stats['median_distance']:>11,.0f} "
            f"{stats['p10_distance']:>11,.0f} "
            f"{stats['crash_rate']:>7.1%} "
            f"{stats['below_10k_rate']:>7.1%} "
            f"{stats['below_25k_rate']:>7.1%} "
            f"{stats['below_50k_rate']:>7.1%} "
            f"{stats['mean_final_speed']:>8.1f}"
        )

    compact = {
        name: {k: v for k, v in stats.items() if k != "rows"}
        for name, stats in results.items()
    }
    print("\nJSON")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
