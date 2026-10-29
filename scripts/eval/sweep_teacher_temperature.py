"""Sweep stochastic policy temperatures on the original simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.gpu_eval import evaluate_original_temperature_sweep
from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, OBS_SIZE


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2],
    )
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--seed-start", type=int, default=10_000)
    p.add_argument("--action-seed-start", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    model.eval()

    config = payload.get("config", {})
    results = evaluate_original_temperature_sweep(
        model,
        temperatures=args.temperatures,
        episodes=args.episodes,
        seed_start=args.seed_start,
        crash_penalty=float(config.get("crash_penalty", 250.0)),
        reward_scale=float(config.get("reward_scale", 1000.0)),
        action_seed_start=args.action_seed_start,
        workers=args.workers,
        include_deterministic=True,
    )

    print("\nORIGINAL SIM TEMPERATURE SWEEP")
    print(
        f"{'mode':>18} {'mean':>12} {'median':>12} "
        f"{'p10':>12} {'crash':>9} {'speed':>10}"
    )
    for name, stats in results.items():
        print(
            f"{name:>18} "
            f"{stats['mean_distance']:>12,.1f} "
            f"{stats['median_distance']:>12,.1f} "
            f"{stats['p10_distance']:>12,.1f} "
            f"{stats['crash_rate']:>8.1%} "
            f"{stats['mean_final_speed']:>10.1f}"
        )

    compact = {
        name: {k: v for k, v in stats.items() if k != "rows"}
        for name, stats in results.items()
    }
    print("\nJSON")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
