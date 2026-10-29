"""Evaluate a trained privileged PPO teacher on fixed seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO

from training.eval_utils import evaluate_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--crash-penalty", type=float, default=25.0)
    parser.add_argument("--reward-scale", type=float, default=1000.0)
    parser.add_argument("--sim", choices=("fast", "original"), default="fast")
    parser.add_argument("--csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(str(args.model), device="auto")

    seeds = range(args.seed_start, args.seed_start + args.episodes)
    stats = evaluate_teacher(
        model,
        seeds,
        crash_penalty=args.crash_penalty,
        reward_scale=args.reward_scale,
        sim=args.sim,
    )

    print(json.dumps({k: v for k, v in stats.items() if k != "rows"}, indent=2))

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["seed", "distance", "ticks", "crashed", "final_speed"],
            )
            writer.writeheader()
            writer.writerows(stats["rows"])
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
