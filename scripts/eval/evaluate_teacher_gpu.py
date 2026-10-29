"""Evaluate a CUDA-native teacher checkpoint with deployment diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from training.gpu_eval import evaluate_policy_diagnostics
from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, OBS_SIZE


def _write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "distance",
                "ticks",
                "crashed",
                "final_speed",
                "speed_cap",
                "speed_cap_overrides",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=20_000)
    p.add_argument("--action-seed-start", type=int, default=1_000_000)
    p.add_argument("--gpu-seed", type=int, default=3_000_000)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Original-simulator CPU workers; 0=auto (up to 24), 1=serial.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Backward-compatible CSV for ORIGINAL deterministic rows.",
    )
    p.add_argument(
        "--csv-prefix",
        type=Path,
        default=None,
        help="Optional prefix; writes one CSV per simulator/policy mode.",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    config = payload.get("config", {})
    stats = evaluate_policy_diagnostics(
        model,
        episodes=args.episodes,
        seed_start=args.seed_start,
        crash_penalty=float(config.get("crash_penalty", 250.0)),
        reward_scale=float(config.get("reward_scale", 1000.0)),
        action_seed_start=args.action_seed_start,
        gpu_seed=args.gpu_seed,
        original_workers=args.workers,
    )

    compact = {
        name: {k: v for k, v in result.items() if k != "rows"}
        for name, result in stats.items()
    }
    print(json.dumps(compact, indent=2))

    if args.csv is not None:
        _write_rows(args.csv, stats["original_deterministic"]["rows"])
        print(f"Wrote {args.csv}")

    if args.csv_prefix is not None:
        prefix = args.csv_prefix
        for name, result in stats.items():
            path = prefix.parent / f"{prefix.name}_{name}.csv"
            _write_rows(path, result["rows"])
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
