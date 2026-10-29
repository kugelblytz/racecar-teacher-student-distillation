"""Sweep inference-time max-speed caps for a privileged teacher."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from training.gpu_eval import evaluate_original_speed_cap_sweep
from training.gpu_ppo import ActorCritic
from training.gpu_sim import ACTIONS, OBS_SIZE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a teacher under inference-time speed caps on identical "
            "exact-original simulator seeds."
        )
    )
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--caps",
        type=float,
        nargs="+",
        default=[40, 50, 60, 70, 80, 100],
        help="Maximum ego vx values to test.",
    )
    p.add_argument("--episodes", type=int, default=1024)
    p.add_argument("--seed-start", type=int, default=90_000)
    p.add_argument("--action-seed-start", type=int, default=9_000_000)
    p.add_argument(
        "--mode",
        choices=["scheduler", "deterministic", "stochastic"],
        default="scheduler",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--no-uncapped",
        action="store_true",
        help="Do not include the normal uncapped policy.",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path for full per-seed results.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for one-row-per-cap summary.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ActorCritic(OBS_SIZE, len(ACTIONS), safe_init=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    config = payload.get("config", {})
    results = evaluate_original_speed_cap_sweep(
        model,
        speed_caps=args.caps,
        episodes=args.episodes,
        seed_start=args.seed_start,
        crash_penalty=float(config.get("crash_penalty", 250.0)),
        reward_scale=float(config.get("reward_scale", 1000.0)),
        action_seed_start=args.action_seed_start,
        workers=args.workers,
        policy_mode=args.mode,
        temperature=args.temperature,
        include_uncapped=not args.no_uncapped,
    )

    print(
        "\nExact-original speed/safety frontier "
        f"({args.mode}, T={args.temperature:g}, n={args.episodes})"
    )
    print(
        f"{'cap':>9} {'mean':>10} {'median':>10} {'p10':>10} "
        f"{'crash':>8} {'<25k':>8} {'avg_v':>8} {'veto%':>8}"
    )
    print("-" * 83)

    summary_rows = []
    for label, result in results.items():
        cap = result.get("speed_cap")
        cap_text = "uncapped" if cap is None else f"{float(cap):g}"
        row = {
            "label": label,
            "speed_cap": cap,
            "episodes": result["episodes"],
            "mean_distance": result["mean_distance"],
            "median_distance": result["median_distance"],
            "p10_distance": result["p10_distance"],
            "crash_rate": result["crash_rate"],
            "below_10k_rate": result["below_10k_rate"],
            "below_25k_rate": result["below_25k_rate"],
            "below_50k_rate": result["below_50k_rate"],
            "below_100k_rate": result["below_100k_rate"],
            "mean_episode_speed": result["mean_episode_speed"],
            "mean_final_speed": result["mean_final_speed"],
            "speed_cap_overrides": result["speed_cap_overrides"],
            "speed_cap_override_tick_rate": result[
                "speed_cap_override_tick_rate"
            ],
        }
        summary_rows.append(row)

        print(
            f"{cap_text:>9} "
            f"{result['mean_distance']:10,.0f} "
            f"{result['median_distance']:10,.0f} "
            f"{result['p10_distance']:10,.0f} "
            f"{result['crash_rate']:8.1%} "
            f"{result['below_25k_rate']:8.1%} "
            f"{result['mean_episode_speed']:8.1f} "
            f"{result['speed_cap_override_tick_rate']:8.1%}"
        )

    safe = [
        row for row in summary_rows if float(row["crash_rate"]) <= 0.05
    ]
    if safe:
        best = max(safe, key=lambda row: float(row["mean_distance"]))
        print(
            "\nBest <=5% crash candidate: "
            f"cap={best['speed_cap']} "
            f"mean={best['mean_distance']:,.0f} "
            f"p10={best['p10_distance']:,.0f} "
            f"crash={best['crash_rate']:.1%}"
        )
    else:
        best = min(
            summary_rows,
            key=lambda row: (
                float(row["crash_rate"]),
                -float(row["mean_distance"]),
            ),
        )
        print(
            "\nNo cap reached <=5% crash. Lowest-crash candidate: "
            f"cap={best['speed_cap']} "
            f"mean={best['mean_distance']:,.0f} "
            f"p10={best['p10_distance']:,.0f} "
            f"crash={best['crash_rate']:.1%}"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
