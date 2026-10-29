"""Evaluate a direct legal-observation MLP or GRU checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.direct_eval import evaluate_direct_original
from training.gpu_ppo import ActorCritic
from training.gpu_recurrent_ppo import (
    RecurrentActorCritic,
    ResidualRecurrentActorCritic,
    recurrent_obs_dim_from_state_dict,
)
from training.gpu_sim import ACTIONS, LEGAL_OBS_SIZE, LEGAL_OBS_SIZE_TRACKED_Y


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=256)
    p.add_argument("--seed-start", type=int, default=50_000)
    p.add_argument("--action-seed-start", type=int, default=5_000_000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--include-rows",
        action="store_true",
        help="Include per-seed collision and action diagnostics in JSON output.",
    )
    p.add_argument(
        "--modes",
        choices=["deterministic", "stochastic", "scheduler", "lane_options"],
        nargs="+",
        default=["deterministic", "stochastic", "scheduler"],
    )
    p.add_argument("--lane-option-threshold", type=float, default=None)
    args = p.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = str(payload.get("arch", "mlp"))
    config = payload.get("config", {})
    hidden_dim = int(payload.get("hidden_dim", config.get("hidden_dim", 128)))
    state = payload["model_state"]
    lane_option_state = payload.get("lane_option_state")

    if arch in ("gru", "residual_gru"):
        obs_dim = recurrent_obs_dim_from_state_dict(state)
    elif arch == "mlp":
        obs_dim = int(state["backbone.0.weight"].shape[1])
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {arch}")

    tracked_y = bool(
        config.get(
            "tracked_y",
            payload.get("obs_version", 1) == 2
            or obs_dim == LEGAL_OBS_SIZE_TRACKED_Y,
        )
    )
    expected = LEGAL_OBS_SIZE_TRACKED_Y if tracked_y else LEGAL_OBS_SIZE
    if obs_dim != expected:
        raise ValueError(
            f"checkpoint obs_dim={obs_dim} conflicts with tracked_y={tracked_y}"
        )

    if arch == "mlp":
        model = ActorCritic(obs_dim, len(ACTIONS), safe_init=False)
    elif arch == "gru":
        model = RecurrentActorCritic(
            obs_dim,
            len(ACTIONS),
            hidden_dim=hidden_dim,
            safe_init=False,
        )
    elif arch == "residual_gru":
        model = ResidualRecurrentActorCritic(
            obs_dim,
            len(ACTIONS),
            hidden_dim=hidden_dim,
        )

    model.load_state_dict(state)
    model.eval()

    stats = evaluate_direct_original(
        model,
        arch=arch,
        episodes=args.episodes,
        seed_start=args.seed_start,
        crash_penalty=float(config.get("crash_penalty", 250.0)),
        wall_crash_penalty=float(
            config.get("wall_crash_penalty", config.get("crash_penalty", 250.0))
        ),
        reward_scale=float(config.get("reward_scale", 1000.0)),
        action_seed_start=args.action_seed_start,
        workers=args.workers,
        hidden_dim=hidden_dim,
        action_repeat=int(config.get("action_repeat", 1)),
        obs_dim=obs_dim,
        tracked_y=tracked_y,
        policy_modes=tuple(args.modes),
        lane_option_state=lane_option_state,
        lane_option_threshold=(
            float(payload.get("lane_option_threshold", 0.5))
            if args.lane_option_threshold is None
            else args.lane_option_threshold
        ),
    )

    report = (
        stats
        if args.include_rows
        else {
            mode: {k: v for k, v in result.items() if k != "rows"}
            for mode, result in stats.items()
        }
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
