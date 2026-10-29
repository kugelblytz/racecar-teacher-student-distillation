"""Assertion-based checks for exact MPC distillation label collection."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from training.direct_eval import _lane_option_primitive, _summarize
from training.gpu_distill import distill_update_recurrent
from training.gpu_sim import (
    ACTIONS,
    CAR_HEIGHT,
    LANE_CENTERS,
    LEGAL_OBS_SIZE,
    LEGAL_OBS_SIZE_TRACKED_Y,
)
from training.gpu_recurrent_ppo import RecurrentActorCritic
from training.teacher_mpc_distill import (
    ExactMpcLabelCollector,
    MpcLabelConfig,
    merge_collection_stats,
)
from scripts.train.train_distilled_gru_mpc import _score


CHECKPOINT = Path("models/gpu_teacher_65k_1b/best_teacher.pt")


def _state():
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    return payload["model_state"]


def _collect(seed: int, steps: int, config: MpcLabelConfig):
    collector = ExactMpcLabelCollector(
        _state(),
        worker_rank=0,
        worker_count=1,
        seed_start=seed,
        config=config,
    )
    try:
        return collector.collect(steps)
    finally:
        collector.close()


def verify_shapes_and_determinism() -> None:
    config = MpcLabelConfig()
    first = _collect(300_417, 160, config)
    second = _collect(300_417, 160, config)
    assert first["observations"].shape == (160, LEGAL_OBS_SIZE_TRACKED_Y)
    assert first["actions"].shape == (160,)
    assert first["probabilities"].shape == (160, len(ACTIONS))
    assert first["dones"].shape == (160,)
    assert first["triggers"].shape == (160,)
    assert first["overrides"].shape == (160,)
    assert first["lane_options"].shape == (160,)
    assert first["behavior_actions"].shape == (160,)
    assert first["student_behavior"].shape == (160,)
    assert first["observations"].dtype == np.float32
    assert first["actions"].dtype == np.int64
    assert first["probabilities"].dtype == np.float32
    assert np.allclose(first["probabilities"].sum(axis=1), 1.0, atol=1e-6)
    assert np.all((first["actions"] >= 0) & (first["actions"] < len(ACTIONS)))
    assert np.array_equal(first["actions"], second["actions"])
    assert np.array_equal(first["dones"], second["dones"])
    assert np.array_equal(first["triggers"], second["triggers"])
    assert np.array_equal(first["overrides"], second["overrides"])
    assert np.array_equal(first["lane_options"], second["lane_options"])
    assert np.array_equal(first["observations"], second["observations"])
    assert first["stats"] == second["stats"]
    print("PASS stable legal-label shapes and deterministic replay")


def verify_mpc_labels_and_diagnostics() -> None:
    # This held-out episode contains the reproduced long-range closing case and
    # exercises the one-accepted-strategic-intervention limit.
    result = _collect(300_417, 3600, MpcLabelConfig())
    stats = result["stats"]
    assert stats["ticks"] == 3600
    assert stats["triggers"] > 0
    assert stats["strategic_candidates"] > 0
    assert stats["strategic_accepted"] == 1
    assert stats["overrides"] > 0
    assert int(result["triggers"].sum()) == stats["triggers"]
    assert int(result["overrides"].sum()) == stats["overrides"]
    assert sum(stats["proposed_counts"]) == 3600
    assert sum(stats["label_counts"]) == 3600
    assert sum(stats["override_action_counts"]) == stats["overrides"]
    assert sum(stats["lane_option_counts"]) == 3600
    assert np.array_equal(
        np.bincount(result["lane_options"], minlength=3),
        np.asarray(stats["lane_option_counts"]),
    )
    assert sum(stats["executed_counts"]) == 3600
    assert sum(stats["trigger_reasons"].values()) == stats["triggers"]
    assert sum(stats["selected_plans"].values()) == stats["triggers"]

    # With the default blend, every actual MPC override is represented as a
    # one-hot soft target as well as a hard action label.
    one_hot_rows = np.isclose(result["probabilities"].max(axis=1), 1.0)
    assert int(one_hot_rows.sum()) >= stats["overrides"]
    print("PASS MPC overrides, strategic limit, and diagnostic accounting")


def verify_episode_boundaries_and_merge() -> None:
    first = _collect(700_000, 32, MpcLabelConfig())
    second = _collect(700_001, 32, MpcLabelConfig())
    merged = merge_collection_stats([first["stats"], second["stats"]])
    assert merged["ticks"] == 64
    assert sum(merged["proposed_counts"]) == 64
    assert sum(merged["executed_counts"]) == 64
    assert sum(merged["override_action_counts"]) == merged["overrides"]
    assert sum(merged["lane_option_counts"]) == 64
    assert math.isclose(
        merged["completed_distance"],
        first["stats"]["completed_distance"]
        + second["stats"]["completed_distance"],
    )
    print("PASS recurrent boundaries and multi-worker diagnostic merge")


def verify_student_behavior_collection() -> None:
    student = RecurrentActorCritic(
        LEGAL_OBS_SIZE_TRACKED_Y,
        len(ACTIONS),
        hidden_dim=32,
        safe_init=True,
    )
    collector = ExactMpcLabelCollector(
        _state(),
        worker_rank=0,
        worker_count=1,
        seed_start=800_000,
        config=MpcLabelConfig(),
        behavior_student_state=student.state_dict(),
        behavior_hidden_dim=32,
        behavior_mode="deterministic",
    )
    try:
        result = collector.collect(64, student_behavior_fraction=1.0)
    finally:
        collector.close()
    assert result["student_behavior"].all()
    assert result["stats"]["student_behavior_ticks"] == 64
    assert sum(result["stats"]["label_counts"]) == 64
    assert sum(result["stats"]["executed_counts"]) == 64
    assert np.all(result["behavior_actions"] == 1)  # safe-init ACCELERATE
    assert np.any(result["actions"] != result["behavior_actions"])
    print("PASS DAgger student-state visitation with MPC labels")


def verify_student_proposal_correction() -> None:
    """A 41-D legal policy can replace PPO proposals without private input."""
    torch.manual_seed(81)
    proposal = RecurrentActorCritic(
        LEGAL_OBS_SIZE,
        len(ACTIONS),
        hidden_dim=32,
        safe_init=True,
    )
    collector = ExactMpcLabelCollector(
        _state(),
        worker_rank=0,
        worker_count=1,
        seed_start=810_000,
        config=MpcLabelConfig(),
        proposal_student_state=proposal.state_dict(),
        proposal_obs_dim=LEGAL_OBS_SIZE,
        proposal_hidden_dim=32,
        proposal_mode="deterministic",
    )
    try:
        result = collector.collect(1600)
    finally:
        collector.close()

    assert result["stats"]["proposed_counts"][1] == 1600
    assert sum(result["stats"]["label_counts"]) == 1600
    assert np.all(result["behavior_actions"] == result["actions"])
    assert np.any(result["actions"] != 1), "MPC never corrected the proposal"

    # The probability-credit proposal decoder is deterministic under replay.
    def scheduled():
        instance = ExactMpcLabelCollector(
            _state(),
            worker_rank=0,
            worker_count=1,
            seed_start=810_001,
            config=MpcLabelConfig(),
            proposal_student_state=proposal.state_dict(),
            proposal_obs_dim=LEGAL_OBS_SIZE,
            proposal_hidden_dim=32,
            proposal_mode="scheduler",
        )
        try:
            return instance.collect(64)["actions"]
        finally:
            instance.close()

    assert np.array_equal(scheduled(), scheduled())
    print("PASS legal student proposals with deterministic MPC corrections")


def verify_invalid_config() -> None:
    invalid = (
        MpcLabelConfig(horizon=0),
        MpcLabelConfig(strategic_horizon=71),
        MpcLabelConfig(temperature=0.0),
        MpcLabelConfig(opponent_uncertainty_sigma=-1.0),
        MpcLabelConfig(override_soft_target=1.1),
    )
    for config in invalid:
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {config}")
    print("PASS collector-boundary input validation")


def verify_student_checkpoint_ranking() -> None:
    stopped = {
        "crash_or_below_25k_rate": 1.0,
        "crash_rate": 0.0,
        "p10_distance": 495.0,
        "mean_distance": 495.0,
    }
    moving = {
        "crash_or_below_25k_rate": 0.75,
        "crash_rate": 0.5,
        "p10_distance": 5_000.0,
        "mean_distance": 40_000.0,
    }
    assert _score(moving) > _score(stopped)
    print("PASS checkpoint ranking rejects the stopped-policy loophole")


def verify_lane_option_controller() -> None:
    """The optional decoder must finish adjacent-lane moves without oscillation."""
    centers = [float(value) for value in LANE_CENTERS.tolist()]
    car_half_height = float(CAR_HEIGHT) * 0.5
    for proposed, target_index in ((3, 1), (4, 3)):
        y = centers[2] - car_half_height
        vy = 0.0
        target_lane = None
        actions = []
        for _ in range(200):
            action, target_lane = _lane_option_primitive(
                proposed if not actions else 0,
                target_lane,
                y=y,
                vy=vy,
            )
            actions.append(action)
            if target_lane is None:
                break
            vy += -0.1 if action == 3 else 0.1 if action == 4 else 0.0
            y += vy
        assert target_lane is None
        assert len(actions) < 200
        assert abs(y - (centers[target_index] - car_half_height)) <= 4.0
        assert abs(vy) <= 0.2
        assert proposed in actions
        assert (4 if proposed == 3 else 3) in actions

    top_y = centers[0] - car_half_height
    assert _lane_option_primitive(3, None, y=top_y, vy=0.0) == (0, None)
    bottom_y = centers[-1] - car_half_height
    assert _lane_option_primitive(4, None, y=bottom_y, vy=0.0) == (0, None)
    for kwargs in (
        {"proposed": -1, "target_lane": None, "y": top_y, "vy": 0.0},
        {"proposed": 0, "target_lane": 5, "y": top_y, "vy": 0.0},
        {"proposed": 0, "target_lane": None, "y": float("nan"), "vy": 0.0},
    ):
        try:
            _lane_option_primitive(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid lane-option input accepted: {kwargs}")
    print("PASS bounded lane-option execution and input validation")


def verify_lane_option_distillation() -> None:
    torch.manual_seed(711)
    model = RecurrentActorCritic(
        LEGAL_OBS_SIZE_TRACKED_Y,
        len(ACTIONS),
        hidden_dim=8,
        safe_init=False,
    )
    option_head = torch.nn.Linear(8, 3)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(option_head.parameters()), lr=1e-3
    )
    t_steps, num_envs = 5, 3
    obs = torch.randn(t_steps, num_envs, LEGAL_OBS_SIZE_TRACKED_Y)
    actions = torch.randint(0, len(ACTIONS), (t_steps, num_envs))
    probs = torch.softmax(torch.randn(t_steps, num_envs, len(ACTIONS)), dim=-1)
    dones = torch.zeros(t_steps, num_envs, dtype=torch.bool)
    targets = torch.tensor(
        [[0, 1, 2], [0, 0, 0], [1, 0, 2], [0, 0, 0], [2, 1, 0]]
    )
    class_weights = torch.tensor([1.0, 8.0, 8.0])
    before = option_head.weight.detach().clone()
    stats = distill_update_recurrent(
        model,
        optimizer,
        obs,
        actions,
        probs,
        dones,
        model.initial_state(num_envs),
        lane_option_head=option_head,
        lane_option_targets=targets,
        lane_option_class_weights=class_weights,
        lane_option_loss_weight=1.0,
        epochs=1,
        env_minibatch_size=2,
        hard_weight=0.0,
        soft_weight=1.0,
        ent_coef=0.0,
        max_grad_norm=1.0,
    )
    assert math.isfinite(stats.lane_option_loss)
    assert 0.0 <= stats.lane_option_accuracy <= 1.0
    assert not torch.equal(before, option_head.weight)

    try:
        distill_update_recurrent(
            model,
            optimizer,
            obs,
            actions,
            probs,
            dones,
            model.initial_state(num_envs),
            lane_option_head=option_head,
            lane_option_targets=targets,
            lane_option_class_weights=torch.tensor([1.0, float("nan"), 1.0]),
            lane_option_loss_weight=1.0,
            epochs=1,
            env_minibatch_size=2,
            hard_weight=0.0,
            soft_weight=1.0,
            ent_coef=0.0,
            max_grad_norm=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite lane-option class weight accepted")
    print("PASS lane-option auxiliary loss and validation")


def verify_evaluation_diagnostics() -> None:
    rows = [
        {
            "distance": 10_000.0,
            "final_speed": 4.0,
            "crashed": True,
            "hit_car": True,
            "hit_wall": False,
            "final_lateral_speed": -0.2,
            "action_counts": [1, 2, 3, 4, 5],
        },
        {
            "distance": 30_000.0,
            "final_speed": 8.0,
            "crashed": True,
            "hit_car": False,
            "hit_wall": True,
            "final_lateral_speed": 0.4,
            "action_counts": [5, 4, 3, 2, 1],
        },
    ]
    stats = _summarize(rows)
    assert stats["car_crash_rate"] == 0.5
    assert stats["wall_crash_rate"] == 0.5
    assert stats["action_counts"] == [6, 6, 6, 6, 6]
    assert math.isclose(sum(stats["action_fractions"]), 1.0)
    assert math.isclose(stats["mean_abs_final_lateral_speed"], 0.3)
    print("PASS stable collision, lateral-state, and action diagnostics")


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    verify_shapes_and_determinism()
    verify_mpc_labels_and_diagnostics()
    verify_episode_boundaries_and_merge()
    verify_student_behavior_collection()
    verify_student_proposal_correction()
    verify_invalid_config()
    verify_student_checkpoint_ranking()
    verify_lane_option_controller()
    verify_lane_option_distillation()
    verify_evaluation_diagnostics()
    print("All MPC distillation checks passed.")


if __name__ == "__main__":
    main()
