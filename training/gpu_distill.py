"""Recurrent supervised distillation for a legal-observation GRU student."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from training.gpu_recurrent_ppo import RecurrentActorCritic


@dataclass
class DistillStats:
    loss: float
    hard_loss: float
    soft_loss: float
    teacher_kl: float
    action_accuracy: float
    entropy: float
    lane_option_loss: float
    lane_option_accuracy: float
    epochs_completed: int


def distill_update_recurrent(
    model: RecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_probs: torch.Tensor,
    dones: torch.Tensor,
    initial_hidden: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    lane_option_head: torch.nn.Module | None = None,
    lane_option_targets: torch.Tensor | None = None,
    lane_option_class_weights: torch.Tensor | None = None,
    lane_option_loss_weight: float = 0.0,
    epochs: int,
    env_minibatch_size: int,
    hard_weight: float,
    soft_weight: float,
    ent_coef: float,
    max_grad_norm: float,
) -> DistillStats:
    """Train on complete recurrent rollout sequences.

    Inputs are [T, N, ...]. Environments are shuffled, not timesteps, preserving
    sequence order and reset boundaries. initial_hidden is the student's
    recurrent state at the beginning of the collected rollout.
    """
    t_steps, num_envs = teacher_actions.shape
    if obs.shape[:2] != (t_steps, num_envs):
        raise ValueError("obs shape does not match teacher_actions")
    if teacher_probs.shape[:2] != (t_steps, num_envs):
        raise ValueError("teacher_probs shape does not match teacher_actions")
    if dones.shape != (t_steps, num_envs):
        raise ValueError("dones shape does not match teacher_actions")
    if sample_weights is not None:
        if sample_weights.shape != (t_steps, num_envs):
            raise ValueError("sample_weights shape does not match teacher_actions")
        if not bool(torch.isfinite(sample_weights).all()) or bool(
            (sample_weights <= 0).any()
        ):
            raise ValueError("sample_weights must be finite and positive")
    option_values = (
        lane_option_head,
        lane_option_targets,
        lane_option_class_weights,
    )
    if any(value is not None for value in option_values) and not all(
        value is not None for value in option_values
    ):
        raise ValueError("lane-option head, targets, and class weights are required")
    if lane_option_targets is not None:
        if lane_option_targets.shape != (t_steps, num_envs):
            raise ValueError("lane_option_targets shape does not match actions")
        if lane_option_class_weights.shape != (3,):
            raise ValueError("lane_option_class_weights must have three values")
        if not bool(torch.isfinite(lane_option_class_weights).all()) or bool(
            (lane_option_class_weights <= 0).any()
        ):
            raise ValueError("lane_option_class_weights must be finite and positive")
        if bool((lane_option_targets < 0).any()) or bool(
            (lane_option_targets > 2).any()
        ):
            raise ValueError("lane_option_targets must be in [0, 2]")
    if not math.isfinite(float(lane_option_loss_weight)) or lane_option_loss_weight < 0:
        raise ValueError("lane_option_loss_weight must be finite and non-negative")
    if env_minibatch_size <= 0:
        raise ValueError("env_minibatch_size must be positive")
    if hard_weight < 0 or soft_weight < 0:
        raise ValueError("distillation weights must be non-negative")
    if hard_weight == 0 and soft_weight == 0:
        raise ValueError("at least one distillation weight must be positive")

    loss_sum = torch.zeros((), device=obs.device)
    hard_sum = torch.zeros((), device=obs.device)
    soft_sum = torch.zeros((), device=obs.device)
    kl_sum = torch.zeros((), device=obs.device)
    acc_sum = torch.zeros((), device=obs.device)
    ent_sum = torch.zeros((), device=obs.device)
    option_loss_sum = torch.zeros((), device=obs.device)
    option_acc_sum = torch.zeros((), device=obs.device)
    updates = 0

    for _ in range(int(epochs)):
        permutation = torch.randperm(num_envs, device=obs.device)

        for start in range(0, num_envs, int(env_minibatch_size)):
            idx = permutation[start : start + int(env_minibatch_size)]
            h = initial_hidden[idx]

            logits_seq = []
            hidden_seq = []
            for t in range(t_steps):
                logits_t, _, h_next = model.step(obs[t, idx], h)
                logits_seq.append(logits_t)
                hidden_seq.append(h_next)
                h = h_next * (~dones[t, idx]).to(h.dtype).unsqueeze(1)

            logits = torch.stack(logits_seq, dim=0)
            target_actions = teacher_actions[:, idx]
            target_probs = teacher_probs[:, idx]
            weights = (
                torch.ones_like(target_actions, dtype=torch.float32)
                if sample_weights is None
                else sample_weights[:, idx].to(torch.float32)
            )
            weight_sum = weights.sum().clamp_min(1e-8)

            log_probs = F.log_softmax(logits, dim=-1)
            hard_per_item = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_actions.reshape(-1),
                reduction="none",
            ).reshape(t_steps, -1)
            hard_loss = (hard_per_item * weights).sum() / weight_sum
            soft_per_item = -(target_probs * log_probs).sum(dim=-1)
            soft_loss = (soft_per_item * weights).sum() / weight_sum

            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()

            option_loss = torch.zeros((), device=obs.device)
            option_accuracy = torch.zeros((), device=obs.device)
            if lane_option_head is not None and lane_option_targets is not None:
                option_logits = lane_option_head(torch.stack(hidden_seq, dim=0))
                option_target = lane_option_targets[:, idx]
                option_per_item = F.cross_entropy(
                    option_logits.reshape(-1, 3),
                    option_target.reshape(-1),
                    reduction="none",
                ).reshape(t_steps, -1)
                option_weights = lane_option_class_weights.to(
                    device=obs.device, dtype=torch.float32
                )[option_target]
                option_loss = (
                    option_per_item * option_weights
                ).sum() / option_weights.sum().clamp_min(1e-8)
                option_accuracy = (
                    (option_logits.argmax(dim=-1) == option_target)
                    .to(torch.float32)
                    .mean()
                )

            loss = (
                float(hard_weight) * hard_loss
                + float(soft_weight) * soft_loss
                - float(ent_coef) * entropy
                + float(lane_option_loss_weight) * option_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clipped_parameters = list(model.parameters())
            if lane_option_head is not None:
                clipped_parameters.extend(lane_option_head.parameters())
            torch.nn.utils.clip_grad_norm_(clipped_parameters, max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                teacher_log_probs = target_probs.clamp_min(1e-8).log()
                teacher_kl = (
                    target_probs * (teacher_log_probs - log_probs)
                ).sum(dim=-1)
                teacher_kl = (teacher_kl * weights).sum() / weight_sum
                correct = (
                    logits.argmax(dim=-1) == target_actions
                ).to(torch.float32)
                accuracy = (correct * weights).sum() / weight_sum

            loss_sum += loss.detach()
            hard_sum += hard_loss.detach()
            soft_sum += soft_loss.detach()
            kl_sum += teacher_kl.detach()
            acc_sum += accuracy.detach()
            ent_sum += entropy.detach()
            option_loss_sum += option_loss.detach()
            option_acc_sum += option_accuracy.detach()
            updates += 1

    denom = max(1, updates)
    metrics = torch.stack(
        (
            loss_sum,
            hard_sum,
            soft_sum,
            kl_sum,
            acc_sum,
            ent_sum,
            option_loss_sum,
            option_acc_sum,
        )
    ) / denom
    loss, hard, soft, kl, acc, ent, option_loss, option_acc = (
        metrics.detach().cpu().tolist()
    )

    return DistillStats(
        loss=float(loss),
        hard_loss=float(hard),
        soft_loss=float(soft),
        teacher_kl=float(kl),
        action_accuracy=float(acc),
        entropy=float(ent),
        lane_option_loss=float(option_loss),
        lane_option_accuracy=float(option_acc),
        epochs_completed=int(epochs),
    )
