"""Minimal CUDA-native PPO components for the privileged teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, safe_init: bool = True) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

        self._init_weights(safe_init=safe_init)

    def _init_weights(self, safe_init: bool) -> None:
        for module in self.backbone:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        # Primitive-action PPO from a uniform random policy tends to destroy its
        # lateral velocity immediately. This initialization still leaves every
        # action available, but starts with a useful full-throttle/lane-hold prior.
        if safe_init:
            with torch.no_grad():
                # NOTHING, ACCELERATE, DECELERATE, LEFT, RIGHT
                self.policy_head.bias.copy_(
                    torch.tensor([0.0, 1.5, -1.0, -2.0, -2.0])
                )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    @torch.no_grad()
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        logits, _ = self(obs)
        return logits.argmax(dim=-1)


@dataclass
class PpoStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    epochs_completed: int


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
    target_kl: float,
) -> PpoStats:
    batch_size = obs.shape[0]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_sum = torch.zeros((), device=obs.device)
    vf_sum = torch.zeros((), device=obs.device)
    ent_sum = torch.zeros((), device=obs.device)
    kl_sum = torch.zeros((), device=obs.device)
    clip_sum = torch.zeros((), device=obs.device)
    updates = 0
    epochs_completed = 0

    for epoch in range(epochs):
        permutation = torch.randperm(batch_size, device=obs.device)
        early_stop = False

        for start in range(0, batch_size, minibatch_size):
            idx = permutation[start : start + minibatch_size]

            logits, values = model(obs[idx])
            dist = Categorical(logits=logits)
            new_logprob = dist.log_prob(actions[idx])
            entropy = dist.entropy().mean()

            logratio = new_logprob - old_logprobs[idx]
            ratio = logratio.exp()

            adv = advantages[idx]
            pg_loss_1 = -adv * ratio
            pg_loss_2 = -adv * torch.clamp(
                ratio, 1.0 - clip_range, 1.0 + clip_range
            )
            policy_loss = torch.maximum(pg_loss_1, pg_loss_2).mean()

            value_loss = 0.5 * (returns[idx] - values).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1.0) > clip_range).float().mean()
                )

            pg_sum += policy_loss.detach()
            vf_sum += value_loss.detach()
            ent_sum += entropy.detach()
            kl_sum += approx_kl.detach()
            clip_sum += clip_fraction.detach()
            updates += 1

        epochs_completed = epoch + 1

        # At most one scalar synchronization per PPO epoch for KL early-stop,
        # rather than one synchronization per minibatch.
        if target_kl > 0 and updates > 0:
            mean_kl_so_far = kl_sum / updates
            if mean_kl_so_far.item() > target_kl:
                early_stop = True

        if early_stop:
            break

    denom = max(1, updates)
    metrics = torch.stack(
        (pg_sum, vf_sum, ent_sum, kl_sum, clip_sum)
    ) / denom
    metrics_cpu = metrics.detach().cpu().tolist()

    return PpoStats(
        policy_loss=float(metrics_cpu[0]),
        value_loss=float(metrics_cpu[1]),
        entropy=float(metrics_cpu[2]),
        approx_kl=float(metrics_cpu[3]),
        clip_fraction=float(metrics_cpu[4]),
        epochs_completed=epochs_completed,
    )
