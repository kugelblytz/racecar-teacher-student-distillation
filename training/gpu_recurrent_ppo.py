"""CUDA recurrent actor-critic and PPO update for direct sensor policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


def recurrent_obs_dim_from_state_dict(state_dict) -> int:
    """Return the input width encoded in a recurrent checkpoint."""
    weight = state_dict.get("encoder.0.weight")
    if weight is None:
        raise ValueError("recurrent checkpoint has no encoder.0.weight")
    return int(weight.shape[1])


def expand_recurrent_obs_state_dict(state_dict, new_obs_dim: int):
    """Append zero-weight observation columns without changing policy behavior.

    Existing feature columns keep exactly the same indices. The newly appended
    inputs start disconnected from the encoder, so logits/values are identical
    until further optimization.
    """
    old_obs_dim = recurrent_obs_dim_from_state_dict(state_dict)
    new_obs_dim = int(new_obs_dim)
    if new_obs_dim < old_obs_dim:
        raise ValueError(
            f"cannot shrink recurrent observation {old_obs_dim} -> {new_obs_dim}"
        )

    migrated = dict(state_dict)
    if new_obs_dim == old_obs_dim:
        return migrated, old_obs_dim

    old_weight = state_dict["encoder.0.weight"]
    new_weight = old_weight.new_zeros((old_weight.shape[0], new_obs_dim))
    new_weight[:, :old_obs_dim].copy_(old_weight)
    migrated["encoder.0.weight"] = new_weight
    return migrated, old_obs_dim


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        safe_init: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(128, self.hidden_dim)
        self.post = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

        self._init_weights(safe_init=safe_init)

    def _init_weights(self, safe_init: bool) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.post:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        if safe_init:
            with torch.no_grad():
                # NOTHING, ACCELERATE, DECELERATE, LEFT, RIGHT
                self.policy_head.bias.copy_(
                    torch.tensor([0.0, 1.5, -1.0, -2.0, -2.0])
                )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            int(batch_size),
            self.hidden_dim,
            dtype=torch.float32,
            device=device,
        )

    def step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs)
        next_hidden = self.gru(encoded, hidden)
        z = self.post(next_hidden)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value, next_hidden

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ):
        logits, value, next_hidden = self.step(obs, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, next_hidden


class ResidualRecurrentActorCritic(nn.Module):
    """MLP policy plus a zero-initialized recurrent residual branch.

    The feed-forward path is architecture-identical to ActorCritic. Loading an
    MLP checkpoint and keeping recurrent_projection at exactly zero makes this
    model produce bit-identical logits and values to that MLP for every state,
    regardless of recurrent hidden state. PPO can then learn temporal corrections
    without throwing away the behavior discovered by the MLP.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(128, self.hidden_dim)
        self.recurrent_projection = nn.Linear(self.hidden_dim, 128)
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

        self._init_recurrent()

    def _init_recurrent(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Critical warm-start property: zero residual means exact MLP behavior.
        nn.init.zeros_(self.recurrent_projection.weight)
        nn.init.zeros_(self.recurrent_projection.bias)

    @torch.no_grad()
    def load_from_mlp_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        required = [
            "backbone.0.weight",
            "backbone.0.bias",
            "backbone.2.weight",
            "backbone.2.bias",
            "backbone.4.weight",
            "backbone.4.bias",
            "policy_head.weight",
            "policy_head.bias",
            "value_head.weight",
            "value_head.bias",
        ]
        missing = [name for name in required if name not in state_dict]
        if missing:
            raise ValueError(
                "MLP checkpoint is missing expected parameters: "
                + ", ".join(missing)
            )

        own = self.state_dict()
        for name in required:
            if own[name].shape != state_dict[name].shape:
                raise ValueError(
                    f"Shape mismatch for {name}: residual={tuple(own[name].shape)} "
                    f"mlp={tuple(state_dict[name].shape)}"
                )
            own[name].copy_(state_dict[name])

        # Reassert exact-zero residual in case this method is reused.
        self.recurrent_projection.weight.zero_()
        self.recurrent_projection.bias.zero_()

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(
            int(batch_size),
            self.hidden_dim,
            dtype=torch.float32,
            device=device,
        )

    def step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.backbone(obs)
        next_hidden = self.gru(base, hidden)
        features = base + self.recurrent_projection(next_hidden)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value, next_hidden

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ):
        logits, value, next_hidden = self.step(obs, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, next_hidden

    def base_parameters(self):
        yield from self.backbone.parameters()
        yield from self.policy_head.parameters()
        yield from self.value_head.parameters()

    def recurrent_parameters(self):
        yield from self.gru.parameters()
        yield from self.recurrent_projection.parameters()


@dataclass
class RecurrentPpoStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    epochs_completed: int


def ppo_update_recurrent(
    model: RecurrentActorCritic | ResidualRecurrentActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    dones: torch.Tensor,
    initial_hidden: torch.Tensor,
    *,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    epochs: int,
    env_minibatch_size: int,
    max_grad_norm: float,
    target_kl: float,
) -> RecurrentPpoStats:
    """Truncated-BPTT PPO over complete rollout sequences.

    obs/actions/... are [T, N, ...]. We shuffle environments, not individual
    timesteps, so temporal ordering is preserved. Hidden state is reset after
    done transitions exactly as during rollout.
    """
    t_steps, num_envs = actions.shape
    if env_minibatch_size <= 0:
        raise ValueError("env_minibatch_size must be positive")

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    pg_sum = torch.zeros((), device=obs.device)
    vf_sum = torch.zeros((), device=obs.device)
    ent_sum = torch.zeros((), device=obs.device)
    kl_sum = torch.zeros((), device=obs.device)
    clip_sum = torch.zeros((), device=obs.device)
    updates = 0
    epochs_completed = 0

    for epoch in range(int(epochs)):
        permutation = torch.randperm(num_envs, device=obs.device)

        for start in range(0, num_envs, int(env_minibatch_size)):
            idx = permutation[start : start + int(env_minibatch_size)]
            h = initial_hidden[idx]

            logits_seq = []
            values_seq = []
            for t in range(t_steps):
                logits_t, values_t, h_next = model.step(obs[t, idx], h)
                logits_seq.append(logits_t)
                values_seq.append(values_t)
                h = h_next * (1.0 - dones[t, idx]).unsqueeze(1)

            logits = torch.stack(logits_seq, dim=0)
            values = torch.stack(values_seq, dim=0)
            dist = Categorical(logits=logits)

            new_logprob = dist.log_prob(actions[:, idx])
            entropy = dist.entropy().mean()

            logratio = new_logprob - old_logprobs[:, idx]
            ratio = logratio.exp()
            adv = advantages[:, idx]

            pg_loss_1 = -adv * ratio
            pg_loss_2 = -adv * torch.clamp(
                ratio, 1.0 - clip_range, 1.0 + clip_range
            )
            policy_loss = torch.maximum(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * (returns[:, idx] - values).pow(2).mean()
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
        if target_kl > 0 and updates > 0:
            if (kl_sum / updates).item() > target_kl:
                break

    denom = max(1, updates)
    metrics = torch.stack((pg_sum, vf_sum, ent_sum, kl_sum, clip_sum)) / denom
    pg, vf, ent, kl, clip = metrics.detach().cpu().tolist()

    return RecurrentPpoStats(
        policy_loss=float(pg),
        value_loss=float(vf),
        entropy=float(ent),
        approx_kl=float(kl),
        clip_fraction=float(clip),
        epochs_completed=epochs_completed,
    )
