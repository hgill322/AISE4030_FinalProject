"""
actor_critic.py
===============
Actor-Critic neural network architecture for the PPO agent.

Defines a shared-backbone MLP Actor-Critic network. The actor outputs the mean
of a Gaussian policy over the continuous action space; the critic outputs a
scalar state-value estimate.

Architecture:
    Input (obs_dim=125)
        -> Shared MLP backbone (configurable hidden layers)
              |-> Actor head  -> action_mean (act_dim=3)
              |                  log_std     (act_dim=3, learned parameter)
              |-> Critic head -> value (scalar)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, List


class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone Actor-Critic network for continuous control with PPO.

    The backbone is a shared MLP. The actor head produces the mean of a
    Gaussian policy; the critic head produces a scalar value estimate.
    Log standard deviation is a separate learned parameter vector.

    Args:
        obs_dim (int): Dimension of the observation vector (125 for AC env).
        act_dim (int): Dimension of the action vector (3 for AC env).
        hidden_dims (List[int]): Hidden layer sizes for the shared backbone.
        activation_fn (str): Activation function name ('tanh' or 'relu').
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int],
                 activation_fn: str = "tanh"):
        """
        Initializes the ActorCriticNetwork: builds shared backbone, actor head,
        critic head, and log_std parameter. Applies orthogonal weight init.

        Args:
            obs_dim (int): Observation space dimensionality.
            act_dim (int): Action space dimensionality.
            hidden_dims (List[int]): Hidden layer sizes, e.g. [256, 256].
            activation_fn (str): 'tanh' or 'relu'. Defaults to 'tanh'.
        """
        super().__init__()

        act_fn = nn.Tanh if activation_fn == "tanh" else nn.ReLU

        # Shared backbone
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Actor head: outputs mean of Gaussian policy
        self.actor_head = nn.Linear(in_dim, act_dim)

        # Learned log std (not state-dependent, standard PPO practice)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head: outputs scalar value estimate
        self.critic_head = nn.Linear(in_dim, 1)

        # Orthogonal weight initialization (improves early training stability)
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Applies orthogonal initialization to all Linear layers.
        Actor head uses gain=0.01 for small initial actions; others use sqrt(2).

        Returns:
            None
        """
        for module in self.backbone.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through backbone, actor head, and critic head.

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch, obs_dim).

        Returns:
            Tuple containing:
                - action_mean (torch.Tensor): Shape (batch, act_dim).
                - log_std (torch.Tensor): Shape (act_dim,).
                - value (torch.Tensor): Shape (batch, 1).
        """
        features = self.backbone(obs)
        action_mean = self.actor_head(features)
        value = self.critic_head(features)
        return action_mean, self.log_std, value

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """
        Constructs the Gaussian action distribution for a given observation.

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch, obs_dim).

        Returns:
            torch.distributions.Normal: Gaussian distribution over actions.
        """
        action_mean, log_std, _ = self.forward(obs)
        std = torch.exp(log_std.clamp(-4.0, 2.0))
        return Normal(action_mean, std)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates stored actions under the current policy for the PPO update.

        Args:
            obs (torch.Tensor): Observation batch, shape (batch, obs_dim).
            actions (torch.Tensor): Action batch, shape (batch, act_dim).

        Returns:
            Tuple containing:
                - log_probs (torch.Tensor): Shape (batch,) — sum over action dims.
                - entropy (torch.Tensor): Shape (batch,) — sum over action dims.
                - values (torch.Tensor): Shape (batch, 1).
        """
        action_mean, log_std, values = self.forward(obs)
        std = torch.exp(log_std.clamp(-4.0, 2.0))
        dist = Normal(action_mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, values

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns only the critic value estimate for bootstrapping.

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch, obs_dim)
                or (obs_dim,).

        Returns:
            torch.Tensor: Value estimate, shape (batch, 1).
        """
        features = self.backbone(obs)
        return self.critic_head(features)
