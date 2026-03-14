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
        pass

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
        pass

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        """
        Constructs the Gaussian action distribution for a given observation.

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch, obs_dim).

        Returns:
            torch.distributions.Normal: Gaussian distribution over actions.
        """
        pass

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates stored actions under the current policy for the PPO update.

        Args:
            obs (torch.Tensor): Observation batch, shape (batch, obs_dim).
            actions (torch.Tensor): Action batch, shape (batch, act_dim).

        Returns:
            Tuple containing:
                - log_probs (torch.Tensor): Shape (batch,).
                - entropy (torch.Tensor): Shape (batch,).
                - values (torch.Tensor): Shape (batch, 1).
        """
        pass

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns only the critic value estimate for bootstrapping.

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch, obs_dim)
                or (obs_dim,).

        Returns:
            torch.Tensor: Value estimate, shape (batch, 1).
        """
        pass
