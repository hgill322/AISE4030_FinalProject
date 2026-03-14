"""
rollout_buffer.py
=================
On-policy rollout buffer for PPO.

Stores a fixed-length trajectory of (obs, action, reward, done, log_prob, value)
tuples collected during environment interaction. After each rollout, computes
Generalized Advantage Estimates (GAE) and returns mini-batches for the PPO
update step.

Reference:
    Schulman et al., "High-Dimensional Continuous Control Using Generalized
    Advantage Estimation" (2015)
"""

import numpy as np
import torch
from typing import Iterator, Tuple


class RolloutBuffer:
    """
    Fixed-capacity on-policy rollout buffer for PPO.

    Collects n_steps transitions, computes GAE advantages and returns,
    then yields randomized mini-batches for the PPO update epochs.

    Args:
        n_steps (int): Number of environment steps per rollout (buffer capacity).
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.
        gamma (float): Discount factor for return computation.
        gae_lambda (float): GAE smoothing parameter (lambda).
        device (torch.device): Torch device for tensor outputs.
    """

    def __init__(self, n_steps: int, obs_dim: int, act_dim: int,
                 gamma: float, gae_lambda: float, device: torch.device):
        """
        Initializes the RolloutBuffer and pre-allocates numpy storage arrays.

        Args:
            n_steps (int): Rollout length before each PPO update.
            obs_dim (int): Observation vector dimensionality.
            act_dim (int): Action vector dimensionality.
            gamma (float): Discount factor, e.g. 0.99.
            gae_lambda (float): GAE lambda, e.g. 0.95.
            device (torch.device): Device for returned tensors.
        """
        pass

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            done: bool, log_prob: float, value: float) -> None:
        """
        Stores a single transition in the buffer at the current pointer position.

        Args:
            obs (np.ndarray): Observation, shape (obs_dim,).
            action (np.ndarray): Action taken, shape (act_dim,).
            reward (float): Scalar reward received.
            done (bool): Whether this step ended the episode.
            log_prob (float): Log probability of the action under the policy.
            value (float): Critic value estimate V(obs).

        Returns:
            None
        """
        pass

    def compute_advantages_and_returns(self, last_value: float) -> None:
        """
        Computes GAE advantages and discounted returns after a full rollout.

        Uses the GAE formula (backwards pass):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = sum_{l=0}^{T} (gamma * lambda)^l * delta_{t+l}

        Must be called before get_batches().

        Args:
            last_value (float): Bootstrapped value V(s_T) after the rollout.
                Pass 0.0 if the rollout ended in a terminal state.

        Returns:
            None
        """
        pass

    def get_batches(self, batch_size: int, normalize_advantages: bool = True
                    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Yields randomized mini-batches of rollout data for PPO update epochs.

        Args:
            batch_size (int): Number of transitions per mini-batch.
            normalize_advantages (bool): If True, normalizes advantages to
                zero mean and unit variance before yielding batches.

        Yields:
            Tuple of torch.Tensors:
                (obs_batch, action_batch, log_prob_batch,
                 advantage_batch, return_batch)
        """
        pass

    def reset(self) -> None:
        """
        Resets the buffer pointer to allow re-use for the next rollout.

        Returns:
            None
        """
        pass

    @property
    def is_full(self) -> bool:
        """
        Returns True when the buffer has collected n_steps transitions.

        Returns:
            bool: Whether the buffer is ready for a PPO update.
        """
        pass
