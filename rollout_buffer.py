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
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Pre-allocate numpy storage
        self.obs       = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.rewards   = np.zeros(n_steps, dtype=np.float32)
        self.dones     = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.values    = np.zeros(n_steps, dtype=np.float32)

        # Computed after rollout ends
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns    = np.zeros(n_steps, dtype=np.float32)

        self._ptr = 0  # current write position

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
        assert self._ptr < self.n_steps, "Buffer is full; call reset() before adding."
        self.obs[self._ptr]       = obs
        self.actions[self._ptr]   = action
        self.rewards[self._ptr]   = reward
        self.dones[self._ptr]     = float(done)
        self.log_probs[self._ptr] = log_prob
        self.values[self._ptr]    = value
        self._ptr += 1

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
        gae = 0.0
        for t in reversed(range(self._ptr)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self._ptr - 1 else self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

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
        adv = self.advantages[:self._ptr].copy()
        if normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = np.random.permutation(self._ptr)
        for start in range(0, self._ptr, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                torch.FloatTensor(self.obs[idx]).to(self.device),
                torch.FloatTensor(self.actions[idx]).to(self.device),
                torch.FloatTensor(self.log_probs[idx]).to(self.device),
                torch.FloatTensor(adv[idx]).to(self.device),
                torch.FloatTensor(self.returns[idx]).to(self.device),
            )

    def reset(self) -> None:
        """
        Resets the buffer pointer to allow re-use for the next rollout.

        Returns:
            None
        """
        self._ptr = 0

    @property
    def is_full(self) -> bool:
        """
        Returns True when the buffer has collected n_steps transitions.

        Returns:
            bool: Whether the buffer is ready for a PPO update.
        """
        return self._ptr >= self.n_steps
