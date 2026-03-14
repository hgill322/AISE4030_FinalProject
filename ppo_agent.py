"""
ppo_agent.py
============
Proximal Policy Optimization (PPO) agent for continuous control in Assetto Corsa.

Implements the clipped surrogate objective PPO algorithm (Schulman et al., 2017)
with Generalized Advantage Estimation (GAE). Designed for the continuous 3-
dimensional action space of the AC racing environment (steering, throttle, brake).

Algorithm overview:
    1. Collect n_steps transitions using the current policy (RolloutBuffer).
    2. Compute GAE advantages and discounted returns.
    3. For n_epochs, iterate over mini-batches and update the policy by:
       a. Computing the clipped surrogate loss (actor).
       b. Computing the MSE value loss (critic).
       c. Adding an entropy bonus to encourage exploration.
       d. Backpropagating the combined loss with gradient clipping.
    4. Repeat from step 1.

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347
"""

import os
import torch
import numpy as np
from typing import Tuple, Dict

from actor_critic import ActorCriticNetwork
from rollout_buffer import RolloutBuffer


class PPOAgent:
    """
    PPO agent for continuous control on the Assetto Corsa racing environment.

    Manages the Actor-Critic network, rollout buffer, optimizer, and implements
    the PPO clipped objective update. Supports exploration/exploitation modes,
    entropy-based exploration scheduling, and model checkpointing.

    Args:
        config (dict): Full config dictionary loaded from config.yaml.
        obs_dim (int): Observation space dimensionality (125 for AC env).
        act_dim (int): Action space dimensionality (3 for AC env).
        device (torch.device): Torch device (CPU / CUDA / MPS).
    """

    def __init__(self, config: dict, obs_dim: int, act_dim: int, device: torch.device):
        """
        Initializes the PPOAgent: builds the ActorCriticNetwork, Adam optimizer,
        RolloutBuffer, and sets all hyperparameters from config.

        Args:
            config (dict): Full config loaded from config.yaml.
            obs_dim (int): Number of observation dimensions.
            act_dim (int): Number of action dimensions.
            device (torch.device): Compute device.
        """
        pass

    def choose_action(self, obs: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        Selects an action for the given observation.

        In training mode, samples from the Gaussian policy distribution
        (exploration). In evaluation mode, takes the distribution mean
        (exploitation).

        Args:
            obs (np.ndarray): Current observation vector, shape (obs_dim,).
            training (bool): If True, sample stochastically. If False, use
                the policy mean for deterministic exploitation.

        Returns:
            Tuple containing:
                - action (np.ndarray): Chosen action, shape (act_dim,).
                - log_prob (float): Log probability of the sampled action.
                - value (float): Critic value estimate V(obs).
        """
        pass

    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                         done: bool, log_prob: float, value: float) -> None:
        """
        Stores a single environment transition in the rollout buffer.

        Args:
            obs (np.ndarray): Observation at this step, shape (obs_dim,).
            action (np.ndarray): Action taken, shape (act_dim,).
            reward (float): Scalar reward received after taking the action.
            done (bool): Whether this step terminated the episode.
            log_prob (float): Log probability of the action at collection time.
            value (float): Critic estimate V(obs) at collection time.

        Returns:
            None
        """
        pass

    def update(self, last_obs: np.ndarray, last_done: bool) -> Dict[str, float]:
        """
        Performs the PPO update using the current rollout buffer contents.

        Steps:
            1. Bootstrap the value of the last observation.
            2. Compute GAE advantages and discounted returns in the buffer.
            3. For n_epochs: iterate over mini-batches and compute the
               clipped PPO surrogate loss, value loss, and entropy bonus.
               Backpropagate with gradient norm clipping.
            4. Reset the buffer.

        Args:
            last_obs (np.ndarray): Observation following the last stored step.
            last_done (bool): Whether the last step ended the episode.

        Returns:
            Dict[str, float]: Training metrics: policy_loss, value_loss,
                entropy, total_loss.
        """
        pass

    def update_entropy_coef(self, new_coef: float) -> None:
        """
        Updates the entropy coefficient used to control exploration level.

        Args:
            new_coef (float): New entropy coefficient value (>= 0).

        Returns:
            None
        """
        pass

    def decay_entropy_coef(self, decay_factor: float = 0.995) -> None:
        """
        Multiplies the entropy coefficient by a decay factor for annealing.

        Args:
            decay_factor (float): Multiplier applied each call. Defaults to 0.995.

        Returns:
            None
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Saves the network weights and optimizer state to a checkpoint file.

        Args:
            path (str): File path for the checkpoint .pt file.

        Returns:
            None
        """
        pass

    def load_model(self, path: str) -> None:
        """
        Loads network weights and optimizer state from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file saved by save_model().

        Returns:
            None

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        pass

    def save_best_if_improved(self, eval_reward: float) -> bool:
        """
        Saves the model as the best checkpoint if eval_reward exceeds the
        previous best evaluation reward.

        Args:
            eval_reward (float): Mean episode reward from the latest evaluation.

        Returns:
            bool: True if this was a new best and the model was saved.
        """
        pass
