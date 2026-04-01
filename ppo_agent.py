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
import torch.nn as nn
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
        self.device = device
        ppo_cfg  = config["ppo"]
        net_cfg  = config["network"]
        path_cfg = config["paths"]["ppo"]

        self.gamma           = ppo_cfg["gamma"]
        self.gae_lambda      = ppo_cfg["gae_lambda"]
        self.clip_epsilon    = ppo_cfg["clip_epsilon"]
        self.value_loss_coef = ppo_cfg["value_loss_coef"]
        self.entropy_coef    = ppo_cfg["entropy_coef"]
        self.max_grad_norm   = ppo_cfg["max_grad_norm"]
        self.n_steps         = ppo_cfg["n_steps"]
        self.n_epochs        = ppo_cfg["n_epochs"]
        self.batch_size      = ppo_cfg["batch_size"]
        self.normalize_adv   = ppo_cfg.get("normalize_advantages", True)
        self.results_dir     = path_cfg["results_dir"]
        self.best_model_path = path_cfg["best_model_path"]

        os.makedirs(self.results_dir, exist_ok=True)

        # Actor-critic network: shared MLP backbone + actor head + critic head
        self.network = ActorCriticNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=net_cfg["hidden_dims"],
            activation_fn=net_cfg.get("activation", "relu"),
        ).to(device)

        # Single Adam optimizer covers all network parameters
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=ppo_cfg["learning_rate"]
        )

        # On-policy rollout buffer — stores one full rollout before each update
        self.buffer = RolloutBuffer(
            n_steps=self.n_steps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=device,
        )

        # Track best evaluation reward for save_best_if_improved()
        self._best_eval_reward = float("-inf")

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
                - action (np.ndarray): Chosen action clipped to [-1, 1], shape (act_dim,).
                - log_prob (float): Log probability of the sampled action.
                - value (float): Critic value estimate V(obs).
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if training:
                dist = self.network.get_distribution(obs_tensor)
                action_tensor = dist.sample()
                log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
            else:
                # Deterministic exploitation: use policy mean
                action_mean, _, _ = self.network(obs_tensor)
                action_tensor = action_mean
                log_prob = 0.0

            value = self.network.get_value(obs_tensor).item()

        action = action_tensor.clamp(-1.0, 1.0).squeeze(0).cpu().numpy()
        return action, log_prob, value

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
        self.buffer.add(obs, action, reward, done, log_prob, value)

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
        # Bootstrap last value for GAE
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(last_obs).unsqueeze(0).to(self.device)
            last_value = 0.0 if last_done else self.network.get_value(last_obs_t).item()

        self.buffer.compute_advantages_and_returns(last_value)

        # Accumulate loss metrics across all epochs and mini-batches
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_loss_sum    = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in self.buffer.get_batches(
                self.batch_size, self.normalize_adv
            ):
                # Evaluate stored actions under current policy
                new_log_probs, entropy, values = self.network.evaluate_actions(obs_b, act_b)
                values = values.squeeze(-1)

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE between critic output and MC returns)
                value_loss = nn.functional.mse_loss(values, ret_b)

                # Entropy bonus encourages continued exploration
                entropy_mean = entropy.mean()

                loss = (policy_loss
                        + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_mean)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy_mean.item()
                total_loss_sum    += loss.item()
                n_updates += 1

        self.buffer.reset()

        denom = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / denom,
            "value_loss":  total_value_loss  / denom,
            "entropy":     total_entropy     / denom,
            "total_loss":  total_loss_sum    / denom,
        }

    def update_entropy_coef(self, new_coef: float) -> None:
        """
        Updates the entropy coefficient used to control exploration level.

        Args:
            new_coef (float): New entropy coefficient value (>= 0).

        Returns:
            None
        """
        self.entropy_coef = max(0.0, new_coef)

    def decay_entropy_coef(self, decay_factor: float = 0.9998) -> None:
        """
        Multiplies the entropy coefficient by a decay factor for annealing.

        Entropy annealing gradually reduces the exploration bonus during
        training so the agent shifts from exploration to exploitation over time.

        Args:
            decay_factor (float): Multiplier applied each call. Defaults to 0.995.

        Returns:
            None
        """
        self.entropy_coef = max(1e-5, self.entropy_coef * decay_factor)

    def save_model(self, path: str) -> None:
        """
        Saves the network weights and optimizer state to a checkpoint file.

        Args:
            path (str): File path for the checkpoint .pt file.

        Returns:
            None
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "network_state_dict":   self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "entropy_coef":         self.entropy_coef,
            "best_eval_reward":     self._best_eval_reward,
        }, path)
        print(f"[Checkpoint] Model saved -> {path}")

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
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.entropy_coef = checkpoint.get("entropy_coef", self.entropy_coef)
        self._best_eval_reward = checkpoint.get("best_eval_reward", float("-inf"))
        print(f"[Checkpoint] Model loaded <- {path}")

    def save_best_if_improved(self, eval_reward: float) -> bool:
        """
        Saves the model as the best checkpoint if eval_reward exceeds the
        previous best evaluation reward.

        Args:
            eval_reward (float): Mean episode reward from the latest evaluation.

        Returns:
            bool: True if this was a new best and the model was saved.
        """
        if eval_reward > self._best_eval_reward:
            self._best_eval_reward = eval_reward
            self.save_model(self.best_model_path)
            print(f"[Best] New best model! Eval reward: {eval_reward:.2f}")
            return True
        return False
