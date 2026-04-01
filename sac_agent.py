"""
sac_agent.py
============
Soft Actor-Critic (SAC) agent for continuous control in Assetto Corsa.

Implements SAC (Haarnoja et al., 2018) with:
    - Twin Q-networks (SAC-v2) to reduce value overestimation
    - Automatic entropy temperature tuning
    - Off-policy replay buffer for high sample efficiency
    - Reparameterization trick for low-variance policy gradients

Algorithm overview:
    1. Collect one transition using the current stochastic policy.
    2. Store transition in the replay buffer (never discarded).
    3. Sample a random mini-batch from the replay buffer.
    4. Update twin Q-networks (critics) using Bellman backup with target nets.
    5. Update the actor by maximizing Q - alpha * log_prob.
    6. Update alpha (temperature) to track target entropy.
    7. Soft-update target Q-networks toward current Q-networks.
    8. Repeat from step 1.

Key difference from PPO:
    - Off-policy: every transition is reused many times (sample efficient)
    - Entropy is part of the objective, not just a bonus (auto-tuned alpha)
    - Updates happen every step, not after a full rollout

Reference:
    Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018)
    arXiv:1812.05905
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
from collections import deque
import random


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Off-policy replay buffer storing (obs, action, reward, next_obs, done)
    transitions. Randomly sampled mini-batches break temporal correlations
    and enable data reuse across many gradient updates.

    Unlike the PPO RolloutBuffer, this buffer retains all transitions up to
    its capacity — nothing is discarded after an update.

    Args:
        capacity (int): Maximum number of transitions to store.
        obs_dim (int): Observation space dimensionality.
        act_dim (int): Action space dimensionality.
        device (torch.device): Device for returned tensors.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int,
                 device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Pre-allocate numpy arrays for fast indexing
        self.obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),        dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),        dtype=np.float32)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool) -> None:
        """
        Adds a single transition to the replay buffer (circular overwrite).

        Args:
            obs (np.ndarray): Observation at this step, shape (obs_dim,).
            action (np.ndarray): Action taken, shape (act_dim,).
            reward (float): Scalar reward received.
            next_obs (np.ndarray): Next observation, shape (obs_dim,).
            done (bool): Whether this transition ended the episode.
        """
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Samples a random mini-batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of tensors: (obs, actions, rewards, next_obs, dones)
                each of shape (batch_size, dim).
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_obs[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# SAC Network Components
# ---------------------------------------------------------------------------
def _mlp(in_dim: int, hidden_dims: List[int], out_dim: int,
         activation: str = "relu") -> nn.Sequential:
    """
    Builds a simple MLP with the given hidden dimensions.

    Args:
        in_dim (int): Input dimension.
        hidden_dims (List[int]): Hidden layer sizes.
        out_dim (int): Output dimension.
        activation (str): 'relu' or 'tanh'.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    act_fn = nn.ReLU if activation == "relu" else nn.Tanh
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act_fn()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SACQNetwork(nn.Module):
    """
    Q-network Q(s, a) for SAC. Takes concatenated (obs, action) as input
    and outputs a scalar Q-value estimate.

    SAC uses two of these (twin critics) to mitigate overestimation bias.

    Args:
        obs_dim (int): Observation dimensionality.
        act_dim (int): Action dimensionality.
        hidden_dims (List[int]): Hidden layer sizes.
        activation (str): Activation function name.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int],
                 activation: str = "relu"):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, hidden_dims, 1, activation)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs (torch.Tensor): Shape (batch, obs_dim).
            action (torch.Tensor): Shape (batch, act_dim).

        Returns:
            torch.Tensor: Q-values, shape (batch, 1).
        """
        return self.net(torch.cat([obs, action], dim=-1))


class SACActor(nn.Module):
    """
    Stochastic actor for SAC using the reparameterization trick.

    Outputs a squashed Gaussian policy via tanh(mu + std * eps).
    This allows gradients to flow through sampled actions directly
    into the policy parameters (unlike REINFORCE-style log_prob gradients).

    The output is squashed to [-1, 1] via tanh, matching the AC action space.

    Args:
        obs_dim (int): Observation dimensionality.
        act_dim (int): Action dimensionality.
        hidden_dims (List[int]): Hidden layer sizes.
        activation (str): Activation function name.
        log_std_min (float): Minimum log std (clamp).
        log_std_max (float): Maximum log std (clamp).
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: List[int],
                 activation: str = "relu"):
        super().__init__()
        # Shared backbone
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        self.backbone = nn.Sequential(*layers)

        # Mean and log_std heads (state-dependent, unlike PPO's fixed log_std)
        self.mean_head    = nn.Linear(prev, act_dim)
        self.log_std_head = nn.Linear(prev, act_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        # Warm-start: bias throttle (dim 1) slightly positive
        with torch.no_grad():
            self.mean_head.bias[1] = 0.3

    def forward(self, obs: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action and computes its log probability.

        Uses the reparameterization trick: action = tanh(mean + std * N(0,1))
        Log prob accounts for the tanh squashing via change of variables.

        Args:
            obs (torch.Tensor): Shape (batch, obs_dim).

        Returns:
            Tuple:
                - action (torch.Tensor): Squashed action in [-1,1], shape (batch, act_dim).
                - log_prob (torch.Tensor): Log probability, shape (batch, 1).
        """
        features = self.backbone(obs)
        mean     = self.mean_head(features)
        log_std  = self.log_std_head(features).clamp(self.LOG_STD_MIN,
                                                      self.LOG_STD_MAX)
        std = log_std.exp()

        # Reparameterization: sample via N(0,1), shift/scale, then squash
        normal = torch.distributions.Normal(mean, std)
        x_t    = normal.rsample()          # rsample = reparameterized sample
        action = torch.tanh(x_t)

        # Log prob with tanh squashing correction
        # log pi(a|s) = log N(x_t) - sum log(1 - tanh^2(x_t) + eps)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        """
        Returns a numpy action for environment interaction (no grad).

        Args:
            obs (torch.Tensor): Shape (1, obs_dim).

        Returns:
            np.ndarray: Action vector, shape (act_dim,).
        """
        with torch.no_grad():
            action, _ = self.forward(obs)
        return action.squeeze(0).cpu().numpy()

    def get_deterministic_action(self, obs: torch.Tensor) -> np.ndarray:
        """
        Returns the deterministic policy mean (tanh(mean)) for evaluation.

        Args:
            obs (torch.Tensor): Shape (1, obs_dim).

        Returns:
            np.ndarray: Deterministic action, shape (act_dim,).
        """
        with torch.no_grad():
            features = self.backbone(obs)
            mean     = self.mean_head(features)
            action   = torch.tanh(mean)
        return action.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------
class SACAgent:
    """
    Soft Actor-Critic agent for continuous control on the AC racing environment.

    Manages the actor, twin critics, target critics, replay buffer, and
    implements the SAC update with automatic entropy tuning.

    Args:
        config (dict): Full config dictionary loaded from config.yaml.
        obs_dim (int): Observation space dimensionality (125 for AC env).
        act_dim (int): Action space dimensionality (3 for AC env).
        device (torch.device): Torch device.
    """

    def __init__(self, config: dict, obs_dim: int, act_dim: int,
                 device: torch.device):
        self.device  = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        sac_cfg  = config["sac"]
        net_cfg  = config["network"]
        path_cfg = config["paths"]["sac"]

        self.gamma           = sac_cfg["gamma"]
        self.tau             = sac_cfg["tau"]                # soft update rate
        self.batch_size      = sac_cfg["batch_size"]
        self.learning_rate   = sac_cfg["learning_rate"]
        self.update_every    = sac_cfg["update_every"]       # steps between updates
        self.updates_per_step= sac_cfg["updates_per_step"]   # gradient steps per update
        self.warmup_steps    = sac_cfg["warmup_steps"]       # random actions before learning
        self.target_entropy  = sac_cfg.get("target_entropy", -float(act_dim))
        self.save_buffer_enabled = sac_cfg.get("save_buffer", False)  # persist buffer to disk
        self.results_dir     = path_cfg["results_dir"]
        self.best_model_path = path_cfg["best_model_path"]

        hidden_dims = net_cfg["hidden_dims"]
        activation  = net_cfg.get("activation", "relu")

        os.makedirs(self.results_dir, exist_ok=True)

        # --- Actor ---
        self.actor = SACActor(obs_dim, act_dim, hidden_dims, activation).to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate)

        # --- Twin critics (Q1, Q2) ---
        self.critic1 = SACQNetwork(obs_dim, act_dim, hidden_dims, activation).to(device)
        self.critic2 = SACQNetwork(obs_dim, act_dim, hidden_dims, activation).to(device)
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=self.learning_rate)

        # --- Target critics (not trained directly, soft-updated) ---
        self.target_critic1 = SACQNetwork(obs_dim, act_dim, hidden_dims, activation).to(device)
        self.target_critic2 = SACQNetwork(obs_dim, act_dim, hidden_dims, activation).to(device)
        self._hard_update_targets()   # initialise targets = critics

        # --- Automatic entropy tuning ---
        # log_alpha is the learnable log temperature parameter
        self.log_alpha = torch.tensor(
            np.log(sac_cfg.get("init_alpha", 0.2)),
            dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.learning_rate)

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(
            capacity=sac_cfg["buffer_size"],
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=device,
        )

        self._best_eval_reward = float("-inf")
        self._step_count = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def choose_action(self, obs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Selects an action for the given observation.

        During warmup (first warmup_steps), returns purely random actions to
        populate the replay buffer. After warmup, samples from the stochastic
        policy during training, or uses the deterministic mean for evaluation.

        Args:
            obs (np.ndarray): Current observation, shape (obs_dim,).
            training (bool): Stochastic (True) vs deterministic (False).

        Returns:
            np.ndarray: Action clipped to [-1, 1], shape (act_dim,).
        """
        if training and self._step_count < self.warmup_steps:
            return np.random.uniform(-1.0, 1.0, size=(self.act_dim,)).astype(np.float32)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        if training:
            return self.actor.get_action(obs_t)
        else:
            return self.actor.get_deterministic_action(obs_t)

    # ------------------------------------------------------------------
    # Transition storage
    # ------------------------------------------------------------------
    def store_transition(self, obs: np.ndarray, action: np.ndarray,
                         reward: float, next_obs: np.ndarray, done: bool) -> None:
        """
        Stores a single transition in the replay buffer and increments
        the step counter.

        Args:
            obs (np.ndarray): Current observation, shape (obs_dim,).
            action (np.ndarray): Action taken, shape (act_dim,).
            reward (float): Reward received.
            next_obs (np.ndarray): Next observation, shape (obs_dim,).
            done (bool): Whether this transition ended the episode.
        """
        self.buffer.add(obs, action, reward, next_obs, done)
        self._step_count += 1

    # ------------------------------------------------------------------
    # SAC update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """
        Performs one or more SAC gradient updates if conditions are met.

        Conditions:
            - Buffer must have at least batch_size transitions.
            - Step count must be a multiple of update_every.

        Runs updates_per_step gradient steps per call. Each step updates:
            1. Twin critics (Bellman backup with target nets)
            2. Actor (maximize Q - alpha * log_prob)
            3. Temperature alpha (track target entropy)
            4. Target networks (soft update)

        Returns:
            Dict[str, float]: Metrics — critic1_loss, critic2_loss,
                actor_loss, alpha_loss, alpha. Empty dict if no update ran.
        """
        if (len(self.buffer) < self.batch_size or
                self._step_count % self.update_every != 0):
            return {}

        metrics = {
            "critic1_loss": 0.0,
            "critic2_loss": 0.0,
            "actor_loss":   0.0,
            "alpha_loss":   0.0,
            "alpha":        0.0,
        }

        for _ in range(self.updates_per_step):
            obs_b, act_b, rew_b, next_obs_b, done_b = self.buffer.sample(
                self.batch_size)

            alpha = self.log_alpha.exp().detach()

            # ---- 1. Critic update ----------------------------------------
            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_obs_b)
                # Target Q: take min of two target critics (clipped double-Q)
                target_q1 = self.target_critic1(next_obs_b, next_action)
                target_q2 = self.target_critic2(next_obs_b, next_action)
                target_q  = torch.min(target_q1, target_q2) - alpha * next_log_prob
                # Bellman backup
                target_q  = rew_b + self.gamma * (1.0 - done_b) * target_q

            current_q1 = self.critic1(obs_b, act_b)
            current_q2 = self.critic2(obs_b, act_b)
            critic1_loss = F.mse_loss(current_q1, target_q)
            critic2_loss = F.mse_loss(current_q2, target_q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # ---- 2. Actor update -----------------------------------------
            new_action, log_prob = self.actor(obs_b)
            q1_new = self.critic1(obs_b, new_action)
            q2_new = self.critic2(obs_b, new_action)
            q_new  = torch.min(q1_new, q2_new)
            # Maximise Q - alpha * log_prob  (minimise negative)
            actor_loss = (alpha * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ---- 3. Alpha (temperature) update ---------------------------
            alpha_loss = -(
                self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # ---- 4. Soft update target critics ---------------------------
            self._soft_update_targets()

            metrics["critic1_loss"] += critic1_loss.item()
            metrics["critic2_loss"] += critic2_loss.item()
            metrics["actor_loss"]   += actor_loss.item()
            metrics["alpha_loss"]   += alpha_loss.item()
            metrics["alpha"]        += self.log_alpha.exp().item()

        n = self.updates_per_step
        return {k: v / n for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Target network updates
    # ------------------------------------------------------------------
    def _hard_update_targets(self) -> None:
        """Copies critic weights directly into target critics (used at init)."""
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def _soft_update_targets(self) -> None:
        """
        Polyak averaging: target = tau * current + (1 - tau) * target.
        Applied after every gradient step.
        """
        for param, target_param in zip(
            self.critic1.parameters(), self.target_critic1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(
            self.critic2.parameters(), self.target_critic2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # ------------------------------------------------------------------
    # Buffer persistence
    # ------------------------------------------------------------------
    def save_buffer(self, path: str) -> None:
        """
        Saves the replay buffer to disk as a compressed numpy archive.

        Only the filled portion of the buffer is written (up to buffer.size),
        so the file size reflects actual collected transitions, not capacity.

        Args:
            path (str): Output .npz file path.
        """
        np.savez_compressed(
            path,
            obs=self.buffer.obs[:self.buffer.size],
            actions=self.buffer.actions[:self.buffer.size],
            rewards=self.buffer.rewards[:self.buffer.size],
            next_obs=self.buffer.next_obs[:self.buffer.size],
            dones=self.buffer.dones[:self.buffer.size],
            ptr=np.array([self.buffer.ptr]),
            size=np.array([self.buffer.size]),
        )
        print(f"[Buffer] Saved {self.buffer.size} transitions -> {path}")

    def load_buffer(self, path: str) -> None:
        """
        Loads a replay buffer from disk into the current buffer.

        If the file does not exist (e.g. first run), silently skips so that
        load_model() can be called unconditionally on resume.

        Args:
            path (str): Path to a .npz file saved by save_buffer().
        """
        if not os.path.isfile(path):
            print(f"[Buffer] No buffer file found at {path}, starting fresh.")
            return
        data = np.load(path)
        size = int(data["size"][0])
        self.buffer.obs[:size]      = data["obs"]
        self.buffer.actions[:size]  = data["actions"]
        self.buffer.rewards[:size]  = data["rewards"]
        self.buffer.next_obs[:size] = data["next_obs"]
        self.buffer.dones[:size]    = data["dones"]
        self.buffer.ptr             = int(data["ptr"][0])
        self.buffer.size            = size
        print(f"[Buffer] Loaded {size} transitions <- {path}")

    def _buffer_path(self, model_path: str) -> str:
        """Derives the buffer .npz path from a model .pt path."""
        return model_path.replace(".pt", "_buffer.npz")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        """
        Saves all network weights, optimizers, and training state.
        If save_buffer is enabled in config, also saves the replay buffer
        to a companion .npz file alongside the checkpoint.

        Args:
            path (str): Output .pt file path.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "actor_state_dict":          self.actor.state_dict(),
            "critic1_state_dict":        self.critic1.state_dict(),
            "critic2_state_dict":        self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer":           self.actor_optimizer.state_dict(),
            "critic1_optimizer":         self.critic1_optimizer.state_dict(),
            "critic2_optimizer":         self.critic2_optimizer.state_dict(),
            "alpha_optimizer":           self.alpha_optimizer.state_dict(),
            "log_alpha":                 self.log_alpha.item(),
            "best_eval_reward":          self._best_eval_reward,
            "step_count":                self._step_count,
        }, path)
        print(f"[Checkpoint] SAC model saved -> {path}")

        if self.save_buffer_enabled:
            self.save_buffer(self._buffer_path(path))

    def load_model(self, path: str) -> None:
        """
        Loads all network weights, optimizers, and training state.
        If save_buffer is enabled in config, also loads the replay buffer
        from the companion .npz file if it exists.

        Args:
            path (str): Path to checkpoint saved by save_model().

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic1.load_state_dict(ckpt["critic1_state_dict"])
        self.critic2.load_state_dict(ckpt["critic2_state_dict"])
        self.target_critic1.load_state_dict(ckpt["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(ckpt["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(ckpt["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(ckpt["critic2_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self.log_alpha = torch.tensor(
            ckpt["log_alpha"], dtype=torch.float32,
            device=self.device, requires_grad=True)
        self._best_eval_reward = ckpt.get("best_eval_reward", float("-inf"))
        self._step_count       = ckpt.get("step_count", 0)
        print(f"[Checkpoint] SAC model loaded <- {path}")

        if self.save_buffer_enabled:
            self.load_buffer(self._buffer_path(path))

    def save_best_if_improved(self, eval_reward: float) -> bool:
        """
        Saves the model only if eval_reward exceeds the previous best.

        Args:
            eval_reward (float): Mean reward from latest evaluation.

        Returns:
            bool: True if a new best was saved.
        """
        if eval_reward > self._best_eval_reward:
            self._best_eval_reward = eval_reward
            self.save_model(self.best_model_path)
            print(f"[Best] New best SAC model! Eval reward: {eval_reward:.2f}")
            return True
        return False