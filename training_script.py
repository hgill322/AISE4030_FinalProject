"""
training_script.py
==================
Main entry point for training, evaluation, and deployment of the PPO or SAC
agent on the Assetto Corsa racing environment.

Switch between algorithms by setting `algorithm: "ppo"` or `algorithm: "sac"`
in config.yaml. Everything else (environment, evaluation, deployment) is shared.

Usage:
    python training_script.py
"""

import numpy as np
import torch

from environment import make_env, load_config
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from utils import get_device, print_device_info, set_global_seed, TrainingLogger, latest_checkpoint


# ---------------------------------------------------------------------------
# Environment factory (no reward shaping)
# ---------------------------------------------------------------------------
class SpinTerminationWrapper:
    """
    Wraps the AC env to detect sustained spinning (donuts) via the yaw rate channel
    (obs index 7, normalized angular_velocity_y) and forces an early
    episode termination with a penalty if the car spins hard for too long.
    """
    _YAW_IDX = 7

    def __init__(self, env, spin_threshold: float = 0.15, max_spin_seconds: float = 7.0, control_freq: int = 25):
        self.env = env
        self.spin_threshold = spin_threshold
        self.max_spin_steps = int(max_spin_seconds * control_freq)
        self.spin_counter = 0

    def reset(self, **kwargs):
        self.spin_counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Unpack the 4 items directly
        obs, reward, done, info = self.env.step(action)

        # Extract yaw rate directly from the observation array
        yaw_rate = abs(obs[self._YAW_IDX])

        if yaw_rate > self.spin_threshold:
            self.spin_counter += 1
        else:
            self.spin_counter = 0

        # Terminate if spinning for too long
        if self.spin_counter >= self.max_spin_steps:
            done = True
            reward -= 50.0  # Severe penalty for spinning out
            info["termination_reason"] = "spun_out"

        # Return the modified 4 items
        return obs, reward, done, info

    def __getattr__(self, name):
        """Pass any missing attributes (like action_space) down to the base env."""
        return getattr(self.env, name)

def _make_env(config: dict) -> SpinTerminationWrapper:
    """Creates the environment and applies the spin termination wrapper."""
    return SpinTerminationWrapper(make_env(config))


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def make_agent(config: dict, device: torch.device):
    """
    Instantiates the correct agent based on config['algorithm'].

    Args:
        config (dict): Full config dictionary.
        device (torch.device): Compute device.

    Returns:
        PPOAgent or SACAgent.
    """
    obs_dim   = config["environment"]["obs_dim"]
    act_dim   = config["environment"]["act_dim"]
    algorithm = config.get("algorithm", "ppo").lower()

    if algorithm == "sac":
        print("[Agent] Using SAC (Soft Actor-Critic)")
        return SACAgent(config=config, obs_dim=obs_dim, act_dim=act_dim, device=device)
    else:
        print("[Agent] Using PPO (Proximal Policy Optimization)")
        return PPOAgent(config=config, obs_dim=obs_dim, act_dim=act_dim, device=device)


# ---------------------------------------------------------------------------
# Task 1: environment confirmation
# ---------------------------------------------------------------------------
def confirm_environment(config: dict) -> None:
    """
    Initializes the environment, prints space information, and confirms
    that at least one environment step executes successfully.

    Args:
        config (dict): Full config dictionary loaded from config.yaml.
    """
    print("\n" + "=" * 60)
    print("  TASK 1: Environment Setup & API Confirmation")
    print("=" * 60)

    env = _make_env(config)
    env.print_space_info()

    device = get_device()
    print_device_info(device)

    print("\n[Step Test] Resetting environment...")
    try:
        obs = env.reset()
        print(f"  reset() -> obs shape: {obs.shape}, dtype: {obs.dtype}")

        action = env.action_space.sample()
        print(f"\n[Step Test] Applying random action: {action}")
        next_obs, reward, done, info = env.step(action)
        print(f"  step()  -> next_obs shape: {next_obs.shape}")
        print(f"             reward: {reward:.4f}")
        print(f"             done  : {done}")
        print(f"             info  : {info}")
        print("\n  [PASS] Environment step executed successfully.")
    except Exception as e:
        print(f"\n  [FAIL] Environment step failed: {e}")
        print("  Ensure Assetto Corsa is running with the AC plugin active.")
    finally:
        env.close()

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------
def train_ppo(config: dict, agent: PPOAgent, device: torch.device, resume_path: str = "") -> None:
    """
    Runs the PPO training loop: collect n_steps rollout -> update -> repeat.

    Args:
        config (dict): Full config dictionary.
        agent (PPOAgent): Initialised PPO agent.
        device (torch.device): Compute device.
    """
    train_cfg = config["training"]
    algorithm = config.get("algorithm", "ppo").lower()
    path_cfg  = config["paths"][algorithm]

    total_timesteps = train_cfg["total_timesteps"]
    eval_interval   = train_cfg["eval_interval"]
    save_interval   = train_cfg["save_interval"]
    log_interval    = train_cfg["log_interval"]
    eval_episodes   = train_cfg["eval_episodes"]

    env = _make_env(config)

    logger = TrainingLogger(
        csv_path=path_cfg["training_history"],
        fields=["timestep", "episode", "episode_reward", "episode_length",
                "policy_loss", "value_loss", "entropy", "entropy_coef"],
    )

    obs            = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_num    = 0
    last_done      = False

    # If resuming, parse the step count from the checkpoint filename so that
    # logging, checkpoint saves, and evals continue from the right offset.
    total_steps = 0
    if resume_path:
        try:
            total_steps = int(resume_path.split("_")[-1].replace(".pt", ""))
            print(f"[PPO Train] Resuming from step {total_steps:,}")
        except (ValueError, IndexError):
            total_steps = 0

    print(f"\n[PPO Train] Starting for {total_timesteps:,} timesteps...")

    while total_steps < total_timesteps:
        # Collect rollout
        for _ in range(agent.n_steps):
            action, log_prob, value = agent.choose_action(obs, training=True)
            next_obs, reward, last_done, _ = env.step(action)
            agent.store_transition(obs, action, reward, last_done, log_prob, value)

            obs             = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps    += 1

            if last_done:
                obs = env.reset()
                episode_num += 1
                if total_steps % log_interval < agent.n_steps:
                    print(f"  [Ep {episode_num:4d} | Step {total_steps:8,}] "
                          f"reward={episode_reward:.2f}  len={episode_length}")
                episode_reward = 0.0
                episode_length = 0

        # PPO update
        metrics = agent.update(last_obs=obs, last_done=last_done)
        agent.decay_entropy_coef(decay_factor=0.9998)

        logger.log({
            "timestep":       total_steps,
            "episode":        episode_num,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "policy_loss":    metrics["policy_loss"],
            "value_loss":     metrics["value_loss"],
            "entropy":        metrics["entropy"],
            "entropy_coef":   agent.entropy_coef,
        }, print_to_console=False)

        if total_steps % eval_interval < agent.n_steps:
            mean_reward = evaluate(config, agent, device, n_episodes=eval_episodes)
            agent.save_best_if_improved(mean_reward)
            obs = env.reset()

        if total_steps % save_interval < agent.n_steps:
            ckpt_path = f"{path_cfg['checkpoint_prefix']}_{total_steps}.pt"
            agent.save_model(ckpt_path)

    env.close()
    print(f"\n[PPO Train] Complete. {total_steps:,} steps over {episode_num} episodes.")


# ---------------------------------------------------------------------------
# SAC training loop
# ---------------------------------------------------------------------------
def train_sac(config: dict, agent: SACAgent, device: torch.device, resume_path: str = "") -> None:
    """
    Runs the SAC training loop: step -> store -> update every N steps.

    SAC updates happen every step (after warmup), unlike PPO's rollout-batch
    approach. The replay buffer accumulates all transitions.

    Args:
        config (dict): Full config dictionary.
        agent (SACAgent): Initialised SAC agent.
        device (torch.device): Compute device.
    """
    train_cfg = config["training"]
    algorithm = config.get("algorithm", "sac").lower()
    path_cfg  = config["paths"][algorithm]

    total_timesteps = train_cfg["total_timesteps"]
    eval_interval   = train_cfg["eval_interval"]
    save_interval   = train_cfg["save_interval"]
    log_interval    = train_cfg["log_interval"]
    eval_episodes   = train_cfg["eval_episodes"]

    env = _make_env(config)

    logger = TrainingLogger(
        csv_path=path_cfg["training_history"],
        fields=["timestep", "episode", "episode_reward", "episode_length",
                "critic1_loss", "critic2_loss", "actor_loss", "alpha"],
    )

    obs            = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_num    = 0

    # If resuming, parse the step count from the checkpoint filename.
    total_steps = 0
    if resume_path:
        try:
            total_steps = int(resume_path.split("_")[-1].replace(".pt", ""))
            print(f"[SAC Train] Resuming from step {total_steps:,}")
        except (ValueError, IndexError):
            total_steps = 0

    # Keep a recent loss dict for logging (SAC update() returns {} when skipped)
    last_metrics = {
        "critic1_loss": 0.0, "critic2_loss": 0.0,
        "actor_loss": 0.0, "alpha": 0.2,
    }

    print(f"\n[SAC Train] Starting for {total_timesteps:,} timesteps...")
    print(f"            Warmup: {agent.warmup_steps:,} random steps before learning.")

    while total_steps < total_timesteps:
        action = agent.choose_action(obs, training=True)
        next_obs, reward, done, _ = env.step(action)

        agent.store_transition(obs, action, reward, next_obs, done)
        metrics = agent.update()
        if metrics:
            last_metrics = metrics

        obs             = next_obs
        episode_reward += reward
        episode_length += 1
        total_steps    += 1
        
        if done:
            obs = env.reset()
            episode_num += 1

            if episode_num % max(1, log_interval // 200) == 0:
                print(f"  [Ep {episode_num:4d} | Step {total_steps:8,}] "
                      f"reward={episode_reward:.2f}  len={episode_length}  "
                      f"alpha={last_metrics['alpha']:.4f}")

            logger.log({
                "timestep":       total_steps,
                "episode":        episode_num,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                "critic1_loss":   last_metrics["critic1_loss"],
                "critic2_loss":   last_metrics["critic2_loss"],
                "actor_loss":     last_metrics["actor_loss"],
                "alpha":          last_metrics["alpha"],
            }, print_to_console=False)

            episode_reward = 0.0
            episode_length = 0

        if total_steps % eval_interval == 0 and total_steps > 0:
            mean_reward = evaluate(config, agent, device, n_episodes=eval_episodes)
            agent.save_best_if_improved(mean_reward)
            obs = env.reset()

        if total_steps % save_interval == 0 and total_steps > 0:
            ckpt_path = f"{path_cfg['checkpoint_prefix']}_{total_steps}.pt"
            agent.save_model(ckpt_path)

    env.close()
    print(f"\n[SAC Train] Complete. {total_steps:,} steps over {episode_num} episodes.")


# ---------------------------------------------------------------------------
# Shared evaluation loop
# ---------------------------------------------------------------------------
def evaluate(config: dict, agent, device: torch.device,
             n_episodes: int = 5) -> float:
    """
    Evaluates the agent in deterministic exploitation mode.

    Works with both PPO and SAC agents — both expose
    choose_action(obs, training=False).

    Args:
        config (dict): Full config dictionary.
        agent: PPOAgent or SACAgent.
        device (torch.device): Compute device.
        n_episodes (int): Number of evaluation episodes.

    Returns:
        float: Mean episode reward.
    """
    env = _make_env(config)
    episode_rewards = []

    for _ in range(n_episodes):
        obs      = env.reset()
        ep_reward = 0.0
        done     = False
        while not done:
            action = agent.choose_action(obs, training=False)
            # PPOAgent returns (action, log_prob, value); SAC returns just action
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        episode_rewards.append(ep_reward)

    env.close()
    mean_reward = float(np.mean(episode_rewards))
    std_reward  = float(np.std(episode_rewards))
    print(f"=== [Eval] === {n_episodes} eps | mean: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------
def deploy(config: dict, device: torch.device) -> None:
    """
    Loads the best saved model and runs it in exploitation mode.

    Args:
        config (dict): Full config dictionary.
        device (torch.device): Compute device.
    """
    algorithm = config.get("algorithm", "ppo").lower()
    agent     = make_agent(config, device)
    best_path = config["paths"][algorithm]["best_model_path"]
    agent.load_model(best_path)

    eval_episodes = config["training"].get("eval_episodes", 5)
    mean_reward   = evaluate(config, agent, device, n_episodes=eval_episodes)
    print(f"[Deploy] Mean reward over {eval_episodes} episodes: {mean_reward:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Loads config, sets seeds, prints env info, then trains and/or evaluates
    based on config mode flags and the selected algorithm.
    """
    config    = load_config("config.yaml")
    algorithm = config.get("algorithm", "ppo").lower()

    set_global_seed(config["training"]["seed"])
    confirm_environment(config)

    device = get_device()

    if config["mode"].get("train", True):
        agent = make_agent(config, device)

        # Resume from checkpoint if configured — resolve path once so
        # train_ppo / train_sac can parse the step count from the filename.
        path_cfg    = config["paths"][algorithm]
        resume_path = ""
        if config["mode"].get("resume"):
            resume_path = config["mode"].get("resume_path", "")
            if not resume_path:
                resume_path = latest_checkpoint(path_cfg["checkpoint_prefix"]) or ""
            if resume_path:
                agent.load_model(resume_path)

        if algorithm == "sac":
            train_sac(config, agent, device, resume_path=resume_path)
        else:
            train_ppo(config, agent, device, resume_path=resume_path)

    if config["mode"].get("evaluate", False):
        deploy(config, device)


if __name__ == "__main__":
    main()