"""
training_script.py
==================
Main entry point for training, evaluation, and deployment of the PPO agent
on the Assetto Corsa racing environment.

Responsibilities:
    - Load configuration from config.yaml.
    - Initialize the environment and verify the API (Task 1).
    - Instantiate the PPOAgent and rollout buffer.
    - Run the main training loop (collect rollouts -> PPO update -> log -> save).
    - Run the evaluation loop.
    - Deploy a trained model in pure exploitation mode.

Usage:
    python training_script.py
"""

import numpy as np
import torch

from environment import make_env, load_config
from ppo_agent import PPOAgent
from utils import get_device, print_device_info, set_global_seed, TrainingLogger, latest_checkpoint


def confirm_environment(config: dict) -> None:
    """
    Initializes the environment, prints space information, and confirms
    that at least one environment step executes successfully.

    Satisfies Task 1 of Phase 2: prints observation space, action space,
    device info, and step confirmation to the console.

    Args:
        config (dict): Full config dictionary loaded from config.yaml.

    Returns:
        None
    """
    print("\n" + "=" * 60)
    print("  TASK 1: Environment Setup & API Confirmation")
    print("=" * 60)

    env = make_env(config)
    env.print_space_info()

    device = get_device()
    print_device_info(device)

    print("\n[Step Test] Resetting environment...")
    try:
        obs = env.reset()
        print(f"  reset() -> obs shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"  obs sample: {obs}")

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


def train(config: dict, device: torch.device) -> None:
    """
    Runs the main PPO training loop.

    Collects rollouts of n_steps transitions, calls the PPO update, logs
    metrics, saves checkpoints, and runs periodic evaluation.
    Terminates after total_timesteps environment steps.

    Args:
        config (dict): Full config dictionary.
        device (torch.device): Compute device for tensor operations.

    Returns:
        None
    """
    train_cfg = config["training"]
    path_cfg  = config["paths"]

    total_timesteps = train_cfg["total_timesteps"]
    eval_interval   = train_cfg["eval_interval"]
    save_interval   = train_cfg["save_interval"]
    log_interval    = train_cfg["log_interval"]
    eval_episodes   = train_cfg["eval_episodes"]

    env = make_env(config)
    obs_dim = config["environment"]["obs_dim"]
    act_dim = config["environment"]["act_dim"]

    agent = PPOAgent(config=config, obs_dim=obs_dim, act_dim=act_dim, device=device)

    # Resume from checkpoint if configured
    if config["mode"].get("resume") and config["mode"].get("resume_path"):
        agent.load_model(config["mode"]["resume_path"])
    elif config["mode"].get("resume"):
        ckpt = latest_checkpoint(path_cfg["checkpoint_prefix"])
        if ckpt:
            agent.load_model(ckpt)

    # CSV logger
    logger = TrainingLogger(
        csv_path=path_cfg["training_history"],
        fields=["timestep", "episode", "episode_reward", "episode_length",
                "policy_loss", "value_loss", "entropy", "entropy_coef"],
    )

    obs = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episode_num    = 0
    total_steps    = 0

    print(f"\n[Train] Starting PPO training for {total_timesteps:,} timesteps...")

    while total_steps < total_timesteps:
        # --- Collect one rollout of n_steps ---
        for _ in range(agent.n_steps):
            action, log_prob, value = agent.choose_action(obs, training=True)
            next_obs, reward, done, _ = env.step(action)

            agent.store_transition(obs, action, reward, done, log_prob, value)

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            total_steps    += 1

            if done:
                obs = env.reset()
                episode_num += 1
                if total_steps % log_interval < agent.n_steps:
                    print(f"  [Ep {episode_num:4d} | Step {total_steps:8,}] "
                          f"reward={episode_reward:.2f}  len={episode_length}")
                episode_reward = 0.0
                episode_length = 0

        # --- PPO update ---
        metrics = agent.update(last_obs=obs, last_done=False)

        # --- Entropy annealing ---
        agent.decay_entropy_coef()

        # --- Log metrics ---
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

        # --- Periodic evaluation ---
        if total_steps % eval_interval < agent.n_steps:
            mean_reward = evaluate(config, agent, device, n_episodes=eval_episodes)
            agent.save_best_if_improved(mean_reward)
            obs = env.reset()

        # --- Periodic checkpoint save ---
        if total_steps % save_interval < agent.n_steps:
            ckpt_path = f"{path_cfg['checkpoint_prefix']}_{total_steps}.pt"
            agent.save_model(ckpt_path)

    env.close()
    print(f"\n[Train] Training complete. {total_steps:,} steps over {episode_num} episodes.")


def evaluate(config: dict, agent: PPOAgent, device: torch.device,
             n_episodes: int = 5) -> float:
    """
    Evaluates the agent in pure exploitation mode (no exploration).

    Runs n_episodes full episodes using the deterministic policy mean,
    and returns the mean episode reward.

    Args:
        config (dict): Full config dictionary.
        agent (PPOAgent): The agent to evaluate.
        device (torch.device): Compute device.
        n_episodes (int): Number of evaluation episodes to run.

    Returns:
        float: Mean episode reward across all evaluation episodes.
    """
    env = make_env(config)
    episode_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # Deterministic exploitation: training=False uses policy mean
            action, _, _ = agent.choose_action(obs, training=False)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward

        episode_rewards.append(ep_reward)

    env.close()
    mean_reward = float(np.mean(episode_rewards))
    std_reward  = float(np.std(episode_rewards))
    print(f"=== [Eval] === {n_episodes} episodes | mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    return mean_reward


def deploy(config: dict, device: torch.device) -> None:
    """
    Loads the best saved model and runs it in pure exploitation mode
    for visual demonstration and deployment.

    Args:
        config (dict): Full config dictionary.
        device (torch.device): Compute device.

    Returns:
        None
    """
    obs_dim = config["environment"]["obs_dim"]
    act_dim = config["environment"]["act_dim"]
    best_path = config["paths"]["best_model_path"]

    agent = PPOAgent(config=config, obs_dim=obs_dim, act_dim=act_dim, device=device)
    agent.load_model(best_path)

    eval_episodes = config["training"].get("eval_episodes", 5)
    mean_reward = evaluate(config, agent, device, n_episodes=eval_episodes)
    print(f"[Deploy] Mean reward over {eval_episodes} exploitation episodes: {mean_reward:.2f}")


def main() -> None:
    """
    Main entry point. Loads config, sets seeds, then runs Task 1 environment
    confirmation (space info + single random step). Proceeds to training
    and/or evaluation based on config mode flags.

    Returns:
        None
    """
    config = load_config("config.yaml")
    set_global_seed(config["training"]["seed"])

    confirm_environment(config)

    device = get_device()

    if config["mode"].get("train", True):
        train(config, device)

    if config["mode"].get("evaluate", False):
        deploy(config, device)


if __name__ == "__main__":
    main()
