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
from utils import get_device, print_device_info, set_global_seed


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
    pass


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
    pass


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
    pass


def main() -> None:
    """
    Main entry point. Loads config, sets seeds, then runs Task 1 environment
    confirmation (space info + single random step).

    Returns:
        None
    """
    config = load_config("config.yaml")
    set_global_seed(config["training"]["seed"])

    confirm_environment(config)


if __name__ == "__main__":
    main()
