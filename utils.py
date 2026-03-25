"""
utils.py
========
Shared utilities for the Autonomous Racing PPO project.

Provides config loading, device detection, seed management, CSV logging,
and training history plotting. Imported by training_script.py and ppo_agent.py.
"""

import os
import csv
import glob
import random
import yaml
import numpy as np
import torch
from typing import Dict, List, Any, Optional


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads and parses the YAML configuration file.

    Args:
        path (str): Path to config.yaml. Defaults to 'config.yaml'.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist at path.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """
    Detects and returns the best available torch compute device.
    Priority order: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """
    Prints the selected compute device and relevant hardware information.

    Args:
        device (torch.device): The device selected by get_device().

    Returns:
        None
    """
    print(f"\n[Device] Using: {device}")
    if device.type == "cuda":
        print(f"  GPU Name : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM     : {mem:.1f} GB")
    elif device.type == "mps":
        print("  Backend  : Apple Metal Performance Shaders (MPS)")
    else:
        print("  Backend  : CPU")


def set_global_seed(seed: int) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): Integer seed value (e.g., 42).

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Seed] Global seed set to {seed}")


class TrainingLogger:
    """
    Logs training metrics to the console and appends them to a CSV file
    for later analysis and plotting.

    Args:
        csv_path (str): Path to the output CSV file.
        fields (List[str]): Column names for the CSV header.
    """

    def __init__(self, csv_path: str, fields: List[str]):
        """
        Initializes the TrainingLogger and writes the CSV header if the
        file does not already exist (allows safe resumption of training).

        Args:
            csv_path (str): File path for the training history CSV.
            fields (List[str]): List of metric names (column headers).
        """
        self.csv_path = csv_path
        self.fields = fields
        dir_ = os.path.dirname(csv_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        if not os.path.isfile(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, metrics: Dict[str, Any], print_to_console: bool = True) -> None:
        """
        Logs a dictionary of metrics to CSV and optionally to the console.

        Args:
            metrics (Dict[str, Any]): Dictionary mapping field names to values.
            print_to_console (bool): If True, also prints metrics to stdout.

        Returns:
            None
        """
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(metrics)
        if print_to_console:
            parts = [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
            print("  " + " | ".join(parts))


def plot_training_history(csv_path: str, plots_dir: str) -> None:
    """
    Reads the training history CSV and saves plots for key metrics.

    Generates episode_rewards.png and losses.png in plots_dir.
    Uses a moving-average smoother on reward curves to reveal trends.

    Args:
        csv_path (str): Path to the training history CSV file.
        plots_dir (str): Directory to save output plots.

    Returns:
        None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[Plot] No data to plot.")
        return

    def _get(key: str) -> List[float]:
        return [float(r[key]) for r in rows if key in r and r[key] != ""]

    steps         = _get("timestep")
    ep_rewards    = _get("episode_reward")
    policy_losses = _get("policy_loss")
    value_losses  = _get("value_loss")
    entropies     = _get("entropy")

    # --- Episode reward curve ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps[: len(ep_rewards)], ep_rewards,
            alpha=0.25, color="steelblue", label="Raw reward")
    if len(ep_rewards) >= 10:
        window = min(50, max(10, len(ep_rewards) // 10))
        smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1: len(smoothed) + window - 1], smoothed,
                color="steelblue", linewidth=2,
                label=f"Smoothed (window={window})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Reward")
    ax.set_title("PPO Training — Episode Reward (Monza / BMW Z4 GT3)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "episode_rewards.png"), dpi=150)
    plt.close(fig)

    # --- Loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, label, color in zip(
        axes,
        [policy_losses, value_losses, entropies],
        ["Policy Loss (Actor)", "Value Loss (Critic)", "Entropy"],
        ["tomato", "orange", "mediumseagreen"],
    ):
        ax.plot(steps[: len(data)], data, color=color, linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(label)
        ax.set_title(f"PPO — {label}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "losses.png"), dpi=150)
    plt.close(fig)

    print(f"[Plot] Saved plots to {plots_dir}")


def latest_checkpoint(checkpoint_prefix: str) -> Optional[str]:
    """
    Finds the most recently saved checkpoint file with the given prefix.

    Checkpoint files are expected to be named:
        {checkpoint_prefix}_{step}.pt   (e.g., ppo_results/ppo_checkpoint_50000.pt)

    Args:
        checkpoint_prefix (str): Common prefix of checkpoint filenames
            (e.g., 'ppo_results/ppo_checkpoint').

    Returns:
        Optional[str]: Path to the latest checkpoint, or None if not found.
    """
    pattern = f"{checkpoint_prefix}_*.pt"
    files = glob.glob(pattern)
    if not files:
        return None

    def _step(path: str) -> int:
        stem = os.path.basename(path).replace(".pt", "")
        parts = stem.split("_")
        try:
            return int(parts[-1])
        except ValueError:
            return 0

    files.sort(key=_step)
    return files[-1]
