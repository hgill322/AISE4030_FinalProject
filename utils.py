"""
utils.py
========
Shared utilities for the Autonomous Racing PPO project.

Provides config loading, device detection, seed management, CSV logging,
and training history plotting. Imported by training_script.py and ppo_agent.py.
"""

import os
import csv
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
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """
    Detects and returns the best available torch compute device.
    Priority order: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device: The selected device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


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
        Initializes the TrainingLogger and writes the CSV header if needed.

        Args:
            csv_path (str): File path for the training history CSV.
            fields (List[str]): List of metric names (column headers).
        """
        pass

    def log(self, metrics: Dict[str, Any], print_to_console: bool = True) -> None:
        """
        Logs a dictionary of metrics to CSV and optionally to the console.

        Args:
            metrics (Dict[str, Any]): Dictionary mapping field names to values.
            print_to_console (bool): If True, also prints metrics to stdout.

        Returns:
            None
        """
        pass


def plot_training_history(csv_path: str, plots_dir: str) -> None:
    """
    Reads the training history CSV and saves plots for key metrics.

    Generates episode_rewards.png and losses.png in plots_dir.

    Args:
        csv_path (str): Path to the training history CSV file.
        plots_dir (str): Directory to save output plots.

    Returns:
        None
    """
    pass


def latest_checkpoint(checkpoint_prefix: str) -> Optional[str]:
    """
    Finds the most recently saved checkpoint file with the given prefix.

    Args:
        checkpoint_prefix (str): Common prefix of checkpoint filenames
            (e.g., 'ppo_results/ppo_checkpoint').

    Returns:
        Optional[str]: Path to the latest checkpoint, or None if not found.
    """
    pass
