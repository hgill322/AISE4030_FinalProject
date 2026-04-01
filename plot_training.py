"""
plot_training.py
================
Plots training performance from the CSV logs produced by training_script.py.

Usage:
    python plot_training.py                        # auto-detects ppo or sac from config.yaml
    python plot_training.py --algo ppo
    python plot_training.py --algo sac
    python plot_training.py --algo ppo --csv ppo_results/training_history.csv

Outputs PNGs to the plots_dir defined in config.yaml (ppo_results/plots/ or sac_results/plots/).

CSV format note:
    The PPO logger writes 8 columns but only 7 headers, so this script
    re-assigns column names on load:
        timestep, update, episode_length, mean_eval_reward,
        policy_loss, value_loss, entropy, entropy_coef
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def rolling(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
STYLE = {
    "figure.facecolor":  "#0d0d0d",
    "axes.facecolor":    "#141414",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#cccccc",
    "axes.titlecolor":   "#ffffff",
    "xtick.color":       "#888888",
    "ytick.color":       "#888888",
    "grid.color":        "#222222",
    "grid.linestyle":    "--",
    "text.color":        "#cccccc",
    "legend.facecolor":  "#1a1a1a",
    "legend.edgecolor":  "#333333",
}

ACCENT  = "#00d4ff"   # cyan
ACCENT2 = "#ff6b35"   # orange
ACCENT3 = "#7fff6b"   # green
ACCENT4 = "#d46bff"   # purple


def _apply_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def save_fig(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# PPO plots
# ---------------------------------------------------------------------------
def plot_ppo(df: pd.DataFrame, plots_dir: str, window: int = 20) -> None:
    x = df["timestep"]

    # Pick whichever reward column exists
    reward_col = "mean_eval_reward" if "mean_eval_reward" in df.columns else "episode_reward"
    reward_label = "Mean Eval Reward" if reward_col == "mean_eval_reward" else "Episode Reward"

    with plt.rc_context(STYLE):

        # 1. Reward
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, df[reward_col], color=ACCENT, alpha=0.3, linewidth=0.8, label="raw")
        ax.plot(x, rolling(df[reward_col], window), color=ACCENT2, linewidth=2.0, label=f"rolling mean (w={window})")
        _apply_style(ax, "PPO - Eval Reward", "Timestep", reward_label)
        ax.legend(fontsize=9)
        save_fig(fig, os.path.join(plots_dir, "ppo_episode_reward.png"))

        # 2. Policy loss + Value loss (dual axis)
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.plot(x, rolling(df["policy_loss"], window), color=ACCENT,  linewidth=2.0, label="policy loss")
        ax2.plot(x, rolling(df["value_loss"],  window), color=ACCENT3, linewidth=2.0, label="value loss", linestyle="--")
        ax1.set_ylabel("Policy Loss", color=ACCENT,  fontsize=10)
        ax2.set_ylabel("Value Loss",  color=ACCENT3, fontsize=10)
        ax1.tick_params(axis="y", colors=ACCENT)
        ax2.tick_params(axis="y", colors=ACCENT3)
        _apply_style(ax1, "PPO - Policy & Value Loss", "Timestep", "")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        save_fig(fig, os.path.join(plots_dir, "ppo_losses.png"))

        # 3. Entropy + Entropy coef
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.plot(x, rolling(df["entropy"], window), color=ACCENT4, linewidth=2.0, label="entropy")
        if "entropy_coef" in df.columns:
            ax2.plot(x, df["entropy_coef"], color=ACCENT2, linewidth=1.5, label="entropy coef", linestyle=":")
        ax1.set_ylabel("Entropy",      color=ACCENT4, fontsize=10)
        ax2.set_ylabel("Entropy Coef", color=ACCENT2, fontsize=10)
        ax1.tick_params(axis="y", colors=ACCENT4)
        ax2.tick_params(axis="y", colors=ACCENT2)
        _apply_style(ax1, "PPO - Entropy & Coefficient", "Timestep", "")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        save_fig(fig, os.path.join(plots_dir, "ppo_entropy.png"))

        # 4. Episode length (optional column)
        if "episode_length" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, df["episode_length"],                  color=ACCENT,  alpha=0.3, linewidth=0.8, label="raw")
            ax.plot(x, rolling(df["episode_length"], window), color=ACCENT3, linewidth=2.0, label=f"rolling mean (w={window})")
            _apply_style(ax, "PPO - Episode Length", "Timestep", "Steps")
            ax.legend(fontsize=9)
            save_fig(fig, os.path.join(plots_dir, "ppo_episode_length.png"))

        # 5. Combined 2x2 overview
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("PPO Training Overview", fontsize=15, fontweight="bold", color="#ffffff", y=1.01)

        axes[0, 0].plot(x, df[reward_col], color=ACCENT, alpha=0.25, linewidth=0.7)
        axes[0, 0].plot(x, rolling(df[reward_col], window), color=ACCENT2, linewidth=2.0)
        _apply_style(axes[0, 0], reward_label, "Timestep", "Reward")

        axes[0, 1].plot(x, rolling(df["policy_loss"], window), color=ACCENT,  linewidth=2.0, label="policy")
        axes[0, 1].plot(x, rolling(df["value_loss"],  window), color=ACCENT3, linewidth=2.0, label="value", linestyle="--")
        _apply_style(axes[0, 1], "Losses", "Timestep", "Loss")
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].plot(x, rolling(df["entropy"], window), color=ACCENT4, linewidth=2.0)
        _apply_style(axes[1, 0], "Policy Entropy", "Timestep", "Entropy")

        if "episode_length" in df.columns:
            axes[1, 1].plot(x, df["episode_length"],                  color=ACCENT,  alpha=0.25, linewidth=0.7)
            axes[1, 1].plot(x, rolling(df["episode_length"], window), color=ACCENT3, linewidth=2.0)
            _apply_style(axes[1, 1], "Episode Length", "Timestep", "Steps")
        else:
            # Fall back to entropy coef if no episode_length
            if "entropy_coef" in df.columns:
                axes[1, 1].plot(x, df["entropy_coef"], color=ACCENT2, linewidth=2.0)
            _apply_style(axes[1, 1], "Entropy Coef", "Timestep", "Coef")

        fig.tight_layout()
        save_fig(fig, os.path.join(plots_dir, "ppo_overview.png"))

    print(f"\n[PPO] All plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# SAC plots
# ---------------------------------------------------------------------------
def plot_sac(df: pd.DataFrame, plots_dir: str, window: int = 20) -> None:
    x = df["timestep"]

    with plt.rc_context(STYLE):

        # 1. Episode reward
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, df["episode_reward"], color=ACCENT, alpha=0.3, linewidth=0.8, label="raw")
        ax.plot(x, rolling(df["episode_reward"], window), color=ACCENT2, linewidth=2.0, label=f"rolling mean (w={window})")
        _apply_style(ax, "SAC - Episode Reward", "Timestep", "Episode Reward")
        ax.legend(fontsize=9)
        save_fig(fig, os.path.join(plots_dir, "sac_episode_reward.png"))

        # 2. Critic losses
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, rolling(df["critic1_loss"], window), color=ACCENT,  linewidth=2.0, label="critic 1")
        ax.plot(x, rolling(df["critic2_loss"], window), color=ACCENT3, linewidth=2.0, label="critic 2", linestyle="--")
        _apply_style(ax, "SAC - Critic Losses", "Timestep", "Loss")
        ax.legend(fontsize=9)
        save_fig(fig, os.path.join(plots_dir, "sac_critic_losses.png"))

        # 3. Actor loss + Alpha (dual axis)
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.plot(x, rolling(df["actor_loss"], window), color=ACCENT2, linewidth=2.0, label="actor loss")
        ax2.plot(x, df["alpha"],                       color=ACCENT4, linewidth=1.5, label="alpha (temp)", linestyle=":")
        ax1.set_ylabel("Actor Loss", color=ACCENT2, fontsize=10)
        ax2.set_ylabel("Alpha",      color=ACCENT4, fontsize=10)
        ax1.tick_params(axis="y", colors=ACCENT2)
        ax2.tick_params(axis="y", colors=ACCENT4)
        _apply_style(ax1, "SAC - Actor Loss & Temperature (alpha)", "Timestep", "")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        save_fig(fig, os.path.join(plots_dir, "sac_actor_alpha.png"))

        # 4. Episode length
        if "episode_length" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x, df["episode_length"],                  color=ACCENT,  alpha=0.3, linewidth=0.8, label="raw")
            ax.plot(x, rolling(df["episode_length"], window), color=ACCENT3, linewidth=2.0, label=f"rolling mean (w={window})")
            _apply_style(ax, "SAC - Episode Length", "Timestep", "Steps")
            ax.legend(fontsize=9)
            save_fig(fig, os.path.join(plots_dir, "sac_episode_length.png"))

        # 5. Combined 2x2 overview
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("SAC Training Overview", fontsize=15, fontweight="bold", color="#ffffff", y=1.01)

        axes[0, 0].plot(x, df["episode_reward"], color=ACCENT, alpha=0.25, linewidth=0.7)
        axes[0, 0].plot(x, rolling(df["episode_reward"], window), color=ACCENT2, linewidth=2.0)
        _apply_style(axes[0, 0], "Episode Reward", "Timestep", "Reward")

        axes[0, 1].plot(x, rolling(df["critic1_loss"], window), color=ACCENT,  linewidth=2.0, label="critic 1")
        axes[0, 1].plot(x, rolling(df["critic2_loss"], window), color=ACCENT3, linewidth=2.0, label="critic 2", linestyle="--")
        _apply_style(axes[0, 1], "Critic Losses", "Timestep", "Loss")
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].plot(x, rolling(df["actor_loss"], window), color=ACCENT2, linewidth=2.0)
        _apply_style(axes[1, 0], "Actor Loss", "Timestep", "Loss")

        axes[1, 1].plot(x, df["alpha"], color=ACCENT4, linewidth=2.0)
        _apply_style(axes[1, 1], "Temperature alpha", "Timestep", "Alpha")

        fig.tight_layout()
        save_fig(fig, os.path.join(plots_dir, "sac_overview.png"))

    print(f"\n[SAC] All plots saved to: {plots_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PPO or SAC training history.")
    parser.add_argument("--algo",   type=str, default=None,         help="'ppo' or 'sac' (default: read from config.yaml)")
    parser.add_argument("--csv",    type=str, default=None,         help="Path to training_history.csv (overrides config)")
    parser.add_argument("--window", type=int, default=20,           help="Rolling mean window size (default: 20)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config    = load_config(args.config)
    algorithm = (args.algo or config.get("algorithm", "ppo")).lower()
    path_cfg  = config["paths"][algorithm]

    csv_path  = args.csv or path_cfg["training_history"]
    plots_dir = path_cfg["plots_dir"]

    print(f"[plot_training] Algorithm : {algorithm.upper()}")
    print(f"[plot_training] CSV       : {csv_path}")
    print(f"[plot_training] Plots dir : {plots_dir}")

    if not os.path.isfile(csv_path):
        print(f"\n[ERROR] CSV not found: {csv_path}")
        print("  Make sure training has started and the logger has written at least one row.")
        return

    # The PPO logger writes 8 data columns but only 7 headers (off-by-one bug).
    # Re-read with explicit column names so everything lines up correctly.
    PPO_COLS = ["timestep", "update", "mean_eval_reward", "episode_length", "policy_loss", "value_loss", "entropy", "entropy_coef"]
    SAC_COLS = ["timestep", "episode", "episode_reward", "episode_length", "critic1_loss", "critic2_loss", "actor_loss", "alpha"]

    raw = pd.read_csv(csv_path, header=None, skiprows=1)  # skip original header row
    expected_cols = PPO_COLS if algorithm == "ppo" else SAC_COLS

    if raw.shape[1] == len(expected_cols):
        raw.columns = expected_cols
    else:
        # Fallback: read normally and hope columns match
        raw = pd.read_csv(csv_path)

    df = raw
    print(f"[plot_training] Rows loaded : {len(df):,}")
    print(f"[plot_training] Columns     : {df.columns.tolist()}")

    if len(df) < 2:
        print("[WARNING] Too few rows to plot meaningfully. Train longer first.")
        return

    if algorithm == "sac":
        plot_sac(df, plots_dir, window=args.window)
    else:
        plot_ppo(df, plots_dir, window=args.window)


if __name__ == "__main__":
    main()