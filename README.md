# Autonomous Racing with Assetto Corsa — PPO & SAC

AISE 4030 Reinforcement Learning — Project Phase 3  
Track: **Indianapolis** | Car: **BMW Z4 GT3**

---

## Project Overview

This project trains reinforcement learning agents to autonomously drive a BMW Z4 GT3
around the Indianapolis circuit in Assetto Corsa using the
[assetto_corsa_gym](https://github.com/dasGringuen/assetto_corsa_gym) plugin.

Two algorithms are implemented, trained, and compared:

- **Base Algorithm: PPO** (Proximal Policy Optimization) — on-policy, clipped surrogate
  objective with GAE, shared Actor-Critic backbone, entropy annealing.
- **Advanced Algorithm: SAC** (Soft Actor-Critic) — off-policy, twin Q-networks,
  reparameterized squashed Gaussian policy, automatic entropy temperature tuning.

---

## File Structure

```
AutonomousRacing/
├── config.yaml          # All hyperparameters and settings (shared by PPO & SAC)
├── environment.py       # AssettoCorsaGym wrapper + space definitions
├── actor_critic.py      # Shared Actor-Critic MLP network (used by PPO)
├── ppo_agent.py         # PPO agent (action selection, update, save/load)
├── rollout_buffer.py    # On-policy rollout buffer with GAE
├── sac_agent.py         # SAC agent (actor, twin critics, replay buffer, auto-alpha)
├── training_script.py   # Main entry point: train / evaluate / deploy (both algos)
├── utils.py             # Config loading, device, seed, logging
├── plot_training.py     # Standalone plotting script for PPO and SAC CSV logs
├── requirements.txt     # Python dependencies
├── README.md
├── ppo_results/         # PPO checkpoints, training_history.csv, plots/
└── sac_results/         # SAC checkpoints, training_history.csv, plots/
```

---

## Prerequisites

1. **Assetto Corsa** installed on Windows with the AC UDP plugin configured.
2. **Python 3.10+** (recommended: Anaconda environment).
3. The `assetto_corsa_gym` package installed (see below).

---

## Installation

```bash
# 1. Clone the AC gym plugin
git clone https://github.com/dasGringuen/assetto_corsa_gym.git
cd assetto_corsa_gym
pip install -e .

# 2. Install project dependencies
cd ../AutonomousRacing
pip install -r requirements.txt
```

> **Note:** The AC plugin must be running inside Assetto Corsa before executing
> any Python scripts. Follow the plugin setup instructions in the
> `assetto_corsa_gym` repository.

---

## Configuration

All settings are in `config.yaml`. Switch algorithms by changing the top-level
`algorithm` field:

```yaml
algorithm: "ppo"   # or "sac"
```

Key parameters:

| Section | Key | Value | Description |
|---|---|---|---|
| — | algorithm | `ppo` / `sac` | Selects which agent to train/evaluate |
| environment | track | `indianapolis_sp` | AC track name |
| environment | car | `bmw_z4_gt3` | AC car identifier |
| ppo | learning_rate | `3e-4` | Adam learning rate |
| ppo | n_steps | `1024` | Rollout length before each PPO update |
| ppo | clip_epsilon | `0.2` | PPO surrogate clip ratio |
| ppo | entropy_coef | `0.015` | Initial entropy bonus coefficient |
| sac | learning_rate | `3e-4` | Adam learning rate (actor & critics) |
| sac | buffer_size | `1_000_000` | Replay buffer capacity |
| sac | warmup_steps | `5000` | Random exploration steps before learning |
| sac | target_entropy | `-3.0` | Target entropy for auto-alpha tuning |
| training | total_timesteps | `500_000` | Total environment steps |
| training | eval_interval | `50_000` | Steps between evaluation runs |
| training | save_interval | `10_000` | Steps between checkpoint saves |

---

## Running

```bash
# Train the selected algorithm (set algorithm: in config.yaml first)
python training_script.py

# Evaluate the best saved checkpoint (set mode.train: false, mode.evaluate: true)
python training_script.py

# Plot training curves from the saved CSV log
python plot_training.py --algo ppo
python plot_training.py --algo sac
```

To resume training from the latest checkpoint:
```yaml
mode:
  train: true
  resume: true
  resume_path: ""   # leave empty to auto-detect latest checkpoint
```

---

## Algorithm Details

### PPO (Proximal Policy Optimization)
- **Network:** Shared MLP backbone → actor head (Gaussian mean) + critic head (scalar value)
- **Exploration:** Stochastic Gaussian sampling during training; entropy coefficient
  annealed by factor `0.9998` each update
- **Update:** Clipped surrogate loss + MSE value loss + entropy bonus, over `n_epochs`
  mini-batch epochs per rollout
- **Buffer:** On-policy `RolloutBuffer`; discarded after each update

### SAC (Soft Actor-Critic)
- **Networks:** Stochastic actor (squashed Gaussian via reparameterization trick) +
  twin Q-critics + twin target critics
- **Exploration:** Entropy is part of the objective; temperature `alpha` auto-tuned
  to maintain `target_entropy = -3.0`
- **Update:** Bellman backup with minimum of twin Q-values; actor maximizes
  `Q - alpha * log_prob`; Polyak soft target updates (`tau = 0.005`)
- **Buffer:** Off-policy `ReplayBuffer` (circular, capacity 1M); optionally saved to
  disk alongside model checkpoints (`save_buffer: true`)

---

## Observation & Action Spaces

**Observation Space:** `(125,)` float32, range `[-1, 1]`

| Block | Contents | Size |
|---|---|---|
| Current frame | 14 telemetry values + 11 ray-cast distances | 25 |
| History frame t-1 | Same 25 features, 1 step ago | 25 |
| History frame t-2 | Same 25 features, 2 steps ago | 25 |
| History frame t-3 | Same 25 features, 3 steps ago | 25 |
| Curvature lookahead | Down-sampled from 300 m of racing line | 12 |
| Past actions (t-1..t-3) | 3 actions × 3 dims | 9 |
| Last applied action | Most recent action | 3 |
| Out-of-track flag | Binary (0 or 1) | 1 |
| **Total** | | **125** |

**Action Space:** Continuous `(3,)` — all dims `[-1, 1]` with `use_relative_actions=True`

| Index | Action | Range |
|---|---|---|
| 0 | Steering | [-1, 1] |
| 1 | Throttle (relative) | [-1, 1] |
| 2 | Brake (relative) | [-1, 1] |