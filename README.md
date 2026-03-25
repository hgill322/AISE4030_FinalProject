# Autonomous Racing with Assetto Corsa — PPO Agent

AISE 4030 Reinforcement Learning — Project Phase 2  
Track: **Monza** | Car: **BMW Z4 GT3**

---

## Project Overview

This project trains a Proximal Policy Optimization (PPO) agent to autonomously
drive a BMW Z4 GT3 around the Monza circuit in Assetto Corsa using the
[assetto_corsa_gym](https://github.com/dasGringuen/assetto_corsa_gym) plugin.

**Phase 2 goal:** Build the full code skeleton (environment wrapper, agent, buffer,
training loop) with complete documentation. No trained agent is required yet.

**Algorithm comparison (Phase 3):**
- Base: **PPO** (this phase)
- Advanced: **SAC** (Soft Actor-Critic) — to be added in Phase 3

---

## File Structure

```
AutonomousRacing/
├── config.yaml          # All hyperparameters and settings
├── environment.py       # AssettoCorsaGym wrapper + space definitions
├── actor_critic.py      # Shared Actor-Critic MLP network
├── ppo_agent.py         # PPO agent (action selection, update, save/load)
├── rollout_buffer.py    # On-policy rollout buffer with GAE
├── training_script.py   # Main entry point: train / evaluate / deploy
├── utils.py             # Config loading, device, seed, logging, plotting
├── requirements.txt     # Python dependencies
├── README.md
└── ppo_results/         # Model checkpoints, training history, plots
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

All settings are in `config.yaml`. Key parameters:

| Section | Key | Default | Description |
|---|---|---|---|
| environment | track | `monza` | AC track name |
| environment | car | `bmw_z4_gt3` | AC car identifier |
| ppo | learning_rate | `3e-4` | Adam learning rate |
| ppo | n_steps | `2048` | Rollout length |
| ppo | clip_epsilon | `0.2` | PPO clip ratio |
| training | total_timesteps | `1_000_000` | Total training steps |

---

## Running

```bash
# Train from scratch
python training_script.py

# The script always prints environment/space info first (Task 1 output).
```

Set `mode.train: false` and `mode.evaluate: true` in `config.yaml` to run
evaluation only using the best saved checkpoint.

---

## Observation & Action Spaces

**Observation Space:** `(125,)` float32, range `[-1, 1]`

The 125-dimensional vector is structured as:

| Block | Contents | Size |
|---|---|---|
| Current frame | 14 telemetry values + 11 ray-cast distances | 25 |
| History frame t-1 | Same 25 features, 1 step ago | 25 |
| History frame t-2 | Same 25 features, 2 steps ago | 25 |
| History frame t-3 | Same 25 features, 3 steps ago | 25 |
| Curvature lookahead | Down-sampled from 300 m of racing line | 12 |
| Past actions (t-1..t-3) | 3 actions x 3 dims | 9 |
| Last applied action | Most recent action | 3 |
| Out-of-track flag | Binary (0 or 1) | 1 |
| **Total** | | **125** |

**Action Space:** Continuous `(3,)` — all dims `[-1, 1]` with `use_relative_actions=True`

| Index | Action | Range |
|---|---|---|
| 0 | steering | [-1, 1] |
| 1 | throttle (relative) | [-1, 1] |
| 2 | brake (relative) | [-1, 1] |
