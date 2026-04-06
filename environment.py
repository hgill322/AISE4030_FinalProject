"""
environment.py
==============
AssettoCorsaGym wrapper for the AISE 4030 RL project.

This module handles environment creation and exposes observation/action space
information for the Assetto Corsa racing simulator using the
assetto_corsa_gym plugin (https://github.com/dasGringuen/assetto_corsa_gym).

Non-Gymnasium API Note:
    The AC gym environment is a gymnasium.Env subclass (AssettoCorsaEnv).
    It is instantiated via assettoCorsa.make_ac_env(cfg, work_dir) where cfg
    is an OmegaConf config object loaded from config.yml.

    API equivalences:
        - gymnasium reset()       <--> env.reset()
        - gymnasium step(action)  <--> env.step(action)
        - gymnasium obs           <--> float32 numpy array, shape (state_dim,)
        - gymnasium reward        <--> float, computed internally by AC env
        - gymnasium done          <--> bool, off-track / max steps / low speed

    Observation space (from ac_env.py, add_previous_obs_to_state=True):
        - state_dim = 125  (25 base features * 3 history frames + 25 current)
        - dtype: float32
        - range: approx [-1, 1] (normalized internally by AC env)

    Action space (from ac_env.py):
        - Continuous Box([-1, -1, -1], [1, 1, 1], shape=(3,), dtype=float32)
        - [0] steering  : [-1, 1]
        - [1] throttle  : [-1, 1]  (AC env uses [-1,1] with use_relative_actions)
        - [2] brake     : [-1, 1]

Track name mapping (from AssettoCorsaConfigs/tracks/config.yaml):
    Barcelona   : ks_barcelona-layout_gp
    Monza       : monza
    Austria     : ks_red_bull_ring-layout_gp
    Silverstone : ks_silverstone-gp
    Indianapolis: indianapolis_sp
"""

import os
import sys
import numpy as np
import yaml
from typing import Tuple, Any, Optional
from omegaconf import OmegaConf

# Add the assetto_corsa_gym subpackage directory to sys.path so that
# AssettoCorsaEnv and AssettoCorsaPlugin are found as top-level modules.
# This mirrors the path setup done inside assettoCorsa.py itself.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AC_GYM_PKG = os.path.join(_THIS_DIR, "assetto_corsa_gym")
if _AC_GYM_PKG not in sys.path:
    sys.path.insert(0, _AC_GYM_PKG)

import AssettoCorsaEnv.assettoCorsa as assettoCorsa


# ---------------------------------------------------------------------------
# AssettoCorsaWrapper
# ---------------------------------------------------------------------------
class AssettoCorsaWrapper:
    """
    Wrapper around the AssettoCorsaEnv gymnasium environment.

    Provides a consistent interface for the PPO agent and exposes
    observation_space and action_space descriptors with human-readable
    print methods for Task 1.
    """

    def __init__(self, ac_config_path: str = "config.yml", work_dir: str = "output"):
        """
        Initializes the AssettoCorsaWrapper.

        Args:
            ac_config_path (str): Path to config.yml for OmegaConf loading.
            work_dir (str): Working directory for AC env telemetry output.
        """
        self.ac_config_path = ac_config_path
        self.work_dir = work_dir

        # Load the OmegaConf config (used by make_ac_env)
        if not os.path.isabs(ac_config_path):
            ac_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ac_config_path)
        self.cfg = OmegaConf.load(ac_config_path)

        # Observation space: state_dim=125 when add_previous_obs_to_state=True
        # (25 base features * 3 history frames + 25 current = 100+25 = 125)
        # Confirmed from ac_env.py log: "state_dim 125"
        self._obs_dim = 125
        self._act_dim = 3

        # Space descriptors
        self.observation_space = _ObservationSpace(self._obs_dim)
        self.action_space = _ActionSpace(self._act_dim)

        # Underlying AC gym env (created on first reset)
        self._env: Optional[Any] = None

    def _build_env(self) -> None:
        """
        Instantiates the AssettoCorsaEnv via the official make_ac_env factory.

        Uses the OmegaConf config loaded from config.yml, exactly as the
        test_gym.ipynb notebook does.

        Returns:
            None
        """
        self._env = assettoCorsa.make_ac_env(cfg=self.cfg, work_dir=self.work_dir)

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the start of a new episode.

        Creates the underlying AC env on the first call. Subsequent calls
        reuse the existing env instance.

        Returns:
            np.ndarray: Initial observation vector, shape (obs_dim,), float32.
        """
        if self._env is None:
            self._build_env()
        result = self._env.reset()
        # AC env reset() returns (obs, info) or just obs depending on gym version
        obs = result[0] if isinstance(result, tuple) else result
        return np.array(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Applies an action to the simulator and returns the resulting transition.

        Args:
            action (np.ndarray): Action vector of shape (3,):
                [steering in [-1,1], throttle in [-1,1], brake in [-1,1]].

        Returns:
            Tuple containing:
                - obs (np.ndarray): Next observation, shape (obs_dim,), float32.
                - reward (float): Reward computed internally by AssettoCorsaEnv.
                - done (bool): True if the episode has terminated.
                - info (dict): Auxiliary telemetry info from AC env.
        """
        result = self._env.step(action)
        # Handle both old gym (obs, reward, done, info) and new gymnasium (obs, reward, terminated, truncated, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        return np.array(obs, dtype=np.float32), float(reward), done, info

    def close(self) -> None:
        """
        Closes the AC environment and releases simulator resources.

        Returns:
            None
        """
        if self._env is not None:
            self._env.close()
            self._env = None

    def print_space_info(self) -> None:
        """
        Prints observation space, action space, device, and environment
        configuration to the console for Task 1 verification.

        Returns:
            None
        """
        ac = self.cfg.AssettoCorsa
        print("=" * 60)
        print("  ENVIRONMENT: Assetto Corsa Gym")
        print(f"  Track : {ac.track}")
        print(f"  Car   : {ac.car}")
        print(f"  Host  : {ac.remote_machine_ip}:{ac.ego_server_port}")
        print("=" * 60)

        print("\n[Observation Space]")
        print(f"  Shape     : {self.observation_space.shape}")
        print(f"  Dtype     : {self.observation_space.dtype}")
        print(f"  Low       : {self.observation_space.low:.4f} (approx, all dims)")
        print(f"  High      : {self.observation_space.high:.4f} (approx, all dims)")
        print(f"  Num obs   : {self.observation_space.n_observations}")
        print(f"  Note      : 25 base features x 3 history frames + 25 current")
        print(f"              = 100 historical + 25 current = 125 total")
        print(f"              (add_previous_obs_to_state=True in config.yml)")

        print("\n[Action Space]")
        print(f"  Type      : Continuous")
        print(f"  Shape     : {self.action_space.shape}")
        print(f"  Dtype     : {self.action_space.dtype}")
        print(f"  Low       : {self.action_space.low}")
        print(f"  High      : {self.action_space.high}")
        print(f"  [0] steering  : [-1, 1]")
        print(f"  [1] throttle  : [-1, 1]  (relative actions)")
        print(f"  [2] brake     : [-1, 1]  (relative actions)")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Space descriptors
# ---------------------------------------------------------------------------
class _ObservationSpace:
    """Descriptor for the AC observation space."""

    def __init__(self, obs_dim: int):
        self.shape: Tuple[int, ...] = (obs_dim,)
        self.dtype = np.float32
        self.low: float = -1.0
        self.high: float = 1.0
        self.n_observations: int = obs_dim


class _ActionSpace:
    """Descriptor for the AC action space."""

    def __init__(self, act_dim: int):
        self.shape: Tuple[int, ...] = (act_dim,)
        self.dtype = np.float32
        self.low: np.ndarray = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self.high: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.n_actions: int = act_dim
        self.is_discrete: bool = False

    def sample(self) -> np.ndarray:
        """
        Returns a random action sampled uniformly from the action space.

        Returns:
            np.ndarray: Random action vector of shape (3,), dtype float32.
        """
        return np.random.uniform(self.low, self.high).astype(np.float32)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def make_env(config: dict) -> AssettoCorsaWrapper:
    """
    Factory function that creates and returns a configured AssettoCorsaWrapper.

    This is the single entry point for environment creation used by
    training_script.py and evaluation scripts.

    Args:
        config (dict): Project config dict loaded from config.yaml.
            Uses config['environment']['ac_config_path'] and
            config['environment']['work_dir'].

    Returns:
        AssettoCorsaWrapper: Ready-to-use environment wrapper.
    """
    env_cfg = config["environment"]
    return AssettoCorsaWrapper(
        ac_config_path=env_cfg.get("ac_config_path", "config.yml"),
        work_dir=env_cfg.get("work_dir", "output"),
    )


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads the project YAML configuration file.

    Args:
        path (str): Path to config.yaml. Defaults to 'config.yaml'.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
