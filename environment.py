import gymnasium as gym
import assetto_corsa_gym

class AssettoCorsaWrapper:
    """
    Wrapper for the Assetto Corsa Gym environment to handle preprocessing 
    and state/action space verification.
    """
    def __init__(self, config):
        """
        Initializes the environment and wraps it.
        Args:
            config (dict): Configuration dictionary containing track/car settings.
        """
        self.env = gym.make("assetto_corsa_gym:AssettoCorsa-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        """Resets the environment to initial state."""
        return self.env.reset()

    def step(self, action):
        """Executes one timestep in the environment."""
        return self.env.step(action)