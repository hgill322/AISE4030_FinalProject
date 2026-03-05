class DDPGAgent:
    """
    Skeleton for the DDPG (Baseline) Agent.
    """
    def __init__(self, state_dim, action_dim, config):
        """
        Constructor for network and buffer initialization.
        """
        pass

    def choose_action(self, state, evaluate=False):
        """
        Chooses an action based on the current policy.
        Args:
            state (np.ndarray): Current observation.
            evaluate (bool): If True, disable exploration noise.
        Returns:
            np.ndarray: Action vector [steering, throttle, brake].
        """
        pass

    def update(self):
        """Performs a gradient update step."""
        pass
    