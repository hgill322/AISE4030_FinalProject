import torch
from environment import AssettoCorsaWrapper

def verify_setup():
    """
    Prints environment details to console for Phase 2 report.
    """
    # Initialize environment
    env = AssettoCorsaWrapper(config={})
    
    print(f"--- Environment Verification ---")
    print(f"Observation Space: {env.observation_space}") # Should be Box(dim,)
    print(f"Action Space: {env.action_space}")           # Should be Box(-1, 1, (3,))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device being used: {device}")
    
    # Run one test step
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print("Confirmation: Successfully executed one environment step.")

if __name__ == "__main__":
    verify_setup()
    