from pettingzoo.utils import wrappers
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from marl_env import MLWorkflowEnv

def make_gym_env():
    # Create the PettingZoo environment
    env = MLWorkflowEnv()

    # You can still apply flattening to the observation space for compatibility with SB3
    env = gym.wrappers.FlattenObservation(env)

    # Vectorize the environment for training with multiple agents
    env = DummyVecEnv([lambda: env])

    return env

