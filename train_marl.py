# train_marl.py

from stable_baselines3 import PPO
from make_gym_env import make_gym_env

# Create the Gym-compatible environment
env = make_gym_env()

# Define the model and train it
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("marl_model")

# Test the model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    env.render()
    if dones:
        break
