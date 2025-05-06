import gymnasium as gym
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO

# Make the environment
env = gym.make("MiniGrid-FourRooms-v0")
env = RGBImgPartialObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)            # Flatten observation space for CNN

# Create PPO model
model = PPO("CnnPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_fourrooms")

# Optional: Test
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()