import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import inv_pend_env

# Parallel environments
vec_env = make_vec_env("inv_pend_env/inv_pendulum_v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/model-sb3.pth")

del model # remove to demonstrate saving and loading

model = PPO.load("checkpoints/model-sb3.pth")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
