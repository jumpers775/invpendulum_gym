import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import inv_pend_env
import sys
import time
# Parallel environments
vec_env = make_vec_env("inv_pend_env/inv_pendulum_v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, device="cuda")



if "train" in sys.argv:
    # Separate env for evaluation
    eval_env = gym.make("inv_pend_env/inv_pendulum_v0")

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Untrained mean_reward={mean_reward:.2f} +/- {std_reward}")

    model.learn(total_timesteps=int(1e7), progress_bar=True)
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/model-sb3.pth")

    eval_env = gym.make("inv_pend_env/inv_pendulum_v0")

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"Final mean_reward={mean_reward:.2f} +/- {std_reward}")

elif "test" in sys.argv:
    del model
    model = PPO.load("checkpoints/model-sb3.pth")
    for _ in range(20):
        obs = vec_env.reset()
        for i in range(20):
            time.sleep(0.1)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = vec_env.step(action)
            vec_env.render("human")
else:
    print("Please provide train or test as argument")
