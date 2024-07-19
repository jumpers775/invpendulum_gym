import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
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
elif "eval" in sys.argv:
    passes = 10
    env = gym.make('inv_pend_env/inv_pendulum_v0')
    conditions = np.linspace(-np.pi/2, np.pi/2, passes)
    observation, info = env.reset(val=conditions[0])
    model = PPO.load("checkpoints/model-sb3.pth")
    timestep = 0
    starts = []

    for i in range(0,passes):
        terminated = False
        truncated = False
        passes = 0
        obs = env.reset(val=conditions[i])
        timestep = 0
        action, _states = model.predict(obs[0])
        while not terminated and not truncated:
            passes +=1
            obs, reward, terminated, truncated, info = env.step(action)
            action, _states = model.predict(obs)
            if terminated or truncated:
                starts.append(passes==200)
        
        timestep += 1
    env.close()

    # Plotting conditions vs starts
    plt.figure()
    plt.scatter(conditions, starts, c=starts, cmap='RdYlGn')
    plt.xlabel('Conditions')
    plt.ylabel('Starts')
    plt.title('Starts vs Conditions')
    plt.grid(True)
    plt.show()
else:
    print("Please provide train or test as argument")
