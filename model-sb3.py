import gymnasium as gym
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import inv_pend_env
import sys
import time

# Parallel environments
vec_env = make_vec_env("inv_pend_env/inv_pendulum_v0", n_envs=4)

model = PPO("MlpPolicy", vec_env, device="cuda")


def hereshwhatihavetosay():
    print("Please provide train, test, or eval (success or initforce) as arguments")


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
    if "success" in sys.argv:
        passes = 100
        env = gym.make("inv_pend_env/inv_pendulum_v0")
        thconditions = np.linspace(-np.pi / 2, np.pi / 2, passes)
        vconditions = np.linspace(-10, 10, passes)
        observation, info = env.reset(thval=thconditions[0], vval=vconditions[0])
        model = PPO.load("checkpoints/model-sb3.pth")
        timestep = 0
        starts = []

        for i in range(0, passes):
            starts.append([])
            for j in range(0, passes):
                terminated = False
                truncated = False
                npasses = 0
                obs, info = env.reset(thval=thconditions[i], vval=vconditions[j])
                timestep = 0
                action, _states = model.predict(obs)
                while not terminated and not truncated:
                    npasses += 1
                    obs, reward, terminated, truncated, info = env.step(action)
                    action, _states = model.predict(obs)
                    if terminated or truncated:
                        starts[-1].append(npasses == 200)
                timestep += 1

        env.close()

        fig, ax = plt.subplots()

        # Calculate width and height for rectangles based on the conditions' spacing
        width = np.diff(thconditions).mean()
        height = np.diff(vconditions).mean()

        # Iterate over each condition pair and plot a rectangle for each
        for i, thval in enumerate(thconditions):
            for j, vvval in enumerate(vconditions):
                color = "green" if starts[i][j] else "red"
                # Create a rectangle patch
                rect = patches.Rectangle(
                    (thval - width / 2, vvval - height / 2),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="none",
                    facecolor=color,
                )
                # Add the rectangle to the Axes
                ax.add_patch(rect)

        # Set the limits of the plot to the min and max of the conditions
        plt.xlim(min(thconditions) - width, max(thconditions) + width)
        plt.ylim(min(vconditions) - height, max(vconditions) + height)

        # Set labels and title
        plt.xlabel("Theta starting conditions")
        plt.ylabel("Velocity starting conditions")
        plt.title("Success by Start Condition")

        # Show the plot
        plt.show()
    if "initforce" in sys.argv:
        passes = 100
        env = gym.make("inv_pend_env/inv_pendulum_v0")
        thconditions = np.linspace(-np.pi / 2, np.pi / 2, passes)
        vconditions = np.linspace(-10, 10, passes)
        observation, info = env.reset(thval=thconditions[0], vval=vconditions[0])
        model = PPO.load("checkpoints/model-sb3.pth")
        timestep = 0
        starts = []

        for i in range(0, passes):
            starts.append([])
            for j in range(0, passes):
                terminated = False
                truncated = False
                npasses = 0
                obs, info = env.reset(thval=thconditions[i], vval=vconditions[j])
                timestep = 0
                action, _states = model.predict(obs)
                starts[-1].append(action)
                timestep += 1

        env.close()

        starts = np.array(starts)

        fig, ax = plt.subplots()

        # Calculate width and height for rectangles based on the conditions' spacing
        width = np.diff(thconditions).mean()
        height = np.diff(vconditions).mean()

        # Adjust width and height for visual clarity if necessary
        width *= 1
        height *= 1

        # Iterate over each condition pair and plot a rectangle for each
        for i, thval in enumerate(thconditions):
            for j, vvval in enumerate(vconditions):
                min_val = min(min(sublist) for sublist in starts)
                max_val = max(max(sublist) for sublist in starts)

                val = max_val[0] if max_val[0] > abs(min_val[0]) else abs(min_val[0])

                m = interp1d([-val, val], [0, 1])

                color = (0, 0, m(starts[i, j])[0], 1)
                # Create a rectangle patch
                rect = patches.Rectangle(
                    (thval - width / 2, vvval - height / 2),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="none",
                    facecolor=color,
                )
                # Add the rectangle to the Axes
                ax.add_patch(rect)

        # Set the limits of the plot to the min and max of the conditions
        plt.xlim(min(thconditions) - width, max(thconditions) + width)
        plt.ylim(min(vconditions) - height, max(vconditions) + height)

        # Set labels and title
        plt.xlabel("Theta starting conditions")
        plt.ylabel("Velocity starting conditions")
        plt.title("Init Force by Start Condition")

        # Show the plot
        plt.show()
    else:
        hereshwhatihavetosay()
else:
    hereshwhatihavetosay()
