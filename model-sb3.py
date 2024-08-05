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
import torch




class PIDController:
    def __init__(self, setpoint=0, kp=10, ki=0, kd=0, dt=0.1):
        self.controlhistory = []
        self.integral = 0
        self.previous_error = 0
        self.setpoint = setpoint
        self.kp = kp 
        self.ki = ki 
        self.kd = kd 
        self.dt = dt

    def control(self, y):
        error = self.setpoint - y

        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt

        if self.ki != 0:
            self.integral = max(-1 / self.ki, min(1 / self.ki, self.integral))

        self.previous_error = error

        u = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        self.controlhistory.append(u)

        return u






# Parallel environments
vec_env = make_vec_env("inv_pend_env/inv_pendulum_v0", n_envs=8)


def find_closest_match(small_list, big_list):
    min_distance = float('inf')
    closest_match = None
    
    for entry in big_list:
        distance = np.linalg.norm(np.array(small_list) - np.array(entry))
        if distance < min_distance:
            min_distance = distance
            closest_match = entry
    
    return closest_match

def hereshwhatihavetosay():
    print("Please provide train, test, or eval (success or initforce) as arguments")


if "train" in sys.argv:

    if "continue" in sys.argv:
        model = PPO.load("checkpoints/model-sb3.pth")
        model.set_env(vec_env)
    else:
    #     policy_kwargs = dict(
    #     activation_fn=torch.nn.ReLU, net_arch=dict(pi=[8, 8], vf=[8, 8])
    # )

        model = PPO("MlpPolicy", vec_env, device="mps")

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

    length = int(10e6)

    # work around a crash that occurs when training for too long
    lengths = [int(5e6) for i in range(0, int(length/int(5e6)))]
    if length%int(5e6) != 0:
        lengths.append(length%int(5e6))
    for i in range(len(lengths)):
        print(f"{i+1}/{len(lengths)}")
        model.learn(total_timesteps=lengths[i], progress_bar=True)

        # work around memory leak
        os.makedirs("checkpoints", exist_ok=True)
        model.save("checkpoints/model-sb3.pth")
        del model
        model = PPO.load("checkpoints/model-sb3.pth")
        model.set_env(vec_env)


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
    if "pid" not in sys.argv:
        model = PPO.load("checkpoints/model-sb3.pth")
        for _ in range(20):
            obs = vec_env.reset()
            for i in range(20):
                time.sleep(0.1)
                action, _states = model.predict(obs)
                obs, rewards, dones, info = vec_env.step(action)
                vec_env.render("human")
    else:
        controller = PIDController()
        for _ in range(20):
            obs = vec_env.reset()
            for i in range(20):
                time.sleep(0.1)
                action = [controller.control(obs[1])]
                obs, rewards, dones, info = vec_env.step(action)
                vec_env.render("human")
elif "eval" in sys.argv:
    env = gym.make("inv_pend_env/inv_pendulum_v0")
    quant = "quant" in sys.argv
    th_staterange = [env.observation_space.low, env.observation_space.high]
    v_staterange = [-1, 1]
    boxinitdimensions = 10

    sqsize = ((abs(th_staterange[0][0]) + abs(th_staterange[1][0]))/ (2*boxinitdimensions), (abs(v_staterange[0]) + abs(v_staterange[1]))/ (2*boxinitdimensions))

    th_conditions = np.linspace(th_staterange[0][0]+sqsize[0], th_staterange[1][0]-sqsize[0], boxinitdimensions-1)
    v_conditions = np.linspace(v_staterange[0]+sqsize[1], v_staterange[1]-sqsize[1], boxinitdimensions-1)
    
    quantpoints = [[i,j] for i in th_conditions for j in v_conditions]

    if "success" in sys.argv:
        passes = 100
        thconditions = np.linspace(-np.pi / 2, np.pi / 2, passes)
        vconditions = np.linspace(-1, 1, passes)
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
                if quant:
                    obs = find_closest_match(obs, quantpoints)
                action, _states = model.predict(obs)
                while not terminated and not truncated:
                    npasses += 1
                    obs, reward, terminated, truncated, info = env.step(action)
                    if quant:
                        obs = find_closest_match(obs, quantpoints)
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
                if quant:
                    obs = find_closest_match(obs, quantpoints)
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
        plt.title("Command by Condition")

        # Show the plot
        plt.show()
    else:
        hereshwhatihavetosay()

elif "verify" in sys.argv:

    quant = "quant" in sys.argv

    model = PPO.load("checkpoints/model-sb3.pth")
    env = gym.make("inv_pend_env/inv_pendulum_v0")

    th_staterange = [env.observation_space.low, env.observation_space.high]
    v_staterange = [-1, 1]
    boxinitdimensions = 10

    sqsize = ((abs(th_staterange[0][0]) + abs(th_staterange[1][0]))/ (2*boxinitdimensions), (abs(v_staterange[0]) + abs(v_staterange[1]))/ (2*boxinitdimensions))

    th_conditions = np.linspace(th_staterange[0][0]+sqsize[0], th_staterange[1][0]-sqsize[0], boxinitdimensions-1)
    v_conditions = np.linspace(v_staterange[0]+sqsize[1], v_staterange[1]-sqsize[1], boxinitdimensions-1)
    
    quantpoints = [[i,j] for i in th_conditions for j in v_conditions]

    pps = 500
    nonquants = []
    for i in quantpoints:
        newentry = []
        for j in range(4): 
            if j ==0: #up
                vals = np.linspace(i[0]-sqsize[0], i[0]+sqsize[0], pps)
                for k in range(pps):
                    newentry.append([vals[k], i[1]+sqsize[1]])
            if j ==1: #right
                vals = np.linspace(i[1]+sqsize[1], i[1]-sqsize[1], pps)
                for k in range(pps):
                    newentry.append([i[0]-sqsize[0], vals[k]])
            if j ==2: #down
                vals = np.linspace(i[0]+sqsize[0], i[0]-sqsize[0], pps)
                for k in range(pps):
                    newentry.append([vals[k], i[1]-sqsize[1]])
            if j ==3: #left
                vals = np.linspace(i[1]-sqsize[1], i[1]+sqsize[1], pps)
                for k in range(pps):
                    newentry.append([i[0]+sqsize[0], vals[k]])
        nonquants.append(newentry)




    model = PPO.load("checkpoints/model-sb3.pth")
    timestep = 0
    transformedpoints = []
    box = 0

    PID = "pid" in sys.argv
    if PID:
        controller = PIDController()
    if not quant:
        for i in range(len(nonquants[box])):
            obs, info = env.reset(thval=nonquants[box][i][0], vval=nonquants[box][i][1])
            if not PID:
                action, _states = model.predict(obs)
            else:
                action = [controller.control(obs[1])]
            obs, reward, terminated, truncated, info = env.step(action)
            transformedpoints.append([obs[1], obs[0]])
    else:
        obs, info = env.reset(thval=quantpoints[box][0], vval=quantpoints[box][1])
        if not PID:
            action, _states = model.predict(obs)
        else:
            action = [controller.control(obs[1])]
        for i in range(len(nonquants[box])):
            obs, info = env.reset(thval=nonquants[box][i][0], vval=nonquants[box][i][1])
            obs, reward, terminated, truncated, info = env.step(action)
            transformedpoints.append([obs[1], obs[0]])
        

    #plt scatter plot of nonquants with stateranges as ranges
    fig, ax = plt.subplots()

    # shade everything within the ranges green, and everything outside red
    rect = patches.Rectangle((2*th_staterange[0][0], 2*v_staterange[0]), 2*th_staterange[1][0]-2*th_staterange[0][0], 2*v_staterange[1]-2*v_staterange[0], linewidth=1, edgecolor='none', facecolor='red')
    rect2 = patches.Rectangle((th_staterange[0][0], 2*v_staterange[0]), th_staterange[1][0]-th_staterange[0][0], 2*v_staterange[1]-2*v_staterange[0], linewidth=1, edgecolor='none', facecolor='green')
    # Add the rectangle to the Axes
    ax.add_patch(rect)
    ax.add_patch(rect2)


    if not quant or True: # this is the correct code to run. The other code demonstrates something interesting, but irelevant.
        x, y = zip(*nonquants[box])
        ax.plot(x, y, marker='none', color="blue")
        
        import scipy.spatial

        # Perform convex hull algorithm
        hull = scipy.spatial.ConvexHull(transformedpoints)
        convex_points = [transformedpoints[i] for i in hull.vertices]

        x, y = zip(*convex_points)
        ax.plot(x, y, marker='none', color="#00FFFF")
        ax.plot([x[0], x[-1]], [y[0], y[-1]], marker='none', color="#00FFFF")

    else:
        qx, qy = zip(*quantpoints)
        tx, ty = zip(*transformedpoints)
        ax.scatter(qx, qy, color='blue', s=10)
        ax.scatter(tx, ty, color='white', s=10)
        for (qx_i, qy_i), (tx_i, ty_i) in zip(quantpoints, transformedpoints):
            ax.plot([qx_i, tx_i], [qy_i, ty_i], color='black')
            # Calculate the midpoint
            mid_x = (qx_i + tx_i) / 2
            mid_y = (qy_i + ty_i) / 2

            # Calculate the direction of the arrow
            dx = (tx_i - qx_i) / 2
            dy = (ty_i - qy_i) / 2

            # Plot the arrow at the midpoint
            arrow = patches.FancyArrowPatch((mid_x, mid_y), (mid_x + dx / 2, mid_y + dy / 2),
                                            arrowstyle='->', mutation_scale=10, color='black')
            # Add the arrow to the plot
            ax.add_patch(arrow)



    # Optionally, you can close the shape by connecting the last point to the first
    ax.set_xlim(2*th_staterange[0][0], 2*th_staterange[1][0])
    ax.set_ylim(2*v_staterange[0], 2*v_staterange[1])
    plt.show()

    env.close()
else:
    hereshwhatihavetosay()
