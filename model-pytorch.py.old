from __future__ import annotations

import random


import inv_pend_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import os
import gymnasium as gym
import time
import sys

plt.rcParams["figure.figsize"] = (10, 5)




class Policy_Network(nn.Module):

    
    
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = 16  # Nothing special with 16, feel free to change
        hidden_space2 = 32  # Nothing special with 32, feel free to change

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specxfific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs



class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int, device):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards
        self.device = device
        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]), device=self.device)
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)


        action = action.cpu()
        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
    def save(self, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)
    def load(self, checkpoint_path):
        self.net.load_state_dict(torch.load(checkpoint_path))

        
if torch.cuda.is_available():
    print("using CUDA")
    device = torch.device("cuda:0")
    x = torch.ones(1, device=device)
elif torch.backends.mps.is_available():
    print("using MPS")
    device = torch.device("mps")
    x = torch.ones(1, device=device)
else:
    print("using CPU")
    device = torch.device("cpu")
    x = torch.ones(1, device=device)

# Create and wrap the environment
env = gym.make("inv_pend_env/inv_pendulum_v0")
#env = gym.make("InvertedPendulum-v4")

wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

total_num_episodes = int(5e5)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)

obs_space_dims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
action_space_dims = env.action_space.shape[0]
rewards_over_seeds = []

agent = REINFORCE(obs_space_dims, action_space_dims, device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent.net.to(device)


training = "train" in sys.argv
try:
    agent.load(f"checkpoints/model-pytorch.pth")
    print("Loaded checkpoint")
except FileNotFoundError:
    print("No checkpoint found")
    training = True
# set seed
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

if training:

    reward_over_episodes = []

    for episode in range(total_num_episodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            reward = torch.tensor(reward, device=device)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrapped_env.return_queue[-1])
        agent.update()

        if episode % 1000 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    rewards_over_seeds.append(reward_over_episodes)

    os.makedirs("checkpoints", exist_ok=True)
    agent.save(f"checkpoints/model-pytorch.pth")


input("PRESS ENTER TO CONTINUE.")

env = gym.make("inv_pend_env/inv_pendulum_v0", render_mode="human")
obs, info = wrapped_env.reset(seed=seed)

done = False
while not done:
    action = agent.sample_action(obs)

    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print("Reward: " + str(reward))

    done = terminated or truncated



rewards_to_plot = [[reward[0] for reward in rewards] for rewards in rewards_over_seeds]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
custom_theme = {
    'style': 'darkgrid',
    'context': 'talk',
    'palette': 'rainbow'
}
sns.set_theme(**custom_theme)
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for Inverted Pendulum"
)

plt.show()
