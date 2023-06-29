import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

env = gym.make("CartPole-v1", render_mode="human")
# env.seed(1)
torch.manual_seed(1)

# Hyperparameters
learning_rate = 0.01
gamma = 0.99

# lets build Policy
# few points for making policy
"""
Policy is a Non linear model which stores 
# Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()
        self.forrward(x)

"""


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        obs_space_shape = env.observation_space.shape[0]
        print(f"observation space {obs_space_shape }")
        action_space = env.action_space.n
        num_hidden = 128
        self.l1 = nn.Linear(obs_space_shape, num_hidden, bias=False)
        self.l2 = nn.Linear(num_hidden, action_space, bias=False)

        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.reset()

    def reset(self):
        """rest your episode history and episode_reward history"""
        # Episode policy and reward history
        self.episode_actions = torch.Tensor([])
        self.episode_rewards = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1, nn.Dropout(p=0.5), nn.ReLU(), self.l2, nn.Softmax(dim=-1)
        )
        return model(x)


def predict(obs):
    obs = torch.FloatTensor(obs)
    # print(obs.size())
    action_probs = policy(obs)
    distribution = Categorical(action_probs)
    action = distribution.sample()
    # TODO append episode action to policy
    policy.episode_actions = torch.cat(
        [policy.episode_actions, distribution.log_prob(action).reshape(1)]
    )
    return action


def discount_future_reward(episode_rewards=[], gamma=0.06):
    rewards = []
    R = 0
    # Discount future rewards back to the present using gamma
    for r in episode_rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)

    # normalise rewards rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    return rewards


def update_policy():
    episode_rewards = policy.episode_rewards
    rewards = discount_future_reward(episode_rewards=episode_rewards, gamma=gamma)
    # Calculate loss
    loss = torch.sum(torch.mul(policy.episode_actions, rewards).mul(-1), -1)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.episode_rewards))
    policy.reset()


def train(episodes):
    list_score = []
    print(env.spec.reward_threshold)
    for episode in range(episodes):
        # TODO: rest Env
        obs, _ = env.reset()

        obs = torch.from_numpy(obs)
        for step in range(0, 1000):
            # TODO  get action from policy prediction
            action = predict(obs)
            env.render()
            # print(action)

            # TODO get next_obs , reward by applyinction action.item()
            next_obs, reward, _is_done, _, _ = env.step(action.item())
            # Save reward
            policy.episode_rewards.append(reward)
            obs = next_obs
            if _is_done:
                break
        # TODO update policy
        update_policy()

        list_score.append(step)
        mean_score = np.mean(list_score[-100:])

        if episode % 50 == 0:
            print(
                "Episode {}\tAverage length (last 100 episodes): {:.2f} time steps.{}".format(
                    episode, mean_score, step
                )
            )

        if mean_score > env.spec.reward_threshold:
            print(
                "Solved after {} episodes! Running average is now {}. Last episode ran to {} time steps.".format(
                    episode, mean_score, step
                )
            )
            break


policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
train(episodes=1000)
window = 50

fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
std = pd.Series(policy.reward_history).rolling(window).std()
ax1.plot(rolling_mean)
ax1.fill_between(
    range(len(policy.reward_history)),
    rolling_mean - std,
    rolling_mean + std,
    color="orange",
    alpha=0.2,
)
ax1.set_title("Episode Length Moving Average ({}-episode window)".format(window))
ax1.set_xlabel("Episode")
ax1.set_ylabel("Episode Length")

ax2.plot(policy.reward_history)
ax2.set_title("Episode Length")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Episode Length")

fig.tight_layout(pad=2)
plt.show()
