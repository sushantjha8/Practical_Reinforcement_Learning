import gym
import numpy as np
import torch
from torch import nn, optim
import ptan
import pygame

class CartPoleEnv(gym.Env):
    def __init__(self):
        super(CartPoleEnv, self).__init__()
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, _is_done, _, _ = self.env.step(action)
        return next_obs, reward, _is_done, _, _

class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(DQN, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = self.fc_layers(x)

        return x
    
    
class LinearAlgebraActionSelector(ptan.actions.ActionSelector):
    def __call__(self, q_values):
        # Round the output to obtain integer predictions
        #q_values = torch.round(q_values)
        print(f"Running Action selector {np.argmax(q_values)}")
        action = np.argmax(q_values)  # Corrected method name
        return action
def unpack_batch(batch):
    states, actions, rewards, dones, next_states = [], [], [], [], []
    print("running unpack")
    for experience in batch:
        states.append(experience.state)
        actions.append(experience.action)
        rewards.append(experience.reward)
        dones.append(experience[-1])  # Corrected accessing done flag
        next_states.append(experience.next_state)

    states_v = torch.tensor(states, dtype=torch.float32)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards, dtype=torch.float32)
    if dones.extend(experience.done):
        print(dones)
        # Convert 'dones' to a NumPy array before creating the PyTorch tensor
        dones_mask = torch.tensor(np.array(dones), dtype=torch.bool)

    next_states_v = torch.tensor(next_states, dtype=torch.float32)

    return states_v, actions_v, rewards_v, dones_mask, next_states_v


