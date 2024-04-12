import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        # The value stream and the advantage stream are combined to produce the final output of the network.
        # This is done by adding the value estimates (val) to the advantages (adv) after centering the advantages by subtracting their
        # mean (adv - adv.mean(dim=1, keepdim=True)). T
        # This step helps in stabilizing the learning process by ensuring that the network can learn relative advantages of different actions
        # while still having a baseline value estimate for each state.The final output of the network is a tensor with actions_n elements, 
        # where each element represents the estimated Q-value for each action, considering both the state value and the advantages of each action.
        return val + (adv - adv.mean(dim=1, keepdim=True))