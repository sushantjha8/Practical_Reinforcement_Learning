import torch
import numpy as np
import gym


class Net(torch.nn.Module):
    def __init__(self, obs_size, hidden_size, n_action):
        super(Net, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_action),
        )

    def forward(self, x):
        return self.net(x)


# Intial ;ise gym env

env = gym.make("CartPole-v1", render_mode="human")
obs_size = env.observation_space.shape[0]
label_action = env.action_space.n
net = Net(obs_size, hidden_size=32, n_action=label_action)
sm = torch.nn.Softmax(dim=1)
obs, _ = env.reset()
print(obs)
total_reward = 0
while True:
    obs_v = torch.FloatTensor([obs])
    action_probility = sm(net(obs_v))
    act_prob = action_probility.data.numpy()[0]
    # select random action from action probalities
    action = np.random.choice(len(act_prob), p=act_prob)
    next_obs, reward, _is_done, _, _ = env.step(action)
    total_reward += reward
    obs = next_obs
    # [Cart Position,Cart Velocity,Pole Angle,Pole Angular Velocity] will be our feture from env obsercation
    print(
        f"Action {action} Reward {reward} Total rerward {total_reward} Action probability ditribution {act_prob}"
    )
    if _is_done:
        print(f" balaced at Pole Angle {obs[2]}")
        break
    if total_reward >= 200:
        print("done")
        break
