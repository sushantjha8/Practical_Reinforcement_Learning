""" 
The objective is train our policy nn Net to give best actions to get max reward to balance game pole.

"""


import torch
import numpy as np
import gym
from collections import namedtuple
import time


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


BATCH_SIZE = 16
Episode = namedtuple("Episode", field_names=["reward", "steps"])
Episode_steps = namedtuple("Episode_Step", field_names=["obs", "act"])
# Intial ;ise gym env
PERCENTILE = 73.0
env = gym.make("CartPole-v1", render_mode="human")
obs_size = env.observation_space.shape[0]
label_action = env.action_space.n
print(f"label {label_action}")
net = Net(obs_size, hidden_size=128, n_action=label_action)


def iterate_batches(env, net, batch_size=BATCH_SIZE):
    sm = torch.nn.Softmax(dim=1)
    obs = env.reset()
    obs, _ = obs
    batch = []
    episode_reward = 0.0
    episode_step = []
    while True:
        # [Cart Position,Cart Velocity,Pole Angle,Pole Angular Velocity] will be our feture from env obsercation
        obs_v = torch.FloatTensor([obs])
        action_probility = sm(net(obs_v))
        act_prob = action_probility.data.numpy()[0]
        # select random action from action probalities
        # action = np.random.choice(len(act_prob), p=act_prob)
        action = act_prob.argmax()
        next_obs, reward, _is_done, _, _ = env.step(action)
        episode_reward += reward
        step = Episode_steps(obs=obs, act=action)
        if _is_done:
            e = Episode(reward=episode_reward, steps=step)
            batch.append(e)
            episode_step = []
            next_obs = env.reset()
            next_obs, _ = next_obs
        if len(batch) == batch_size:
            yield batch
            batch = []
            break

        # print(next_obs)
        episode_step.append(step)
        obs = next_obs


def filter_batch(batch, pencentile=PERCENTILE):
    _, batch = batch
    rewards = list(map(lambda s: s.reward, batch))
    # print(f"filter reward {rewards}")
    reward_bound = np.percentile(rewards, pencentile)
    reward_mean = float(np.mean(rewards))
    # print(f"meean reawrd filter from batch {reward_mean}")
    train_obs = []
    train_act = []
    for reward, step in batch:
        if reward < reward_bound:
            continue
        train_obs.append(step.obs)
        train_act.append(step.act)
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.FloatTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


objective = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)

count = 1


for e in range(0, 20):
    itrator = iterate_batches(env, net)
    for batch in enumerate(iterate_batches(env, net)):
        train_obs_v, train_act_v, reward_bound, reward_mean = filter_batch(batch)
        optimizer.step()
        obsv = net(train_obs_v)
        loss_v = objective(obsv, train_act_v.to(dtype=torch.long))
        loss_v.backward()
        l = loss_v.item()
        optimizer.step()
        count += count
        print(f"Action reward,loss {reward_mean, l}")
        # if loss_v.item() < 0.01:
        if reward_mean > 100:
            print(f" Final Epoch {e} loss {loss_v.item(), reward_mean}")
            break
    if l < 0.0001:
        break


# torch.save(net, "model/chapter4a")
# print("----------------Inference------------------------------------------\n")
# sm = torch.nn.Softmax(dim=1)
# obs, _ = env.reset()
# total_reward = 0
# rl = []
# rm = 0
# while True:
#     obs_v = torch.FloatTensor([obs])
#     action_probility = sm(net(obs_v))
#     act_prob = action_probility.data.numpy()[0]
#     # select random action from action probalities
#     # action = np.random.choice(len(act_prob), p=act_prob)
#     action = act_prob.argmax()
#     print(f" Print Action {action}")
#     next_obs, reward, _is_done, _, _ = env.step(action)
#     print(reward)
#     if reward == 0:
#         total_reward -= 1
#     if reward == 1:
#         total_reward += 1
#     # # [Cart Position,Cart Velocity,Pole Angle,Pole Angular Velocity] will be our feture from env obsercation
#     print(
#         f"Action {action} Reward {reward} Total rerward {total_reward} Action probability ditribution {act_prob}"
#     )
#     if _is_done:
#         # rm = float(np.mean(rl))
#         # print(f" balaced at Pole Angle {rm}")
#         obs, _ = env.reset()
#         # if rm > 100:
#         #     rl = 0
#         #     total_reward = 0
#         total_reward = 0

#     if total_reward > 100:
#         break
