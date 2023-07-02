import gym
import collections
import numpy as np


ENV_NAME = "FrozenLake-v1"

GAMA = 0.9

TEST_EPISODE = 20


# # print()
class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME, render_mode="human", is_slippery=True)
        self.state, _ = self.env.reset()
        self.reward_bank = collections.defaultdict(float)
        self.tansition = collections.defaultdict(collections.Counter)
        self.value = collections.defaultdict(float)

    def Q(self, p, s, a):
        (ts, ta) = np.shape(p)
        reward = 0
        action_value = 0.0
        for details in p:
            s_ = details[1]
            p = details[0]
            r = details[2]
            reward += float(r)
            bellman_value = reward + 0.9 * self.value[s_]
            action_value += p * bellman_value
        return action_value

    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()

        while True:
            action = self.select_action(state)
            n_state, reward, is_done, _, prob = self.env.step(action)
            self.reward_bank[(state, action, n_state)] = reward
            self.tansition[(state, action)][n_state] += 1
            total_reward += reward
            if is_done:
                self.env.reset()
                break
            state = n_state
        return total_reward

    def value_itrations(self):
        for s in range(self.env.observation_space.n):
            Vs = []
            for a in range(self.env.action_space.n):
                P = np.array(self.env.env.P[s][a])
                Vs.append(self.Q(P, s, a))

            self.value[s] = max(Vs)

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            p = self.env.env.P[state][action]
            p = np.array(p)
            action_value = self.Q(p, state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        self.value[state] = best_value
        return best_action

    def random_play(self, count):
        # t = 1
        for i in range(count):
            action = self.env.action_space.sample()
            n_state, reward, is_done, _, info = self.env.step(action)
            self.tansition[(self.state, action)][n_state] += 1
            self.reward_bank[(self.state, action, n_state)] = reward
            if is_done:
                self.state, _ = self.env.reset()

            else:
                self.state = n_state


agent = Agent()

i = 0
b_r = 0.0
reward = 0
while True:
    agent.random_play(100)
    TEST_EPISODE = 20
    agent.value_itrations()
    print(f"Value of state {i} ...........................\n{agent.value}")
    i += 0
    for _ in range(TEST_EPISODE):
        reward += agent.play_episode(t_env)
        reward /= TEST_EPISODE
    if reward > 0.8:
        print(f"Done with Final reard {reward}")
        break
