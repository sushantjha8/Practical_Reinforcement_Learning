import gym
import collections
import numpy as np


ENV_NAME = "FrozenLake-v1"

GAMA = 0.9

TEST_EPISODE = 20
t_env = gym.make(ENV_NAME, render_mode="human")


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
        # print(p.shape)
        reward = 0
        action_value = 0.0
        for details in p:
            s_ = details[1]
            p = details[0]
            r = details[2]
            reward += float(r)
            # print(total)
            bellman_value = reward + 0.9 * self.value[s_]
            action_value += p * bellman_value
        #     print(
        #         f"Reward : {reward}   Bellmen_value :{bellman_value} t_state : {s_} action_value : {action_value}"
        #     )
        #     print(f"Reward{reward} Action_value:{action_value}")

        return action_value

    def play_episode(self, env):
        total_reward = 0.0
        state, _ = env.reset()

        while True:
            action = self.select_action(state)
            n_state, reward, is_done, _, prob = self.env.step(action)
            # reward = reward + 0.6 * n_state
            self.reward_bank[(state, action, n_state)] = reward
            self.tansition[(state, action)][n_state] += 1
            # print(f" Episode palyed counter {n_state}, {action},  {reward}")
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
                # [self.Q(p) for p in range(np.shape(np.array(self.env.env.P[s][a])))]
                P = np.array(self.env.env.P[s][a])
                #     (x, y) = np.shape(P)
                #     for i in range(x):
                #         s_ = P[i][1]
                #         p = P[i][0]
                #         r = P[i][2]
                Vs.append(self.Q(P, s, a))

            self.value[s] = max(Vs)

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            # print(f"env action_space {action}")
            p = self.env.env.P[state][action]
            p = np.array(p)
            action_value = self.Q(p, state, action)
            if best_value is None or best_value < action_value:
                # print(f"state :{state} Q : {action_value} Best Value : {best_value}")
                best_value = action_value
                best_action = action
        self.value[state] = best_value
        return best_action

    def random_play(self, count):
        # t = 1
        for i in range(count):
            action = self.env.action_space.sample()
            # print(f" Guruji palyed counter {self.env.observation_space.n}")
            n_state, reward, is_done, _, info = self.env.step(action)
            # if reward != 1:
            #     reward = reward + 0.6 * self.state
            # else:
            #     reward = reward + self.state

            self.tansition[(self.state, action)][n_state] += 1
            self.reward_bank[(self.state, action, n_state)] = reward
            # print(f"Reward {rd} n state {n_state} action {action} info{info}|")
            # print(f" Guruji palyed counter {n_state}, {action},  {reward}")
            # print(self.tansition)
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
    print(agent.value)
    for _ in range(TEST_EPISODE):
        reward += agent.play_episode(t_env)
        reward /= TEST_EPISODE
    if reward > 0.8:
        print(f"Done with Final reard {reward}")
        break
