import gym
import collections

reward_bank = collections.defaultdict(float)
tansition = collections.defaultdict(collections.Counter)
value = collections.defaultdict(float)

ENV_NAME = "FrozenLake-v1"

GAMA = 0.9

TEST_EPISODE = 20
env = gym.make(ENV_NAME, render_mode="human")

state = env.reset()


def Q(s, a):
    t_c = tansition[(state, action)]
    total = sum(t_c.values())

    action_value = 0.0
    for tgt_state, count in t_c.items():
        reward = reward_bank[(s, a, tgt_state)]
        bellman_value = reward + 0.9 * value[tgt_state]
        # print(f"Reward : {reward}   Bellmen_value :{bellman_value}")
        action_value += (count / total) * bellman_value
        print(f"Reward{reward} Action_value:{action_value}")
    return action_value


def select_action(state):
    best_action, best_value = None, None
    for action in range(env.action_space.n):
        print(f"env action_space {action}")
        action_value = Q(state, action)
        if best_value is None or best_value < action_value:
            print(f"yes {action_value}")
            best_value = action_value
            best_action = action
    value[state] = best_value
    return best_action


# random play
# for i in range(1000):
#     action = env.action_space.sample()

#     try:
#         st, _ = state
#     except:
#         st = state

#     n_state, rd, is_done, _, _ = env.step(action)
#     print(f"Reward {rd} n state {n_state} action {action}")
#     reward_bank[(st, action, n_state)] = rd
#     tansition[(st, action)][n_state] += 1
#     state = env.reset() if is_done else n_state


# initi game
while True:
    # reward_bank = collections.defaultdict(float)
    # tansition = collections.defaultdict(collections.Counter)
    # value = collections.defaultdict(float)
    state = env.reset()
    for i in range(100):
        try:
            st, _ = state
        except:
            st = state
        try:
            print(tansition[(state, action)])
            action = select_action(state)
            n_state, rd, is_done, _, _ = env.step(action)
            print(f"Reward {rd} n state {n_state} action {action}")
            reward_bank[(st, action, n_state)] = rd
            tansition[(st, action)][n_state] += 1
            state = env.reset() if is_done else n_state

        except:
            action = env.action_space.sample()

            n_state, rd, is_done, _, _ = env.step(action)
            print(f"Reward {rd} n state {n_state} action {action}")
            reward_bank[(st, action, n_state)] = rd
            tansition[(st, action)][n_state] += 1
            state = env.reset() if is_done else n_state


# print(reward)

# print()
