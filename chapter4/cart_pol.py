"""
https://www.gymlibrary.dev/environments/classic_control/cart_pole/
"""


import gym

# Intial ;ise gym env

env = gym.make("CartPole-v1", render_mode="human")
for i in range(0, 100):
    env.reset()
    # print(env.observation_space)
    total_reward = 0
    while total_reward < 200:
        next_obs, reward, _is_done, _, _ = env.step(env.action_space.sample())
        total_reward += reward
        obs = next_obs
        # [Cart Position,Cart Velocity,Pole Angle,Pole Angular Velocity] will be our feture from env obsercation
        print(next_obs)
        if _is_done:
            print("done")
            break
