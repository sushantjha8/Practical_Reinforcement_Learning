import random


class Envoiranment:
    def __init__(self):
        self.step_left = 15

    def get_observation(self) -> list[float]:
        return [0.0, 0.0, 0.0]

    def get_action(self) -> list[int]:
        return [0, 1]

    def is_done(self) -> bool:
        return self.step_left == 0

    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception("game over")
        if action > 0:
            self.step_left -= 1
            return 0.25
        else:
            return -0.23


class Agent:
    def __init__(self):
        self.total_reward = 0

    def step(self, envoiranment: Envoiranment):
        current_obs = envoiranment.get_observation()
        action = envoiranment.get_action()
        step_left = envoiranment.step_left
        for step in range(step_left):
            act = random.choice(action)
            reward = envoiranment.action(act)
            self.total_reward += reward
            step_left = envoiranment.step_left
            print(f" Here is current obs {current_obs}")
            print(f" Here is current act {act}")
            print(f" Here is total reward {self.total_reward}")
