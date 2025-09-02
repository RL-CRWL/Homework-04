import gymnasium as gym
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class CliffWalker:
    def __init__(
        self, 
        env: gym.Env, 
        alpha: float,
        epsilon: float,
        lambda_v: float,
        gamma: float
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.alpha = alpha # learning rate
        self.epsilon = epsilon # exploration rate
        self.lambda_v = lambda_v # eligibility trace decay
        self.gamma = gamma # discount factor
        self.e = defaultdict(lambda: np.zeros(self.env.action_space.n)) # eligibility trace

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = int(np.argmax(self.q_values[next_state]))
        td_target = reward + self.gamma * self.q_values[next_state][best_next_action]
        td_delta = td_target - self.q_values[state][action]
        self.q_values[state][action] += self.alpha * td_delta

def main():
    env = gym.make("CliffWalking-v0")
    lambda_v = 0.5
    agent = CliffWalker(
        env,
        alpha=0.1,
        epsilon=0.1,
        lambda_v=lambda_v,
        gamma=1.0,
    )
    

if __name__ == "__main__":
    main()