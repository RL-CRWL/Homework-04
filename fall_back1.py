import gymnasium as gym
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class CliffWalkerAgent:
    def __init__(
        self, 
        env: gym.Env, 
        learning_rate: float, 
        epsilon_value: float, 
        discount_factor: float, 
        lambda_value: float):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon_value = epsilon_value
        self.discount_factor = discount_factor
        self.lambda_value = lambda_value
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.e_trace = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def take_action(self, state):
        # take a random action if less than epsilon
        if np.random.random() < self.epsilon_value:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))
    
    def update(self, observation, action, next_observation, next_action, reward, done):
        if done:
            td_target = reward
        else: 
            td_target = reward + self.discount_factor*self.q_values[next_observation][next_action]
        td_difference = td_target - self.q_values[observation][action]
    
        self.e_trace[observation][action] += 1

        for o in list(self.e_trace.keys()):
            for a in range(self.env.action_space.n):
                if self.e_trace[o][a] > 0:
                    self.q_values[o][a] += self.learning_rate*td_difference*self.e_trace[o][a]
                    self.e_trace[o][a] *= self.discount_factor * self.lambda_value

    def clean_e_trace(self):
        self.e_trace.clear()

def main():
    num_episodes = 200
    num_runs = 100
    lambda_values = [0.0, 0.3, 0.5]
    env = gym.make("CliffWalking-v0")

    # Store returns for each lambda, run, and episode
    all_returns = {lmbd: np.zeros((num_runs, num_episodes)) for lmbd in lambda_values}

    for run in tqdm(range(num_runs), desc="Runs"):
        for idx, lambda_val in enumerate(lambda_values):
            agent = CliffWalkerAgent(
                env=env,
                learning_rate=0.5,
                epsilon_value=0.1,
                discount_factor=1.0,
                lambda_value=lambda_val
            )
            for episode in range(num_episodes):
                observation, info = env.reset()
                agent.clean_e_trace()
                action = agent.take_action(observation)
                total_reward = 0
                done = False

                while not done:
                    next_observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                    if not done:
                        next_action = agent.take_action(next_observation)
                        agent.update(observation, action, next_observation, next_action, reward, done)
                        observation = next_observation
                        action = next_action
                    else:
                        agent.update(observation, action, next_observation, None, reward, done)

                all_returns[lambda_val][run, episode] = total_reward

    # Average returns over runs for each episode and lambda
    avg_returns = {lmbd: np.mean(all_returns[lmbd], axis=0) for lmbd in lambda_values}

    # Example: plot average returns
    plt.figure(figsize=(10, 6))
    for lmbd in lambda_values:
        plt.plot(avg_returns[lmbd], label=f"λ={lmbd}")
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.legend()
    plt.title("SARSA(λ) on CliffWalking-v0 (Average over 100 runs)")
    plt.grid(True)
    plt.show()
        

if __name__ == "__main__":
    main()