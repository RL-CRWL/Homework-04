# Contributors
# 2602515   Taboka Dube
# 2541693   Wendy Maboa
# 2596852   Liam Brady
# 2333776   Refiloe Mopeloa

import gymnasium as gym
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class CliffWalker:
    def __init__(
        self, 
        env: gym.Env, 
        learning_rate: float, 
        initial_epsilon: float, 
        epsilon_decay: float, 
        final_epsilon: float, 
        discount_factor: float = 0.95
    ):
        # sets environment and initialises Q table
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # sets hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # whether we care about the future rewards
        self.discount_factor = discount_factor

        # keeps track of the training progress
        self.training_error = []

    def take_action(self, observation: int) -> int:
        # returns action taken by the agent
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[observation]))

    def update(
        self,
        observation: int,
        action: int,
        reward: float,
        terminated: bool,
        next_observation: int,
    ):
        future_q_value = (not terminated)*np.max(self.q_values[next_observation])

        # bellman equation to see what the q values should be
        target = reward + self.discount_factor*future_q_value

        # temporal difference to see how wrong the current estimate was
        td = target - self.q_values[observation][action]

        self.q_values[observation][action] = (
            self.q_values[observation][action] + self.learning_rate*td
        )

        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def train_agent(agent, env, num_episodes):
    # get observation from environment
    for episode in tqdm(range(num_episodes)):
        observation, info = env.reset()
        done = False

        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.take_action(observation)

            # Take action and observe result
            next_observation, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(observation, action, reward, terminated, next_observation)

            # Move to next state
            done = terminated or truncated
            observation = next_observation

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()
    return agent

def test_agent(agent, env, num_episodes=1000):
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.take_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

def main():

    # Training hyperparameters
    learning_rate = 0.5        # How fast to learn (higher = faster but less stable)
    num_episodes = 100_000        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (num_episodes / 4)  # Reduce exploration over time
    final_epsilon = 0.01         # Always keep some exploration

    # Create environment and agent
    env = gym.make('CliffWalking-v0')
    agent = CliffWalker(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    agent = train_agent(agent, env, num_episodes)
    test_agent(agent, env)

if __name__ == "__main__":
    main()