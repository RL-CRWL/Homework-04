import gymnasium as gym
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os

class SarsaLambdaAgent:
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
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_v = lambda_v
        self.gamma = gamma
        self.e = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))

    def update(self, state, action, reward, next_state, next_action, done):
        # SARSA update: use the actual next action, not the max
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * self.q_values[next_state][next_action]
        
        td_delta = td_target - self.q_values[state][action]
        
        # Update eligibility traces FIRST
        # Decay all traces
        for s in list(self.e.keys()):
            for a in range(self.env.action_space.n):
                self.e[s][a] *= self.gamma * self.lambda_v
        
        # Increment current state-action trace
        self.e[state][action] += 1
        
        # Update ALL Q-values based on eligibility traces
        for s in list(self.q_values.keys()):
            for a in range(self.env.action_space.n):
                if self.e[s][a] > 0:  # Only update if trace is non-zero
                    self.q_values[s][a] += self.alpha * td_delta * self.e[s][a]

    def reset_eligibility_traces(self):
        self.e = defaultdict(lambda: np.zeros(self.env.action_space.n))

def plot_value_function(q_values, episode, lambda_v, save_dir):
    V = np.zeros((4, 12))
    for i in range(4):
        for j in range(12):
            state = i * 12 + j
            V[i, j] = np.max(q_values[state])
    
    plt.figure(figsize=(8, 4))
    plt.imshow(V, cmap="viridis", origin="upper")
    plt.title(f"Value Function - Episode {episode+1}, λ={lambda_v}")
    plt.colorbar(label="Max Q-value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"heatmap_ep{episode+1:03d}.png"))
    plt.close()

def main():
    env = gym.make("CliffWalking-v0")
    lambda_v = 0.5
    agent = SarsaLambdaAgent(
        env=env,
        alpha=0.5,  # Changed to 0.5 as per instructions
        epsilon=0.1,
        lambda_v=lambda_v,
        gamma=1.0,
    )
    num_episodes = 200
    rewards = []

    save_dir = f"images_lambda_{lambda_v}"
    os.makedirs(save_dir, exist_ok=True)

    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.reset_eligibility_traces()
        action = agent.take_action(state)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Get next action if not done
            next_action = agent.take_action(next_state) if not done else None
            
            # Update Q-values and eligibility traces
            agent.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            total_reward += reward

        rewards.append(total_reward)
        
        # Plot value function every 10 episodes to reduce overhead
        plot_value_function(agent.q_values, episode, lambda_v, save_dir)
        
        print(f"Episode {episode+1}: Total reward = {total_reward}")

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"SARSA(λ) Learning Curve (λ={lambda_v})")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "learning_curve.png"))
    plt.close()

if __name__ == "__main__":
    main()