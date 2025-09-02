# Contributors
# 2602515   Taboka Dube
# 2541693   Wendy Maboa
# 2596852   Liam Brady
# 2333776   Refiloe Mopeloa

import gymnasium as gym
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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
        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * self.q_values[next_observation][next_action]
        
        # TD error
        td_error = td_target - self.q_values[observation][action]
        
        # updates eligibility trace for current state-action pair
        self.e_trace[observation][action] += 1
        
        # updates all q-values and decay eligibility traces
        for state in list(self.e_trace.keys()):
            for a in range(self.env.action_space.n):
                if self.e_trace[state][a] > 0:
                    # Update Q-value
                    self.q_values[state][a] += self.learning_rate * td_error * self.e_trace[state][a]
                    # Decay eligibility trace
                    self.e_trace[state][a] *= self.discount_factor * self.lambda_value

    def clean_e_trace(self):
        self.e_trace.clear()
        
    def get_value_function(self):
        rows, cols = 4, 12  # CliffWalking grid dimensions
        value_func = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                state = i * cols + j
                value_func[i, j] = np.max(self.q_values[state])
                
        return value_func

def create_animation(value_history, lambda_values, filename="value_function_evolution.gif"):
    # makes animation for runs
    fig, axes = plt.subplots(1, len(lambda_values), figsize=(15, 5))
    
    # colour scaling
    vmin = min(np.min(values) for values_list in value_history.values() for values in values_list)
    vmax = max(np.max(values) for values_list in value_history.values() for values in values_list)
    
    # plots
    ims = []
    for idx, lmbd in enumerate(lambda_values):
        im = axes[idx].imshow(value_history[lmbd][0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"λ = {lmbd} - Episode 1")
        axes[idx].set_xlabel("Column")
        axes[idx].set_ylabel("Row")
        ims.append(im)
        
        plt.colorbar(im, ax=axes[idx])
    
    fig.suptitle("Value Function Evolution (V(s) = max_a Q(s, a))")
    plt.tight_layout()
    
    def update(frame):
        for idx, lmbd in enumerate(lambda_values):
            ims[idx].set_array(value_history[lmbd][frame])
            axes[idx].set_title(f"λ = {lmbd} - Episode {frame+1}")
        return ims
    
    ani = FuncAnimation(fig, update, frames=len(value_history[lambda_values[0]]), 
                        interval=200, blit=True)
    
    # save gif
    ani.save(filename, writer=PillowWriter(fps=5), dpi=100)
    plt.close()
    print(f"Animation saved as {filename}")

def main():
    num_episodes = 200
    num_runs = 100
    lambda_values = [0.0, 0.3, 0.5]
    env = gym.make("CliffWalking-v0")

    # stores returns for each lambda, run, and episode
    all_returns = {lmbd: np.zeros((num_runs, num_episodes)) for lmbd in lambda_values}
    
    # store for animation
    value_history = {lmbd: [] for lmbd in lambda_values}
    
    # Running once for the animation
    print("Running single episode for animation...")
    for idx, lambda_val in enumerate(lambda_values):
        agent = CliffWalkerAgent(
            env=env,
            learning_rate=0.5,
            epsilon_value=0.1,
            discount_factor=1.0,
            lambda_value=lambda_val
        )
        for episode in tqdm(range(num_episodes), desc=f"λ={lambda_val}"):
            observation, info = env.reset()
            agent.clean_e_trace()
            action = agent.take_action(observation)
            done = False

            while not done:
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if not done:
                    next_action = agent.take_action(next_observation)
                    agent.update(observation, action, next_observation, next_action, reward, done)
                    observation = next_observation
                    action = next_action
                else:
                    agent.update(observation, action, next_observation, None, reward, done)
            
            # stores value function after each episode
            value_history[lambda_val].append(agent.get_value_function())
    
    truncated_history = {lmbd: value_history[lmbd] for lmbd in lambda_values}
    create_animation(truncated_history, lambda_values)
    
    # running multiple times for the graph
    print("Running multiple episodes for statistics...")
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

    # average returns over runs for each episode and lambda
    avg_returns = {lmbd: np.mean(all_returns[lmbd], axis=0) for lmbd in lambda_values}
    std_returns = {lmbd: np.std(all_returns[lmbd], axis=0) for lmbd in lambda_values}

    # combined plot
    plt.figure(figsize=(12, 8))
    colors = ['red', 'green', 'blue']
    
    for idx, lmbd in enumerate(lambda_values):
        episodes = np.arange(1, num_episodes + 1)
        plt.plot(episodes, avg_returns[lmbd], label=f"λ={lmbd}", color=colors[idx], linewidth=2)
        plt.fill_between(episodes, 
                         avg_returns[lmbd] - std_returns[lmbd], 
                         avg_returns[lmbd] + std_returns[lmbd], 
                         alpha=0.2, color=colors[idx])
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Return", fontsize=12)
    plt.legend(fontsize=12)
    plt.title("SARSA(λ) on CliffWalking-v0 (Average over 100 runs with std deviation)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig("sarsa_lambda_results.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()