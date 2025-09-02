# Contributors
# 2602515   Taboka Dube
# 2541693   Wendy Maboa
# 2596852   Liam Brady
# 2333776   Refiloe Mopeloa

import gymnasium as gym
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

class SarsaLambdaAgent:
    def __init__(
        self, 
        env: gym.Env, 
        learning_rate: float, 
        epsilon: float,
        discount_factor: float,
        lambd: float
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.lambd = lambd
        
        # Eligibility traces
        self.eligibility_traces = defaultdict(lambda: np.zeros(self.env.action_space.n))
        
        # For tracking
        self.episode_returns = []
        self.value_function_history = []

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[state]))

    def update_eligibility_traces(self, state, action):
        # Decay all eligibility traces
        for s in self.eligibility_traces:
            self.eligibility_traces[s] *= self.discount_factor * self.lambd
        
        # Increment eligibility trace for current state-action pair
        self.eligibility_traces[state][action] += 1

    def learn(self, state, action, reward, next_state, next_action, done):
        # Calculate TD error
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * self.q_values[next_state][next_action]
        
        td_error = target - self.q_values[state][action]
        
        # Update eligibility traces
        self.update_eligibility_traces(state, action)
        
        # Update Q-values for all states based on eligibility traces
        for s in list(self.eligibility_traces.keys()):
            for a in range(self.env.action_space.n):
                if self.eligibility_traces[s][a] > 0:
                    self.q_values[s][a] += self.learning_rate * td_error * self.eligibility_traces[s][a]

    def reset_eligibility_traces(self):
        self.eligibility_traces = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def get_value_function(self):
        # Get the maximum Q-value for each state
        rows, cols = 4, 12
        value_func = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                state = i * cols + j
                value_func[i, j] = np.max(self.q_values[state])
        return value_func

def run_sarsa_lambda(env, lambd, num_episodes=200, num_runs=100):
    # For storing results across runs
    all_returns = []
    
    for run in range(num_runs):
        agent = SarsaLambdaAgent(
            env=env,
            learning_rate=0.5,
            epsilon=0.1,
            discount_factor=1.0,  # Changed to 1.0 as per CliffWalking standard
            lambd=lambd
        )
        
        returns = []
        value_history = []  # Store value function history for this run
        
        for episode in range(num_episodes):
            # Reset eligibility traces at the start of each episode
            agent.reset_eligibility_traces()
            
            state, info = env.reset()
            action = agent.get_action(state)
            
            episode_return = 0
            done = False
            
            while not done:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                next_action = agent.get_action(next_state) if not done else None
                
                # Learn from the experience
                agent.learn(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                episode_return += reward
            
            returns.append(episode_return)
            
            # Record the value function after each episode for the first run
            if run == 0:
                value_func = agent.get_value_function()
                value_history.append(value_func.copy())
        
        all_returns.append(returns)
        
        # For the first run, save the value function history
        if run == 0:
            agent.value_function_history = value_history
            agent.episode_returns = returns
    
    # Calculate average returns across all runs
    avg_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)
    
    return agent, avg_returns, std_returns

def main():
    env = gym.make('CliffWalking-v0')
    
    lambdas = [0, 0.3, 0.5]
    num_episodes = 200
    num_runs = 100
    
    # For storing results
    all_avg_returns = []
    all_std_returns = []
    agents = []
    
    # Run SARSA(λ) for each λ value
    for lambd in lambdas:
        print(f"Running SARSA(λ) with λ = {lambd}")
        agent, avg_returns, std_returns = run_sarsa_lambda(env, lambd, num_episodes, num_runs)
        agents.append(agent)
        all_avg_returns.append(avg_returns)
        all_std_returns.append(std_returns)
    
    # Plot the average returns with error bars
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']
    
    for i, lambd in enumerate(lambdas):
        plt.plot(all_avg_returns[i], color=colors[i], label=f"λ = {lambd}")
        plt.fill_between(
            range(num_episodes),
            all_avg_returns[i] - all_std_returns[i],
            all_avg_returns[i] + all_std_returns[i],
            color=colors[i],
            alpha=0.2
        )
    
    plt.xlabel('Episode')
    plt.ylabel('Average Return')
    plt.title('SARSA(λ) Performance on CliffWalking')
    plt.legend()
    plt.grid(True)
    plt.savefig('sarsa_lambda_comparison.png')
    plt.show()
    
    # Print final results
    for i, lambd in enumerate(lambdas):
        print(f"λ = {lambd}: Final Average Return = {all_avg_returns[i][-1]:.2f} ± {all_std_returns[i][-1]:.2f}")

if __name__ == "__main__":
    main()