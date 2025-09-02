# Contributors
# 2602515   Taboka Dube
# 2541693   Wendy Maboa
# 2596852   Liam Brady
# 2333776   Refiloe Mopeloa


import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SarsaLambda:
    def __init__(self, env, lambda_val=0.0, alpha=0.5, gamma=1.0, epsilon=0.1):
        self.env = env
        self.lambda_val = lambda_val
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table and eligibility traces
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.Q = np.zeros((self.nS, self.nA))
        self.E = np.zeros((self.nS, self.nA))
        
        # For tracking returns and value functions
        self.returns = []
        self.value_functions = []
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state, next_action, done):
        # Calculate TD error
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state][next_action]
        
        delta = target - self.Q[state][action]
        
        # Update eligibility trace for current state-action pair
        self.E[state][action] += 1
        
        # Update all state-action pairs
        for s in range(self.nS):
            for a in range(self.nA):
                if self.E[s][a] > 0:
                    self.Q[s][a] += self.alpha * delta * self.E[s][a]
                    self.E[s][a] *= self.gamma * self.lambda_val
        
        if done:
            self.E.fill(0)
    
    def train(self, episodes):
        for episode in range(episodes):
            state, _ = self.env.reset()
            action = self.choose_action(state)
            
            total_reward = 0
            done = False
            truncated = False
            
            while not done and not truncated:
                next_state, reward, done, truncated, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                state = next_state
                action = next_action
                total_reward += reward
            
            self.returns.append(total_reward)
            
            # Store max Q values for heatmap
            max_q_values = np.max(self.Q, axis=1).reshape(4, 12)
            self.value_functions.append(max_q_values.copy())
            
            # Reset eligibility traces at the end of episode
            self.E.fill(0)
        
        return self.returns, self.value_functions

def run_experiment(lambdas, num_runs=100, num_episodes=200):
    results = {}
    
    for lambda_val in lambdas:
        print(f"Running SARSA(λ) with λ={lambda_val}")
        all_returns = []
        all_value_functions = []
        
        for run in tqdm(range(num_runs)):
            env = gym.make('CliffWalking-v1')
            agent = SarsaLambda(env, lambda_val=lambda_val)
            returns, value_functions = agent.train(num_episodes)
            all_returns.append(returns)
            
            if run == 0:  # Store value functions only for first run
                all_value_functions = value_functions
            
            env.close()
        
        results[lambda_val] = {
            'returns': np.array(all_returns),
            'value_functions': all_value_functions
        }
    
    return results

def create_animation(value_history, lambda_values, filename="value_function_evolution.gif"):
    # makes animation for runs
    fig, axes = plt.subplots(1, len(lambda_values), figsize=(15, 5))
    
    # colour scaling
    vmin = min(np.min(values) for values_list in value_history for values in values_list)
    vmax = max(np.max(values) for values_list in value_history for values in values_list)
    
    # plots
    ims = []
    for idx, lmbd in enumerate(lambda_values):
        im = axes[idx].imshow(value_history[idx][0], cmap='viridis', vmin=vmin, vmax=vmax)
        axes[idx].set_title(f"λ = {lmbd} - Episode 1")
        axes[idx].set_xlabel("Column")
        axes[idx].set_ylabel("Row")
        ims.append(im)
        
        plt.colorbar(im, ax=axes[idx])
    
    fig.suptitle("Value Function Evolution (V(s) = max_a Q(s, a))")
    plt.tight_layout()
    
    def update(frame):
        for idx, lmbd in enumerate(lambda_values):
            ims[idx].set_array(value_history[idx][frame])
            axes[idx].set_title(f"λ = {lmbd} - Episode {frame+1}")
        return ims
    
    ani = FuncAnimation(fig, update, frames=len(value_history[lambda_values[0]]), 
                        interval=200, blit=True)
    
    # save gif
    ani.save(filename, writer=PillowWriter(fps=5), dpi=100)
    plt.close()
    print(f"Animation saved as {filename}")

def plot_average_returns(results, lambdas):
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    for i, lambda_val in enumerate(lambdas):
        returns = results[lambda_val]['returns']
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)
        
        episodes = np.arange(1, len(mean_returns) + 1)
        
        plt.plot(episodes, mean_returns, label=f'λ = {lambda_val}', color=colors[i], linewidth=2)
        plt.fill_between(episodes, mean_returns - std_returns, 
                        mean_returns + std_returns, alpha=0.2, color=colors[i])
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Return', fontsize=12)
    plt.title('SARSA(λ) Performance on CliffWalking\n(Averaged over 100 runs)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sarsa_lambda_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_individual_plots(results, lambdas):
    """Create individual plots for each lambda value"""
    for lambda_val in lambdas:
        plt.figure(figsize=(10, 6))
        
        returns = results[lambda_val]['returns']
        mean_returns = np.mean(returns, axis=0)
        std_returns = np.std(returns, axis=0)
        
        episodes = np.arange(1, len(mean_returns) + 1)
        
        plt.plot(episodes, mean_returns, label=f'λ = {lambda_val}', color='blue', linewidth=2)
        plt.fill_between(episodes, mean_returns - std_returns, 
                        mean_returns + std_returns, alpha=0.3, color='blue')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Return', fontsize=12)
        plt.title(f'SARSA(λ={lambda_val}) Performance on CliffWalking\n(Averaged over 100 runs)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'sarsa_lambda_{lambda_val}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

# Main execution
if __name__ == "__main__":
    # Parameters
    lambdas = [0, 0.3, 0.5]
    num_runs = 100
    num_episodes = 200
    
    print("Starting SARSA(λ) experiment with Gymnasium...")
    print(f"Lambda values: {lambdas}")
    print(f"Number of runs: {num_runs}")
    print(f"Episodes per run: {num_episodes}")
    
    # Run the experiment
    results = run_experiment(lambdas, num_runs, num_episodes)
    
    # Create animation for first run
    print("Creating animation...")
    value_functions_list = [results[lambda_val]['value_functions'] for lambda_val in lambdas]
    create_animation(value_functions_list, lambdas)
    
    # Plot average returns
    print("Creating performance plots...")
    plot_average_returns(results, lambdas)
    create_individual_plots(results, lambdas)
    
    # Save results
    np.savez('sarsa_lambda_results.npz', 
             results=results, 
             lambdas=lambdas)
    
    print("Experiment completed successfully!")
    print("Files created:")
    print("- sarsa_lambda_animation.gif (animation of value functions)")
    print("- sarsa_lambda_performance.png (combined performance plot)")
    print("- sarsa_lambda_X_performance.png (individual performance plots)")
    print("- sarsa_lambda_results.npz (raw data)")
