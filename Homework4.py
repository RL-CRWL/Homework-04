# Contributors
# 2602515   Taboka Dube
# 2541693   Wendy Maboa
# 2596852   Liam Brady
# 2333776   Refiloe Mopeloa

import gymnasium as gym

def main():
    env = gym.make('CliffWalking-v0', render_mode="human")
    observation, info = env.reset()
    print(f"Starting observation: {observation}")
    total_reward = 0
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done or truncated:
            print(f"Episode finished with total reward: {total_reward}")
            break
    env.close()

if __name__ == "__main__":
    main()