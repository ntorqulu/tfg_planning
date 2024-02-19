import gymnasium as gym
import numpy as np



def main():
    env = gym.make("FrozenLake-v1")
    num_episodes = 10000

    Q_sarsa = sarsa(env, num_episodes)
    avg_reward = evaluate_policy(env, Q_sarsa, num_episodes)
    print(f"Average reward after SARSA: {avg_reward}")

    visual_env = gym.make('FrozenLake-v1', render_mode='human')
    demo_agent(visual_env, Q_sarsa, 3)


if __name__ == '__main__':
    main()