import gymnasium as gym

# Random agent that takes random actions from the action space
def random_agent(env, observations):
    return env.action_space.sample()

def training(env, num_episodes):
    total_reward = 0
    for episode in range(num_episodes):
        observation = env.reset()
        print("Observation: ", observation)
        done = False
        episode_reward = 0
        while not done:
            action = random_agent(env, observation)
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    print("Average reward over {} episodes: {}".format(num_episodes, total_reward / num_episodes))

def demo_agent(env, num_episodes=1):
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            env.render()
            action = random_agent(env, observation)
            observation, reward, done, _, _ = env.step(action)
        env.render()

def main():
    env = gym.make("FrozenLake-v1")
    num_episodes = 10000
    training(env, num_episodes)
    visual_env = gym.make("FrozenLake-v1", render_mode='human')
    demo_agent(visual_env, 3)

if __name__ == '__main__':
    main()


