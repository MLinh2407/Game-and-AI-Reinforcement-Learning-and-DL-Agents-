import numpy as np

from levels import LEVELS
from gridworld import GridWorld
from rl_agents import QLearningAgent, SARSAAgent

def train_agent(agent, env, num_episodes=1000, max_steps=100):
    rewards = []
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Episode {episode}")
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        while not done and step < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
        rewards.append(total_reward)
    return rewards

def save_rewards(rewards, level, agent_name):
    with open(f'{agent_name}_level_{level}_rewards.txt', 'w') as f:
        for r in rewards:
            f.write(f"{r}\n")

def main():
    levels_to_train = [4, 5]
    agents = {
        'QLearning': QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.1),
        'SARSA': SARSAAgent(alpha=0.1, gamma=0.9, epsilon=0.1)
    }

    for level in levels_to_train:
        env = GridWorld(LEVELS[level])
        for agent_name, agent in agents.items():
            print(f"Training {agent_name} on Level {level}")
            rewards = train_agent(agent, env, num_episodes=1000)
            save_rewards(rewards, level, agent_name)
            print(f"Average reward: {np.mean(rewards)}")

if __name__ == "__main__":
    main()
