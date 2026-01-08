import random
import numpy as np
import pickle
import os
import config
from constants import ACTIONS

class BaseAgent:
    """
    Base class shared by Q-Learning and SARSA agents.

    This class handles:
    - Q-table storage
    - Epsilon-greedy action selection
    - Episode tracking
    - Saving/loading learned Q-values
    """
    def __init__(self):
        self.Q = {}
        self.actions = list(ACTIONS.keys())
        self.episode = 0
        self.step_count = 0
        self.state_visits = {}

    # Ensure that a state exists in the Q-table
    def ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))

    # Compute the current epsilon value for epsilon-greedy exploration
    def epsilon(self):
        if self.episode >= config.EPSILON_DECAY_EPISODES:
            return config.EPSILON_END
        decay = (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY_EPISODES
        return config.EPSILON_START - decay * self.episode

    def select_action(self, state):
        self.ensure_state(state)
        eps = self.epsilon()

        # Exploration
        if random.random() < eps:
            return random.choice(self.actions)

        # Exploitation
        q_vals = self.Q[state]
        max_q = np.max(q_vals)
        
        # Random tie-breaking when multiple actions have equal value
        best = [a for a in self.actions if q_vals[a] == max_q]
        return random.choice(best)

    def new_episode(self):
        self.episode += 1
        self.step_count = 0
        self.state_visits = {}
        
    # Save the Q-table and episode counter
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "Q": self.Q,
                "episode": self.episode
            }, f)

    # Load a previously saved Q-table and episode counter
    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.Q = data["Q"]
            self.episode = data["episode"]

class QLearningAgent(BaseAgent):
    def update(self, state, action, reward, next_state):
        self.ensure_state(state)
        self.ensure_state(next_state)

        self.state_visits[state] = self.state_visits.get(state, 0) + 1
        intrinsic_reward = config.INTRINSIC_REWARD_STRENGTH / np.sqrt(self.state_visits[state] + 1)
        total_reward = reward + intrinsic_reward

        # Best possible future value from next state
        best_next = np.max(self.Q[next_state])
        
        # Temporal Difference update
        self.Q[state][action] += config.ALPHA * (
            total_reward + config.GAMMA * best_next - self.Q[state][action]
        )
        self.step_count += 1

class SarsaAgent(BaseAgent):
    def update(self, state, action, reward, next_state, next_action):
        self.ensure_state(state)
        self.ensure_state(next_state)

        self.state_visits[state] = self.state_visits.get(state, 0) + 1
        intrinsic_reward = config.INTRINSIC_REWARD_STRENGTH / np.sqrt(self.state_visits[state] + 1)
        total_reward = reward + intrinsic_reward

        # TD target uses the next action actually chosen
        td_target = total_reward + config.GAMMA * self.Q[next_state][next_action]
        
        # Temporal Difference update
        self.Q[state][action] += config.ALPHA * (
            td_target - self.Q[state][action]
        )
        self.step_count += 1
