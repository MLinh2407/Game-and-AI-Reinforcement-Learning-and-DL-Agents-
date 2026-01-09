import random
import numpy as np
import pickle
import os
import config
from constants import ACTIONS

# Base class shared by Q-Learning and SARSA agents
class BaseAgent:
    def __init__(self, epsilon_decay_episodes, intrinsic_reward=False, beta=0.1):
        self.Q = {}
        self.actions = list(ACTIONS.keys())
        self.episode = 0
        self.step_count = 0
        self.epsilon_decay_episodes = epsilon_decay_episodes

    # Intrinsic reward settings
        self.use_intrinsic_reward = intrinsic_reward
        self.beta = beta 
        self.state_visit_count = {} 
        self.total_state_visits = {} 
        
    # Ensure that a state exists in the Q-table
    def ensure_state(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(len(self.actions))

    # Compute the current epsilon value for epsilon greedy exploration
    def epsilon(self):
        if self.episode >= self.epsilon_decay_episodes:
            return config.EPSILON_END
        decay = (config.EPSILON_START - config.EPSILON_END) / self.epsilon_decay_episodes
        return config.EPSILON_START - decay * self.episode

    def select_action(self, state):
        self.ensure_state(state)

        # Exploration
        if random.random() < self.epsilon():
            return random.choice(self.actions)

        # Exploitation
        q_vals = self.Q[state]
        max_q = np.max(q_vals)
        
        # Random tie-breaking when multiple actions have equal value
        best = [a for a in self.actions if q_vals[a] == max_q]
        return random.choice(best)

    def compute_intrinsic_reward(self, state):
        if not self.use_intrinsic_reward:
            return 0.0
        
        # Increment visit count for this state in current episode
        if state not in self.state_visit_count:
            self.state_visit_count[state] = 0
        self.state_visit_count[state] += 1
        
        # Track total visits across all episodes
        if state not in self.total_state_visits:
            self.total_state_visits[state] = 0
        self.total_state_visits[state] += 1
        
        # Number of times the current state has been visited
        n_s = self.state_visit_count[state]
        
        intrinsic_reward = self.beta / np.sqrt(n_s)
        return intrinsic_reward

    def new_episode(self):
        self.episode += 1
        self.step_count = 0
        self.state_visit_count = {}
        
    # Save the Q-table, training progress and exploration related parameters
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "Q": self.Q,
                "episode": self.episode,
                "intrinsic_reward": self.intrinsic_reward,
                "beta": self.beta,
                "total_state_visits": self.total_state_visits
            }, f)

    # Load a previously saved Q-table and restore training state
    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.Q = data["Q"]
            self.episode = data["episode"]
            if "intrinsic_reward" in data:
                self.intrinsic_reward = data["intrinsic_reward"]
                self.beta = data.get("beta", 0.1)
                self.total_state_visits = data.get("total_state_visits", {})

class QLearningAgent(BaseAgent):
    def update(self, state, action, env_reward, next_state):
        self.ensure_state(state)
        self.ensure_state(next_state)

        intrinsic_reward = self.compute_intrinsic_reward(state)
        total_reward = env_reward + intrinsic_reward
        # Best possible future value from next state
        best_next = np.max(self.Q[next_state])
        
        # Temporal Difference update
        self.Q[state][action] += config.ALPHA * (
            total_reward + config.GAMMA * best_next - self.Q[state][action]
        )
        self.step_count += 1
        return intrinsic_reward

class SarsaAgent(BaseAgent):
    def update(self, state, action, env_reward, next_state, next_action):
        self.ensure_state(state)
        self.ensure_state(next_state)

        intrinsic_reward = self.compute_intrinsic_reward(state)
        total_reward = env_reward + intrinsic_reward
        
        # TD target uses the next action actually chosen
        td_target = total_reward + config.GAMMA * self.Q[next_state][next_action]
        
        # Temporal Difference update
        self.Q[state][action] += config.ALPHA * (
            td_target - self.Q[state][action]
        )
        self.step_count += 1
        return intrinsic_reward