import random
import numpy as np
from constants import ACTIONS

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def get_best_action(self, state):
        actions = list(ACTIONS.keys())
        q_values = [self.get_q_value(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        else:
            return self.get_best_action(state)

    def learn(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            next_action = self.get_best_action(next_state)
            target = reward + self.gamma * self.get_q_value(next_state, next_action)
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (target - current_q)
        self.set_q_value(state, action, new_q)

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state, action, value):
        self.q_table[(state, action)] = value

    def get_best_action(self, state):
        actions = list(ACTIONS.keys())
        q_values = [self.get_q_value(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        else:
            return self.get_best_action(state)

    def learn(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            next_action = self.choose_action(next_state)
            target = reward + self.gamma * self.get_q_value(next_state, next_action)
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (target - current_q)
        self.set_q_value(state, action, new_q)
