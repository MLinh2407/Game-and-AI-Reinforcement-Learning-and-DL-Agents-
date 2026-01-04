import random
from collections import defaultdict

from gridworld import GridWorld
from levels import LEVELS
from config import EPISODES, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY_EPISODES, MAX_STEPS_PER_EPISODE, SEED
from constants import UP, DOWN, LEFT, RIGHT

ACTIONS_LIST = [UP, DOWN, LEFT, RIGHT]

random.seed(SEED)


def get_q_list(Q, state):
    """Return the action-value list for state, initializing if necessary."""
    if state not in Q:
        Q[state] = [0.0 for _ in ACTIONS_LIST]
    return Q[state]


def epsilon_for_episode(i):
    # Linear decay from start to end over EPSILON_DECAY_EPISODES
    if i >= EPSILON_DECAY_EPISODES:
        return EPSILON_END
    ratio = i / float(EPSILON_DECAY_EPISODES)
    eps = EPSILON_START + ratio * (EPSILON_END - EPSILON_START)
    return max(EPSILON_END, eps)


def epsilon_greedy_action(Q, state, epsilon):
    qvals = get_q_list(Q, state)
    if random.random() < epsilon:
        return random.choice(ACTIONS_LIST)

    # Greedy with random tie-breaking
    maxv = max(qvals)
    max_actions = [a for a, v in enumerate(qvals) if v == maxv]
    return random.choice(max_actions)


def train_sarsa(level_id=1, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA):
    """Train a SARSA agent on the given level and return Q-values.

    Non-visual: prints per-episode summary only. UI/visualization should be
    implemented in the caller (e.g., `main.py`) if live HUD is required.
    """
    random.seed(SEED)
    env = GridWorld(LEVELS[level_id])

    Q = {}

    for ep in range(episodes):
        state = env.reset()
        epsilon = epsilon_for_episode(ep)
        action = epsilon_greedy_action(Q, state, epsilon)
        total_reward = 0.0

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_action(Q, next_state, epsilon)

            q_s = get_q_list(Q, state)
            q_next = get_q_list(Q, next_state)

            # SARSA update
            td_target = reward + (0 if done else gamma * q_next[next_action])
            td_error = td_target - q_s[action]
            q_s[action] += alpha * td_error

            state, action = next_state, next_action
            total_reward += reward

            if step % 50 == 0:
                print(f"Ep {ep+1}/{episodes} step {step} eps {epsilon:.3f} return {total_reward:.2f}")

            if done:
                break

        # episode summary
        print(f"Episode {ep+1}/{episodes} finished in {step} steps, return {total_reward:.2f}, eps {epsilon:.3f}")

    print("SARSA training completed")
    return Q


def greedy_action(Q, state):
    qvals = get_q_list(Q, state)
    maxv = max(qvals)
    max_actions = [a for a, v in enumerate(qvals) if v == maxv]
    return random.choice(max_actions)
