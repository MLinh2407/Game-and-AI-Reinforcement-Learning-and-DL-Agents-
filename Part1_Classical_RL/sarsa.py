import random
from typing import Dict, List, Tuple, Optional, Callable

from gridworld import GridWorld
from levels import LEVELS
from config import (
    EPISODES,
    ALPHA,
    GAMMA,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_EPISODES,
    MAX_STEPS_PER_EPISODE,
    SEED,
)
from constants import UP, DOWN, LEFT, RIGHT

ACTIONS_LIST = [UP, DOWN, LEFT, RIGHT]

random.seed(SEED)


def get_q_list(Q: Dict, state) -> List[float]:
    """Return the action-value list for a state, initializing with zeros.

    Keeps Q as a plain dict mapping states to lists of action-values.
    """
    if state not in Q:
        Q[state] = [0.0 for _ in ACTIONS_LIST]
    return Q[state]


def epsilon_for_episode(i: int) -> float:
    """Compute epsilon for episode i with linear decay and clamping.

    Handles the case EPSILON_DECAY_EPISODES == 0 and clamps epsilon between
    the start and end values regardless of their ordering.
    """
    if EPSILON_DECAY_EPISODES <= 0:
        return EPSILON_END
    if i >= EPSILON_DECAY_EPISODES:
        return EPSILON_END
    ratio = i / float(EPSILON_DECAY_EPISODES)
    eps = EPSILON_START + ratio * (EPSILON_END - EPSILON_START)
    low, high = min(EPSILON_START, EPSILON_END), max(EPSILON_START, EPSILON_END)
    return max(low, min(high, eps))


def epsilon_greedy_action(Q: Dict, state, epsilon: float, rng: Optional[random.Random] = None) -> int:
    """Select an action using an epsilon-greedy policy.

    rng may be provided for deterministic testing (e.g., random.Random(seed)).
    Returns an action index (0..len(ACTIONS_LIST)-1) which matches the
    action constants (UP, DOWN, ...).
    """
    rng = rng or random
    qvals = get_q_list(Q, state)
    if rng.random() < epsilon:
        return rng.choice(ACTIONS_LIST)

    # Greedy with random tie-breaking
    maxv = max(qvals)
    max_actions = [a for a, v in enumerate(qvals) if v == maxv]
    return rng.choice(max_actions)


def train_sarsa(
    level_id: int = 1,
    episodes: int = EPISODES,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    rng: Optional[random.Random] = None,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Tuple[Dict, List[Dict]]:
    """Train a SARSA agent on the given level and return (Q, history).

    History is a list of dicts with keys: 'episode', 'steps', 'return', 'epsilon'.
    An optional progress_callback(ep, step, total_reward, epsilon) can be
    provided to receive updates (useful for UI integration).
    """
    rng = rng or random
    # Re-seed the RNG if a global SEED was provided to get reproducible runs
    if hasattr(rng, "seed"):
        rng.seed(SEED)

    env = GridWorld(LEVELS[level_id])

    Q: Dict = {}
    history: List[Dict] = []

    for ep in range(episodes):
        state = env.reset()
        epsilon = epsilon_for_episode(ep)
        action = epsilon_greedy_action(Q, state, epsilon, rng=rng)
        total_reward = 0.0

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_action(Q, next_state, epsilon, rng=rng)

            q_s = get_q_list(Q, state)
            q_next = get_q_list(Q, next_state)

            # SARSA update
            td_target = reward + (0 if done else gamma * q_next[next_action])
            td_error = td_target - q_s[action]
            q_s[action] += alpha * td_error

            state, action = next_state, next_action
            total_reward += reward

            if step % 50 == 0 and progress_callback is None:
                print(f"Ep {ep+1}/{episodes} step {step} eps {epsilon:.3f} return {total_reward:.2f}")

            if progress_callback is not None:
                progress_callback(ep + 1, step, total_reward, epsilon)

            if done:
                break

        # episode summary
        history.append({"episode": ep + 1, "steps": step, "return": total_reward, "epsilon": epsilon})
        print(f"Episode {ep+1}/{episodes} finished in {step} steps, return {total_reward:.2f}, eps {epsilon:.3f}")

    print("SARSA training completed")
    return Q, history


def greedy_action(Q: Dict, state, rng: Optional[random.Random] = None) -> int:
    """Return the greedy action (random tie-breaking supported via rng)."""
    rng = rng or random
    qvals = get_q_list(Q, state)
    maxv = max(qvals)
    max_actions = [a for a, v in enumerate(qvals) if v == maxv]
    return rng.choice(max_actions)
