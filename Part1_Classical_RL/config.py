EPISODES_PER_LEVEL = {
    0: 600,    # Apples only
    1: 1000,   # Fire hazard
    2: 2000,   # Apples + Key + Chest
    3: 2000,   # Fire + Apples + key + Chest
    4: 2500,   # Monster
    5: 2500,   # Multiple monsters
    6: 2000,   # Intrinsic reward
}
DEFAULT_EPISODES = 1500

ALPHA = 0.1
GAMMA = 0.95  

# Intrinsic reward strength
INTRINSIC_REWARD_STRENGTH = 0.1

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.01

DEATH_PENALTY = -1
MAX_STEPS_PER_EPISODE = 500
FPS_VISUAL = 30    
FPS_FAST = 240  
SEED = 42