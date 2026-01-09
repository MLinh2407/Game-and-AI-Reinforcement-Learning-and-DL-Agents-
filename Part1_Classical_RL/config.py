EPISODES_PER_LEVEL = {
    0: 800,    # Apples only
    1: 1000,   # Fire hazard
    2: 1200,   # Apples + Key + Chest
    3: 1500,   # Fire + Apples + key + Chest
    4: 2000,   # Monster
    5: 2500,   # Multiple monsters
    6: 2500,   # Intrinsic reward
}
DEFAULT_EPISODES = 1500

ALPHA = 0.2 
GAMMA = 0.95  

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05

DEATH_PENALTY = -1
MAX_STEPS_PER_EPISODE = 500
FPS_VISUAL = 30    
FPS_FAST = 240  
SEED = 42