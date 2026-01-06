EPISODES = 100
ALPHA = 0.2 
GAMMA = 0.95  

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
# Gradually reduce exploration during the first 85% of training
EPSILON_DECAY_EPISODES = int(0.85 * EPISODES)

DEATH_PENALTY = -1
MAX_STEPS_PER_EPISODE = 500
FPS_VISUAL = 30    
FPS_FAST = 240  
SEED = 42