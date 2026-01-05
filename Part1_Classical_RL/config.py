EPISODES = 2000
ALPHA = 0.2 
GAMMA = 0.95  

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 700

MAX_STEPS_PER_EPISODE = 400
FPS_VISUAL = 30    
FPS_FAST = 240  
# Penalty applied when agent dies stepping into fire or hit by monster
DEATH_PENALTY = -5
SEED = 42