import numpy as np
import config

class Spawner:
    def __init__(self, x, y, phase=1):
        self.pos = np.array([x, y], dtype=np.float32)
        self.health = config.SPAWNER_HEALTH + phase * config.SPAWNER_HEALTH_INCREASE_PER_PHASE
        self.max_health = self.health
        self.phase = phase
        
        # Spawning
        self.spawn_timer = 0
        self.spawn_rate = max(
            config.SPAWNER_SPAWN_RATE - phase * config.SPAWNER_SPAWN_RATE_DECREASE,
            config.SPAWNER_MIN_SPAWN_RATE
        )
    
    # Update spawner timer
    def update(self):
        self.spawn_timer += 1
    
    # Check if spawner should spawn an enemy
    def should_spawn(self):
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_timer = 0
            return True
        return False
    
    def take_damage(self, damage):
        self.health -= damage
        return self.health > 0
    
    def is_alive(self):
        return self.health > 0
    
    # Get health as a ratio 
    def health_ratio(self):
        return self.health / self.max_health
    
    # Check collision with a position and radius
    def collides_with(self, pos, radius):
        distance = np.linalg.norm(self.pos - pos)
        return distance < (radius + config.SPAWNER_COLLISION_RADIUS)