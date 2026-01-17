import numpy as np
import config

class Enemy:
    def __init__(self, x, y, phase=1):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.angle = 0
        self.health = config.ENEMY_HEALTH
        self.max_health = config.ENEMY_HEALTH
        self.speed = config.ENEMY_SPEED + phase * config.ENEMY_SPEED_INCREASE_PER_PHASE
        self.phase = phase
    
    # Update enemy position, move toward player
    def update(self, player_pos):
        direction = player_pos - self.pos
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            # Normalize and apply speed
            direction = direction / distance
            self.vel = direction * self.speed
            self.pos += self.vel
            self.angle = np.arctan2(direction[1], direction[0])
    
    # Take damage and return True if still alive
    def take_damage(self, damage):
        self.health -= damage
        return self.health > 0
    
    # Check if enemy is alive
    def is_alive(self):
        return self.health > 0
    
    # Get health as a ratio (0-1)
    def health_ratio(self):
        return self.health / self.max_health
    
    # Check collision with a position and radius
    def collides_with(self, pos, radius):
        distance = np.linalg.norm(self.pos - pos)
        return distance < (radius + config.ENEMY_COLLISION_RADIUS)