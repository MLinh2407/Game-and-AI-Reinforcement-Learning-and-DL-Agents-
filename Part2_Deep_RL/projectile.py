import numpy as np
import config

# Handles bullet movement and collisions
class Projectile:
    def __init__(self, x, y, vx, vy, angle, damage=config.BULLET_DAMAGE):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([vx, vy], dtype=np.float32)
        self.angle = angle
        self.damage = damage
    
    # Update projectile position
    def update(self):
        self.pos += self.vel
    
    # Check if projectile is out of bounds
    def is_out_of_bounds(self, width, height):
        return (self.pos[0] < 0 or self.pos[0] > width or
                self.pos[1] < 0 or self.pos[1] > height)
    
    # Check collision with a position and radius
    def collides_with(self, pos, radius):
        distance = np.linalg.norm(self.pos - pos)
        return distance < (radius + config.BULLET_COLLISION_RADIUS)