"""
Particle effects system
Handles explosions, trails, and other visual effects
"""

import numpy as np
import config


class Particle:
    def __init__(self, x, y, vx, vy, color, life):
        self.pos = np.array([x, y], dtype=np.float32)
        self.vel = np.array([vx, vy], dtype=np.float32)
        self.color = color
        self.life = life
        self.max_life = life

    # Update particle position and lifetime
    def update(self):
        self.life -= 1
        self.pos += self.vel
        self.vel *= 0.95  # Friction

    def is_alive(self):
        return self.life > 0

    # Get alpha value based on remaining life
    def get_alpha(self):
        return int(255 * (self.life / self.max_life))

    # Get size based on remaining life
    def get_size(self):
        return max(1, int(3 * (self.life / self.max_life)))


class ParticleSystem:
    def __init__(self):
        self.particles = []

    # Update all particles and remove dead ones
    def update(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for particle in self.particles:
            particle.update()

    # Create an explosion effect
    def create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1, 4)
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            life = np.random.randint(config.PARTICLE_MIN_LIFE, config.PARTICLE_MAX_LIFE)

            particle = Particle(pos[0], pos[1], vx, vy, color, life)
            self.particles.append(particle)

    # Create a thrust trail particle
    def thrust_particle(self, pos, vel):
        vx = -vel[0] * 0.5 + np.random.randn() * 0.5
        vy = -vel[1] * 0.5 + np.random.randn() * 0.5

        particle = Particle(
            pos[0], pos[1], vx, vy, config.COLOR_THRUST, config.PARTICLE_THRUST_LIFE
        )
        self.particles.append(particle)

    # Create explosion for destroyed enemy
    def enemy_explosion(self, pos):
        self.create_explosion(
            pos, config.COLOR_EXPLOSION_ENEMY, config.PARTICLE_EXPLOSION_COUNT_MEDIUM
        )

    # Create explosion for destroyed spawner
    def spawner_explosion(self, pos):
        self.create_explosion(
            pos, config.COLOR_EXPLOSION_SPAWNER, config.PARTICLE_EXPLOSION_COUNT_LARGE
        )

    # Create explosion for player death
    def player_explosion(self, pos):
        self.create_explosion(
            pos, config.COLOR_EXPLOSION_PLAYER, config.PARTICLE_EXPLOSION_COUNT_PLAYER
        )

    # Create effect for completing a phase
    def phase_complete_effect(self, pos):
        # Use spawner explosion color and phase particle count so
        # phase completion looks like a big spawner-style blast.
        self.create_explosion(
            pos, config.COLOR_EXPLOSION_SPAWNER, config.PARTICLE_EXPLOSION_COUNT_PHASE
        )

    # Create effect when enemy spawns
    def spawn_effect(self, pos):
        self.create_explosion(pos, config.COLOR_SPAWN, config.PARTICLE_SPAWN_COUNT)

    # Create small hit effect
    def hit_effect(self, pos, color):
        self.create_explosion(pos, color, config.PARTICLE_EXPLOSION_COUNT_SMALL)

    def clear(self):
        self.particles = []

    def get_particles(self):
        return self.particles